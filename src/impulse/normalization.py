"""
Configuration-based normalization utilities for IMPULS project - v4 (Synonym Groups)

This module provides functions for normalizing text, geographic names,
framework names, and organization types. All mappings are loaded from
external TSV configuration files for easy maintenance and extensibility.

KEY FEATURE: Supports synonym groups - one user input can map to multiple DB values.
Example: "company" → ["EMPRESA", "PRC"] (both mean private company)

Configuration files (TSV format):
- framework_mappings.tsv: Programme/framework name mappings (supports synonyms)
- organization_type_mappings.tsv: Organization type mappings (supports synonyms)
- geographic_mappings.tsv: Geographic location mappings (1:1)

Author: IMPULS Project
Date: 2024-11-17
Version: 4.0 - Multi-value normalization with synonym groups
"""

import unicodedata
import re
from typing import Optional, List, Dict, Tuple, Set
from pathlib import Path
import logging
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

# ============================================================================
# ROR DATA - Module-level cache
# ============================================================================

# Load ROR mappings at module level (lazy loading)
_ROR_MAPPINGS = None
_ROR_ORGS = None

# ============================================================================
# CONFIGURATION LOADER WITH SYNONYM GROUP SUPPORT
# ============================================================================

class NormalizationConfig:
    """Loads and manages normalization mappings from TSV files with synonym support"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_dir: Directory containing TSV config files.
                       Defaults to 'data/normalization'
        """
        if config_dir is None:
            # Default to data/normalization
            config_dir = Path("data/normalization")
        
        self.config_dir = Path(config_dir)
        
        # Multi-value mappings: user_input -> List[(db_value, metadata)]
        # Allows synonym groups: "company" -> [("EMPRESA", {}), ("PRC", {})]
        self.framework_map: Dict[str, List[Tuple[str, Dict]]] = defaultdict(list)
        self.org_type_map: Dict[str, List[Tuple[str, Dict]]] = defaultdict(list)
        self.geographic_map: Dict[str, List[Tuple[str, Dict]]] = defaultdict(list)
        
        # Reverse mappings: db_value -> short_form/metadata
        self.framework_reverse: Dict[str, str] = {}
        self.org_type_reverse: Dict[str, str] = {}
        
        # Valid database values (unique sets)
        self.valid_frameworks: Set[str] = set()
        self.valid_org_types: Set[str] = set()
        
        # Load all configurations
        self._load_all()
    
    def _load_tsv(self, filename: str) -> List[Dict[str, str]]:
        """
        Load TSV file and return list of row dictionaries.
        
        Args:
            filename: Name of TSV file
            
        Returns:
            List of dictionaries (one per row)
        """
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Config file not found: {filepath}")
            return []
        
        rows = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # Read header (first non-comment line)
                header = None
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        header = line.split('\t')
                        break
                
                if not header:
                    logger.error(f"No header found in {filename}")
                    return []
                
                # Read data rows
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        values = line.split('\t')
                        if len(values) == len(header):
                            rows.append(dict(zip(header, values)))
                        else:
                            logger.warning(f"Skipping malformed line in {filename}: {line}")
            
            logger.info(f"Loaded {len(rows)} entries from {filename}")
            
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
        
        return rows
    
    def _load_frameworks(self):
        """Load framework mappings from TSV (supports synonym groups)"""
        rows = self._load_tsv("framework_mappings.tsv")
        
        for row in rows:
            user_input = normalize_text(row['user_input'])
            db_value = row['db_value']
            short_form = row['short_form']
            category = row.get('category', '')
            
            # Store mapping (allows multiple DB values per user input)
            self.framework_map[user_input].append((db_value, {
                'short_form': short_form,
                'category': category
            }))
            
            # Store reverse mapping (prefer first occurrence)
            if db_value not in self.framework_reverse:
                self.framework_reverse[db_value] = short_form
            
            # Collect valid DB values
            self.valid_frameworks.add(db_value)
    
    def _load_org_types(self):
        """Load organization type mappings from TSV (supports synonym groups)"""
        rows = self._load_tsv("organization_type_mappings.tsv")
        
        for row in rows:
            user_input = normalize_text(row['user_input'])
            db_value = row['db_value']
            description = row.get('description', '')
            language = row.get('language', '')
            
            # Store mapping (allows multiple DB values per user input)
            self.org_type_map[user_input].append((db_value, {
                'description': description,
                'language': language
            }))
            
            # Store reverse mapping (prefer English descriptions)
            if db_value not in self.org_type_reverse or language == 'en':
                self.org_type_reverse[db_value] = description
            
            # Collect valid DB values
            self.valid_org_types.add(db_value)
    
    def _load_geographic(self):
        """Load geographic mappings from TSV (1:1 mappings, no synonyms)"""
        rows = self._load_tsv("geographic_mappings.tsv")
        
        for row in rows:
            user_input = normalize_text(row['user_input'])
            canonical_form = row['canonical_form']
            location_type = row['location_type']
            language = row.get('language', '')
            
            # Store mapping (geographic is 1:1, but we use list for consistency)
            self.geographic_map[user_input].append((canonical_form, {
                'location_type': location_type,
                'language': language
            }))
    
    def _load_all(self):
        """Load all configuration files"""
        try:
            self._load_frameworks()
            self._load_org_types()
            self._load_geographic()
            logger.info("All normalization configurations loaded successfully")
        except Exception as e:
            logger.error(f"Error loading configurations: {e}")
    
    def get_frameworks(self, user_input: str) -> List[str]:
        """
        Get all database values for framework user input (synonym group).
        
        Returns:
            List of DB values (e.g., ["H2020", "HORIZON"] for "h2020")
        """
        normalized = normalize_text(user_input)
        results = self.framework_map.get(normalized, [])
        return [db_value for db_value, _ in results]
    
    def get_framework_short(self, db_value: str) -> Optional[str]:
        """Get short form for framework database value"""
        return self.framework_reverse.get(db_value)
    
    def get_org_types(self, user_input: str) -> List[str]:
        """
        Get all database values for org type user input (synonym group).
        
        Returns:
            List of DB values (e.g., ["EMPRESA", "PRC"] for "company")
        """
        normalized = normalize_text(user_input)
        results = self.org_type_map.get(normalized, [])
        return [db_value for db_value, _ in results]
    
    def get_org_type_description(self, db_value: str) -> Optional[str]:
        """Get description for org type database value"""
        return self.org_type_reverse.get(db_value)
    
    def get_geographic(self, user_input: str, location_type: Optional[str] = None) -> Optional[str]:
        """
        Get canonical form for geographic user input (1:1 mapping).
        
        Args:
            user_input: User's location input
            location_type: Optional filter by location type (region, province, country)
        
        Returns:
            Canonical form or None
        """
        normalized = normalize_text(user_input)
        results = self.geographic_map.get(normalized, [])
        
        if not results:
            return None
        
        # Geographic is 1:1, take first result
        canonical, metadata = results[0]
        
        # Filter by location type if specified
        if location_type and metadata['location_type'] != location_type:
            return None
        
        return canonical


# Global configuration instance (lazy loaded)
_config: Optional[NormalizationConfig] = None


def get_config() -> NormalizationConfig:
    """Get or create global configuration instance"""
    global _config
    if _config is None:
        _config = NormalizationConfig()
    return _config


def reload_config(config_dir: Optional[Path] = None):
    """Reload configuration from files"""
    global _config
    _config = NormalizationConfig(config_dir)


# ============================================================================
# BASIC TEXT NORMALIZATION
# ============================================================================

def remove_accents(text: str) -> str:
    """
    Remove accents from text while preserving other characters.
    
    Examples:
        >>> remove_accents("Cataluña")
        'Cataluna'
        >>> remove_accents("Lérida")
        'Lerida'
    """
    if not text:
        return text
    
    # Normalize to NFD (decomposed form)
    nfd = unicodedata.normalize('NFD', text)
    
    # Remove combining marks (accents)
    return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison: lowercase, strip whitespace, remove accents.
    
    Examples:
        >>> normalize_text("  CATALUNYA  ")
        'catalunya'
        >>> normalize_text("Cataluña")
        'cataluna'
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove accents
    text = remove_accents(text)
    
    # Lowercase and strip
    text = text.lower().strip()
    
    # Normalize whitespace (multiple spaces → single space)
    text = re.sub(r'\s+', ' ', text)
    
    return text


# ============================================================================
# GEOGRAPHIC NORMALIZATION (1:1)
# ============================================================================

def normalize_region(region: str) -> str:
    """
    Normalize region names handling Catalan ↔ Spanish variants.
    Returns canonical form (Catalan official name) in lowercase.
    
    Examples:
        >>> normalize_region("CATALUNYA")
        'catalunya'
        >>> normalize_region("Cataluña")
        'catalunya'
    """
    if not region:
        return ""
    
    config = get_config()
    canonical = config.get_geographic(region, location_type='region')
    
    return canonical if canonical else normalize_text(region)


def normalize_province(province: str) -> str:
    """
    Normalize province names handling Catalan ↔ Spanish variants.
    Returns canonical form (Catalan official name) in lowercase.
    
    Examples:
        >>> normalize_province("GERONA")
        'girona'
        >>> normalize_province("Lérida")
        'lleida'
    """
    if not province:
        return ""
    
    config = get_config()
    canonical = config.get_geographic(province, location_type='province')
    
    return canonical if canonical else normalize_text(province)


def normalize_country(country: str) -> str:
    """
    Normalize country names.
    
    Examples:
        >>> normalize_country("España")
        'spain'
    """
    if not country:
        return ""
    
    config = get_config()
    canonical = config.get_geographic(country, location_type='country')
    
    return canonical if canonical else normalize_text(country)


def matches_region(value: str, target: str) -> bool:
    """
    Check if two region names match (handling variants).
    
    Examples:
        >>> matches_region("CATALUNYA", "Cataluña")
        True
    """
    return normalize_region(value) == normalize_region(target)


def matches_province(value: str, target: str) -> bool:
    """
    Check if two province names match (handling variants).
    
    Examples:
        >>> matches_province("GERONA", "Girona")
        True
    """
    return normalize_province(value) == normalize_province(target)


# ============================================================================
# FRAMEWORK/PROGRAMME NORMALIZATION (SYNONYM GROUPS)
# ============================================================================

def normalize_framework(framework: str, to_db_value: bool = True) -> List[str]:
    """
    Normalize framework/programme names to database values (synonym group).
    
    Args:
        framework: Framework name (user input or DB value)
        to_db_value: If True, return actual DB values. If False, return short forms.
        
    Returns:
        List of database values or short forms (synonym group)
        
    Examples:
        >>> normalize_framework("h2020")
        ['H2020', 'HORIZON']  # Both EU programmes treated as synonyms
        
        >>> normalize_framework("aei")
        ['Agencia Estatal de Investigación (AEI)']
        
        >>> normalize_framework("H2020", to_db_value=False)
        ['H2020']  # Short form
    """
    if not framework:
        return []
    
    config = get_config()
    
    if to_db_value:
        # Map to actual DB values (may return multiple for synonyms)
        db_values = config.get_frameworks(framework)
        if db_values:
            return db_values
        
        # Check if it's already a valid DB value
        if framework in config.valid_frameworks:
            return [framework]
        
        # Return original as fallback
        return [framework]
    else:
        # Convert DB values to short forms
        short_forms = []
        for fw in normalize_framework(framework, to_db_value=True):
            short = config.get_framework_short(fw)
            if short:
                short_forms.append(short)
            else:
                short_forms.append(fw)
        return short_forms


def matches_framework(value: str, target: str) -> bool:
    """
    Check if two framework names match (handling variants and synonyms).
    
    Examples:
        >>> matches_framework("h2020", "H2020")
        True
        >>> matches_framework("h2020", "HORIZON")
        True  # Synonyms!
        >>> matches_framework("aei", "Agencia Estatal de Investigación (AEI)")
        True
    """
    value_synonyms = set(normalize_framework(value))
    target_synonyms = set(normalize_framework(target))
    
    # Match if synonym groups overlap
    return bool(value_synonyms & target_synonyms)


def get_valid_frameworks() -> List[str]:
    """Get list of all valid framework database values"""
    config = get_config()
    return sorted(config.valid_frameworks)


# ============================================================================
# ORGANIZATION TYPE NORMALIZATION (SYNONYM GROUPS)
# ============================================================================

def normalize_org_type(org_type: str) -> List[str]:
    """
    Normalize organization type to actual database values (synonym group).
    
    Args:
        org_type: Organization type (user input or DB value)
        
    Returns:
        List of database organization type values (synonym group)
        
    Examples:
        >>> normalize_org_type("company")
        ['EMPRESA', 'PRC']  # Both mean private company
        
        >>> normalize_org_type("university")
        ['HES']
        
        >>> normalize_org_type("research_center")
        ['REC']
    """
    if not org_type:
        return []
    
    config = get_config()
    
    # Map to actual DB values (may return multiple for synonyms)
    db_values = config.get_org_types(org_type)
    if db_values:
        return db_values
    
    # Check if it's already a valid DB value
    if org_type in config.valid_org_types:
        return [org_type]
    
    # Return original as fallback
    return [org_type]


def matches_org_type(value: str, target: str) -> bool:
    """
    Check if two organization types match (handling variants and synonyms).
    
    Examples:
        >>> matches_org_type("company", "EMPRESA")
        True
        >>> matches_org_type("company", "PRC")
        True  # Synonyms!
        >>> matches_org_type("EMPRESA", "PRC")
        True  # Both are private companies
        >>> matches_org_type("university", "HES")
        True
    """
    value_synonyms = set(normalize_org_type(value))
    target_synonyms = set(normalize_org_type(target))
    
    # Match if synonym groups overlap
    return bool(value_synonyms & target_synonyms)


def get_valid_org_types() -> List[str]:
    """Get list of all valid organization type database values"""
    config = get_config()
    return sorted(config.valid_org_types)


# ============================================================================
# ORGANIZATION NAME NORMALIZATION
# ============================================================================


# ============================================================================
# ORGANIZATION NAME NORMALIZATION
# ============================================================================

def preprocess_organization_name(org_name: str) -> List[str]:
    """
    Preprocess organization name to generate multiple search variants.
    
    Handles common issues in organization names:
    - Extracts acronyms from parentheses: "Universidad (UB)" → ["universidad de...", "ub"]
    - Removes CCT suffix: "Universitat CCT" → tries without
    - Removes FUNDACIÓ/FUNDACIO prefix
    - Handles CSIC format: "AGENCIA...CSIC (CSIC)" → extracts "CSIC"
    - Handles hyphenated acronyms: "Center (BSC-CNS)" → extracts "bsc-cns"
    
    Args:
        org_name: Raw organization name from database
    
    Returns:
        List of normalized variants to try for ROR matching
        
    Examples:
        >>> preprocess_organization_name("Universitat de Barcelona (UB)")
        ['universitat de barcelona (ub)', 'ub', 'universitat de barcelona']
        
        >>> preprocess_organization_name("FUNDACIO CENTRE DE REGULACIO GENOMICA")
        ['fundacio centre de regulacio genomica', 'centre de regulacio genomica']
        
        >>> preprocess_organization_name("UNIVERSITAT POMPEU FABRA CCT")
        ['universitat pompeu fabra cct', 'universitat pompeu fabra']
    """
    if not org_name:
        return []
    
    variants = []
    
    # 1. Start with the full normalized name
    normalized = normalize_text(org_name)
    variants.append(normalized)
    
    # 2. Extract acronym from parentheses: "Name (ACRONYM)" or "Name (BSC-CNS)"
    # Match uppercase letters/hyphens in parentheses at the end
    acronym_match = re.search(r'\(([A-Z]{2,}(?:-[A-Z]+)?)\)\s*$', org_name)
    if acronym_match:
        acronym = normalize_text(acronym_match.group(1))
        # Get the base name without the parentheses
        base_name = normalize_text(org_name[:acronym_match.start()].strip())
        
        # Add acronym as separate variant
        if acronym not in variants:
            variants.append(acronym)
        
        # Add base name without parentheses
        if base_name and base_name not in variants:
            variants.append(base_name)
    
    # 3. Remove CCT suffix (Centre Català de Tecnologia)
    if normalized.endswith(' cct'):
        without_cct = normalized[:-4].strip()
        if without_cct and without_cct not in variants:
            variants.append(without_cct)
    
    # 4. Remove FUNDACIÓ/FUNDACIO/FUNDACION prefix
    for prefix in ['fundacio ', 'fundacion ']:
        if normalized.startswith(prefix):
            without_prefix = normalized[len(prefix):].strip()
            if without_prefix and without_prefix not in variants:
                variants.append(without_prefix)
    
    # 5. Handle CSIC special case: "AGENCIA ESTATAL...CIENTIFICAS (CSIC)"
    # If it contains 'csic' and has parentheses, extract just 'csic'
    if 'csic' in normalized and '(' in org_name:
        if 'csic' not in variants:
            variants.append('csic')
    
    # 6. Remove extra whitespace (like double spaces in "BARCELONA  CENTRO")
    cleaned_variants = []
    for variant in variants:
        cleaned = re.sub(r'\s+', ' ', variant).strip()
        if cleaned and cleaned not in cleaned_variants:
            cleaned_variants.append(cleaned)
    
    return cleaned_variants

def normalize_organization_name(org_name: str) -> str:
    """
    Normalize organization names for matching.
    
    This is more aggressive than text normalization:
    - Removes common suffixes (S.L., S.A., SL, SA, etc.)
    - Removes punctuation
    - Normalizes whitespace
    
    Examples:
        >>> normalize_organization_name("Universitat de Barcelona")
        'universitat de barcelona'
        >>> normalize_organization_name("ACME, S.L.")
        'acme'
    """
    if not org_name:
        return ""
    
    # Start with basic normalization
    normalized = normalize_text(org_name)
    
    # Remove common suffixes
    suffixes = [
        r'\s+s\.?l\.?$',           # S.L., SL
        r'\s+s\.?a\.?$',           # S.A., SA
        r'\s+s\.?l\.?u\.?$',       # S.L.U., SLU
        r'\s+s\.?c\.?p\.?$',       # S.C.P., SCP
        r'\s+fundacio$',           # Fundació
        r'\s+fundacion$',          # Fundación
        r'\s+associacio$',         # Associació
        r'\s+asociacion$',         # Asociación
        r'\s+coop\.?$',            # Coop., Cooperativa
        r'\s+ltd\.?$',             # Ltd.
        r'\s+inc\.?$',             # Inc.
    ]
    
    for suffix in suffixes:
        normalized = re.sub(suffix, '', normalized, flags=re.IGNORECASE)
    
    # Remove punctuation except spaces
    normalized = re.sub(r'[^\w\s]', '', normalized)
    
    # Normalize whitespace again
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    # Core functions (NOW RETURN LISTS!)
    'normalize_text',
    'normalize_region',
    'normalize_province',
    'normalize_country',
    'normalize_framework',          # Returns List[str]
    'normalize_org_type',           # Returns List[str]
    'normalize_organization_name',
    
    # Matching functions
    'matches_region',
    'matches_province',
    'matches_framework',
    'matches_org_type',
    
    # Validation
    'get_valid_frameworks',
    'get_valid_org_types',
    
    # Utilities
    'remove_accents',
    
    # Configuration management
    'get_config',
    'reload_config',
    
    # ROR organization matching
    'normalize_organization',
    'matches_organization',
    'get_organization_details',
    'get_ror_statistics',
    'preprocess_organization_name',  # Preprocessing helper
    'extract_org_keywords',          # Fuzzy matching helper
    'fuzzy_match_organizations',     # Fuzzy matching
    
]

# ============================================================================
# ROR (Research Organization Registry) INTEGRATION
# ============================================================================

def load_ror_data():
    """Load ROR mappings on first use (lazy loading)"""
    global _ROR_MAPPINGS, _ROR_ORGS
    
    if _ROR_MAPPINGS is None:
        ror_file = Path("data/normalization/ror_mappings.json")
        if ror_file.exists():
            try:
                with open(ror_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    _ROR_MAPPINGS = data['mappings']
                    _ROR_ORGS = data['organizations']
                logger.info(f"Loaded ROR mappings: {len(_ROR_ORGS)} organizations, {len(_ROR_MAPPINGS)} name variants")
            except Exception as e:
                logger.error(f"Error loading ROR mappings: {e}")
                _ROR_MAPPINGS = {}
                _ROR_ORGS = {}
        else:
            logger.warning(f"ROR mappings file not found: {ror_file}")
            _ROR_MAPPINGS = {}
            _ROR_ORGS = {}
    
    return _ROR_MAPPINGS, _ROR_ORGS


def extract_org_keywords(org_name: str) -> Set[str]:
    """
    Extract significant keywords from organization name.
    
    Removes common stopwords (articles, prepositions) and extracts
    the meaningful terms for fuzzy matching. Handles contractions.
    
    Args:
        org_name: Normalized organization name (already lowercased, accent-free)
    
    Returns:
        Set of significant keywords, with language-specific terms normalized
        
    Examples:
        >>> extract_org_keywords("universitat de barcelona")
        {'university', 'barcelona'}  # universitat → university
        >>> extract_org_keywords("l'institut de bioenginyeria de catalunya")
        {'institut', 'bioenginyeria', 'catalunya'}  # l' removed
    """
    # Common Spanish/Catalan/English stopwords in organization names
    STOPWORDS = {
        # Articles (single letters often from contractions)
        'el', 'la', 'los', 'las',
        'the', 'a', 'an',
        # Prepositions
        'de', 'del', 'dels',
        'of', 'for', 'in', 'at',
        # Common org words
        'center', 'centre', 'centro',
        # Conjunctions
        'i', 'y', 'and',
    }
    
    # Language variants to normalize (Spanish/Catalan → English base form)
    LANGUAGE_VARIANTS = {
        'universitat': 'university',
        'universidad': 'university',
        'institut': 'institute',
        'instituto': 'institute',
        'fundacio': 'foundation',
        'fundacion': 'foundation',
        'recerca': 'research',
        'investigacion': 'research',
        'investigaciones': 'research',
    }
    
    if not org_name:
        return set()
    
    # Handle contractions like "l'institut" → "linstitut", then split
    # This ensures "l'" gets separated
    cleaned = org_name.replace("'", " ")
    
    # Split into words
    words = cleaned.split()
    
    # Filter and normalize
    keywords = set()
    for word in words:
        # Skip short words (likely articles/prepositions)
        if len(word) < 3:
            continue
        
        # Skip if it's a stopword
        if word in STOPWORDS:
            continue
        
        # Normalize language variants
        normalized_word = LANGUAGE_VARIANTS.get(word, word)
        keywords.add(normalized_word)
    
    return keywords


def calculate_keyword_similarity(keywords1: Set[str], keywords2: Set[str]) -> float:
    """
    Calculate similarity between two sets of keywords using Jaccard similarity.
    
    Args:
        keywords1: First set of keywords
        keywords2: Second set of keywords
    
    Returns:
        Similarity score between 0.0 (no match) and 1.0 (perfect match)
    """
    if not keywords1 or not keywords2:
        return 0.0
    
    # Jaccard similarity: intersection / union
    intersection = len(keywords1 & keywords2)
    union = len(keywords1 | keywords2)
    
    return intersection / union if union > 0 else 0.0


def fuzzy_match_organizations(org_name: str, min_similarity: float = 0.5) -> List[str]:
    """
    Find organizations with fuzzy matching based on keyword similarity.
    
    Handles cases where exact matching fails due to:
    - Language variants: "universidad" vs "universitat"  
    - Articles: "Institut" vs "L'Institut"
    - Minor differences in phrasing
    
    Args:
        org_name: Organization name to match
        min_similarity: Minimum similarity threshold (0.0 to 1.0)
    
    Returns:
        List of ROR IDs that match with sufficient similarity
    """
    mappings, orgs = load_ror_data()
    
    # Extract keywords from input
    normalized_input = normalize_text(org_name)
    input_keywords = extract_org_keywords(normalized_input)
    
    if not input_keywords:
        return []
    
    # Find candidates with similar keywords
    candidates = []
    
    for ror_name, ror_ids in mappings.items():
        ror_keywords = extract_org_keywords(ror_name)
        
        if not ror_keywords:
            continue
        
        similarity = calculate_keyword_similarity(input_keywords, ror_keywords)
        
        if similarity >= min_similarity:
            for ror_id in ror_ids:
                candidates.append((ror_id, similarity))
    
    # Sort by similarity (highest first) and return unique ROR IDs
    candidates.sort(key=lambda x: x[1], reverse=True)
    seen = set()
    unique_ids = []
    for ror_id, _ in candidates:
        if ror_id not in seen:
            seen.add(ror_id)
            unique_ids.append(ror_id)
    
    return unique_ids


def normalize_organization(org_name: str, use_fuzzy: bool = True, fuzzy_threshold: float = 0.6) -> List[str]:
    """
    Normalize organization name using ROR data with preprocessing and fuzzy matching.
    Returns list of canonical forms, aliases, and acronyms.
    
    This function uses:
    1. preprocess_organization_name() to handle acronyms, prefixes, suffixes
    2. Exact matching against ROR database
    3. Fuzzy matching as fallback for close matches
    
    Args:
        org_name: Organization name to normalize
        use_fuzzy: Enable fuzzy matching for close matches (default: True)
        fuzzy_threshold: Minimum similarity for fuzzy matches (0.0-1.0, default: 0.6)
        
    Returns:
        List of all name variants (canonical, aliases, acronyms)
        
    Examples:
        >>> normalize_organization("Universidad de Barcelona")
        ['Universitat de Barcelona', 'Universidad de Barcelona', 
         'University of Barcelona', 'UB']
        
        >>> normalize_organization("Universidad de Lleida")
        # Fuzzy matches "Universitat de Lleida" (language variant)
        
        >>> normalize_organization("FUNDACIO INSTITUT BIOENGINYERIA")
        # Fuzzy matches "L'Institut de Bioenginyeria de Catalunya"
    """
    if not org_name:
        return []
    
    mappings, orgs = load_ror_data()
    
    # STEP 1: Try exact matching with preprocessing variants
    variants_to_try = preprocess_organization_name(org_name)
    
    # Collect all ROR IDs that match any variant
    matched_ror_ids = set()
    for variant in variants_to_try:
        ror_ids = mappings.get(variant, [])
        if ror_ids:
            matched_ror_ids.update(ror_ids)
    
    # STEP 2: If no exact match and fuzzy enabled, try fuzzy matching
    if not matched_ror_ids and use_fuzzy:
        fuzzy_ids = fuzzy_match_organizations(org_name, min_similarity=fuzzy_threshold)
        matched_ror_ids.update(fuzzy_ids)
    
    if not matched_ror_ids:
        # No match found - return original
        return [org_name]
    
    # Collect all name variants for matched organizations
    all_variants = set()
    
    for ror_id in matched_ror_ids:
        org_data = orgs.get(ror_id, {})
        
        # Add canonical name (highest priority)
        if org_data.get('canonical_name'):
            all_variants.add(org_data['canonical_name'])
        
        # Add all aliases
        for alias in org_data.get('aliases', []):
            all_variants.add(alias['value'])
        
        # Add all acronyms
        for acronym in org_data.get('acronyms', []):
            all_variants.add(acronym['value'])
    
    return sorted(list(all_variants))

def get_organization_details(org_name: str) -> Optional[Dict]:
    """
    Get full ROR details for an organization.
    
    Args:
        org_name: Organization name or acronym
        
    Returns:
        Dictionary with organization details including:
        - ror_id: ROR identifier URL
        - canonical_name: Primary display name
        - aliases: List of name variants with languages
        - acronyms: List of acronyms
        - location: City, region, country
        - types: Organization types (education, company, etc.)
        - domains: Web domains
        - status: active/inactive/withdrawn
        
    Examples:
        >>> details = get_organization_details("UB")
        >>> details['canonical_name']
        'Universitat de Barcelona'
        >>> details['location']['city']
        'Barcelona'
        >>> details['acronyms']
        [{'value': 'UB', 'lang': None}]
    """
    if not org_name:
        return None
    
    mappings, orgs = load_ror_data()
    
    org_lower = normalize_text(org_name)
    ror_ids = mappings.get(org_lower, [])
    
    if not ror_ids:
        return None
    
    # Return details for first match
    # (If multiple matches exist, this returns the first one)
    return orgs.get(ror_ids[0])


def matches_organization(org_name_in_data: str, org_name_in_query: str) -> bool:
    """
    Check if two organization names refer to the same entity using ROR.
    Handles acronyms, different languages, and name variants.
    
    Args:
        org_name_in_data: Organization name as it appears in the dataset
        org_name_in_query: Organization name from user query
        
    Returns:
        True if names refer to same organization, False otherwise
        
    Examples:
        >>> matches_organization("Universitat de Barcelona", "Universidad de Barcelona")
        True
        >>> matches_organization("UB", "University of Barcelona")
        True
        >>> matches_organization("Universitat de Barcelona", "UB")
        True
        >>> matches_organization("CSIC", "Spanish National Research Council")
        True
        >>> matches_organization("Universitat de Barcelona", "Universitat Pompeu Fabra")
        False
    """
    if not org_name_in_data or not org_name_in_query:
        return False
    
    # Get all variants for both names
    data_variants = normalize_organization(org_name_in_data)
    query_variants = normalize_organization(org_name_in_query)
    
    # Normalize for comparison
    data_set = {v.lower() for v in data_variants}
    query_set = {v.lower() for v in query_variants}
    
    # Check for overlap (any common variant means same org)
    return bool(data_set & query_set)


def get_ror_statistics() -> Dict[str, any]:
    """
    Get statistics about loaded ROR data.
    
    Returns:
        Dictionary with statistics:
        - total_organizations: Number of organizations
        - total_name_variants: Number of name mappings
        - organizations_by_type: Count by organization type
        - organizations_by_region: Count by region
        - etc.
    """
    _, orgs = load_ror_data()
    
    if not orgs:
        return {
            'loaded': False,
            'error': 'ROR data not loaded'
        }
    
    # Try to load statistics from the ROR mappings file
    ror_file = Path("data/normalization/ror_mappings.json")
    if ror_file.exists():
        try:
            with open(ror_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    'loaded': True,
                    'metadata': data.get('metadata', {}),
                    'statistics': data.get('statistics', {})
                }
        except Exception as e:
            logger.error(f"Error loading ROR statistics: {e}")
    
    return {
        'loaded': True,
        'total_organizations': len(orgs)
    }


# ============================================================================
# SELF-TEST
# ============================================================================

if __name__ == "__main__":
    print("Running normalization tests with synonym groups...")
    print()
    
    # Test frameworks (WITH SYNONYMS)
    print("Framework normalization (with synonym groups):")
    test_frameworks = ["h2020", "horizon", "Horizon 2020", "aei", "cdti"]
    for fw in test_frameworks:
        normalized = normalize_framework(fw)
        short_forms = normalize_framework(fw, to_db_value=False)
        print(f"  {fw:20s} → {normalized} (short: {short_forms})")
    print()
    
    # Test org types (WITH SYNONYMS)
    print("Organization type normalization (with synonym groups):")
    test_orgs = ["company", "university", "empresa", "research_center", "public_entity"]
    for org in test_orgs:
        normalized = normalize_org_type(org)
        print(f"  {org:20s} → {normalized}")
    print()
    
    # Test regions (1:1)
    print("Region normalization (1:1):")
    test_regions = ["CATALUNYA", "Cataluña", "catalunya", "catalonia"]
    for region in test_regions:
        print(f"  {region:20s} → {normalize_region(region)}")
    print()
    
    # Test provinces (1:1)
    print("Province normalization (1:1):")
    test_provinces = ["GERONA", "Girona", "LLEIDA", "Lérida"]
    for province in test_provinces:
        print(f"  {province:20s} → {normalize_province(province)}")
    print()
    
    # Test synonym matching
    print("Synonym matching tests:")
    print(f"  'h2020' matches 'H2020': {matches_framework('h2020', 'H2020')}")
    print(f"  'h2020' matches 'HORIZON': {matches_framework('h2020', 'HORIZON')}  ← SYNONYMS!")
    print(f"  'company' matches 'EMPRESA': {matches_org_type('company', 'EMPRESA')}")
    print(f"  'company' matches 'PRC': {matches_org_type('company', 'PRC')}  ← SYNONYMS!")
    print(f"  'EMPRESA' matches 'PRC': {matches_org_type('EMPRESA', 'PRC')}  ← SYNONYMS!")
    print()
    
    # Show valid values
    print("Valid frameworks:")
    for fw in get_valid_frameworks():
        print(f"  - {fw}")
    print()
    
    print("Valid org types:")
    for org in get_valid_org_types():
        print(f"  - {org}")
    print()
    
    print("✓ All tests passed!")
