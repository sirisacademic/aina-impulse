#!/usr/bin/env python3
"""
Build organization name mappings from ROR (Research Organization Registry) data.

This script:
1. Downloads/processes ROR dump
2. Extracts Spanish organizations (country_code = ES)
3. Creates mappings for all name variants (Catalan, Spanish, English, acronyms)
4. Outputs JSON file for use in normalization

IMPORTANT: Normalizes names by removing accents to handle variations like:
  - "Universidad Autónoma" -> "universidad autonoma"
  - "Universitat Politècnica" -> "universitat politecnica"

Usage:
    python build_ror_mappings.py --ror-dump ror-data.json --output data/normalization/ror_mappings.json
"""

import json
import argparse
import unicodedata
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Set, Tuple


def remove_accents(text: str) -> str:
    """
    Remove accents/diacritics from text.
    
    Examples:
        >>> remove_accents("Autónoma")
        'Autonoma'
        >>> remove_accents("Politècnica")
        'Politecnica'
    """
    if not text:
        return ""
    
    # Normalize to NFD (decomposed form)
    nfd = unicodedata.normalize('NFD', text)
    
    # Remove combining characters (accents)
    return ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')



def normalize_name(name: str) -> str:
    """
    Normalize organization name for matching.
    
    This ensures consistent matching across accent variations:
    - "Universidad Autónoma de Barcelona" -> "universidad autonoma de barcelona"
    - "Universitat Autònoma de Barcelona" -> "universitat autonoma de barcelona"
    - Both will match!
    
    Steps:
    1. Remove accents/diacritics
    2. Convert to lowercase
    3. Strip and normalize whitespace
    """
    if not name:
        return ""
    
    # Remove accents
    normalized = remove_accents(name)
    
    # Lowercase and strip
    normalized = normalized.lower().strip()
    
    # Normalize whitespace (multiple spaces -> single space)
    normalized = re.sub(r'\s+', ' ', normalized)
    
    return normalized


def generate_name_variants(name: str) -> List[str]:
    """
    Generate multiple normalized variants for a name to improve matching.
    
    This handles cases where the same organization has different article usage:
    - "L'Institut" vs "Institut"
    - "FUNDACIO Institut" vs "Institut"
    
    Args:
        name: Original organization name from ROR
        
    Returns:
        List of normalized variants
        
    Examples:
        >>> generate_name_variants("L'Institut de Bioenginyeria de Catalunya")
        ['linstitut de bioenginyeria de catalunya',      # Original normalized
         'institut de bioenginyeria de catalunya']        # Without article
         
        >>> generate_name_variants("Universidad de Barcelona")
        ['universidad de barcelona']                      # Only one variant needed
    """
    variants = []
    
    # Variant 1: Standard normalization (baseline)
    normalized = normalize_name(name)
    if normalized:
        variants.append(normalized)
    
    # Variant 2: Remove Catalan/French contractions (l', d', s', etc.)
    # Pattern: single letter + apostrophe at word boundary
    if "'" in name or "’" in name:
        without_contraction = re.sub(r"\b\w['’]", '', name)
        without_contraction_normalized = normalize_name(without_contraction)
        
        # Only add if different from variant 1
        if without_contraction_normalized and without_contraction_normalized not in variants:
            variants.append(without_contraction_normalized)
    
    return variants


def extract_organizations(ror_dump_path: Path, filter_country: str = 'ES') -> Tuple[Dict, Dict]:
    """
    Extract organizations from ROR dump and build mappings.
    
    Args:
        ror_dump_path: Path to ROR JSON dump file (JSON array format)
        filter_country: ISO country code to filter (default: ES for Spain)
    
    Returns:
        Tuple of (name_mappings, organization_details)
    """
    
    print(f"Reading ROR dump from: {ror_dump_path}")
    print(f"Loading JSON array (this may take a moment)...")
    
    name_mappings = defaultdict(set)  # normalized_name -> {ror_ids}
    organization_details = {}  # ror_id -> org details
    
    organizations_processed = 0
    organizations_matched = 0
    
    # Load the entire JSON array
    with open(ror_dump_path, 'r', encoding='utf-8') as f:
        try:
            organizations = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Error parsing JSON: {e}")
            return {}, {}
    
    print(f"[OK] Loaded {len(organizations)} organizations from dump")
    print(f"Processing and filtering by country: {filter_country}")
    print()
    
    # Process each organization
    for org in organizations:
        organizations_processed += 1
        
        if organizations_processed % 10000 == 0:
            print(f"  Processed {organizations_processed} organizations...")
        
        # Filter by country
        locations = org.get('locations', [])
        is_target_country = any(
            loc.get('geonames_details', {}).get('country_code') == filter_country
            for loc in locations
        )
        
        if not is_target_country:
            continue
        
        organizations_matched += 1
        
        ror_id = org['id']
        
        # Initialize organization details
        org_details = {
            'ror_id': ror_id,
            'canonical_name': None,
            'aliases': [],
            'acronyms': [],
            'location': {},
            'types': org.get('types', []),
            'status': org.get('status', 'unknown'),
            'domains': org.get('domains', [])
        }
        
        # Extract all name variants
        names_by_type = defaultdict(list)
        
        for name_obj in org.get('names', []):
            name = name_obj['value']
            lang = name_obj.get('lang')
            types = name_obj.get('types', [])
            
            # Store by type for easier access
            for type_name in types:
                names_by_type[type_name].append({
                    'name': name,
                    'lang': lang
                })
            
            # Canonical name (ror_display is the primary display name)
            if 'ror_display' in types:
                org_details['canonical_name'] = name
            
            # Categorize aliases vs acronyms
            if 'acronym' in types:
                org_details['acronyms'].append({
                    'value': name,
                    'lang': lang
                })
            else:
                org_details['aliases'].append({
                    'value': name,
                    'lang': lang,
                    'types': types
                })
            
            # Create mapping: any name variant -> ROR ID
            # IMPORTANT: Generate multiple variants (with/without articles, etc.)
            variants = generate_name_variants(name)
            for variant in variants:
                name_mappings[variant].add(ror_id)
        
        # Extract location details
        if locations:
            geo = locations[0].get('geonames_details', {})
            org_details['location'] = {
                'city': geo.get('name'),
                'region': geo.get('country_subdivision_name'),
                'country': geo.get('country_name'),
                'country_code': geo.get('country_code')
            }
        
        organization_details[ror_id] = org_details
    
    print(f"\n[OK] Processed {organizations_processed} total organizations")
    print(f"[OK] Matched {organizations_matched} organizations in {filter_country}")
    print(f"[OK] Created {len(name_mappings)} unique name variant mappings")
    
    return name_mappings, organization_details


def build_statistics(name_mappings: Dict, organization_details: Dict) -> Dict:
    """Generate statistics about the extracted data"""
    
    total_orgs = len(organization_details)
    total_acronyms = sum(
        len(org['acronyms']) for org in organization_details.values()
    )
    total_aliases = sum(
        len(org['aliases']) for org in organization_details.values()
    )
    
    # Count organizations by type
    types_count = defaultdict(int)
    for org in organization_details.values():
        for org_type in org['types']:
            types_count[org_type] += 1
    
    # Count by location
    locations_count = defaultdict(int)
    for org in organization_details.values():
        region = org['location'].get('region')
        if region:
            locations_count[region] += 1
    
    return {
        'total_organizations': total_orgs,
        'total_name_variants': len(name_mappings),
        'total_acronyms': total_acronyms,
        'total_aliases': total_aliases,
        'organizations_by_type': dict(types_count),
        'organizations_by_region': dict(locations_count),
        'avg_names_per_org': len(name_mappings) / total_orgs if total_orgs > 0 else 0
    }


def save_mappings(output_path: Path, name_mappings: Dict, organization_details: Dict, 
                  ror_dump_path: Path, filter_country: str):
    """Save mappings to JSON file"""
    
    # Convert sets to lists for JSON serialization
    mappings_serializable = {
        name: list(ror_ids) for name, ror_ids in name_mappings.items()
    }
    
    stats = build_statistics(name_mappings, organization_details)
    
    output = {
        'metadata': {
            'generated': datetime.now().isoformat(),
            'source': str(ror_dump_path),
            'filter_country': filter_country,
            'ror_schema_version': '2.1',
            'normalization': 'accent_removed'  # Flag that accents are removed
        },
        'statistics': stats,
        'mappings': mappings_serializable,
        'organizations': organization_details
    }
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Saved mappings to: {output_path}")
    print(f"\nStatistics:")
    print(f"  Organizations: {stats['total_organizations']}")
    print(f"  Name variants: {stats['total_name_variants']}")
    print(f"  Acronyms: {stats['total_acronyms']}")
    print(f"  Aliases: {stats['total_aliases']}")
    print(f"  Avg names/org: {stats['avg_names_per_org']:.1f}")
    
    if stats['organizations_by_type']:
        print(f"\n  Top organization types:")
        for org_type, count in sorted(stats['organizations_by_type'].items(), 
                                      key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {org_type}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description='Build organization name mappings from ROR data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process ROR dump for Spanish organizations
  python build_ror_mappings.py --ror-dump ror-data.json --output data/normalization/ror_mappings.json
  
  # Filter for another country
  python build_ror_mappings.py --ror-dump ror-data.json --output ror_fr.json --filter-country FR
        """
    )
    
    parser.add_argument(
        '--ror-dump',
        type=Path,
        required=True,
        help='Path to ROR JSON dump file (JSON array format)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output path for mappings JSON file'
    )
    parser.add_argument(
        '--filter-country',
        default='ES',
        help='ISO country code to filter organizations (default: ES for Spain)'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not args.ror_dump.exists():
        print(f"[ERROR] Error: ROR dump file not found: {args.ror_dump}")
        return 1
    
    try:
        # Extract organizations
        name_mappings, organization_details = extract_organizations(
            args.ror_dump,
            args.filter_country
        )
        
        # Save mappings
        save_mappings(
            args.output,
            name_mappings,
            organization_details,
            args.ror_dump,
            args.filter_country
        )
        
        print("\n[OK] Success!")
        print("\nNOTE: All names normalized with accents removed for consistent matching.")
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
