#!/usr/bin/env python3
"""
Test Data Quality Checker - validates training/test examples
Fixed false positives and encoding issues
"""

import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


def check_location_language_consistency(example: Dict[Any, Any]) -> List[str]:
    """Check if locations in original_query appear in schema - with smart filtering"""
    issues = []
    
    original = example.get('meta', {}).get('original_query', '').lower()
    filters = example.get('filters', {})
    orgs = example.get('organisations', [])
    
    # Skip if location is in programme name (Horizon Europe, Generalitat de Catalunya, etc)
    programme = filters.get('programme', '')
    if programme and any(x in original for x in ['europe', 'europa', 'generalitat', 'governo']):
        # Location might be part of programme name, not a filter
        if 'catalunya' in original and 'generalitat' in original:
            return issues  # Catalunya is in programme name
        if 'europe' in original or 'europa' in original:
            return issues  # Europe in programme name
    
    # Define location patterns by language
    location_patterns = {
        'CA': ['frança', 'alemanya', 'itàlia', 'espanya', 'eslovènia', 'catalunya',
               'barcelona', 'girona', 'tarragona', 'lleida'],
        'ES': ['francia', 'alemania', 'italia', 'españa', 'eslovenia', 'cataluña',
               'barcelona', 'girona', 'tarragona', 'lérida'],
        'EN': ['france', 'germany', 'italy', 'spain', 'slovenia', 'catalonia',
               'barcelona', 'girona', 'tarragona', 'lleida']
    }
    
    # Find locations mentioned in original query
    found_locations = []
    for locations in location_patterns.values():
        for loc in locations:
            # Use word boundaries to avoid partial matches
            if re.search(r'\b' + re.escape(loc) + r'\b', original):
                found_locations.append(loc)
    
    # Check each found location appears in schema
    for location in found_locations:
        found_in_schema = False
        
        # Check project location filter
        if filters.get('location'):
            if location in filters['location'].lower():
                found_in_schema = True
        
        # Check organisation locations
        for org in orgs:
            if org.get('location'):
                if location in org['location'].lower():
                    found_in_schema = True
        
        # Check if it's an organisation name containing location
        for org in orgs:
            if org.get('name') and location in org['name'].lower():
                found_in_schema = True
        
        if not found_in_schema:
            issues.append(f"Location '{location}' in query not found in schema")
    
    return issues


def check_semantic_query_quality(example: Dict[Any, Any]) -> List[str]:
    """Validate semantic_query field - only flag real issues"""
    issues = []
    semantic = example.get('semantic_query', '')
    
    if semantic:
        # Check for stop words only
        stop_words = ['the', 'a', 'an', 'in', 'on', 'at', 'de', 'la', 'el', 'en']
        if semantic.lower() in stop_words:
            issues.append("Semantic query contains only stop words")
        
        # Check for excessive length
        if len(semantic.split()) > 20:
            issues.append(f"Semantic query very long ({len(semantic.split())} words)")
        
        # Boolean operators: 'and'/'or' are fine in natural language queries
        # Only flag if there's a clear boolean query pattern with lowercase operators
        if ' OR ' in semantic and ' or ' in semantic.lower():
            # Mixed case - flag it
            issues.append("Inconsistent boolean operators (mix of 'OR' and 'or')")
    
    return issues


def check_org_type_consistency(example: Dict[Any, Any]) -> List[str]:
    """Check valid organisation types"""
    issues = []
    valid_types = ['university', 'company', 'research_center', 'hospital', 
                   'ngo', 'public_entity', None]
    
    for i, org in enumerate(example.get('organisations', [])):
        org_type = org.get('type')
        if org_type not in valid_types:
            issues.append(f"Organisation {i}: invalid type '{org_type}'")
    
    return issues


def check_null_field_logic(example: Dict[Any, Any]) -> List[str]:
    """Check if null fields make sense given the query - with smart filtering"""
    issues = []
    
    original = example.get('meta', {}).get('original_query', '').lower()
    filters = example.get('filters', {})
    
    # Programme checks - only flag if it's mentioned as a filter, not just in org names
    programme_keywords = {
        'horizon 2020': ['horizon 2020', 'h2020'],
        'horizon europe': ['horizon europe'],
        'erc': [r'\berc\b'],  # Use word boundary to avoid matching "recerca"
        'msca': ['msca', 'marie curie'],
        'feder': ['feder'],
        'sifecat': ['sifecat']
    }
    
    for prog_name, keywords in programme_keywords.items():
        for keyword in keywords:
            # Handle regex patterns
            if keyword.startswith(r'\b'):
                if re.search(keyword, original):
                    match_found = True
                else:
                    match_found = False
            else:
                match_found = keyword in original
            
            if match_found and filters.get('programme') is None:
                # Check if it's really asking about the programme or just mentioned
                # e.g., "ERC projects" vs "the ERC"
                if any(x in original for x in ['projectes', 'proyectos', 'projects', 'funding', 'grants']):
                    # Likely meant to filter by programme, but check context
                    # Skip if programme name is in organisation name
                    orgs = example.get('organisations', [])
                    if any(org.get('name') and keyword in org.get('name', '').lower() for org in orgs):
                        continue  # It's an org name, not a programme filter
                    issues.append(f"Programme '{keyword}' mentioned in query but filter is null")
                break
    
    # Year checks
    year_patterns = [r'\b(19|20)\d{2}\b', r'\b\d{4}\b']
    for pattern in year_patterns:
        matches = re.findall(pattern, original)
        if matches and filters.get('year') is None:
            # Check if it's really a year or just a number
            if any(x in original for x in ['year', 'any', 'del ', 'en ', 'in ']):
                issues.append(f"Year {matches[0]} mentioned in query but filter is null")
                break
    
    return issues


def check_query_rewrite_completeness(example: Dict[Any, Any]) -> List[str]:
    """Check if query_rewrite reflects all key elements - minimal false positives"""
    issues = []
    query_rewrite = example.get('query_rewrite', '').lower()
    
    filters = example.get('filters', {})
    orgs = example.get('organisations', [])
    
    # Check if project location is mentioned
    if filters.get('location'):
        location = filters['location'].lower()
        # Allow variations (e.g., "País Vasco" vs "País Basc")
        location_base = location.replace('ï', 'i').replace('ó', 'o').replace('á', 'a')
        query_base = query_rewrite.replace('ï', 'i').replace('ó', 'o').replace('á', 'a')
        
        if location_base not in query_base and location not in query_rewrite:
            # Check base name without diacritics
            loc_words = location.split()
            if not any(word in query_rewrite for word in loc_words if len(word) > 3):
                issues.append(f"Location '{filters['location']}' not in query_rewrite")
    
    # Check organisations - only flag if SPECIFIC org details missing
    if orgs:
        for org in orgs:
            # Only check if org has specific name
            if org.get('name'):
                name_lower = org['name'].lower()
                # Check for name or common abbreviation
                if name_lower not in query_rewrite:
                    # Check if abbreviated form present (e.g., UPC, CSIC)
                    words = name_lower.split()
                    if len(words) > 1:
                        abbrev = ''.join(w[0] for w in words if w not in ['de', 'di', 'of'])
                        if abbrev not in query_rewrite:
                            issues.append(f"Organisation name '{org['name']}' not in query_rewrite")
    
    return issues


def check_schema_validity(example: Dict[Any, Any]) -> List[str]:
    """Check basic schema structure"""
    issues = []
    
    # Required fields
    required = ['doc_type', 'filters', 'organisations', 'semantic_query', 'query_rewrite', 'meta']
    for field in required:
        if field not in example:
            issues.append(f"Missing required field: {field}")
    
    # Check filters structure
    if 'filters' in example:
        required_filters = ['programme', 'funding_level', 'year', 'location', 'location_level']
        for f in required_filters:
            if f not in example['filters']:
                issues.append(f"Missing filter field: {f}")
    
    # Check meta structure
    if 'meta' in example:
        required_meta = ['id', 'lang', 'original_query']
        for m in required_meta:
            if m not in example['meta']:
                issues.append(f"Missing meta field: {m}")
    
    # Check location_level values
    valid_location_levels = ['city', 'province', 'region', 'country', 'other', None]
    if 'filters' in example:
        level = example['filters'].get('location_level')
        if level not in valid_location_levels:
            issues.append(f"Invalid location_level in filters: {level}")
    
    for org in example.get('organisations', []):
        level = org.get('location_level')
        if level not in valid_location_levels:
            issues.append(f"Invalid location_level in organisation: {level}")
    
    return issues


def check_language_consistency(example: Dict[Any, Any]) -> List[str]:
    """Check if language fields match"""
    issues = []
    
    lang = example.get('meta', {}).get('lang')
    original = example.get('meta', {}).get('original_query', '')
    query_rewrite = example.get('query_rewrite', '')
    
    if lang:
        # Check language indicators
        indicators = {
            'CA': ['llista', 'projectes'],
            'ES': ['lista', 'proyectos'],
            'EN': ['list', 'projects']
        }
        
        if lang in indicators:
            found_indicator = any(ind in query_rewrite.lower() for ind in indicators[lang])
            if not found_indicator:
                issues.append(f"query_rewrite language doesn't match meta.lang={lang}")
    
    return issues


def check_resolvability_annotation(example: Dict[Any, Any]) -> List[str]:
    """Check if resolvability makes sense"""
    issues = []
    
    resolvability = example.get('meta', {}).get('resolvability')
    notes = example.get('meta', {}).get('notes')
    
    if resolvability == 'Partial' and not notes:
        issues.append("Resolvability='Partial' but no notes explaining why")
    
    if resolvability == 'Adapted' and not notes:
        issues.append("Resolvability='Adapted' but no notes explaining adaptation")
    
    return issues


def check_duplicates(examples: List[Dict]) -> List[str]:
    """Check for duplicate IDs"""
    issues = []
    ids = [ex.get('meta', {}).get('id') for ex in examples]
    seen = set()
    
    for id in ids:
        if id in seen:
            issues.append(f"Duplicate ID: {id}")
        seen.add(id)
    
    return issues


def run_quality_checks(examples: List[Dict]) -> Dict[str, List]:
    """Run all quality checks"""
    all_issues = defaultdict(list)
    
    # Check duplicates
    dup_issues = check_duplicates(examples)
    if dup_issues:
        all_issues['duplicate_ids'] = dup_issues
    
    # Per-example checks
    for example in examples:
        test_id = example.get('meta', {}).get('id', 'UNKNOWN')
        
        # Schema
        schema_issues = check_schema_validity(example)
        if schema_issues:
            all_issues['schema'].append({'id': test_id, 'issues': schema_issues})
        
        # Query rewrite
        qr_issues = check_query_rewrite_completeness(example)
        if qr_issues:
            all_issues['query_rewrite'].append({
                'id': test_id,
                'issues': qr_issues,
                'query_rewrite': example.get('query_rewrite', '')
            })
        
        # Language
        lang_issues = check_language_consistency(example)
        if lang_issues:
            all_issues['language'].append({'id': test_id, 'issues': lang_issues})
        
        # Location language
        loc_lang_issues = check_location_language_consistency(example)
        if loc_lang_issues:
            all_issues['location_language'].append({
                'id': test_id,
                'issues': loc_lang_issues,
                'original': example.get('meta', {}).get('original_query', '')
            })
        
        # Semantic query
        sem_issues = check_semantic_query_quality(example)
        if sem_issues:
            all_issues['semantic_query'].append({
                'id': test_id,
                'issues': sem_issues,
                'semantic_query': example.get('semantic_query', '')
            })
        
        # Org types
        org_issues = check_org_type_consistency(example)
        if org_issues:
            all_issues['org_types'].append({'id': test_id, 'issues': org_issues})
        
        # Null fields
        null_issues = check_null_field_logic(example)
        if null_issues:
            all_issues['null_fields'].append({'id': test_id, 'issues': null_issues})
        
        # Resolvability
        res_issues = check_resolvability_annotation(example)
        if res_issues:
            all_issues['resolvability'].append({'id': test_id, 'issues': res_issues})
    
    return all_issues


def print_report(examples: List[Dict], issues: Dict[str, List]):
    """Print formatted quality report"""
    print("=" * 80)
    print("TEST DATA QUALITY REPORT")
    print("=" * 80)
    print(f"\nTotal examples: {len(examples)}")
    
    if not any(issues.values()):
        print("\n✓ No issues found - dataset looks good!")
        print("=" * 80)
        return
    
    issue_count = sum(len(v) for v in issues.values())
    
    # Print issues by category
    categories = {
        'duplicate_ids': 'DUPLICATE IDS',
        'schema': 'SCHEMA ERRORS',
        'query_rewrite': 'QUERY_REWRITE COMPLETENESS',
        'language': 'LANGUAGE CONSISTENCY',
        'location_language': 'LOCATION LANGUAGE CONSISTENCY',
        'semantic_query': 'SEMANTIC QUERY QUALITY',
        'org_types': 'ORGANISATION TYPE VALIDITY',
        'null_fields': 'NULL FIELD LOGIC',
        'resolvability': 'RESOLVABILITY ANNOTATION'
    }
    
    for key, title in categories.items():
        if key in issues and issues[key]:
            print(f"\n*  {title}: {len(issues[key])} issues")
            for item in issues[key][:10]:  # Show first 10
                if isinstance(item, str):
                    print(f"  {item}")
                else:
                    print(f"  {item['id']}:")
                    for issue in item.get('issues', []):
                        print(f"    - {issue}")
                    for extra_key in ['query_rewrite', 'original', 'semantic_query']:
                        if extra_key in item:
                            print(f"    ({extra_key}: {item[extra_key]})")
            
            if len(issues[key]) > 10:
                print(f"  ... and {len(issues[key]) - 10} more")
        else:
            print(f"\n✓ {title}: No issues")
    
    print("\n" + "=" * 80)
    print(f"*  TOTAL ISSUES: {issue_count}")
    print("\nRecommendation: Review and fix issues before training")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Check test data quality")
    parser.add_argument("input_file", help="JSON file to check")
    parser.add_argument("--verbose", action="store_true", help="Show all issues")
    args = parser.parse_args()
    
    # Load data
    with open(args.input_file, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    
    # Run checks
    issues = run_quality_checks(examples)
    
    # Print report
    print_report(examples, issues)
    
    # Exit code
    return 1 if any(issues.values()) else 0


if __name__ == "__main__":
    exit(main())
