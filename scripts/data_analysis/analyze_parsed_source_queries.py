#!/usr/bin/env python3
"""
Query Analysis Script for IMPULS Project
Analyzes parsed queries for test set sampling strategy
"""

import json
from collections import Counter, defaultdict
from typing import List, Dict
       
def load_queries(filepath: str) -> List[Dict]:
    """Load queries from JSON array or JSONL file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        f.seek(0)
        
        # Check if it's a JSON array
        if first_line.startswith('['):
            return json.load(f)
        
        # Otherwise treat as JSONL
        queries = []
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
        return queries

def print_statistics(queries: List[Dict]):
    """Print comprehensive statistics"""
    total = len(queries)
    
    print("=" * 80)
    print("QUERY ANALYSIS - COMPONENT AND RESOLVABILITY STATISTICS")
    print("=" * 80)
    print(f"\nTotal queries analyzed: {total}\n")

    missing_meta = []
    for i, q in enumerate(queries):
        if 'meta' not in q:
            missing_meta.append((i, q.get('doc_type', 'unknown')))

    if missing_meta:
        print(f"WARNING: {len(missing_meta)} queries missing 'meta' field:")
        for idx, doc_type in missing_meta[:20]:
            print(f"  Line {idx}: doc_type={doc_type}")
        if len(missing_meta) > 20:
            print(f"  ... and {len(missing_meta) - 20} more")
        print("\nFiltering them out...\n")
        queries = [q for q in queries if 'meta' in q]
        total = len(queries)

    # 1. Resolvability
    print("-" * 80)
    print("1. RESOLVABILITY DISTRIBUTION")
    print("-" * 80)
    resolve_counts = Counter(q['meta']['resolvability'] for q in queries)
    for status in ['Direct', 'Adapted', 'Partial', 'Unsupported']:
        if status in resolve_counts:
            count = resolve_counts[status]
            pct = (count/total)*100
            print(f"{status:20s}: {count:3d} ({pct:5.1f}%)")
    
    # 2. Intent
    print("\n" + "-" * 80)
    print("2. INTENT DISTRIBUTION")
    print("-" * 80)
    intent_counts = Counter(q['meta']['intent'] for q in queries)
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
        pct = (count/total)*100
        print(f"{intent:20s}: {count:3d} ({pct:5.1f}%)")
    
    # 3. Source
    print("\n" + "-" * 80)
    print("3. SOURCE DISTRIBUTION")
    print("-" * 80)
    source_counts = Counter(q['meta']['source'] for q in queries)
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = (count/total)*100
        print(f"{source:20s}: {count:3d} ({pct:5.1f}%)")
    
    # 4. Component frequency
    print("\n" + "-" * 80)
    print("4. COMPONENT FREQUENCY")
    print("-" * 80)
    component_counts = Counter()
    for q in queries:
        for component in q['meta']['components']:
            component_counts[component] += 1
    
    for component, count in component_counts.most_common():
        pct = (count/total)*100
        print(f"{component:30s}: {count:3d} ({pct:5.1f}%)")
    
    # 5. Top combinations
    print("\n" + "-" * 80)
    print("5. TOP 20 COMPONENT COMBINATIONS")
    print("-" * 80)
    combo_counts = Counter()
    for q in queries:
        combo = tuple(sorted(q['meta']['components']))
        combo_counts[combo] += 1
    
    for i, (combo, count) in enumerate(combo_counts.most_common(20), 1):
        pct = (count/total)*100
        comp_str = " + ".join(combo)
        print(f"{i:2d}. {count:3d} ({pct:4.1f}%): {comp_str}")
    
    # 6. Complexity
    print("\n" + "-" * 80)
    print("6. QUERY COMPLEXITY (by number of components)")
    print("-" * 80)
    complexity_counts = Counter(len(q['meta']['components']) for q in queries)
    for num, count in sorted(complexity_counts.items()):
        pct = (count/total)*100
        print(f"{num} components: {count:3d} ({pct:5.1f}%)")
    
    # 7. Cross-dimensional matrix
    print("\n" + "-" * 80)
    print("7. RESOLVABILITY Ã— INTENT MATRIX")
    print("-" * 80)
    
    cross_matrix = defaultdict(lambda: defaultdict(int))
    for q in queries:
        cross_matrix[q['meta']['resolvability']][q['meta']['intent']] += 1
    
    intents = sorted(set(q['meta']['intent'] for q in queries))
    print(f"{'Resolvability':<15} | ", end="")
    for intent in intents:
        print(f"{intent:<10}", end=" ")
    print()
    print("-" * (15 + 2 + len(intents) * 11))
    
    for resolve in ['Direct', 'Adapted', 'Partial', 'Unsupported']:
        if resolve in cross_matrix:
            print(f"{resolve:<15} | ", end="")
            for intent in intents:
                count = cross_matrix[resolve].get(intent, 0)
                print(f"{count:<10}", end=" ")
            print()

    # 8. Semantic query analysis
    print("\n" + "-" * 80)
    print("8. SEMANTIC QUERY PRESENCE")
    print("-" * 80)
    null_semantic = sum(1 for q in queries if q['semantic_query'] is None)
    non_null_semantic = total - null_semantic
    print(f"Has semantic_query:  {non_null_semantic:3d} ({non_null_semantic/total*100:5.1f}%)")
    print(f"Null semantic_query: {null_semantic:3d} ({null_semantic/total*100:5.1f}%)")

    # 9. Organization filter usage
    print("\n" + "-" * 80)
    print("9. ORGANIZATION FILTERS")
    print("-" * 80)
    no_org = sum(1 for q in queries if len(q['organisations']) == 0)
    has_org = total - no_org
    print(f"No organisations:    {no_org:3d} ({no_org/total*100:5.1f}%)")
    print(f"Has organisations:   {has_org:3d} ({has_org/total*100:5.1f}%)")

    org_type_counts = Counter()
    for q in queries:
        for org in q['organisations']:
            if org.get('type'):
                org_type_counts[org['type']] += 1
    if org_type_counts:
        print("\nOrganization types:")
        for org_type, count in org_type_counts.most_common():
            print(f"  {org_type:20s}: {count:3d}")
            
    # 10. Organization attributes
    print("\n" + "-" * 80)
    print("10. ORGANIZATION ATTRIBUTES")
    print("-" * 80)

    org_with_type = sum(1 for q in queries for org in q['organisations'] if org.get('type'))
    org_with_name = sum(1 for q in queries for org in q['organisations'] if org.get('name'))
    org_with_location = sum(1 for q in queries for org in q['organisations'] if org.get('location'))
    total_orgs = sum(len(q['organisations']) for q in queries)

    if total_orgs > 0:
        print(f"Total org entries: {total_orgs}")
        print(f"  With type:      {org_with_type:3d} ({org_with_type/total_orgs*100:5.1f}%)")
        print(f"  With name:      {org_with_name:3d} ({org_with_name/total_orgs*100:5.1f}%)")
        print(f"  With location:  {org_with_location:3d} ({org_with_location/total_orgs*100:5.1f}%)")
        
        # Location levels for orgs
        org_loc_levels = Counter(org.get('location_level') for q in queries 
                                 for org in q['organisations'] 
                                 if org.get('location_level'))
        if org_loc_levels:
            print("\n  Org location levels:")
            for level, count in org_loc_levels.most_common():
                print(f"    {level:15s}: {count:3d}")

    # 11. Project location filters
    print("\n" + "-" * 80)
    print("11. PROJECT LOCATION FILTERS")
    print("-" * 80)

    with_project_loc = sum(1 for q in queries if q['filters'].get('location'))
    print(f"Queries with project location: {with_project_loc:3d} ({with_project_loc/total*100:5.1f}%)")

    if with_project_loc > 0:
        loc_levels = Counter(q['filters'].get('location_level') for q in queries 
                            if q['filters'].get('location_level'))
        print("\nProject location levels:")
        for level, count in loc_levels.most_common():
            print(f"  {level:15s}: {count:3d}")

            

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_queries.py <json_file>")
        sys.exit(1)
    
    queries = load_queries(sys.argv[1])
    
    # Debug
    print(f"DEBUG: Loaded {len(queries)} items")
    if queries:
        print(f"DEBUG: First item type: {type(queries[0])}")
        print(f"DEBUG: First item keys: {list(queries[0].keys())}")
        if 'meta' in queries[0]:
            print(f"DEBUG: First item has 'meta'")
        else:
            print(f"DEBUG: First item MISSING 'meta'")
            print(f"DEBUG: First item: {json.dumps(queries[0], indent=2)[:500]}")
    print()
    
    print_statistics(queries)
