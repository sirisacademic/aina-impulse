#!/usr/bin/env python3
"""
Query Expansion Test Script v3.0

Tests:
- Centroid-based clustering
- Different thresholds for aliases vs parents
- Language-aware representative selection

Author: IMPULSE Project
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.impulse.query_expansion.loader import load_kb
from src.impulse.query_expansion.expansion import (
    expand_query_with_vectors,
    ExpansionConfig,
    build_kb_index,
    get_expansion_summary
)
from src.impulse.embedding.embedder import Embedder
from src.impulse.settings import settings


TEST_QUERIES = [
    # (query, language) - language simulates parser output
    ("machine learning", "en"),
    ("aprendizaje automático", "es"),
    ("deep learning", "en"),
    ("blockchain", "en"),
    ("cancer", "en"),
    ("renewable energy", "en"),
    ("artificial intelligence", "en"),
    ("inteligencia artificial", "es"),  # Same concept, Spanish query
    ("patient", "en"),
    ("network", "en"),
]


def print_separator(char="=", length=70):
    print(char * length)


def format_cluster(cluster: dict, indent: str = "    ") -> str:
    """Format a cluster for display."""
    rep = cluster["representative"]
    sim = cluster["query_similarity"]
    lang = cluster.get("representative_language", "?")
    terms = cluster["terms"]
    
    # Show representative with language tag
    result = f'{indent}● "{rep}" [{lang.upper()}] (sim={sim:.3f})'
    
    # Show other terms if any
    other_terms = [t for t in terms if t != rep]
    if other_terms:
        # Truncate if too many
        if len(other_terms) > 5:
            display_terms = other_terms[:5] + [f"+{len(other_terms)-5} more"]
        else:
            display_terms = other_terms
        result += f'\n{indent}  └─ {", ".join(display_terms)}'
    
    return result


def test_single_query(
    query: str,
    query_language: str,
    kb: list,
    kb_index: dict,
    embedder: Embedder,
    config: ExpansionConfig
):
    """Test expansion for a single query."""
    print_separator()
    print(f'QUERY: "{query}" [lang={query_language}]')
    print_separator("-", 50)
    
    result = expand_query_with_vectors(
        query=query,
        kb=kb,
        encoder=embedder,
        config=config,
        kb_index=kb_index,
        query_language=query_language
    )
    
    # Matched concepts
    matched = result.get("matched_concepts", [])
    if matched:
        print(f"\n✓ Matched {len(matched)} concept(s):")
        for m in matched:
            print(f"    {m['wikidata_id']}: {m['keyword']}")
    else:
        print("\n✗ No matching concepts in KB")
        return result
    
    # Parent concepts
    parents = result.get("parent_concepts", [])
    if parents:
        print(f"\n↑ Parent concepts ({len(parents)}):")
        for p in parents:
            print(f"    {p['wikidata_id']}: {p['keyword']}")
    
    # Alias expansion
    print(f"\n📖 ALIAS EXPANSION (thresholds: {config.alias_level_boundaries}):")
    alias_exp = result.get("alias_expansion", {})
    for level_num, label in [(1, "exact"), (2, "related"), (3, "broader")]:
        level_key = f"level_{level_num}"
        clusters = alias_exp.get(level_key, [])
        if clusters:
            print(f"  Level {level_num} ({label}): {len(clusters)} cluster(s)")
            for cluster in clusters:
                print(format_cluster(cluster))
        else:
            print(f"  Level {level_num} ({label}): (empty)")
    
    # Parent expansion
    print(f"\n🔗 PARENT EXPANSION (thresholds: {config.parent_level_boundaries}):")
    parent_exp = result.get("parent_expansion", {})
    for level_num, label in [(1, "exact"), (2, "related"), (3, "broader")]:
        level_key = f"level_{level_num}"
        clusters = parent_exp.get(level_key, [])
        if clusters:
            print(f"  Level {level_num} ({label}): {len(clusters)} cluster(s)")
            for cluster in clusters:
                print(format_cluster(cluster))
        else:
            print(f"  Level {level_num} ({label}): (empty)")
    
    print()
    return result


def main():
    print("\n" + "=" * 70)
    print("IMPULSE Query Expansion Test (v3.0)")
    print("Centroid clustering + Different thresholds + Language-aware reps")
    print("=" * 70)
    
    # Load KB
    kb_path = settings.KB_PATH
    print(f"\n📂 Loading KB from: {kb_path}")
    kb = load_kb(kb_path)
    print(f"   ✓ Loaded {len(kb)} records")
    
    # Build index
    kb_index = build_kb_index(kb)
    print(f"   ✓ Indexed {len(kb_index)} unique IDs")
    
    # Initialize embedder
    print(f"\n🧠 Initializing embedder: {settings.embedder_model_name}")
    embedder = Embedder(model_name=settings.embedder_model_name, device=None)
    print(f"   ✓ Embedder ready")
    
    # Configuration
    config = ExpansionConfig(
        use_aliases=True,
        use_parents=True,
        use_definitions=False,
        only_parents_in_kb=True,
        max_parents=3,
        languages=["en", "es", "ca"],
        alias_level_boundaries=[0.85, 0.50, 0.30],
        parent_level_boundaries=[0.75, 0.50, 0.30],
        #cluster_threshold=0.70,
        cluster_threshold=0.60,
        max_clusters_per_level=5,
        language_preference_tolerance=0.05
    )
    
    print(f"\n⚙️  Configuration:")
    print(f"   Alias thresholds: {config.alias_level_boundaries}")
    print(f"   Parent thresholds: {config.parent_level_boundaries}")
    print(f"   Cluster threshold: {config.cluster_threshold}")
    print(f"   Language tolerance: {config.language_preference_tolerance}")
    
    # Test queries
    print("\n")
    
    for query, lang in TEST_QUERIES:
        test_single_query(query, lang, kb, kb_index, embedder, config)
    
    print_separator("=")
    print("TEST COMPLETE")
    print_separator("=")


if __name__ == "__main__":
    main()
