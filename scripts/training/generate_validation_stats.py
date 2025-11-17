#!/usr/bin/env python3
"""Regenerate error analysis and stats from existing validation_detailed JSON"""
import sys
import json
from pathlib import Path
from collections import defaultdict

# Add project root to path to import validate module
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import from validate module
from validate import analyze_errors_detailed, save_error_analysis_detailed

def compute_stats_from_predictions(predictions):
    """Compute complete statistics from prediction data"""
    total = len(predictions)
    valid_json = 0
    language_correct = 0
    all_correct_strict = 0
    all_correct_relaxed = 0
    
    filters = defaultdict(int)
    organisations_exact = 0
    organisations_relaxed = 0
    semantic_query = 0
    
    by_language = defaultdict(lambda: {
        'total': 0,
        'language_correct': 0,
        'strict': 0,
        'relaxed': 0
    })
    
    for pred in predictions:
        val = pred.get('validation', {})
        gold = pred.get('gold', {})
        
        if val.get('valid_json', False):
            valid_json += 1
        
        if val.get('language_correct', False):
            language_correct += 1
        
        if val.get('all_correct_strict', False):
            all_correct_strict += 1
        
        if val.get('all_correct_relaxed', False):
            all_correct_relaxed += 1
        
        # Count filter correctness
        for filter_name, is_correct in val.get('filters', {}).items():
            if is_correct:
                filters[filter_name] += 1
        
        if val.get('organisations_exact', False):
            organisations_exact += 1
        
        if val.get('organisations_relaxed', False):
            organisations_relaxed += 1
        
        if val.get('semantic_query', False):
            semantic_query += 1
        
        # By language stats
        lang = gold.get('meta', {}).get('lang', 'UNKNOWN')
        by_language[lang]['total'] += 1
        if val.get('language_correct', False):
            by_language[lang]['language_correct'] += 1
        if val.get('all_correct_strict', False):
            by_language[lang]['strict'] += 1
        if val.get('all_correct_relaxed', False):
            by_language[lang]['relaxed'] += 1
    
    return {
        'total': total,
        'valid_json': valid_json,
        'language_correct': language_correct,
        'all_correct_strict': all_correct_strict,
        'all_correct_relaxed': all_correct_relaxed,
        'filters': dict(filters),
        'organisations_exact': organisations_exact,
        'organisations_relaxed': organisations_relaxed,
        'semantic_query': semantic_query,
        'by_language': dict(by_language)
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/training/generate_validation_stats.py <validation_detailed.json>")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    output_dir = input_file.parent
    timestamp = input_file.stem.replace('validation_detailed_', '')
    
    print(f"Processing: {input_file}")
    
    # Load predictions
    with open(input_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    print(f"Loaded {len(predictions)} predictions")
    
    # Generate error analysis
    print("Analyzing errors...")
    errors = analyze_errors_detailed(predictions)
    save_error_analysis_detailed(errors, output_dir / f'error_analysis_{timestamp}.txt')
    print(f"✓ Generated: error_analysis_{timestamp}.txt")
    
    # Generate stats
    print("Computing statistics...")
    stats = compute_stats_from_predictions(predictions)
    stats['error_counts'] = {
        'invalid_json': len(errors['invalid_json']),
        'language_mismatch': len(errors['language_mismatch']),
        'critical': {k: len(v) for k, v in errors['critical'].items()},
        'moderate': {k: len(v) for k, v in errors['moderate'].items()},
        'minor': {k: len(v) for k, v in errors['minor'].items()}
    }
    
    with open(output_dir / f'validation_stats_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Generated: validation_stats_{timestamp}.json")
    print(f"\nSummary:")
    print(f"  Total: {stats['total']}")
    print(f"  Valid JSON: {stats['valid_json']}/{stats['total']} ({stats['valid_json']/stats['total']*100:.1f}%)")
    print(f"  Strict: {stats['all_correct_strict']}/{stats['total']} ({stats['all_correct_strict']/stats['total']*100:.1f}%)")
    print(f"  Relaxed: {stats['all_correct_relaxed']}/{stats['total']} ({stats['all_correct_relaxed']/stats['total']*100:.1f}%)")

if __name__ == "__main__":
    main()
