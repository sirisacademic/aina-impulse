#!/usr/bin/env python3
"""
Generate enhanced comparative evaluation report with multilingual metrics
"""
import json
import sys
from pathlib import Path
from collections import defaultdict


def main():
    if len(sys.argv) < 2:
        print("Usage: generate_comparative_report.py <result_dir1> [result_dir2 ...] [--output <file>]")
        sys.exit(1)
    
    result_dirs = sys.argv[1:]
    output_file = None
    
    # Check for --output flag
    if "--output" in result_dirs:
        idx = result_dirs.index("--output")
        if idx + 1 < len(result_dirs):
            output_file = result_dirs[idx + 1]
            result_dirs = result_dirs[:idx] + result_dirs[idx + 2:]
    
    # Load model data
    models_data = []
    for result_dir in result_dirs:
        result_path = Path(result_dir)
        if not result_path.exists():
            continue
        
        stats_files = sorted(result_path.glob("validation_stats_*.json"))
        if stats_files:
            with open(stats_files[-1]) as f:
                model_name = result_path.parent.name
                models_data.append({
                    'name': model_name,
                    'stats': json.load(f),
                    'path': str(result_path)
                })
    
    if not models_data:
        print("No results found")
        sys.exit(0)
    
    # Generate report
    output = []
    output.append("=" * 100)
    output.append("COMPARATIVE MODEL EVALUATION - MULTILINGUAL METRICS")
    output.append("=" * 100 + "\n")
    
    # Overall accuracy table
    output.append("OVERALL ACCURACY")
    output.append("-" * 100)
    output.append(f"{'Model':<35} {'Valid JSON':<15} {'Lang Match':<15} {'Strict':<15} {'Relaxed':<15}")
    output.append("-" * 100)
    
    for m in sorted(models_data, key=lambda x: x['stats'].get('all_correct_relaxed', x['stats'].get('all_correct', 0))/x['stats']['total'], reverse=True):
        s = m['stats']
        total = s['total']
        
        json_pct = s['valid_json'] / total * 100
        lang_pct = s.get('language_correct', 0) / total * 100
        
        # Handle both old and new stats format
        if 'all_correct_strict' in s:
            strict_pct = s['all_correct_strict'] / total * 100
            relaxed_pct = s['all_correct_relaxed'] / total * 100
            strict_str = f"{s['all_correct_strict']}/{total} ({strict_pct:>5.1f}%)"
            relaxed_str = f"{s['all_correct_relaxed']}/{total} ({relaxed_pct:>5.1f}%)"
        else:
            # Old format - use all_correct as fallback
            acc_pct = s.get('all_correct', 0) / total * 100
            strict_str = f"{s.get('all_correct', 0)}/{total} ({acc_pct:>5.1f}%)"
            relaxed_str = "N/A"
        
        output.append(
            f"{m['name'][:33]:<35} "
            f"{s['valid_json']}/{total} ({json_pct:>5.1f}%)  "
            f"{s.get('language_correct', 0)}/{total} ({lang_pct:>5.1f}%)  "
            f"{strict_str:<14} "
            f"{relaxed_str}"
        )
    
    # By-language breakdown
    output.append("\n\nACCURACY BY LANGUAGE")
    output.append("-" * 100)
    
    # Collect all languages
    all_langs = set()
    for m in models_data:
        if 'by_language' in m['stats']:
            all_langs.update(m['stats']['by_language'].keys())
    
    if all_langs:
        for lang in sorted(all_langs):
            output.append(f"\n{lang}:")
            output.append(f"{'Model':<35} {'Total':<10} {'Lang Match':<15} {'Strict':<15} {'Relaxed'}")
            output.append("-" * 100)
            
            for m in sorted(models_data, key=lambda x: x['name']):
                s = m['stats']
                if 'by_language' not in s or lang not in s['by_language']:
                    continue
                
                lang_stats = s['by_language'][lang]
                total = lang_stats['total']
                
                lang_match = lang_stats.get('language_correct', 0)
                lang_pct = lang_match / total * 100 if total > 0 else 0
                
                strict = lang_stats.get('strict', 0)
                strict_pct = strict / total * 100 if total > 0 else 0
                
                relaxed = lang_stats.get('relaxed', lang_stats.get('all_correct', 0))
                relaxed_pct = relaxed / total * 100 if total > 0 else 0
                
                output.append(
                    f"{m['name'][:33]:<35} "
                    f"{total:<10} "
                    f"{lang_match}/{total} ({lang_pct:>5.1f}%)  "
                    f"{strict}/{total} ({strict_pct:>5.1f}%)  "
                    f"{relaxed}/{total} ({relaxed_pct:>5.1f}%)"
                )
    
    # Component accuracy
    output.append("\n\nCOMPONENT ACCURACY")
    output.append("-" * 100)
    output.append(f"{'Model':<30} {'Prog':<8} {'Year':<8} {'Loc':<8} {'Orgs(E)':<10} {'Orgs(R)':<10} {'Semantic'}")
    output.append("-" * 100)
    
    for m in sorted(models_data, key=lambda x: x['name']):
        s = m['stats']
        total = s['total']
        
        prog_pct = s['filters']['programme'] / total * 100
        year_pct = s['filters']['year'] / total * 100
        loc_pct = s['filters']['location'] / total * 100
        
        # Handle both old and new format
        orgs_exact = s.get('organisations_exact', s.get('organisations', 0))
        orgs_relaxed = s.get('organisations_relaxed', s.get('organisations', 0))
        
        orgs_e_pct = orgs_exact / total * 100
        orgs_r_pct = orgs_relaxed / total * 100
        sem_pct = s['semantic_query'] / total * 100
        
        output.append(
            f"{m['name'][:28]:<30} "
            f"{prog_pct:>5.1f}%   "
            f"{year_pct:>5.1f}%   "
            f"{loc_pct:>5.1f}%   "
            f"{orgs_e_pct:>5.1f}%     "
            f"{orgs_r_pct:>5.1f}%     "
            f"{sem_pct:>5.1f}%"
        )
    
    # Error analysis
    output.append("\n\nERROR ANALYSIS")
    output.append("-" * 100)
    output.append(f"{'Model':<30} {'Invalid':<10} {'Lang Mis':<10} {'Critical':<10} {'Moderate':<10} {'Minor'}")
    output.append("-" * 100)
    
    for m in sorted(models_data, key=lambda x: x['name']):
        if 'error_counts' not in m['stats']:
            continue
        
        ec = m['stats']['error_counts']
        invalid = ec.get('invalid_json', 0)
        lang_mis = ec.get('language_mismatch', 0)
        critical_total = sum(ec['critical'].values())
        moderate_total = sum(ec['moderate'].values())
        minor_total = sum(ec['minor'].values())
        
        output.append(
            f"{m['name'][:28]:<30} "
            f"{invalid:<10} "
            f"{lang_mis:<10} "
            f"{critical_total:<10} "
            f"{moderate_total:<10} "
            f"{minor_total}"
        )
    
    # Detailed error breakdown
    output.append("\n\nDETAILED ERROR BREAKDOWN")
    output.append("-" * 100)
    
    for m in sorted(models_data, key=lambda x: x['name']):
        if 'error_counts' not in m['stats']:
            continue
        
        ec = m['stats']['error_counts']
        output.append(f"\n{m['name']}:")
        
        if ec.get('invalid_json', 0) > 0:
            output.append(f"  Invalid JSON: {ec['invalid_json']}")
        
        if ec.get('language_mismatch', 0) > 0:
            output.append(f"  Language Mismatch (content correct): {ec['language_mismatch']}")
        
        if ec['critical']:
            output.append(f"  Critical errors:")
            for err_type, count in sorted(ec['critical'].items()):
                if count > 0:
                    output.append(f"    - {err_type.replace('_', ' ').title()}: {count}")
        
        if ec['moderate']:
            output.append(f"  Moderate errors:")
            for err_type, count in sorted(ec['moderate'].items()):
                if count > 0:
                    output.append(f"    - {err_type.replace('_', ' ').title()}: {count}")
        
        if ec['minor']:
            output.append(f"  Minor errors:")
            for err_type, count in sorted(ec['minor'].items()):
                if count > 0:
                    output.append(f"    - {err_type.replace('_', ' ').title()}: {count}")
    
    # Summary statistics
    output.append("\n\nSUMMARY STATISTICS")
    output.append("-" * 100)
    
    best_strict = max(models_data, key=lambda x: x['stats'].get('all_correct_strict', x['stats'].get('all_correct', 0))/x['stats']['total'])
    best_relaxed = max(models_data, key=lambda x: x['stats'].get('all_correct_relaxed', x['stats'].get('all_correct', 0))/x['stats']['total'])
    best_lang = max(models_data, key=lambda x: x['stats'].get('language_correct', 0)/x['stats']['total'])
    
    output.append(f"Best strict accuracy: {best_strict['name']} ({best_strict['stats'].get('all_correct_strict', 0)}/{best_strict['stats']['total']})")
    output.append(f"Best relaxed accuracy: {best_relaxed['name']} ({best_relaxed['stats'].get('all_correct_relaxed', 0)}/{best_relaxed['stats']['total']})")
    output.append(f"Best language consistency: {best_lang['name']} ({best_lang['stats'].get('language_correct', 0)}/{best_lang['stats']['total']})")
    
    # Average scores
    avg_strict = sum(m['stats'].get('all_correct_strict', m['stats'].get('all_correct', 0))/m['stats']['total'] for m in models_data) / len(models_data) * 100
    avg_relaxed = sum(m['stats'].get('all_correct_relaxed', m['stats'].get('all_correct', 0))/m['stats']['total'] for m in models_data) / len(models_data) * 100
    avg_lang = sum(m['stats'].get('language_correct', 0)/m['stats']['total'] for m in models_data) / len(models_data) * 100
    
    output.append(f"\nAverage strict accuracy: {avg_strict:.1f}%")
    output.append(f"Average relaxed accuracy: {avg_relaxed:.1f}%")
    output.append(f"Average language consistency: {avg_lang:.1f}%")
    
    # Result locations
    output.append("\n\nRESULT LOCATIONS")
    output.append("-" * 100)
    for m in sorted(models_data, key=lambda x: x['name']):
        output.append(f"{m['name']}: {m['path']}")
    
    output.append("\n" + "=" * 100)
    
    result_text = "\n".join(output)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result_text)
        print(f"✓ Report: {output_file}")
    else:
        print(result_text)


if __name__ == "__main__":
    main()
