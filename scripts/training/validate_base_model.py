#!/usr/bin/env python3
"""
Test BASE models (no LoRA) using the same validation framework
Usage matches validate.py but loads base models without fine-tuning
"""

import sys
import argparse
from pathlib import Path

# Add project root to path to import validate module
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import everything from validate
from validate import (
    load_equivalences, EQUIVALENCES, SAMPLE_QUERIES,
    get_model_config, generate_prediction, parse_json_response,
    test_model, save_detailed_json, save_table_tsv,
    analyze_errors_detailed, save_error_analysis_detailed,
    print_results, print_error_summary_detailed,
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    torch, json, datetime
)

def load_base_model_only(model_path: str, quantize: str = None):
    """Load ONLY base model, no LoRA adapters"""
    print(f"Loading BASE model from {model_path}...")
    print("  Type: Base model (no fine-tuning)")
    
    quantization_config = None
    if quantize == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quantize == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    return model, tokenizer, model_path

def main():
    parser = argparse.ArgumentParser(description="Validate BASE model (no fine-tuning)")
    
    # Positional argument (for compatibility with shell script)
    parser.add_argument("model_path", nargs='?', default=None,
                       help="HuggingFace model name (e.g., ministral/Ministral-3b-instruct)")
    
    # Named arguments (matching validate.py)
    parser.add_argument("--model-path", dest="model_path_named", default=None,
                       help="Alternative way to specify model path")
    parser.add_argument("--test-file", default=None,
                       help="Test JSON file (default: use sample queries)")
    parser.add_argument("--output-dir", required=True,
                       help="Output directory for results")
    parser.add_argument("--prompt-file", default="data/training/salamandra_finetuning_prompt.txt",
                       help="System prompt file")
    parser.add_argument("--equiv-dir", default=None,
                       help="Equivalences directory (default: next to test file)")
    parser.add_argument("--temperature", type=float, default=0,
                       help="Generation temperature")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                       help="Maximum new tokens to generate")
    parser.add_argument("--quantize", choices=["none", "8bit", "4bit"], default="none",
                       help="Quantization mode")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    parser.add_argument("--debug", action="store_true",
                       help="Debug mode (show model responses)")
    
    args = parser.parse_args()
    
    # Determine model path (positional or named)
    model_path = args.model_path or args.model_path_named
    if not model_path:
        parser.error("model_path is required (positional or --model-path)")
    
    # Load equivalences (must be done before validation)
    global EQUIVALENCES
    test_file_for_equiv = args.test_file or "data/test/impulse_test.json"
    EQUIVALENCES = load_equivalences(test_file_for_equiv, args.equiv_dir)
    if args.verbose:
        equiv_source = args.equiv_dir or Path(test_file_for_equiv).parent / "equivalences"
        print(f"[OK] Loaded equivalences from {equiv_source}")
    
    # Load prompt
    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        system_prompt = f.read().strip()
    print(f"[OK] Loaded prompt ({len(system_prompt)} chars)")
    
    # Load BASE model only (no LoRA)
    model, tokenizer, base_model = load_base_model_only(
        model_path,
        None if args.quantize == "none" else args.quantize
    )
    
    # Get model config
    model_config = get_model_config(model_path)
    print(f"[OK] Model config: merge_system={model_config.get('merge_system_into_user', False)}")
    
    # Load test data
    if args.test_file:
        with open(args.test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        print(f"[OK] Loaded {len(test_data)} test cases")
    else:
        test_data = SAMPLE_QUERIES
        print(f"[OK] Using {len(test_data)} sample queries")
    
    # Run tests (reuse test_model from validate.py)
    results = test_model(model, tokenizer, test_data, system_prompt, args, model_config)
    
    # Save results
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_detailed_json(results['predictions'], output_dir / f'validation_detailed_{current_datetime}.json')
    save_table_tsv(results['predictions'], output_dir / f'validation_table_{current_datetime}.tsv')
    
    # Error analysis
    errors = analyze_errors_detailed(results['predictions'])
    save_error_analysis_detailed(errors, output_dir / f'error_analysis_{current_datetime}.txt')
    
    # Save stats with error counts
    with open(output_dir / f'validation_stats_{current_datetime}.json', 'w', encoding='utf-8') as f:
        stats_out = {k: v for k, v in results.items() if k != 'predictions'}
        stats_out['error_counts'] = {
            'invalid_json': len(errors['invalid_json']),
            'critical': {k: len(v) for k, v in errors['critical'].items()},
            'moderate': {k: len(v) for k, v in errors['moderate'].items()},
            'minor': {k: len(v) for k, v in errors['minor'].items()}
        }
        json.dump(stats_out, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Results saved to {output_dir}/")
    print_results(results)
    print_error_summary_detailed(errors)
    
    if results.get('total', 0) == 0:
        return 1
    return 0 if results.get('all_correct', 0) / results['total'] >= 0.5 else 1

if __name__ == "__main__":
    exit(main())
