#!/usr/bin/env python3
"""
IMPULSE Training Data Preparation for Salamandra Fine-tuning
Prepara datos en formato ChatML con el prompt completo de Salamandra
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import random
from src.impulse.config.model_config import get_model_config


def load_system_prompt(prompt_file: str) -> str:
    """
    Load system prompt from file (required)
    """
    if not prompt_file:
        raise ValueError(
            "System prompt file is required. "
            "Use --prompt-file to specify the path (e.g., data/training/salamandra_finetuning_prompt.txt)"
        )
    
    prompt_path = Path(prompt_file)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    if not content:
        raise ValueError(f"Prompt file is empty: {prompt_file}")
    
    return content


def create_training_example(query: Dict[Any, Any], system_prompt: str, model_name: str) -> Dict[str, Any]:
    """
    Crea un ejemplo en formato ChatML para el modelo especificado
    """
    
    # Input del usuario
    user_input = query['meta']['original_query']
    
    # Output esperado (JSON limpio sin campos internos)
    output_json = {
        "doc_type": query['doc_type'],
        "filters": query['filters'],
        "organisations": query['organisations'],
        "semantic_query": query['semantic_query'],
        "query_rewrite": query['query_rewrite'],
        "meta": {
            "lang": query['meta']['lang'],
            "original_query": query['meta']['original_query'],
            "notes": query['meta'].get('notes')
        }
    }
    
    # Get model-specific configuration
    model_config = get_model_config(model_name)
    
    # Create messages based on model requirements
    if model_config.get("merge_system_into_user", False):
        # Models like Gemma that don't support system role
        messages = [
            {"role": "user", "content": f"{system_prompt}\n\n{user_input}"},
            {"role": "assistant", "content": json.dumps(output_json, ensure_ascii=False, indent=2)}
        ]
    else:
        # Standard format (BSC, Salamandra, Llama, etc.)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": json.dumps(output_json, ensure_ascii=False, indent=2)}
        ]
    
    return {
        "id": query['meta']['id'],
        "messages": messages,
        "lang": query['meta']['lang']
    }


def load_batch_files(input_dir: str, pattern: str = "batch*.json") -> List[Dict[Any, Any]]:
    """Carga todos los archivos batch"""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    batch_files = sorted(input_path.glob(pattern))
    
    if not batch_files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {input_dir}")
    
    all_queries = []
    
    for batch_file in batch_files:
        try:
            with open(batch_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Handle both list and dict with 'queries' key
                if isinstance(data, list):
                    all_queries.extend(data)
                elif isinstance(data, dict) and 'queries' in data:
                    all_queries.extend(data['queries'])
                else:
                    print(f"Warning: Unexpected format in {batch_file.name}, skipping...")
        except Exception as e:
            print(f"Error loading {batch_file.name}: {e}")
    
    return all_queries


def validate_query(query: Dict[Any, Any]) -> List[str]:
    """Valida la estructura básica de un query"""
    errors = []
    
    required_fields = ['doc_type', 'filters', 'organisations', 'semantic_query', 
                      'query_rewrite', 'meta']
    
    for field in required_fields:
        if field not in query:
            errors.append(f"Missing field '{field}'")
    
    if 'meta' in query:
        required_meta = ['id', 'lang', 'original_query']
        for field in required_meta:
            if field not in query['meta']:
                errors.append(f"Missing meta.{field}")
    
    return errors


def compute_statistics(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calcula estadísticas del dataset"""
    stats = {
        'total': len(examples),
        'by_language': defaultdict(int),
    }
    
    for example in examples:
        stats['by_language'][example['lang']] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepare IMPULSE training data for model fine-tuning"
    )
    parser.add_argument("--input-dir", required=True, 
                       help="Directory containing batch JSON files")
    parser.add_argument("--output-file", required=True, 
                       help="Output JSONL file path")
    parser.add_argument("--prompt-file", required=True,
                       help="Path to system prompt file (REQUIRED)")
    parser.add_argument("--model", required=True,
                       help="Model name (e.g., google/gemma-7b-it, BSC-LT/salamandra-7b-instruct)")
    parser.add_argument("--pattern", default="batch*.json", 
                       help="File pattern to match (default: batch*.json)")
    parser.add_argument("--validate", action="store_true", 
                       help="Validate JSON structure")
    parser.add_argument("--shuffle", action="store_true", 
                       help="Shuffle training examples")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for shuffling")
    
    args = parser.parse_args()
    
    print("IMPULSE Training Data Preparation")
    print("=" * 60)
    
    # Get model configuration
    model_config = get_model_config(args.model)
    print(f"✓ Model: {args.model}")
    print(f"  Message format: {'merged system+user' if model_config.get('merge_system_into_user') else 'separate system/user/assistant'}")
    
    # Load system prompt
    try:
        system_prompt = load_system_prompt(args.prompt_file)
        print(f"✓ Loaded system prompt from: {args.prompt_file}")
        print(f"  Prompt size: {len(system_prompt)} chars")
    except (ValueError, FileNotFoundError) as e:
        print(f"✗ Error loading prompt: {e}")
        return 1
    
    # Load all batch files
    print(f"\nLoading batch files from: {args.input_dir}")
    print(f"Pattern: {args.pattern}")
    
    try:
        queries = load_batch_files(args.input_dir, args.pattern)
        print(f"✓ Loaded {len(queries)} queries")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return 1
    
    # Validate if requested
    if args.validate:
        print("\nValidating queries...")
        total_errors = 0
        for i, query in enumerate(queries):
            errors = validate_query(query)
            if errors:
                total_errors += len(errors)
                print(f"  Query {i}: {', '.join(errors)}")
        
        if total_errors > 0:
            print(f"✗ Found {total_errors} validation errors")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return 1
        else:
            print("✓ All queries valid")
    
    # Create training examples
    print("\nCreating training examples...")
    examples = []
    for query in queries:
        example = create_training_example(query, system_prompt, args.model)
        examples.append(example)
    
    # Shuffle if requested
    if args.shuffle:
        print(f"Shuffling with seed {args.seed}...")
        random.seed(args.seed)
        random.shuffle(examples)
    
    # Compute statistics
    stats = compute_statistics(examples)
    
    # Save to JSONL
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUCCESS")
    print("=" * 60)
    print(f"✓ Created {len(examples)} training examples")
    print(f"✓ Languages: {dict(stats['by_language'])}")
    print(f"✓ Model: {args.model}")
    print(f"✓ Format: {'2 messages (user+system, assistant)' if model_config.get('merge_system_into_user') else '3 messages (system, user, assistant)'}")
    print(f"✓ Output: {args.output_file}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
