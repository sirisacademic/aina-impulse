#!/usr/bin/env python3
"""
Script to merge LoRA adapters back into the base model
Creates a standalone model without requiring PEFT for inference
"""

import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import shutil


def load_adapter_config(adapter_path: str) -> dict:
    """Load adapter configuration"""
    config_path = Path(adapter_path) / "adapter_config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Adapter config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)


def merge_lora_model(adapter_path: str, output_path: str, base_model: str = None):
    """
    Merge LoRA adapters into base model
    
    Args:
        adapter_path: Path to LoRA adapter
        output_path: Where to save merged model
        base_model: Base model name/path (optional, will try to detect)
    """
    
    print("LoRA Adapter Merger")
    print("=" * 60)
    
    # Load adapter config
    print(f"Loading adapter config from {adapter_path}...")
    adapter_config = load_adapter_config(adapter_path)
    
    # Determine base model
    if not base_model:
        base_model = adapter_config.get("base_model_name_or_path")
        if not base_model:
            raise ValueError("Base model not found in config. Please specify with --base-model")
    
    print(f"Base model: {base_model}")
    print(f"Output path: {output_path}")
    
    # Load base model
    print("\nLoading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Load LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, adapter_path)
    
    # Print adapter info
    print("\nAdapter information:")
    print(f"  LoRA r: {adapter_config.get('r', 'N/A')}")
    print(f"  LoRA alpha: {adapter_config.get('lora_alpha', 'N/A')}")
    print(f"  Target modules: {adapter_config.get('target_modules', [])}")
    
    # Merge and unload
    print("\nMerging LoRA weights into base model...")
    merged_model = model.merge_and_unload()
    
    # Save merged model
    print(f"Saving merged model to {output_path}...")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Save model
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="5GB"
    )
    
    # Save tokenizer
    tokenizer.save_pretrained(output_path)
    
    # Copy training info if exists
    training_args_path = Path(adapter_path) / "training_args.json"
    if training_args_path.exists():
        shutil.copy(training_args_path, Path(output_path) / "training_args.json")
        print("Copied training arguments")
    
    # Create merge info
    merge_info = {
        "base_model": base_model,
        "adapter_path": str(adapter_path),
        "adapter_config": adapter_config,
        "merge_dtype": "bfloat16",
        "merged_by": "impulse_merge_lora.py"
    }
    
    with open(Path(output_path) / "merge_info.json", 'w') as f:
        json.dump(merge_info, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Merge complete!")
    print("=" * 60)
    print(f"Merged model saved to: {output_path}")
    print(f"Model size: ~{sum(p.numel() for p in merged_model.parameters()) / 1e9:.1f}B parameters")
    print("\nThe merged model can be used directly without PEFT:")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{output_path}')")
    

def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters into base model for standalone deployment"
    )
    
    parser.add_argument(
        "--adapter-path",
        required=True,
        help="Path to LoRA adapter directory"
    )
    
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output path for merged model"
    )
    
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model name/path (optional, will try to detect from adapter config)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output directory if it exists"
    )
    
    args = parser.parse_args()
    
    # Check paths
    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        print(f"Error: Adapter path does not exist: {adapter_path}")
        return 1
    
    output_path = Path(args.output_path)
    if output_path.exists() and not args.force:
        print(f"Error: Output path already exists: {output_path}")
        print("Use --force to overwrite")
        return 1
    
    try:
        merge_lora_model(
            str(adapter_path),
            str(output_path),
            args.base_model
        )
        return 0
    except Exception as e:
        print(f"\nError during merge: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
