#!/usr/bin/env python3
"""
Average existing checkpoints without retraining
"""

import argparse
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer
from safetensors.torch import load_file, save_file
import shutil


def average_checkpoints(checkpoint_paths: list, output_path: str):
    """Average multiple checkpoint weights"""
    print(f"\nAveraging {len(checkpoint_paths)} checkpoints...")
    
    # Load adapter config to get base model
    adapter_config_path = Path(checkpoint_paths[0]) / "adapter_config.json"
    with open(adapter_config_path, 'r') as f:
        config = json.load(f)
        base_model_name = config.get("base_model_name_or_path")
    
    print(f"Base model: {base_model_name}")
    
    # Load all adapter weights (adapter only, not base model)
    print("Loading checkpoint weights...")
    state_dicts = []
    for i, path in enumerate(checkpoint_paths):
        print(f"  [{i+1}/{len(checkpoint_paths)}] {Path(path).name}")
        # Load only adapter weights from safetensors
        from safetensors.torch import load_file
        adapter_path = Path(path) / "adapter_model.safetensors"
        adapter_weights = load_file(str(adapter_path))
        state_dicts.append({k: v.cpu() for k, v in adapter_weights.items()})
        del adapter_weights
        torch.cuda.empty_cache()
    
    # Average weights
    print("Averaging weights...")
    avg_state_dict = {}
    for key in state_dicts[0].keys():
        avg_state_dict[key] = torch.stack([sd[key] for sd in state_dicts]).mean(dim=0)
    
    del state_dicts
    torch.cuda.empty_cache()
    
    # Save averaged adapter
    print("Saving averaged adapter...")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Save adapter weights
    save_file(avg_state_dict, Path(output_path) / "adapter_model.safetensors")
    
    # Copy adapter config and other files
    for file in ["adapter_config.json", "README.md"]:
        src = Path(checkpoint_paths[0]) / file
        if src.exists():
            shutil.copy(src, Path(output_path) / file)
    
    # Copy tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_paths[0])
    tokenizer.save_pretrained(output_path)
    
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Average existing checkpoints")
    parser.add_argument("--model-dir", required=True, 
                       help="Model directory containing checkpoints")
    parser.add_argument("--output-dir", required=True,
                       help="Output directory for averaged model")
    parser.add_argument("--num-checkpoints", type=int, default=3,
                       help="Number of last checkpoints to average")
    
    args = parser.parse_args()
    
    # Find checkpoints
    model_path = Path(args.model_dir)
    checkpoint_dirs = sorted(
        [d for d in model_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda x: int(x.name.split("-")[-1])
    )
    
    if len(checkpoint_dirs) < args.num_checkpoints:
        print(f"Error: Found {len(checkpoint_dirs)} checkpoints, need {args.num_checkpoints}")
        return 1
    
    checkpoints_to_avg = checkpoint_dirs[-args.num_checkpoints:]
    print(f"Averaging checkpoints:")
    for cp in checkpoints_to_avg:
        print(f"  - {cp.name}")
    
    average_checkpoints([str(d) for d in checkpoints_to_avg], args.output_dir)
    
    print(f"\nDone! Merge with:")
    print(f"  python scripts/training/merge_lora.py \\")
    print(f"    --adapter-path {args.output_dir} \\")
    print(f"    --output-path models/your-model-averaged-merged")
    
    return 0


if __name__ == "__main__":
    exit(main())
