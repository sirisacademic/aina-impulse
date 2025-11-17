#!/usr/bin/env python3
"""
Enhanced training script with early stopping and better monitoring
Supports multiple model architectures via configuration dictionary
"""

import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import argparse
import numpy as np

# Import shared model configuration
from src.impulse.config.model_config import get_model_config

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_dataset(file_path: str) -> Dataset:
    """Load JSONL dataset"""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))
    
    print(f"Loaded {len(examples)} training examples")
    return Dataset.from_list(examples)


def validate_training_data_format(dataset: Dataset, model_config: dict, sample_size: int = 5):
    """Verify training data matches model's expected format"""
    expected_roles = ["user", "assistant"] if model_config.get("merge_system_into_user") else ["system", "user", "assistant"]
    
    errors = []
    for i in range(min(sample_size, len(dataset))):
        messages = dataset[i]['messages']
        actual_roles = [m['role'] for m in messages]
        
        if actual_roles != expected_roles:
            errors.append(f"Example {i}: expected {expected_roles}, got {actual_roles}")
    
    if errors:
        print("\nWARNING: Training data format mismatch!")
        for error in errors:
            print(f"  {error}")
        print(f"\nModel expects: {expected_roles}")
        print("Regenerate training data with correct --model parameter:")
        print(f"  python scripts/training/prepare_training_data.py --model {model_config.get('model_name', 'MODEL_NAME')}")
        raise ValueError("Training data format doesn't match model requirements")
    
    print(f"✓ Training data format validated ({expected_roles})")


def print_trainable_parameters(model):
    """Print trainable vs total parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"Trainable params: {trainable:,} || "
          f"Total params: {total:,} || "
          f"Trainable%: {100 * trainable / total:.2f}%")


def compute_metrics(eval_pred):
    """Compute perplexity metric"""
    logits, labels = eval_pred
    
    # Calculate loss manually
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
    
    # Flatten for loss calculation
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    
    # Filter out padding (-100)
    mask = shift_labels != -100
    shift_logits = shift_logits[mask]
    shift_labels = shift_labels[mask]
    
    loss = loss_fct(shift_logits, shift_labels)
    perplexity = torch.exp(loss)
    
    return {"perplexity": perplexity.item()}


def average_checkpoints(checkpoint_paths: list, output_path: str):
    """Average multiple checkpoint weights to create a more robust model"""
    from safetensors.torch import load_file, save_file
    import shutil
    
    print(f"\nAveraging {len(checkpoint_paths)} checkpoints...")
    
    # Load adapter config to get base model
    adapter_config_path = Path(checkpoint_paths[0]) / "adapter_config.json"
    with open(adapter_config_path, 'r') as f:
        config = json.load(f)
        base_model_name = config.get("base_model_name_or_path")
    
    # Load all adapter weights (adapter only, not base model)
    print("Loading checkpoint weights...")
    state_dicts = []
    for i, path in enumerate(checkpoint_paths):
        print(f"  Loading checkpoint {i+1}/{len(checkpoint_paths)}: {Path(path).name}")
        adapter_path = Path(path) / "adapter_model.safetensors"
        adapter_weights = load_file(str(adapter_path))
        state_dicts.append({k: v.cpu() for k, v in adapter_weights.items()})
        del adapter_weights
        torch.cuda.empty_cache()
    
    # Average the weights
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
    
    print(f"Averaged model saved to {output_path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM for IMPULSE")
    
    # Model arguments
    parser.add_argument("--model", default="BSC-LT/salamandra-7b-instruct", help="Base model")
    parser.add_argument("--data", required=True, help="Training JSONL file")
    parser.add_argument("--output-dir", default="./impulse-llm-ft", help="Output directory")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--quantize", choices=["none", "8bit", "4bit"], default="none", help="Quantization")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    
    # LoRA configuration
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA r")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    
    # Other options
    parser.add_argument("--eval-split", type=float, default=0.1, help="Validation split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-wandb", action="store_true", help="Use W&B logging")
    parser.add_argument("--early-stopping-patience", type=int, default=3, 
                       help="Early stopping patience (0 to disable)")
    parser.add_argument("--eval-accumulation-steps", type=int, default=None,
                       help="Number of eval steps to accumulate before moving to CPU (saves GPU memory)")
    parser.add_argument("--skip-perplexity", action="store_true",
                       help="Skip perplexity calculation to save memory")
    
    # Improvements
    parser.add_argument("--lr-scheduler", choices=["linear", "cosine"], default="cosine",
                       help="Learning rate scheduler type")
    parser.add_argument("--gradient-clip", type=float, default=1.0,
                       help="Gradient clipping max norm")
    parser.add_argument("--average-checkpoints", type=int, default=3,
                       help="Average last N checkpoints (0 to disable)")
    
    args = parser.parse_args()
    
    print("╔" + "═"*58 + "╗")
    print("║  IMPULSE LLM Fine-tuning with LoRA                      ║")
    print("╚" + "═"*58 + "╝")
    print(f"\nModel: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output_dir}\n")
    
    # Get model-specific configuration
    model_config = get_model_config(args.model)
    model_config['model_name'] = args.model  # Store for validation error messages
    print(f"Model config:")
    print(f"  Target modules: {model_config['target_modules']}")
    print(f"  8-bit support: {model_config['supports_8bit']}")
    print(f"  Message format: {'2-role (merged)' if model_config.get('merge_system_into_user') else '3-role (standard)'}")
    
    # Validate quantization
    if args.quantize == "8bit" and not model_config["supports_8bit"]:
        print(f"\nWARNING: {args.model} may not support 8-bit quantization well")
        print("   Consider using 4-bit or no quantization instead")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return 1
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load model
    print("Loading model...")
    quantization_config = None
    if args.quantize == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif args.quantize == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config
    )
    
    # Configure LoRA
    print("\nConfiguring LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=model_config["target_modules"]
    )
    
    print(f"  LoRA r: {args.lora_r}")
    print(f"  LoRA alpha: {args.lora_alpha}")
    print(f"  Target modules: {model_config['target_modules']}")
    
    # Prepare for quantization and apply LoRA
    if args.quantize in ["4bit", "8bit"]:
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        # Enable gradient checkpointing only with quantization
        if args.quantize != "8bit":
            model.gradient_checkpointing_enable()
    else:
        # No gradient checkpointing for full precision
        model = get_peft_model(model, peft_config)

    print_trainable_parameters(model)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(args.data)
    
    # Validate data format matches model
    validate_training_data_format(dataset, model_config)
    
    def tokenize_function(examples):
        """Tokenize messages - data already in correct format"""
        texts = []
        
        for messages in examples['messages']:
            # Messages already in correct format from prepare_training_data.py
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        
        model_inputs = tokenizer(
            texts,
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None
        )
        
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    # Split train/eval
    print(f"Splitting dataset (eval={args.eval_split*100:.0f}%)...")
    split_dataset = tokenized_dataset.train_test_split(
        test_size=args.eval_split, seed=args.seed
    )
    
    print(f"  Train: {len(split_dataset['train'])} examples")
    print(f"  Eval: {len(split_dataset['test'])} examples")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=args.average_checkpoints if args.average_checkpoints > 0 else 3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if args.use_wandb else "none",
        run_name=f"impulse-{args.epochs}ep" if args.use_wandb else None,
        push_to_hub=False,
        logging_first_step=True,
        seed=args.seed,
        prediction_loss_only=True,
        eval_accumulation_steps=args.eval_accumulation_steps,
        max_grad_norm=args.gradient_clip,
        lr_scheduler_type=args.lr_scheduler,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
    )
    
    # Callbacks
    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience
        ))
        print(f"\nEarly stopping enabled (patience={args.early_stopping_patience})")
    
    print(f"Gradient clipping enabled (max_norm={args.gradient_clip})")
    print(f"LR scheduler: {args.lr_scheduler}")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=None if args.skip_perplexity else compute_metrics,
    )
    
    if args.skip_perplexity:
        print("  Perplexity calculation disabled (--skip-perplexity)")
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    train_result = trainer.train()
    
    # Save model
    print(f"\nSaving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Save training arguments with model config
    training_info = vars(args)
    training_info['model_config'] = model_config
    with open(Path(args.output_dir) / "training_args.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    # Checkpoint averaging
    if args.average_checkpoints > 0:
        checkpoint_dirs = sorted(
            [d for d in Path(args.output_dir).iterdir() 
             if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda x: int(x.name.split("-")[-1])
        )
        
        if len(checkpoint_dirs) >= args.average_checkpoints:
            print(f"\n{'='*60}")
            print(f"Averaging last {args.average_checkpoints} checkpoints...")
            print('='*60)
            
            checkpoints_to_avg = checkpoint_dirs[-args.average_checkpoints:]
            avg_output = Path(args.output_dir) / "averaged_model"
            
            try:
                average_checkpoints([str(d) for d in checkpoints_to_avg], str(avg_output))
                print(f"\nAveraged model can be merged with:")
                print(f"   python scripts/training/merge_lora.py \\")
                print(f"     --adapter-path {avg_output} \\")
                print(f"     --output-path models/your-model-averaged-merged")
            except Exception as e:
                print(f" Checkpoint averaging failed: {e}")
        else:
            print(f"\n Not enough checkpoints for averaging ({len(checkpoint_dirs)} < {args.average_checkpoints})")
    
    print("\n" + "="*60)
    print("Fine-tuning complete!")
    print("="*60)
    print(f"Model saved to: {args.output_dir}")
    print(f"Training loss: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"\nValidate: python scripts/training/validate.py")
    print(f"  --model-path {args.output_dir}")
    print(f"  --prompt-file data/training/salamandra_finetuning_prompt.txt")


if __name__ == "__main__":
    exit(main())
