# IMPULSE Fine-tuning Setup and Execution Guide

This guide provides complete instructions for fine-tuning Salamandra 7B for the IMPULSE query parsing task.

## Prerequisites

### Hardware Requirements

- **Minimum**: 1x GPU with 24GB VRAM (e.g., RTX 3090, RTX 4090, A5000)
- **Recommended**: 1x GPU with 40GB+ VRAM (e.g., A100, H100)
- **RAM**: 32GB+ system memory
- **Storage**: 50GB+ free disk space

### Software Requirements

- Python 3.10+
- CUDA 11.8+ or 12.1+
- Git

## Installation

### 1. Create Virtual Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

### 2. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CUDA 12.1 example - adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Axolotl and dependencies
pip install axolotl[flash-attn,deepspeed]

# Install additional requirements
pip install transformers==4.36.2
pip install datasets
pip install peft
pip install accelerate
pip install bitsandbytes
pip install wandb  # Optional, for experiment tracking
```

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.1.0+cu121
CUDA available: True
```

## Project Structure

Your project should have this structure:

```
.
├── configs/
│   └── axolotl_config.yml          # Axolotl configuration
├── data/
│   └── training/
│       ├── batches/                # Raw batch files
│       └── impulse_training_data.jsonl  # Prepared training data
├── outputs/
│   ├── impulse-salamandra-lora/    # LoRA adapters (generated)
│   └── impulse-salamandra-merged/  # Merged model (generated)
├── scripts/
│   ├── prepare_training_data.py    # Data preparation script
│   ├── train.sh                    # Training script
│   ├── merge_lora.sh               # LoRA merge script
│   └── validate.py                 # Validation script
└── src/
    └── impulse/
        └── ...                     # Your application code
```

## Step-by-Step Execution

### Step 1: Prepare Training Data

First, combine all batch files into a single JSONL file:

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Prepare training data
python scripts/prepare_training_data.py \
    --input-dir data/training/batches \
    --output-file data/training/impulse_training_data.jsonl \
    --validate
```

**Expected output:**
```
IMPULSE Training Data Preparation
============================================================

Searching for batch files in: data/training/batches
Pattern: batch*.json
Found 31 batch files

Loading 31 batch files...
...
Loaded 500 total queries

============================================================
DATASET STATISTICS
============================================================

Total queries: 500

By Language:
  CA:  167 ( 33.4%)
  EN:  167 ( 33.4%)
  ES:  166 ( 33.2%)

By Resolvability:
  Direct      :  400 ( 80.0%)
  Adapted     :   75 ( 15.0%)
  Partial     :   25 (  5.0%)

...

Saved 500 examples to data/training/impulse_training_data.jsonl
```

**Verify the output:**
```bash
# Check file was created
ls -lh data/training/impulse_training_data.jsonl

# View first example
head -n 1 data/training/impulse_training_data.jsonl | jq .
```

### Step 2: Configure Training

The `axolotl_config.yml` file contains all training parameters. Key settings to adjust:

```yaml
# Batch size (adjust based on your GPU memory)
micro_batch_size: 2              # Decrease if OOM
gradient_accumulation_steps: 4   # Increase if decreasing micro_batch_size

# Training duration
num_epochs: 3                    # 3 epochs is usually enough

# LoRA parameters
lora_r: 16                       # Rank (16 is good balance)
lora_alpha: 32                   # Alpha (typically 2*r)

# Precision
bf16: true                       # Use bfloat16 if supported
```

**GPU Memory Guidelines:**

| GPU VRAM | micro_batch_size | gradient_accumulation_steps | Effective batch size |
|----------|------------------|----------------------------|---------------------|
| 24GB     | 1                | 8                          | 8                   |
| 40GB     | 2                | 4                          | 8                   |
| 80GB     | 4                | 2                          | 8                   |

### Step 3: Start Training

```bash
# Run training
bash scripts/train.sh
```

**What to expect:**

- Training will start with model loading (1-2 minutes)
- You'll see progress bars for each epoch
- Loss should decrease over time
- Training takes approximately:
  - 24GB GPU: ~3-4 hours for 3 epochs
  - 40GB GPU: ~2-3 hours for 3 epochs
  - 80GB GPU: ~1-2 hours for 3 epochs

**Monitor training:**

```bash
# In another terminal, watch the logs
tail -f outputs/impulse-salamandra-lora/training_log.txt

# Or if using Weights & Biases
wandb login
# Then view at https://wandb.ai
```

**Stopping and resuming:**

```bash
# To stop training: Ctrl+C

# To resume from checkpoint:
# Training automatically resumes from the last checkpoint in output_dir
bash scripts/train.sh
```

### Step 4: Merge LoRA Adapters

After training completes, merge the LoRA adapters back into the base model:

```bash
bash scripts/merge_lora.sh
```

This creates a standalone model at `outputs/impulse-salamandra-merged/` that can be used without the base model.

### Step 5: Validate the Model

Test the fine-tuned model on example queries:

```bash
# Test with default queries
python scripts/validate.py \
    --model-path outputs/impulse-salamandra-merged

# Or test with custom queries
python scripts/validate.py \
    --model-path outputs/impulse-salamandra-merged \
    --test-queries data/test/test_queries.jsonl
```

**Expected output:**

```
IMPULSE Model Validation
================================================================================

Loading model...
  Path: outputs/impulse-salamandra-merged
  Mode: Merged model
Model loaded successfully

Testing 5 queries

Test 1/5
================================================================================
Query ID: TEST_001
Input: recerca sobre intel·ligència artificial
--------------------------------------------------------------------------------
Response:
{
  "doc_type": "projects",
  "filters": {
    "programme": null,
    ...
  },
  ...
}
--------------------------------------------------------------------------------
Status: PASSED
================================================================================

...

================================================================================
VALIDATION SUMMARY
================================================================================
Total queries: 5
Passed: 5 (100.0%)
Failed: 0 (0.0%)
  - Invalid JSON: 0
  - Schema errors: 0
================================================================================
```

## Troubleshooting

### Out of Memory (OOM) Errors

If you get OOM errors during training:

1. **Reduce batch size:**
   ```yaml
   # In axolotl_config.yml
   micro_batch_size: 1
   gradient_accumulation_steps: 8
   ```

2. **Enable gradient checkpointing:**
   ```yaml
   gradient_checkpointing: true
   ```

3. **Use 8-bit or 4-bit quantization:**
   ```yaml
   load_in_8bit: true
   # or
   load_in_4bit: true
   ```

### CUDA Errors

If you get CUDA errors:

```bash
# Check CUDA version
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
```

### Slow Training

If training is slower than expected:

1. **Enable flash attention** (already enabled in config):
   ```yaml
   flash_attention: true
   ```

2. **Check GPU utilization:**
   ```bash
   watch -n 1 nvidia-smi
   # GPU utilization should be 90%+
   ```

3. **Increase batch size** if GPU is underutilized

### Model Not Learning

If validation loss doesn't decrease:

1. **Check learning rate** - try increasing:
   ```yaml
   learning_rate: 0.0003  # Increase from 0.0002
   ```

2. **Increase training epochs:**
   ```yaml
   num_epochs: 5
   ```

3. **Increase LoRA rank:**
   ```yaml
   lora_r: 32
   lora_alpha: 64
   ```

## Using the Fine-tuned Model

### In Python

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

# Load model
model_path = "outputs/impulse-salamandra-merged"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)

# Parse a query
def parse_query(query_text):
    messages = [{"role": "user", "content": query_text}]
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract JSON from response
    response = response.split("<|im_start|>assistant")[-1]
    response = response.replace("<|im_end|>", "").strip()
    
    return json.loads(response)

# Example
result = parse_query("projectes sobre energia renovable a Catalunya")
print(json.dumps(result, indent=2, ensure_ascii=False))
```

### Integration with IMPULSE API

```python
# In your API code (src/impulse/api/main.py)
from transformers import AutoModelForCausalLM, AutoTokenizer

class QueryParser:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )
    
    def parse(self, query: str) -> dict:
        # Same implementation as above
        ...
```

## Advanced Options

### Multi-GPU Training

If you have multiple GPUs:

```bash
# Modify train.sh
NUM_GPUS=4

accelerate launch --num_processes=$NUM_GPUS -m axolotl.cli.train configs/axolotl_config.yml
```

### Experiment Tracking with Weights & Biases

Enable in `axolotl_config.yml`:

```yaml
wandb_project: impulse-finetuning
wandb_entity: your-username
wandb_watch: gradients
wandb_log_model: checkpoint
```

Then:

```bash
wandb login
bash scripts/train.sh
```

View experiments at https://wandb.ai

### Custom System Prompt

To add a system prompt during inference:

```python
messages = [
    {"role": "system", "content": "You are a query parser..."},
    {"role": "user", "content": query_text}
]
```

## Next Steps

After successful fine-tuning:

1. **Evaluate on test set** - Use your test set queries for comprehensive evaluation
2. **Deploy model** - Integrate into your IMPULSE API
3. **Monitor performance** - Track accuracy, latency, JSON validity in production
4. **Iterate** - Fine-tune further if needed with additional data

## Support

For issues or questions:
- Check the [Axolotl documentation](https://github.com/OpenAccess-AI-Collective/axolotl)
- Review [Salamandra model card](https://huggingface.co/BSC-LT/salamandra-7b)
- Check CUDA/PyTorch compatibility
