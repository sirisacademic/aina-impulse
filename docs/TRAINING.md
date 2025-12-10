# Model Training Guide

This document describes how to train and evaluate the IMPULS models.

## Query Parser (Salamandra-7B)

### Overview

The query parser converts natural language queries into structured JSON with:
- `semantic_query`: Thematic content for semantic search
- `filters`: Structured metadata filters (framework, year, location, etc.)
- `query_rewrite`: Human-readable interpretation
- `meta`: Language and processing notes

### Training Data

**Location:** `data/training/`

| Dataset | Size | Description |
|---------|------|-------------|
| Training | 682 queries | Synthetic, template-generated |
| Test | 100 queries | Real queries from domain experts |

**Language Distribution:** ~33% Catalan, ~33% Spanish, ~33% English

**Query Types:**
- Discover (88%): Find projects on a topic
- Quantify (12%): Counts and aggregations

### Training Setup

**Requirements:**
- GPU with 24GB+ VRAM (A100 recommended)
- 32GB system RAM
- CUDA 11.8+

**Dependencies:**
```bash
pip install axolotl[flash-attn] transformers peft bitsandbytes wandb
```

### Training Configuration

We use LoRA (Low-Rank Adaptation) for efficient fine-tuning:

```yaml
# configs/axolotl_salamandra.yml
base_model: BSC-LT/salamandra-7b-instruct-tools
model_type: LlamaForCausalLM

load_in_8bit: false
load_in_4bit: true  # QLoRA

adapter: lora
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj

dataset_prepared_path: data/training/prepared
datasets:
  - path: data/training/impulse_training_salamandra.jsonl
    type: sharegpt

sequence_len: 2048
micro_batch_size: 2
gradient_accumulation_steps: 8
num_epochs: 3

learning_rate: 2e-4
optimizer: adamw_torch
lr_scheduler: cosine
warmup_ratio: 0.1

output_dir: outputs/salamandra-impuls-lora
```

### Training Steps

1. **Prepare training data:**
```bash
python scripts/training/prepare_training_data.py \
  --input-dir data/training/batches \
  --output-file data/training/impulse_training_salamandra.jsonl
```

2. **Run training:**
```bash
accelerate launch -m axolotl.cli.train configs/axolotl_salamandra.yml
```

3. **Merge LoRA weights:**
```bash
python scripts/training/merge_lora.py \
  --base-model BSC-LT/salamandra-7b-instruct-tools \
  --lora-path outputs/salamandra-impuls-lora \
  --output-path outputs/salamandra-impuls-merged
```

### Evaluation

```bash
python scripts/training/validate.py \
  --model outputs/salamandra-impuls-merged \
  --test-file data/test/impulse_test.json \
  --output results/evaluation_report.json
```

**Metrics:**

| Metric | Description |
|--------|-------------|
| JSON Validity | % outputs that are valid JSON |
| Strict Accuracy | All fields exactly match gold |
| Relaxed Accuracy | Core fields match (allowing minor variations) |
| Language Match | Output language matches query language |
| Component Accuracy | Per-field accuracy (programme, year, location, etc.) |

**Results (Salamandra-7B fine-tuned):**

| Metric | Base | Fine-tuned |
|--------|------|------------|
| JSON Validity | 100% | 100% |
| Strict Accuracy | 15% | 51% |
| Relaxed Accuracy | 29% | 65% |
| Language Match | 53% | 87% |
| Semantic Query | 44% | 86% |

---

## Embedding Model (mRoBERTA)

### Overview

Fine-tuned for multilingual semantic retrieval of R&D content.

### Training Data

**Location:** `data/retrieval/`

| Dataset | Size | Description |
|---------|------|-------------|
| Query-Passage Pairs | 76k | CA/ES/EN scientific texts |
| Classification | 19k | Abstracts with 19 categories |

**Pair Types:**
- Keyword → Abstract (89.9%)
- Title → Abstract (10.1%)

### Training Setup

```bash
pip install sentence-transformers
```

### Training Configuration

```python
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

# Load base model
model = SentenceTransformer('langtech-innovation/mRoBERTA_retrieval')

# Prepare data
train_examples = [
    InputExample(texts=[query, passage]) 
    for query, passage in pairs
]
train_dataloader = DataLoader(train_examples, batch_size=32, shuffle=True)

# Loss function
train_loss = losses.MultipleNegativesRankingLoss(model)

# Train
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=500,
    evaluation_steps=500,
    output_path='outputs/mroberta-impuls'
)
```

### Evaluation

```python
from sentence_transformers import evaluation

evaluator = evaluation.InformationRetrievalEvaluator(
    queries, corpus, relevant_docs,
    mrr_at_k=[10], recall_at_k=[1, 5, 10]
)
results = evaluator(model)
```

**Results (mRoBERTA fine-tuned):**

| Metric | Base | Fine-tuned |
|--------|------|------------|
| Recall@1 | 34% | 65% |
| Recall@5 | 59% | 86% |
| Recall@10 | 71% | 91% |
| MRR | 0.46 | 0.74 |

**Cross-lingual Performance:**

| Setting | Base R@1 | Fine-tuned R@1 |
|---------|----------|----------------|
| Monolingual | 25% | 58% |
| Cross-lingual | 19% | 49% |

---

## Inference Optimization

### 4-bit Quantization (Query Parser)

Reduces memory from ~14GB to ~3.5GB:

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="auto"
)
```

### Batch Processing (Embeddings)

```python
# Process documents in batches
embeddings = embedder.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)
```

## Reproducibility

All training uses deterministic settings where possible:

```python
import torch
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
```

Training logs are tracked with Weights & Biases (optional).
