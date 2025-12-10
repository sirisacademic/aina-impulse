# AINA Models Integration Guide

## Methodology for integrating AINA project models in IMPULS

---

## 1. Overview

IMPULS integrates two models from the AINA ecosystem (Barcelona Supercomputing Center):

| Base Model | Purpose in IMPULS | Fine-tuned Version |
|------------|-------------------|-------------------|
| `BSC-LT/salamandra-7b-instruct-tools` | Query parsing | [SIRIS-Lab/impuls-salamandra-7b-query-parser](https://huggingface.co/SIRIS-Lab/impuls-salamandra-7b-query-parser) |
| `langtech-innovation/mRoBERTA_retrieval` | Semantic embeddings | [nicolauduran45/mRoBERTA_retrieval-scientific_domain](https://huggingface.co/nicolauduran45/mRoBERTA_retrieval-scientific_domain) |

---

## 2. Salamandra-7B for Query Parsing

### 2.1 Model Description

Salamandra is a family of multilingual language models developed by BSC with special emphasis on Iberian languages (Catalan, Spanish, Galician, Basque, Portuguese). The `salamandra-7b-instruct-tools` version is specifically trained for instruction following and structured output generation.

### 2.2 Why Salamandra?

| Criterion | Salamandra | Alternatives |
|-----------|------------|--------------|
| Catalan support | ✅ Native, high quality | ❌ Limited or translated |
| Structured outputs | ✅ -tools version optimized | ⚠️ Variable |
| Size | 7B parameters | Similar |
| License | Apache 2.0 | Variable |
| AINA community | ✅ Active support | ❌ None |

### 2.3 Fine-tuning Methodology

**Objective**: Convert natural language queries to structured JSON for search.

**Technique**: LoRA (Low-Rank Adaptation)
- Trainable parameters: ~1% of model (~50MB adapter)
- Memory efficient: Can train on single 24GB GPU
- Modular: Adapters can be swapped or combined

**LoRA Configuration**:
```python
LoraConfig(
    r=16,                    # Rank of LoRA matrices
    lora_alpha=32,           # Scaling factor
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    task_type=TaskType.CAUSAL_LM
)
```

**Training Hyperparameters**:
```python
TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch: 16
    learning_rate=2e-4,
    warmup_steps=100,
    max_length=2048,
    fp16=True
)
```

### 2.4 Training Data

| Dataset | Size | Description |
|---------|------|-------------|
| Training | 682 queries | Synthetic, template-generated |
| Test | 100 queries | Real queries from domain experts |

**Language distribution**: ~33% Catalan, ~33% Spanish, ~33% English

**Dataset available**: [SIRIS-Lab/impuls-query-parsing](https://huggingface.co/datasets/SIRIS-Lab/impuls-query-parsing)

### 2.5 Results

| Metric | Base Model | Fine-tuned | Improvement |
|--------|------------|------------|-------------|
| JSON Validity | 100% | 100% | - |
| Strict Accuracy | 15% | 51% | +36pp |
| Relaxed Accuracy | 29% | 65% | +36pp |
| Language Match | 53% | 87% | +34pp |
| Semantic Query | 44% | 86% | +42pp |

**Component-level accuracy**:
| Component | Accuracy |
|-----------|----------|
| Programme | 96% |
| Year | 98% |
| Location | 91% |
| Organizations | 77% |
| Semantic Query | 86% |

### 2.6 Inference Optimization

**4-bit Quantization** reduces memory from ~14GB to ~3.5GB:

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

**Inference parameters**:
- Temperature: 0.1 (low for deterministic JSON)
- Max tokens: 512
- Greedy decoding (do_sample=True with low temp)

---

## 3. mRoBERTA for Semantic Embeddings

### 3.1 Model Description

`mRoBERTA_retrieval` is a multilingual sentence encoder from the AINA project, pre-trained on Catalan, Spanish, and English text for retrieval tasks.

We fine-tuned this model on scientific domain data, published as [nicolauduran45/mRoBERTA_retrieval-scientific_domain](https://huggingface.co/nicolauduran45/mRoBERTA_retrieval-scientific_domain).

### 3.2 Fine-tuning Results

| Metric | Base | Fine-tuned |
|--------|------|------------|
| Recall@1 | 34% | **65%** |
| Recall@5 | 59% | **86%** |
| Recall@10 | 71% | **91%** |
| MRR | 0.46 | **0.74** |
| Cross-lingual R@1 | 19% | **49%** |

### 3.3 Usage in IMPULS

- **768-dimensional embeddings** for queries and documents
- **Cosine similarity** for semantic matching
- **Cross-lingual retrieval**: Query in one language finds documents in any language

### 3.4 Integration

```python
from sentence_transformers import SentenceTransformer

# Use the fine-tuned model
model = SentenceTransformer('nicolauduran45/mRoBERTA_retrieval-scientific_domain')

# Encode queries and documents
query_embedding = model.encode("intel·ligència artificial")
doc_embeddings = model.encode(documents)

# Compute similarities
similarities = cosine_similarity([query_embedding], doc_embeddings)
```

### 3.5 Cross-lingual Performance

Cross-lingual retrieval enables finding relevant documents regardless of language:

| Query Language | Document Language | Works? |
|----------------|-------------------|--------|
| Catalan | English | ✅ |
| Spanish | Catalan | ✅ |
| English | Spanish | ✅ |

---

## 4. Knowledge Base (Wikidata)

### 4.1 Purpose

The KB enables query expansion with multilingual synonyms and related concepts.

### 4.2 Structure

- **4,265 R&D concepts** from Wikidata
- **Multilingual labels**: CA, ES, EN, IT
- **Hierarchical relations**: subclass_of, instance_of

**Available**: [SIRIS-Lab/impuls-wikidata-kb](https://huggingface.co/datasets/SIRIS-Lab/impuls-wikidata-kb)

### 4.3 Expansion Algorithm

```
1. User query: "blockchain"
2. Find matching KB concepts (exact match on label/alias)
3. Extract aliases in all languages:
   - "cadena de blocs" (CA)
   - "cadena de bloques" (ES)
   - "distributed ledger" (EN)
4. Optionally traverse parents (subclass_of)
5. Cluster similar terms (cosine > 0.85)
6. Return expanded query terms
```

---

## 5. System Architecture Integration

```
┌─────────────────────────────────────────────────────────────┐
│                     User Query (CA/ES/EN)                   │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Query Parser (Salamandra)                  │
│                                                             │
│  Input: "projectes H2020 sobre IA des de 2020"              │
│  Output: {semantic_query: "IA", filters: {year: ">=2020"}}  │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                Query Expansion (Wikidata KB)                 │
│                                                             │
│  Input: "IA"                                                │
│  Output: ["IA", "intel·ligència artificial", "AI", ...]     │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               Semantic Search (mRoBERTA + HNSW)              │
│                                                             │
│  Embed query terms → Search vector index → Apply filters    │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Ranked Results                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Replication Guide

### 6.1 Using Pre-trained Models

```python
# Query Parser
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "SIRIS-Lab/impuls-salamandra-7b-query-parser"
)
tokenizer = AutoTokenizer.from_pretrained(
    "SIRIS-Lab/impuls-salamandra-7b-query-parser"
)

# Embeddings
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("langtech-innovation/mRoBERTA_retrieval")
```

### 6.2 Fine-tuning Your Own

See [TRAINING.md](TRAINING.md) for complete instructions on:
- Preparing training data
- Configuring LoRA
- Running training with Axolotl
- Merging adapters
- Evaluation

---

## 7. Published Resources

| Resource | URL |
|----------|-----|
| Query Parser Model | https://huggingface.co/SIRIS-Lab/impuls-salamandra-7b-query-parser |
| Query Parsing Dataset | https://huggingface.co/datasets/SIRIS-Lab/impuls-query-parsing |
| Wikidata Knowledge Base | https://huggingface.co/datasets/SIRIS-Lab/impuls-wikidata-kb |
| Source Code | https://github.com/sirisacademic/aina-impulse |

---

## 8. Acknowledgments

- **Barcelona Supercomputing Center (BSC)** - Salamandra and mRoBERTA models
- **AINA Project** - Infrastructure and community support
- **Generalitat de Catalunya** - Funding and RIS3-MCAT data

---

*IMPULS - AINA Challenge 2024*
