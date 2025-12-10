# IMPULS System Architecture

## Overview

IMPULS implements a modular multilingual semantic search system with three main components:

1. **Query Parser** - Converts natural language to structured queries
2. **Semantic Search** - Vector similarity search with metadata filtering
3. **Query Expansion** - Broadens search with related concepts

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Client Layer                                │
│                    Web UI (impuls_ui.html) / API Clients                 │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │ HTTP/REST
┌─────────────────────────────────▼───────────────────────────────────────┐
│                            API Gateway                                   │
│                     FastAPI (src/impulse/api/main.py)                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   /health   │  │   /search   │  │   /parse    │  │  /kb/*      │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└────────┬────────────────┬────────────────┬────────────────┬────────────┘
         │                │                │                │
┌────────▼────────┐ ┌─────▼─────┐ ┌───────▼───────┐ ┌──────▼──────┐
│  Query Parser   │ │  Embedder │ │   Expansion   │ │ Vector Store │
│  (Salamandra)   │ │ (mRoBERTA)│ │  (Wikidata)   │ │   (HNSW)    │
└────────┬────────┘ └─────┬─────┘ └───────┬───────┘ └──────┬──────┘
         │                │               │                │
    ┌────▼────┐     ┌─────▼─────┐   ┌─────▼─────┐   ┌──────▼──────┐
    │Salamandra│    │ mRoBERTA  │   │ Wikidata  │   │   Project   │
    │  7B-IT   │    │ Retrieval │   │    KB     │   │   Index     │
    └──────────┘    └───────────┘   └───────────┘   └─────────────┘
```

## Components

### 1. Query Parser (`src/impulse/parser/`)

Converts natural language queries into structured JSON using fine-tuned Salamandra-7B.

**Input:**
```
"projectes d'IA en salut finançats per H2020 des de 2020"
```

**Output:**
```json
{
  "semantic_query": "IA salut",
  "filters": {
    "framework": ["Horizon 2020"],
    "year_from": 2020
  },
  "query_rewrite": "Projectes sobre IA en salut del programa H2020 des de 2020",
  "meta": {"lang": "CA"}
}
```

**Key Features:**
- LoRA fine-tuned on 682 synthetic + 100 real queries
- 4-bit quantization support (reduces memory from 14GB to 3.5GB)
- 100% JSON validity rate
- Multilingual support (CA/ES/EN)

**Files:**
- `query_parser.py` - Main parser class
- `data/training/salamandra_inference_prompt.txt` - System prompt

### 2. Semantic Search (`src/impulse/embedding/`, `src/impulse/vector_store/`)

Dense retrieval using fine-tuned mRoBERTA embeddings.

**Architecture:**
```
Query → Embedding → HNSW Search → Metadata Filter → Results
```

**Key Features:**
- 768-dimensional embeddings
- HNSW index for fast approximate nearest neighbor search
- Post-filtering on metadata (framework, year, location, etc.)
- Score-based ranking with de-duplication

**Files:**
- `embedder.py` - Text embedding using mRoBERTA
- `hnsw_store.py` - HNSW vector index implementation
- `base.py` - Abstract vector store interface

### 3. Query Expansion (`src/impulse/query_expansion/`)

Expands queries using a Wikidata-derived knowledge base of 4,265 R&D concepts.

**Expansion Types:**

| Type | Description | Example |
|------|-------------|---------|
| **Aliases** | Same concept, different terms | "blockchain" → "cadena de bloques" |
| **Parents** | Broader concepts via subclass_of | "blockchain" → "database" |

**Distance Levels:**

| Level | Similarity | Included |
|-------|------------|----------|
| 1 (Exact) | ≥ 0.85 | Near-exact matches |
| 2 (Related) | 0.50-0.85 | Translations, synonyms |
| 3 (Broad) | 0.30-0.50 | Parent concepts |

**Key Features:**
- Centroid-based clustering (threshold 0.60)
- Language-aware representative selection
- Only parents existing in KB included (filters generic Wikidata categories)

**Files:**
- `expansion.py` - Main expansion logic
- `loader.py` - KB loading and indexing
- `data/kb/wikidata_kb.jsonl` - Knowledge base

## Data Flow

### Search Request Flow

```
1. Client sends POST /search
   │
2. API validates request
   │
3. [If use_parsing=true]
   │  └─→ Query Parser extracts filters and semantic query
   │
4. [If expansion.enabled=true]
   │  └─→ Expansion module adds related terms
   │
5. Embedder generates query vector(s)
   │
6. Vector Store performs HNSW search
   │
7. Metadata filter applied
   │
8. Results ranked and de-duplicated
   │
9. Response returned with matched_by attribution
```

### Expansion Flow

```
1. Receive semantic query ("blockchain")
   │
2. Generate query embedding
   │
3. Find matching KB concepts (cosine similarity)
   │
4. For each matched concept:
   │  ├─→ Collect aliases at each level
   │  └─→ Collect parents (if in KB)
   │
5. Cluster similar terms (threshold 0.85)
   │
6. Select representatives per cluster
   │
7. Generate centroids for search
   │
8. Return expansion structure
```

## Configuration

### Environment Variables

```bash
# Models
EMBEDDER_MODEL_NAME=langtech-innovation/mRoBERTA_retrieval
QUERY_PARSER_MODEL=BSC-LT/salamandra-7b-instruct-tools
QUERY_PARSER_QUANTIZE=true

# Paths
INDEX_DIR=data/index
KB_PATH=data/kb/wikidata_kb.jsonl
PROJECTS_METADATA_PATH=data/ris3cat/project_db.parquet

# API
API_HOST=0.0.0.0
API_PORT=8000
API_KEY=your-secret-key

# HNSW Index
HNSW_M=16
HNSW_EF_CONSTRUCTION=200
HNSW_EF_SEARCH=100
```

### Index Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| HNSW M | 16 | Number of connections per node |
| EF Construction | 200 | Search depth during indexing |
| EF Search | 100 | Search depth during query |
| Embedding Dim | 768 | mRoBERTA output dimension |

## Performance Characteristics

| Operation | Typical Latency |
|-----------|-----------------|
| Health check | <10ms |
| Search (no parsing) | ~100ms |
| Search (with parsing) | ~2-3s (GPU) / ~5-8s (CPU) |
| Query expansion | ~50ms |
| KB search | ~20ms |

## Scalability Considerations

**Current Design (Prototype):**
- Single-node deployment
- In-memory index (~100MB for 27k documents)
- Models loaded at startup

**For Production Scale:**
- Separate parsing service (GPU)
- Distributed vector store (Milvus, Qdrant)
- Query caching
- Load balancing for API
