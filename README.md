# IMPULS - Multilingual Semantic Search for R&D Ecosystems

**AINA Challenge 2024** | SIRIS Academic

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🎯 Overview

IMPULS is a **multilingual semantic search system** for research and innovation ecosystems, developed as part of the AINA Challenge 2024. It enables natural language queries in **Catalan, Spanish, and English** over the RIS3-MCAT corpus of ~27,000 R&D projects.

**🌐 Live Demo**: http://impuls-aina.sirisacademic.com:8080/impuls_ui.html  
**📖 API Docs**: http://impuls-aina.sirisacademic.com:8000/docs

### Key Features

- **🤖 Intelligent Query Parsing**: Converts natural language queries into structured filters using fine-tuned Salamandra-7B
- **🔍 Semantic Search**: Cross-lingual retrieval using fine-tuned mRoBERTA embeddings
- **🔗 Query Expansion**: Automatic expansion with multilingual synonyms and broader concepts (4,265 Wikidata R&D concepts)
- **📊 Knowledge Graph**: Interactive visualization of concept relationships
- **🌍 Multilingual**: Full support for Catalan, Spanish, and English queries and documents

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Web UI / API Clients                  │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                      FastAPI Gateway                         │
│              Orchestration & Component Coordination          │
└──────┬──────────────────────┬──────────────────────┬────────┘
       │                      │                      │
┌──────▼──────┐      ┌───────▼───────┐      ┌──────▼──────┐
│   Query     │      │   Semantic    │      │  Query      │
│   Parser    │      │   Search      │      │  Expansion  │
│ Salamandra  │      │  mRoBERTA     │      │  Wikidata   │
└─────────────┘      └───────────────┘      └─────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- 16GB RAM (32GB recommended for query parser)
- GPU with 8GB+ VRAM (optional, for faster inference)
- ~5GB disk space

### Installation

```bash
# Clone repository
git clone https://github.com/sirisacademic/aina-impulse.git
cd aina-impulse

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"

# Copy environment configuration
cp .env.example .env
# Edit .env with your settings
```

### Configuration

Edit `.env` file:

```bash
# Embedding model
EMBEDDER_MODEL_NAME=langtech-innovation/mRoBERTA_retrieval

# Query parser (Salamandra)
QUERY_PARSER_MODEL=BSC-LT/salamandra-7b-instruct-tools
QUERY_PARSER_QUANTIZE=true  # Use 4-bit quantization to reduce memory

# Data paths
INDEX_DIR=data/index
KB_PATH=data/kb/wikidata_kb.jsonl

# API settings
API_HOST=0.0.0.0
API_PORT=8000
API_KEY=your-secret-key
```

### Build the Index

```bash
# Build vector index from project data
python scripts/build_indices/build_index.py \
  --input data/ris3cat/project_db.parquet \
  --index-dir data/index
```

### Run the API

```bash
# Development
python run_api.py

# Production
./run_api.sh
```

API available at `http://localhost:8000`, docs at `http://localhost:8000/docs`

## 📖 Usage Examples

### Basic Semantic Search

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{"query": "machine learning for healthcare", "k": 10}'
```

### Search with Intelligent Parsing

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{
    "query": "projectes IA en salut finançats per H2020 des de 2020",
    "k": 10,
    "use_parsing": true
  }'
```

The parser automatically extracts:
- **Semantic query**: "IA salut" (thematic content)
- **Filters**: framework=H2020, year_from=2020

### Search with Query Expansion

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{
    "query": "blockchain",
    "k": 20,
    "use_parsing": true,
    "expansion": {
      "enabled": true,
      "alias_levels": [1, 2],
      "parent_levels": [1],
      "return_details": true
    }
  }'
```

Expansion adds multilingual synonyms (cadena de bloques, cadena de blocs) and broader concepts (distributed systems, cryptography).

### Knowledge Base Exploration

```bash
# Search concepts
curl "http://localhost:8000/kb/search?q=machine+learning&limit=10"

# Get concept details with parents/children
curl "http://localhost:8000/kb/concept/Q2539"
```

## 📁 Project Structure

```
aina-impulse/
├── data/
│   ├── index/                    # Vector index (HNSW)
│   ├── kb/                       # Knowledge base
│   │   └── wikidata_kb.jsonl     # 4,265 R&D concepts
│   ├── normalization/            # Mapping tables
│   ├── ris3cat/                  # Project data
│   ├── test/                     # Evaluation data
│   └── training/                 # Training data for parser
├── docs/                         # Documentation
├── html/                         # Web UI
│   └── impuls_ui.html
├── scripts/
│   ├── build_indices/            # Index building
│   ├── data_analysis/            # Data analysis tools
│   ├── data_preparation/         # Data preparation
│   └── training/                 # Model training scripts
├── src/impulse/
│   ├── api/main.py               # FastAPI application
│   ├── embedding/embedder.py     # mRoBERTA embeddings
│   ├── parser/query_parser.py    # Salamandra query parser
│   ├── query_expansion/          # Wikidata-based expansion
│   └── vector_store/             # HNSW vector storage
├── .env.example                  # Environment template
├── requirements.txt
└── run_api.py
```

## 🔧 API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System status and statistics |
| `/search` | POST | Semantic search with optional parsing and expansion |
| `/parse` | POST | Parse query without searching |
| `/kb/search` | GET | Search knowledge base concepts |
| `/kb/concept/{id}` | GET | Get concept details with relationships |

### Search Request Schema

```json
{
  "query": "string",
  "k": 10,
  "k_factor": 5,
  "filters": {
    "framework": ["H2020", "Horizon Europe"],
    "year_from": 2018,
    "year_to": 2024,
    "country": ["Spain"],
    "region": ["Catalunya"],
    "organization_type": ["HES", "REC"]
  },
  "use_parsing": true,
  "expansion": {
    "enabled": true,
    "alias_levels": [1, 2],
    "parent_levels": [1],
    "excluded_terms": [],
    "return_details": false
  }
}
```

### Search Response Schema

```json
{
  "query": "original query",
  "query_used": "parsed semantic query",
  "filters": { ... },
  "feedback": {
    "query_rewrite": "human-readable interpretation"
  },
  "expansion": {
    "query_language": "CA",
    "alias_levels": { ... },
    "parent_levels": { ... }
  },
  "total_matching": 150,
  "returned": 10,
  "results": [
    {
      "id": "project_123",
      "title": "Project Title",
      "abstract": "...",
      "score": 0.87,
      "matched_by": ["query", "machine learning", "IA"],
      "metadata": { ... }
    }
  ]
}
```

## 🧪 Evaluation Results

### Query Parser (Salamandra-7B fine-tuned)

| Metric | Score |
|--------|-------|
| JSON Validity | 100% |
| Relaxed Accuracy | 65% |
| Language Match | 87% |
| Semantic Query Accuracy | 86% |

### Semantic Retrieval (mRoBERTA fine-tuned)

| Metric | Base | Fine-tuned |
|--------|------|------------|
| Recall@1 | 34% | 65% |
| Recall@10 | 71% | 91% |
| MRR | 0.46 | 0.74 |
| Cross-lingual R@1 | 19% | 49% |

## 🤝 AINA Models Used

This project fine-tunes and extends models from the AINA project (BSC):

| Base Model | Purpose | Fine-tuned Version |
|------------|---------|-------------------|
| `BSC-LT/salamandra-7b-instruct-tools` | Query parsing | Coming soon |
| `langtech-innovation/mRoBERTA_retrieval` | Semantic embeddings | Coming soon |

### Datasets Generated (to be published)

- **Query-passage pairs**: 76k multilingual pairs (CA/ES/EN)
- **Query parsing dataset**: 682 training + 100 test queries
- **Scientific classification**: 19k abstracts, 19 categories
- **R&D Knowledge Base**: 4,265 Wikidata concepts with multilingual labels

## 🏢 Acknowledgments

- **SIRIS Academic** - Project development
- **Barcelona Supercomputing Center (BSC)** - AINA models and infrastructure
- **Generalitat de Catalunya** - RIS3-MCAT platform and funding

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [API Reference](docs/API.md) | Complete API documentation with examples |
| [Architecture](docs/ARCHITECTURE.md) | System design and component details |
| [Training Guide](docs/TRAINING.md) | How to train and evaluate models |
| [Deployment Guide](docs/DEPLOYMENT.md) | Production deployment instructions |

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/sirisacademic/aina-impulse/issues)
- **AINA Challenge**: [projecteaina.cat](https://projecteaina.cat/)

---

*IMPULS is part of the AINA Challenge 2024, advancing Catalan language technology for research and innovation.*
