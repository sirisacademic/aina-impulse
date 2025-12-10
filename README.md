# IMPULS - Multilingual Semantic Search for R&D Projects

[![AINA Challenge](https://img.shields.io/badge/AINA%20Challenge-2024-blue)](https://projecteaina.cat/)
[![HuggingFace Collection](https://img.shields.io/badge/🤗%20HuggingFace-Collection-yellow)](https://huggingface.co/collections/SIRIS-Lab/aina-impuls)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

IMPULS is a multilingual semantic search system for R&D projects, enabling natural language queries in **Catalan, Spanish, and English** over the RIS3-MCAT corpus (~27,000 projects).

**🔗 Live Demo**: http://impuls-aina.sirisacademic.com:8080/impuls_ui.html  
**📡 API**: http://impuls-aina.sirisacademic.com:8000/docs  
**🤗 HuggingFace**: https://huggingface.co/collections/SIRIS-Lab/aina-impuls

---

## ✨ Features

- **Intelligent Query Parsing**: Understands complex queries like *"projectes d'IA en salut finançats per H2020 des de 2020"* and automatically extracts filters
- **Multilingual Semantic Search**: Find relevant projects regardless of document language
- **Query Expansion**: Automatically adds synonyms and translations using a Wikidata-based knowledge base (4,265 R&D concepts)
- **Interactive Knowledge Graph**: Explore concept relationships visually

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Query                            │
│                     (CA / ES / EN)                           │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                      API Gateway                             │
│                  (FastAPI - Port 8000)                       │
│                                                             │
│   /health    /search    /parse    /kb/search                │
└──────┬──────────┬──────────┬──────────┬─────────────────────┘
       │          │          │          │
┌──────▼────┐ ┌───▼────┐ ┌───▼────┐ ┌───▼────┐
│  Parser   │ │Embedder│ │Expansion│ │  Index │
│Salamandra │ │mRoBERTA│ │Wikidata │ │  HNSW  │
└───────────┘ └────────┘ └─────────┘ └────────┘
```

## 🤗 Published Resources (HuggingFace)

All models and datasets are available in our [HuggingFace Collection](https://huggingface.co/collections/SIRIS-Lab/aina-impuls).

### Fine-tuned Models

| Base Model (AINA/BSC) | Fine-tuned Model | Purpose | Results |
|----------------------|------------------|---------|---------|
| `BSC-LT/salamandra-7b-instruct-tools` | [SIRIS-Lab/impuls-salamandra-7b-query-parser](https://huggingface.co/SIRIS-Lab/impuls-salamandra-7b-query-parser) | Query parsing: NL → JSON | 65% accuracy, 100% valid JSON |
| `langtech-innovation/mRoBERTA_retrieval` | [nicolauduran45/mRoBERTA_retrieval-scientific_domain](https://huggingface.co/nicolauduran45/mRoBERTA_retrieval-scientific_domain) | Multilingual semantic embeddings | R@10: 91%, MRR: 0.74 |

### Datasets

| Dataset | Description | Link |
|---------|-------------|------|
| **Query Parsing** | 682 training + 100 test queries (CA/ES/EN) | [SIRIS-Lab/impuls-query-parsing](https://huggingface.co/datasets/SIRIS-Lab/impuls-query-parsing) |
| **R&D Knowledge Base** | 4,265 Wikidata concepts with multilingual labels | [SIRIS-Lab/impuls-wikidata-kb](https://huggingface.co/datasets/SIRIS-Lab/impuls-wikidata-kb) |

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- 16GB RAM (32GB recommended for query parser)
- GPU with 8GB+ VRAM (optional, for faster inference)

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
```

### Configuration

Edit `.env` file:

```bash
# Embedding model (fine-tuned for scientific domain)
EMBEDDER_MODEL_NAME=nicolauduran45/mRoBERTA_retrieval-scientific_domain

# Query parser (fine-tuned Salamandra)
QUERY_PARSER_MODEL=SIRIS-Lab/impuls-salamandra-7b-query-parser
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
      "parent_levels": [1]
    }
  }'
```

## 🧪 Evaluation Results

### Query Parser (Salamandra-7B fine-tuned)

| Metric | Base | Fine-tuned |
|--------|------|------------|
| JSON Validity | 100% | 100% |
| Strict Accuracy | 15% | **51%** |
| Relaxed Accuracy | 29% | **65%** |
| Language Match | 53% | **87%** |

### Semantic Retrieval (mRoBERTA fine-tuned)

| Metric | Base | Fine-tuned |
|--------|------|------------|
| Recall@1 | 34% | **65%** |
| Recall@10 | 71% | **91%** |
| MRR | 0.46 | **0.74** |
| Cross-lingual R@1 | 19% | **49%** |

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [User Manual (CA)](docs/MANUAL_USUARI.md) | Complete guide for end users (in Catalan) |
| [API Reference](docs/API.md) | Complete API documentation with examples |
| [Architecture](docs/ARCHITECTURE.md) | System design and component details |
| [Training Guide](docs/TRAINING.md) | How to train and evaluate models |
| [AINA Integration](docs/AINA_INTEGRATION.md) | Methodology for integrating AINA models |
| [Deployment Guide](docs/DEPLOYMENT.md) | Production deployment instructions |
| [Maintenance Guide](docs/MAINTENANCE.md) | Operations and maintenance manual |

## 📁 Project Structure

```
aina-impulse/
├── data/
│   ├── index/                    # Vector index (HNSW)
│   ├── kb/                       # Knowledge base
│   │   └── wikidata_kb.jsonl     # 4,265 R&D concepts
│   ├── ris3cat/                  # Project data
│   ├── test/                     # Evaluation data
│   └── training/                 # Training data for parser
├── docs/                         # Documentation
├── html/                         # Web UI
│   └── impuls_ui.html
├── scripts/
│   ├── build_indices/            # Index building
│   ├── data_analysis/            # Data analysis tools
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

## 🏢 Acknowledgments

- **[SIRIS Academic](https://sirisacademic.com/)** - Project development
- **[Barcelona Supercomputing Center (BSC)](https://www.bsc.es/)** - AINA models and infrastructure
- **[Generalitat de Catalunya](https://web.gencat.cat/)** - RIS3-MCAT platform and funding
- **[AINA Project](https://projecteaina.cat/)** - AINA Challenge framework

## 📄 License

Apache 2.0 License - see [LICENSE](LICENSE) for details.

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/sirisacademic/aina-impulse/issues)
- **AINA Challenge**: [projecteaina.cat](https://projecteaina.cat/)

---

*IMPULS is part of the AINA Challenge, advancing Catalan language technology for research and innovation.*
