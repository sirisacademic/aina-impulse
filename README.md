# IMPULS - Multilingual Semantic Search for Catalonia's R&D Ecosystem

**AINA Challenge 2024** - Building intelligent search infrastructure for the RIS3-MCAT platform

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🎯 Project Overview

IMPULS develops a multilingual semantic search system for Catalonia's research and innovation ecosystem, enabling natural language queries in Catalan, Spanish, Italian, and English over the RIS3-MCAT corpus of ~1,000 R&D projects.

This repository contains the **baseline implementation (WP1)** featuring:
- **Semantic search** using multilingual embeddings (`langtech-innovation/mRoBERTA_retrieval`)
- **Hybrid filtering** combining vector similarity with metadata constraints
- **RESTful API** for integration with dashboards and applications
- **Chunking strategies** for handling long project abstracts
- **De-duplication** to show best matching content per project

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- 4GB RAM minimum (8GB recommended)
- ~2GB disk space for models and indexes

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/impuls.git
cd impuls

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (for sentence tokenization)
python -c "import nltk; nltk.download('punkt')"
```

### Build the Index

```bash
# Index the RIS3CAT dataset
python scripts/build_index_from_pickle.py \
  --input data/R3C_data.pkl \
  --index-dir data/index \
  --use-sentence-chunking \
  --sentences-per-chunk 6 \
  --include-context
```

### Run the API

```bash
# Start the FastAPI server
python run_api.py

# API will be available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### Test the Search

```bash
# Simple semantic search
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "renewable energy solar panels", "k": 5}'

# With metadata filters
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "artificial intelligence healthcare",
    "k": 10,
    "filters": {
      "framework": ["H2020"],
      "year_from": 2020,
      "year_to": 2024
    }
  }'
```

## 📁 Project Structure

```
impuls/
├── data/
│   ├── index/               # Vector index and metadata
│   │   ├── vectors.hnsw     # HNSW vector index
│   │   └── metadata.json    # Document metadata
│   ├── meta/                # Enrichment tables
│   │   ├── projects.parquet # Project-level metadata
│   │   └── project_orgs.parquet # Organization relationships
│   └── R3C_data.pkl        # Source dataset (not in repo)
├── src/
│   └── impulse/
│       ├── api/            # FastAPI application
│       ├── embedding/      # Text embedding module
│       ├── vector_store/   # Vector storage backends
│       └── settings.py     # Configuration
├── scripts/
│   ├── build_index_from_pickle.py  # Index builder
│   ├── query_api_search.py         # CLI search client
│   └── inspect_pickle.py           # Dataset inspector
├── html/
│   └── impuls_demo_fastapi.html    # Web UI demo
└── requirements.txt
```

## 🔧 API Endpoints

### Health Check
```http
GET /health
```
Returns system status and index statistics.

### Semantic Search
```http
POST /search
{
  "query": "string",           # Search query
  "k": 5,                      # Number of results
  "k_factor": 5,               # Oversampling factor for filtering
  "filters": {                 # Optional metadata filters
    "framework": ["H2020"],
    "year_from": 2020,
    "year_to": 2024,
    "ris3cat_ambit": ["Energia i recursos"],
    "ris3cat_tft": ["Materials avançats"]
  }
}
```

### Add Documents
```http
POST /add_documents
{
  "documents": [
    {
      "id": "doc1",
      "text": "Document content...",
      "metadata": {"key": "value"}
    }
  ]
}
```

## 🌐 Web Interface

Open `html/impuls_demo_fastapi.html` in a browser for an interactive search interface:

1. Edit the `API_URL` variable in the HTML to match your server
2. Open the file in a web browser
3. Search with natural language queries
4. Apply metadata filters for precise results

## 🛠️ Configuration

Environment variables (`.env` file):
```bash
EMBEDDER_MODEL_NAME=langtech-innovation/mRoBERTA_retrieval
INDEX_DIR=data/index
VECTOR_BACKEND=hnsw
API_HOST=0.0.0.0
API_PORT=8000
```

## 📊 Data Processing Pipeline

### 1. Document Chunking
Projects are intelligently chunked to handle long abstracts while preserving context:
- **Sentence-based chunking**: 6 sentences per chunk with 1-sentence overlap
- **Title preservation**: Project title included in each chunk for context
- **De-duplication**: Only best-scoring chunk per project returned

### 2. Metadata Enrichment
- **Framework normalization**: H2020, Horizon Europe, ERDF variants mapped
- **Year extraction**: From `startingYear` or `startingDate` fields
- **RIS3CAT categories**: Sectoral areas and transversal technologies preserved

### 3. Hybrid Search
- **Semantic similarity**: Cosine similarity in embedding space
- **Metadata filtering**: Post-filters on framework, year, categories
- **Score-based ranking**: Results ordered by relevance score

## 🔄 Development Roadmap

### Current Release (WP1 - Baseline)
- ✅ Multilingual semantic search
- ✅ Metadata filtering
- ✅ RESTful API
- ✅ Web interface
- ✅ Document chunking

### Upcoming Features (WP2-WP4)
- 🚧 **Query parsing**: Natural language to structured JSON (Salamandra fine-tuning)
- 🚧 **Named Entity Recognition**: Extract organizations, locations (DEBERTA_CIEL)
- 🚧 **Query expansion**: WordNet and co-occurrence based expansion
- 🚧 **Knowledge graph**: Navigable relationships between projects
- 🚧 **Cross-lingual search**: Improved Catalan-English-Spanish-Italian alignment

## 🤝 Contributing

This project is part of the AINA Challenge 2024. Contributions welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 Documentation

- [API Documentation](http://localhost:8000/docs) - Interactive API docs (when running)
- [AINA Kit Models](https://huggingface.co/projecte-aina) - Language models used
- [RIS3-MCAT Platform](https://ris3mcat.gencat.cat/) - Data source

## 🏢 Partners & Acknowledgments

**SIRIS Academic** - Project implementation

**Barcelona Supercomputing Center (BSC)** - AINA Kit models and infrastructure

**Generalitat de Catalunya** - RIS3-MCAT platform and funding

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions about the project:
- Create an issue in this repository
- Contact SIRIS Academic team
- Visit the [AINA Challenge](https://projecteaina.cat/) website

---

*IMPULS is part of the AINA Challenge 2024, advancing Catalan language technology for research and innovation.*