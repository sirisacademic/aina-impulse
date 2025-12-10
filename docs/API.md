# IMPULS API Documentation

## Overview

The IMPULS API provides multilingual semantic search over R&D project databases with intelligent query parsing and concept expansion.

**Base URL**: `http://localhost:8000` (development) or `http://impuls-aina.sirisacademic.com:8000` (production)

## Authentication

All endpoints (except `/health`) require an API key:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/search
```

## Endpoints

### Health Check

```
GET /health
```

Returns system status and component information.

**Response:**
```json
{
  "status": "ok",
  "index_size": 26941,
  "projects_metadata_loaded": 26941,
  "kb_concepts": 4265,
  "kb_indexed": 4265,
  "parser_loaded": true
}
```

---

### Search

```
POST /search
```

Main search endpoint with optional query parsing and semantic expansion.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | Natural language search query |
| `k` | integer | No | Number of results (default: 10, max: 100) |
| `k_factor` | integer | No | Oversampling factor for filtering (default: 5) |
| `filters` | object | No | Metadata filters (see below) |
| `use_parsing` | boolean | No | Enable intelligent query parsing (default: false) |
| `expansion` | object | No | Query expansion settings (see below) |

**Filters Object:**

| Field | Type | Description |
|-------|------|-------------|
| `framework` | string[] | Funding programmes: "H2020", "Horizon Europe", "FEDER", "AEI", etc. |
| `instrument` | string | Funding instrument |
| `year_from` | integer | Start year (inclusive) |
| `year_to` | integer | End year (inclusive) |
| `country` | string[] | Countries |
| `region` | string[] | Regions |
| `province` | string[] | Provinces |
| `organization_type` | string[] | Organization types: "HES", "REC", "PRC", "PUB", etc. |

**Expansion Object:**

| Field | Type | Description |
|-------|------|-------------|
| `enabled` | boolean | Enable query expansion |
| `alias_levels` | integer[] | Synonym levels to include: 1=exact, 2=related, 3=broad |
| `parent_levels` | integer[] | Parent concept levels: 1=direct, 2=related, 3=general |
| `excluded_terms` | string[] | Terms to exclude from expansion |
| `return_details` | boolean | Include expansion details in response |

**Example Request:**

```json
{
  "query": "projectes d'intel·ligència artificial en salut finançats per H2020",
  "k": 20,
  "use_parsing": true,
  "expansion": {
    "enabled": true,
    "alias_levels": [1, 2],
    "parent_levels": [1],
    "return_details": true
  }
}
```

**Response:**

```json
{
  "query": "projectes d'intel·ligència artificial en salut finançats per H2020",
  "query_used": "intel·ligència artificial salut",
  "k": 20,
  "returned": 20,
  "total_matching": 156,
  "filters": {
    "framework": ["Horizon 2020"],
    "year_from": null,
    "year_to": null,
    "country": null,
    "region": null
  },
  "feedback": {
    "query_rewrite": "Projectes sobre intel·ligència artificial en l'àmbit de la salut del programa Horizon 2020",
    "notes": null
  },
  "expansion": {
    "query_language": "CA",
    "alias_levels": {
      "1": {
        "terms_count": 3,
        "representatives": ["intel·ligència artificial"]
      },
      "2": {
        "terms_count": 8,
        "representatives": ["inteligencia artificial", "artificial intelligence", "IA"]
      }
    },
    "parent_levels": {
      "1": {
        "terms_count": 2,
        "representatives": ["computer science"]
      }
    }
  },
  "results": [
    {
      "id": "proj_12345",
      "title": "AI-Health: Artificial Intelligence for Early Disease Detection",
      "abstract": "This project develops machine learning algorithms...",
      "score": 0.923,
      "matched_by": ["query", "artificial intelligence", "IA"],
      "metadata": {
        "framework_name": "Horizon 2020",
        "instrument_name": "RIA",
        "year": 2021
      },
      "participants": [
        {
          "organization_name": "Universitat de Barcelona",
          "organization_type": "HES",
          "country": "Spain",
          "region": "Catalunya"
        }
      ]
    }
  ]
}
```

---

### Parse Query

```
POST /parse
```

Parse a natural language query without performing search.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | Natural language query |

**Response:**

```json
{
  "original_query": "projectes H2020 sobre blockchain des de 2020",
  "parsed": {
    "doc_type": "project",
    "filters": {
      "framework": ["Horizon 2020"],
      "year_from": 2020
    },
    "semantic_query": "blockchain",
    "query_rewrite": "Projectes del programa Horizon 2020 sobre blockchain des de l'any 2020",
    "meta": {
      "lang": "CA"
    }
  }
}
```

---

### Knowledge Base Search

```
GET /kb/search
```

Search for concepts in the knowledge base.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `q` | string | Yes | Search query |
| `limit` | integer | No | Max results (default: 10) |

**Response:**

```json
[
  {
    "wikidata_id": "Q2539",
    "keyword": "machine learning",
    "label_en": "machine learning",
    "label_es": "aprendizaje automático",
    "label_ca": "aprenentatge automàtic",
    "score": 0.95
  }
]
```

---

### Get Concept

```
GET /kb/concept/{wikidata_id}
```

Get detailed information about a concept including relationships.

**Response:**

```json
{
  "wikidata_id": "Q2539",
  "keyword": "machine learning",
  "labels": {
    "en": "machine learning",
    "es": "aprendizaje automático",
    "ca": "aprenentatge automàtic"
  },
  "aliases": {
    "en": ["ML", "statistical learning"],
    "es": ["ML", "aprendizaje de máquina"],
    "ca": ["aprenentatge de màquines"]
  },
  "definition": "Branch of artificial intelligence...",
  "parents": [
    {
      "wikidata_id": "Q11660",
      "keyword": "artificial intelligence",
      "label_en": "artificial intelligence",
      "in_kb": true
    }
  ],
  "children": [
    {
      "wikidata_id": "Q197536",
      "keyword": "deep learning",
      "label_en": "deep learning",
      "in_kb": true
    }
  ]
}
```

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error description"
}
```

**Common HTTP Status Codes:**

| Code | Description |
|------|-------------|
| 400 | Bad request (invalid parameters) |
| 401 | Unauthorized (missing/invalid API key) |
| 404 | Resource not found |
| 500 | Internal server error |

## Rate Limits

- Default: 100 requests/minute per API key
- Search with parsing: 30 requests/minute (GPU-intensive)

## Query Language Support

The system automatically detects query language and returns results in all languages. The `expansion.query_language` field indicates the detected language:

- `CA` - Catalan
- `ES` - Spanish  
- `EN` - English
