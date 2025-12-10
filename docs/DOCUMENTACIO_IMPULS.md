# IMPULS - Documentació del Projecte

## Sistema de Cerca Semàntica Multilingüe per a Projectes d'R+D

**Projecte AINA Challenge 2024**  
**Desenvolupat per SIRIS Academic per a la Generalitat de Catalunya**

---

## 1. Descripció del Sistema

IMPULS és un sistema de cerca semàntica intel·ligent que permet trobar projectes d'R+D de l'ecosistema RIS3-MCAT utilitzant llenguatge natural en català, castellà o anglès.

### Característiques principals

- **Cerca multilingüe**: Consultes en qualsevol dels tres idiomes troben resultats rellevants independentment de l'idioma del document
- **Parsing intel·ligent**: Interpreta consultes complexes i extreu filtres automàticament
- **Expansió semàntica**: Afegeix sinònims i traduccions per millorar els resultats
- **Filtres estructurats**: Permet refinar per programa de finançament, any, ubicació, etc.

### Tecnologies AINA utilitzades

| Model | Ús | Font |
|-------|-----|------|
| Salamandra-7B | Parsing de consultes | BSC-LT/salamandra-7b-instruct-tools |
| mRoBERTA | Embeddings semàntics | langtech-innovation/mRoBERTA_retrieval |

---

## 2. Accés al Sistema

### Demostrador en línia

| Recurs | URL |
|--------|-----|
| **Interfície web** | http://impuls-aina.sirisacademic.com:8080/impuls_ui.html |
| **API REST** | http://impuls-aina.sirisacademic.com:8000 |
| **Documentació API** | http://impuls-aina.sirisacademic.com:8000/docs |

### Repositori de codi

**GitHub**: https://github.com/sirisacademic/aina-impulse

---

## 3. Recursos Publicats a HuggingFace

Tots els recursos estan disponibles a la col·lecció de HuggingFace:
**https://huggingface.co/collections/SIRIS-Lab/aina-impuls**

### Models

| Recurs | Descripció | Enllaç |
|--------|------------|--------|
| **Query Parser** | Model Salamandra fine-tuned per parsing de consultes | [SIRIS-Lab/impuls-salamandra-7b-query-parser](https://huggingface.co/SIRIS-Lab/impuls-salamandra-7b-query-parser) |
| **Retrieval Model** | Model mRoBERTA fine-tuned per cerca semàntica | [nicolauduran45/mRoBERTA_retrieval-scientific_domain](https://huggingface.co/nicolauduran45/mRoBERTA_retrieval-scientific_domain) |

### Datasets

| Recurs | Descripció | Enllaç |
|--------|------------|--------|
| **Dataset de consultes** | 682 consultes d'entrenament + 100 de test | [SIRIS-Lab/impuls-query-parsing](https://huggingface.co/datasets/SIRIS-Lab/impuls-query-parsing) |
| **Knowledge Base** | 4.265 conceptes R+D multilingües | [SIRIS-Lab/impuls-wikidata-kb](https://huggingface.co/datasets/SIRIS-Lab/impuls-wikidata-kb) |

---

## 4. Documentació Tècnica

Tota la documentació tècnica està disponible al repositori GitHub en anglès:

### Per a usuaris

| Document | Descripció | Enllaç |
|----------|------------|--------|
| **Manual d'Usuari** | Guia completa per a usuaris finals (en català) | [MANUAL_USUARI.md](docs/MANUAL_USUARI.md) |

### Per a desenvolupadors

| Document | Descripció | Enllaç |
|----------|------------|--------|
| **README** | Visió general i guia d'inici ràpid | [README.md](README.md) |
| **Arquitectura** | Disseny del sistema i components | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| **API Reference** | Documentació completa de l'API REST | [docs/API.md](docs/API.md) |
| **Training Guide** | Com entrenar i avaluar els models | [docs/TRAINING.md](docs/TRAINING.md) |
| **AINA Integration** | Metodologia d'integració de models AINA | [docs/AINA_INTEGRATION.md](docs/AINA_INTEGRATION.md) |

### Per a operadors

| Document | Descripció | Enllaç |
|----------|------------|--------|
| **Deployment Guide** | Desplegament en producció | [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) |
| **Maintenance Guide** | Operació i manteniment del sistema | [docs/MAINTENANCE.md](docs/MAINTENANCE.md) |

---

## 5. Resultats d'Avaluació

### Query Parser (Salamandra fine-tuned)

| Mètrica | Resultat |
|---------|----------|
| JSON Vàlid | 100% |
| Accuracy Estricta | 51% |
| Accuracy Relaxada | 65% |
| Coincidència d'Idioma | 87% |
| Semantic Query | 86% |

### Comparativa amb altres models

| Model | Accuracy | JSON Vàlid |
|-------|----------|------------|
| **Salamandra-7B (nostre)** | **51%** | 100% |
| Qwen 2.5-7B | 47% | 100% |
| Mistral-7B | 24% | 100% |

---

## 6. Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                        Client                                │
│              (Interfície Web / Clients API)                  │
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
│  Parser   │ │Embedder│ │Expansió│ │  Index │
│Salamandra │ │mRoBERTA│ │Wikidata│ │  HNSW  │
└───────────┘ └────────┘ └────────┘ └────────┘
```

---

## 7. Requisits del Sistema

### Mínims (sense Query Parser)
- CPU: 4 cores
- RAM: 8 GB
- Disc: 10 GB

### Recomanats (sistema complet)
- CPU: 8+ cores
- RAM: 32 GB
- GPU: NVIDIA amb 8GB+ VRAM
- Disc: 50 GB SSD

---

## 8. Llicència

Tot el codi i els models es publiquen sota llicència **Apache 2.0**.

---

## 9. Contacte i Suport

- **Issues**: https://github.com/sirisacademic/aina-impulse/issues
- **SIRIS Academic**: https://sirisacademic.com/
- **Projecte AINA**: https://projecteaina.cat/

---

## 10. Agraïments

- **Barcelona Supercomputing Center (BSC)** - Models Salamandra i mRoBERTA
- **Generalitat de Catalunya** - Finançament i plataforma RIS3-MCAT
- **Projecte AINA** - Infraestructura i suport

---

*IMPULS - Projecte AINA Challenge 2024*  
*Desembre 2024*
