from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
import logging

from src.impulse.settings import settings
from src.impulse.embedding.embedder import Embedder
from src.impulse.vector_store.factory import build_store

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="IMPULSE WP1 Retrieval API", version="0.2.1")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Maybe to replace with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Document(BaseModel):
    id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    metadata: Optional[Dict[str, Any]] = None

class AddRequest(BaseModel):
    documents: List[Document]

class MetaFilters(BaseModel):
    framework: Optional[List[str]] = None
    ris3cat_ambit: Optional[List[str]] = None
    ris3cat_tft: Optional[List[str]] = None
    year_from: Optional[int] = None
    year_to: Optional[int] = None

class SearchRequest(BaseModel):
    query: str
    k: int = 5
    k_factor: int = 5
    filters: Optional[MetaFilters] = None

DATA_DIR = Path(settings.index_dir)
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = DATA_DIR / "vectors.hnsw"
META_PATH = DATA_DIR / "metadata.json"

# Add paths for parquet files
META_DIR = DATA_DIR.parent / "meta"  # Assuming meta folder is at same level as index
PROJECTS_PARQUET = META_DIR / "projects.parquet"
PROJECT_ORGS_PARQUET = META_DIR / "project_orgs.parquet"

_EMBEDDER = None
_STORE = None
_PROJECTS_METADATA = None

def load_projects_metadata():
    """Load project metadata from parquet files once at startup"""
    global _PROJECTS_METADATA
    
    if _PROJECTS_METADATA is not None:
        return _PROJECTS_METADATA
    
    _PROJECTS_METADATA = {}
    
    try:
        if PROJECTS_PARQUET.exists():
            logger.info(f"Loading projects metadata from {PROJECTS_PARQUET}")
            projects_df = pd.read_parquet(PROJECTS_PARQUET)
            
            # Create a dictionary for fast lookup
            for _, row in projects_df.iterrows():
                project_id = row['projectId']
                _PROJECTS_METADATA[project_id] = {
                    'title': row.get('projectTitle', ''),
                    'abstract': row.get('projectAbstract', ''),
                    'frameworkName': row.get('frameworkName', ''),
                    'startingYear': row.get('startingYear'),
                    'RIS3CAT_Ambit': row.get('RIS3CAT Àmbit Sectorial Líder', ''),
                    'RIS3CAT_TFT': row.get('RIS3CAT Tecnologia Facilitadora Transversal', ''),
                    'framework_norm': row.get('framework_norm', ''),
                    'year_norm': row.get('year_norm'),
                }
            
            logger.info(f"Loaded metadata for {len(_PROJECTS_METADATA)} projects")
        else:
            logger.warning(f"Projects parquet file not found at {PROJECTS_PARQUET}")
    
    except Exception as e:
        logger.error(f"Error loading projects metadata: {e}")
        _PROJECTS_METADATA = {}
    
    return _PROJECTS_METADATA

def get_embedder() -> Embedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = Embedder(model_name=settings.embedder_model_name, device=None)
    return _EMBEDDER

def get_store():
    global _STORE
    if _STORE is None:
        emb = get_embedder()
        store = build_store(
            backend=settings.vector_backend,
            index_path=str(INDEX_PATH),
            meta_path=str(META_PATH),
            space=settings.hnsw_space,
            M=settings.hnsw_m,
            ef_construct=settings.hnsw_ef_construct,
            ef_query=settings.hnsw_ef_query,
        )
        if INDEX_PATH.exists() and META_PATH.exists():
            store.load()
        else:
            if settings.require_index:
                raise RuntimeError("Index required but not found. Run scripts/build_index_from_pickle.py first.")
            store.init(dim=emb.get_dim())
        _STORE = store
    return _STORE

def enrich_result_with_metadata(result: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich a search result with project metadata from parquet files"""
    projects_meta = load_projects_metadata()
    
    # Get the project ID without chunk suffix
    doc_id = result.get('id', '')
    project_id = doc_id.split('#')[0] if '#' in doc_id else doc_id
    
    # Get metadata from parquet if available
    if project_id in projects_meta:
        project_data = projects_meta[project_id]
        
        # Ensure metadata dict exists
        if 'metadata' not in result:
            result['metadata'] = {}
        
        # Update metadata with project data
        result['metadata'].update(project_data)
        
        # Also add title and abstract at top level for easier access in UI
        result['title'] = project_data.get('title', '')
        result['abstract'] = project_data.get('abstract', '')
    
    return result

@app.on_event("startup")
async def startup_event():
    """Load metadata on startup"""
    logger.info("Loading projects metadata on startup...")
    load_projects_metadata()
    logger.info("Startup complete")

@app.get("/health")
def health():
    store = get_store()
    size = len(getattr(store, "id_map", []))
    exists = INDEX_PATH.exists() and META_PATH.exists()
    projects_loaded = len(_PROJECTS_METADATA) if _PROJECTS_METADATA else 0
    
    return {
        "status": "ok",
        "index_exists": exists,
        "index_size": size,
        "projects_metadata_loaded": projects_loaded
    }

@app.post("/add_documents")
def add_documents(req: AddRequest):
    if not req.documents:
        raise HTTPException(status_code=400, detail="No documents provided")

    emb = get_embedder()
    store = get_store()

    texts = [d.text for d in req.documents]
    ids = [d.id for d in req.documents]
    metas = [d.metadata or {} for d in req.documents]

    vecs = emb.encode(texts)
    store.add(vecs, ids, metas)
    store.save()
    return {"added": len(ids)}

def _norm(x):
    return (str(x) if x is not None else "").strip().lower()

def _passes_filters(meta: Dict[str, Any], f: Optional[MetaFilters]) -> bool:
    if not f:
        return True

    if f.framework:
        m = _norm(meta.get("frameworkName", ""))
        if not any(_norm(v) == m for v in f.framework):
            mn = _norm(meta.get("framework_norm", ""))
            if not any(_norm(v) == mn for v in f.framework):
                return False

    if f.ris3cat_ambit:
        m = _norm(meta.get("RIS3CAT_Ambit", ""))
        if not any(_norm(v) == m for v in f.ris3cat_ambit):
            return False

    if f.ris3cat_tft:
        m = _norm(meta.get("RIS3CAT_TFT", ""))
        if not any(_norm(v) == m for v in f.ris3cat_tft):
            return False

    year_val = meta.get("year_norm", meta.get("startingYear"))
    try:
        y = int(float(year_val))
    except Exception:
        y = None

    if f.year_from is not None and (y is None or y < f.year_from):
        return False
    if f.year_to is not None and (y is None or y > f.year_to):
        return False

    return True

@app.post("/search")
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    emb = get_embedder()
    store = get_store()

    total = len(getattr(store, "id_map", []))
    # Increase k_factor to get more results before deduplication
    kk = min(req.k * max(1, req.k_factor * 2), max(1, total))

    qv = emb.encode([req.query])[0]
    hits = store.query(qv, k=kk)

    # Enrich results with metadata from parquet files
    enriched_hits = [enrich_result_with_metadata(h) for h in hits]

    # Apply filters on the enriched metadata
    filtered = [h for h in enriched_hits if _passes_filters(h.get("metadata", {}), req.filters)]
    
    # De-duplicate: keep only the best scoring chunk per project
    seen_projects = {}
    deduplicated = []
    
    for hit in filtered:
        # Extract base project ID (without chunk suffix)
        doc_id = hit.get('id', '')
        project_id = doc_id.split('#')[0] if '#' in doc_id else doc_id
        
        # Keep only the highest scoring chunk for each project
        if project_id not in seen_projects:
            seen_projects[project_id] = hit
            deduplicated.append(hit)
        elif hit.get('score', 0) > seen_projects[project_id].get('score', 0):
            # Replace with higher scoring chunk
            deduplicated.remove(seen_projects[project_id])
            seen_projects[project_id] = hit
            deduplicated.append(hit)
    
    # Sort by score and take top k
    deduplicated.sort(key=lambda x: x.get('score', 0), reverse=True)
    results = deduplicated[:req.k]

    return {
        "query": req.query,
        "k": req.k,
        "returned": len(results),
        "filters": req.filters.dict() if req.filters else None,
        "results": results,
    }
