from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
import logging

from src.impulse.parser import QueryParser
from src.impulse.settings import settings
from src.impulse.embedding.embedder import Embedder
from src.impulse.vector_store.factory import build_store
from src.impulse.query_expansion.loader import load_kb
from src.impulse.query_expansion.expansion import expand_query_with_vectors

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

# Load the knowledge base for query expansion
KB = load_kb(settings.KB_PATH)

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
    use_parsing: bool = False
    query_expansion: bool = False
    return_expansion_vectors: bool = False

class ParseRequest(BaseModel):
    query: str
    system_prompt: Optional[str] = None

class ParseResponse(BaseModel):
    success: bool
    parsed: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    raw_output: Optional[str] = None

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
_PARSER = None

def get_parser():
    global _PARSER
    if _PARSER is None:
        _PARSER = QueryParser()
    return _PARSER

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

def pick_best_definition(definition_vectors, priority):
    """Pick the first available definition in priority order."""
    for lang in priority:
        for d in definition_vectors:
            if d["language"] == lang:
                return d
    return None

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

def map_parsed_filters(parsed: Dict[str, Any]) -> Optional[MetaFilters]:
    """Convert parsed JSON filters to API filter format"""
    filters_data = parsed.get("filters", {})
    
    mapped = {}
    
    # Programme -> framework
    if filters_data.get("programme"):
        mapped["framework"] = [filters_data["programme"]]
    
    # Year - handle various formats
    year_val = filters_data.get("year")
    if year_val:
        year_str = str(year_val).strip()
        
        # Range: "2015-2020"
        if '-' in year_str and not year_str.startswith(('<', '>', '=')):
            parts = year_str.split('-')
            try:
                mapped["year_from"] = int(parts[0])
                mapped["year_to"] = int(parts[1])
            except (ValueError, IndexError):
                pass
        # Greater than or equal: ">=2018"
        elif year_str.startswith('>='):
            try:
                mapped["year_from"] = int(year_str[2:])
            except ValueError:
                pass
        # Less than or equal: "<=2018"
        elif year_str.startswith('<='):
            try:
                mapped["year_to"] = int(year_str[2:])
            except ValueError:
                pass
        # Greater than: ">2018"
        elif year_str.startswith('>'):
            try:
                mapped["year_from"] = int(year_str[1:])
            except ValueError:
                pass
        # Less than: "<2018"
        elif year_str.startswith('<'):
            try:
                mapped["year_to"] = int(year_str[1:])
            except ValueError:
                pass
        # Single year
        else:
            try:
                year = int(year_str)
                mapped["year_from"] = year
                mapped["year_to"] = year
            except ValueError:
                pass
    
    return MetaFilters(**mapped) if mapped else None

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
    parser_loaded = _PARSER is not None
    
    return {
        "status": "ok",
        "index_exists": exists,
        "index_size": size,
        "projects_metadata_loaded": projects_loaded,
        "parser_loaded": parser_loaded
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

@app.post("/parse", response_model=ParseResponse)
def parse_query(req: ParseRequest):
    """Parse natural language query to structured JSON"""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")
    
    parser = get_parser()
    result = parser.parse(req.query, req.system_prompt)
    
    return ParseResponse(**result)

@app.post("/search")
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")
    
    query_text = req.query
    filters = req.filters
    feedback = None
    
    # Parse query if requested
    if req.use_parsing:
        parser = get_parser()
        parse_result = parser.parse(req.query)
        
        if parse_result["success"]:
            parsed = parse_result["parsed"]
            
            # Extract feedback for user
            meta = parsed.get("meta", {})
            feedback = {
                "query_rewrite": parsed.get("query_rewrite"),
                "notes": meta.get("notes")
            }
            
            # Use semantic_query if available, fallback to original
            query_text = parsed.get("semantic_query") or req.query
            
            # Map parsed filters (merge with existing)
            parsed_filters = map_parsed_filters(parsed)
            if parsed_filters:
                filters = parsed_filters if not filters else filters
        else:
            # Parsing failed, continue with original query
            logger.warning(f"Parsing failed: {parse_result.get('error')}")
            feedback = {"error": "Could not parse query, using direct search"}
    
    # -------- Query Expansion (Definition-based) -------------------
    expansion_data = None
    definition_vectors = []   # list of (vector, language, definition)

    # Execute search
    emb = get_embedder()

    if req.query_expansion:
        
        expansion_data = expand_query_with_vectors(
            query=req.query,
            kb=KB,
            encoder=emb,   # dense embedder
            languages=["en", "es", "ca"] # "it"
        )
        
        # Extract vectors for search
        definition_vectors = expansion_data["definition_vectors"]

    store = get_store()
    total = len(getattr(store, "id_map", []))
    kk = min(req.k * max(1, req.k_factor * 2), max(1, total))
    # --- Base query embedding ---
    qv_base = emb.encode([query_text])[0]
    # First search: base query
    hits_base = store.query(qv_base, k=kk)

    # --- Definition-expanded searches ---
    hits_expanded = []

    # Select ONLY the best definition using priority
    PRIORITY = ["en", "es", "ca"]
    best_def = pick_best_definition(definition_vectors, PRIORITY)

    hits_expanded = []

    if best_def:
        vec = best_def["vector"]
        h = store.query(vec, k=kk)

        # annotate for explainability
        for item in h:
            item["expansion_language"] = best_def["language"]
            item["expansion_definition"] = best_def["definition"]

        hits_expanded.extend(h)
    # Combine: base + expanded
    hits = hits_base + hits_expanded
    
    # Enrich results with metadata
    enriched_hits = [enrich_result_with_metadata(h) for h in hits]
    
    # Apply filters
    filtered = [h for h in enriched_hits if _passes_filters(h.get("metadata", {}), filters)]
    
    # De-duplicate by project
    seen_projects = {}
    deduplicated = []
    
    for hit in filtered:
        doc_id = hit.get('id', '')
        project_id = doc_id.split('#')[0] if '#' in doc_id else doc_id
        
        if project_id not in seen_projects:
            seen_projects[project_id] = hit
            deduplicated.append(hit)
        elif hit.get('score', 0) > seen_projects[project_id].get('score', 0):
            deduplicated.remove(seen_projects[project_id])
            seen_projects[project_id] = hit
            deduplicated.append(hit)
    
    # Sort and limit
    deduplicated.sort(key=lambda x: x.get('score', 0), reverse=True)
    results = deduplicated[:req.k]
    
    response = {
        "query": req.query,
        "query_used": query_text,
        "k": req.k,
        "returned": len(results),
        "filters": filters.dict() if filters else None,
        "results": results
    }
    
    if feedback:
        response["feedback"] = feedback

    if req.return_expansion_vectors and expansion_data:
        response["expansion"] = {
            "query_vector": expansion_data["query_vector"],
            "definition_vectors": expansion_data["definition_vectors"],
            "aliases": expansion_data["aliases"]
        }
    
    return response
