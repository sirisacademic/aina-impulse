from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
import logging
import numpy as np
import traceback

from src.impulse.parser import QueryParser
from src.impulse.settings import settings
from src.impulse.embedding.embedder import Embedder
from src.impulse.vector_store.factory import build_store
from src.impulse.query_expansion.loader import load_kb
from src.impulse.query_expansion.expansion import expand_query_with_vectors
from src.impulse.normalization import (
    normalize_framework,
    normalize_org_type,
    normalize_province,
    normalize_region,
    matches_framework,
    matches_org_type,
    matches_province,
    matches_region,
    normalize_organization,
    matches_organization,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="IMPULSE WP1 Retrieval API", version="0.3.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Global exception handler to ensure CORS headers on errors
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Catch all exceptions and return with CORS headers to prevent CORS errors from hiding real issues"""
    logger.error(f"Unhandled exception in {request.url.path}: {exc}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "type": type(exc).__name__,
            "path": str(request.url.path)
        },
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
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
    """Metadata filters with synonym support"""
    framework: Optional[List[str]] = None
    instrument: Optional[str] = None  # Substring match
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    country: Optional[List[str]] = None
    region: Optional[List[str]] = None
    province: Optional[List[str]] = None
    organization_type: Optional[List[str]] = None
    organisations: Optional[List[Dict[str, Optional[str]]]] = None

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

# Paths for parquet metadata files
META_DIR = DATA_DIR.parent / "meta"
PROJECTS_PARQUET = META_DIR / "projects.parquet"
PARTICIPANTS_PARQUET = META_DIR / "participants.parquet"

_EMBEDDER = None
_STORE = None
_PROJECTS_METADATA = None
_PARTICIPANTS_METADATA = None
_PARSER = None

def get_parser():
    global _PARSER
    if _PARSER is None:
        _PARSER = QueryParser()
    return _PARSER

def load_metadata():
    """Load project and participant metadata from parquet files at startup"""
    global _PROJECTS_METADATA, _PARTICIPANTS_METADATA
    
    if _PROJECTS_METADATA is not None:
        return _PROJECTS_METADATA, _PARTICIPANTS_METADATA
    
    _PROJECTS_METADATA = {}
    _PARTICIPANTS_METADATA = []
    
    try:
        # Load projects
        if PROJECTS_PARQUET.exists():
            logger.info(f"Loading projects metadata from {PROJECTS_PARQUET}")
            projects_df = pd.read_parquet(PROJECTS_PARQUET)
            
            # Create dictionary for fast project lookup
            for _, row in projects_df.iterrows():
                project_id = row['project_id']
                _PROJECTS_METADATA[project_id] = {
                    'title': row.get('title', ''),
                    'abstract': row.get('abstract', ''),
                    'framework_name': row.get('framework_name', ''),
                    'instrument_name': row.get('instrument_name', ''),
                    'year': row.get('year_numeric'),
                    'total_investment': row.get('total_investment'),
                    'total_grant': row.get('total_grant'),
                }
            
            logger.info(f"Loaded metadata for {len(_PROJECTS_METADATA)} projects")
        else:
            logger.warning(f"Projects parquet not found: {PROJECTS_PARQUET}")
        
        # Load participants for organization filtering
        if PARTICIPANTS_PARQUET.exists():
            logger.info(f"Loading participants metadata from {PARTICIPANTS_PARQUET}")
            participants_df = pd.read_parquet(PARTICIPANTS_PARQUET)
            
            # Store as list of dicts for easy filtering
            _PARTICIPANTS_METADATA = participants_df.to_dict('records')
            
            logger.info(f"Loaded {len(_PARTICIPANTS_METADATA)} participant records")
        else:
            logger.warning(f"Participants parquet not found: {PARTICIPANTS_PARQUET}")
    
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        _PROJECTS_METADATA = {}
        _PARTICIPANTS_METADATA = []
    
    return _PROJECTS_METADATA, _PARTICIPANTS_METADATA

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
                raise RuntimeError("Index required but not found. Run build_index.py first.")
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
    """Enrich search result with project metadata from parquet files"""
    projects_meta, participants_meta = load_metadata()
    
    # Extract project ID (remove chunk suffix if present)
    doc_id = result.get('id', '')
    project_id = doc_id.split('#')[0] if '#' in doc_id else doc_id
    
    # Get metadata from parquet
    if project_id in projects_meta:
        project_data = projects_meta[project_id]
        
        if 'metadata' not in result:
            result['metadata'] = {}
        
        # Add title and abstract at top level
        result['title'] = project_data.get('title', '')
        result['abstract'] = project_data.get('abstract', '')
        
        # Add remaining fields to metadata (exclude title/abstract to avoid duplication)
        for key, value in project_data.items():
            if key not in ('title', 'abstract'):
                result['metadata'][key] = value
    
    # Add participants
    project_participants = [p for p in participants_meta 
                           if p.get('project_id') == project_id]
    result['participants'] = project_participants
    
    return result

def map_parsed_filters(parsed: Dict[str, Any]) -> Optional[MetaFilters]:
    """
    Convert parsed JSON filters to API filter format WITH synonym expansion.
    
    Returns filters with Lists of database values (synonym groups).
    """
    filters_data = parsed.get("filters", {})
    mapped = {}
    
    # Framework WITH SYNONYM EXPANSION
    # User: "h2020" → ["H2020", "HORIZON"]
    if filters_data.get("programme"):
        frameworks = normalize_framework(filters_data["programme"])
        if frameworks:
            mapped["framework"] = frameworks
    
    # Year handling
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
        # Comparisons: ">=2018", "<=2020", etc.
        elif year_str.startswith('>='):
            try:
                mapped["year_from"] = int(year_str[2:])
            except ValueError:
                pass
        elif year_str.startswith('<='):
            try:
                mapped["year_to"] = int(year_str[2:])
            except ValueError:
                pass
        elif year_str.startswith('>'):
            try:
                mapped["year_from"] = int(year_str[1:]) + 1
            except ValueError:
                pass
        elif year_str.startswith('<'):
            try:
                mapped["year_to"] = int(year_str[1:]) - 1
            except ValueError:
                pass
        # Exact year
        else:
            try:
                y = int(year_str)
                mapped["year_from"] = y
                mapped["year_to"] = y
            except ValueError:
                pass
    
    # Instrument (substring match, no normalization needed)
    if filters_data.get("instrument"):
        mapped["instrument"] = filters_data["instrument"]
    
    # Location filters WITH NORMALIZATION
    if filters_data.get("location"):
        loc = filters_data["location"]
        
        # Country
        country = loc.get("country")
        if country:
            mapped["country"] = [country] if isinstance(country, str) else country
        
        # Region WITH SYNONYM EXPANSION
        region = loc.get("region")
        if region:
            regions = normalize_region(region)
            if regions:
                mapped["region"] = regions
        
        # Province WITH SYNONYM EXPANSION
        province = loc.get("province")
        if province:
            provinces = normalize_province(province)
            if provinces:
                mapped["province"] = provinces
    
    # Organization type WITH SYNONYM EXPANSION
    if filters_data.get("organization_type"):
        org_types = normalize_org_type(filters_data["organization_type"])
        if org_types:
            mapped["organization_type"] = org_types
    
    # Organizations (top-level field, not in filters)
    if parsed.get("organisations"):
        orgs = parsed["organisations"]
        if not isinstance(orgs, list):
            orgs = [orgs]
        
        # Store for filtering
        mapped["organisations"] = orgs
    
    return MetaFilters(**mapped) if mapped else None

def _norm(x):
    """Simple normalization helper"""
    return (str(x) if x is not None else "").strip().lower()

def _passes_filters(meta: Dict[str, Any], f: Optional[MetaFilters]) -> bool:
    """
    Check if metadata passes filters WITH SYNONYM MATCHING.
    
    For framework and organization_type, checks if metadata value matches
    ANY value in the filter's synonym list.
    """
    if not f:
        return True

    # Framework filter WITH SYNONYM MATCHING
    # meta might be "HORIZON", filter has ["H2020", "HORIZON"]
    if f.framework:
        meta_framework = meta.get("framework_name", "")
        
        # Check if metadata matches ANY synonym in filter list
        if not any(matches_framework(meta_framework, filter_fw) 
                   for filter_fw in f.framework):
            return False

    # Instrument filter (substring match, case-insensitive)
    if f.instrument:
        meta_instrument = meta.get("instrument_name", "").lower()
        if f.instrument.lower() not in meta_instrument:
            return False

    # Year filters
    year_val = meta.get("year_numeric") or meta.get("year")
    try:
        y = int(float(year_val)) if year_val else None
    except (ValueError, TypeError):
        y = None

    if f.year_from is not None and (y is None or y < f.year_from):
        return False
    if f.year_to is not None and (y is None or y > f.year_to):
        return False

    # Country filter (simple matching)
    if f.country:
        meta_country = meta.get("country", "").strip().upper()
        if not any(c.strip().upper() == meta_country for c in f.country):
            return False

    # Region filter WITH NORMALIZATION
    if f.region:
        meta_region = meta.get("region_norm") or meta.get("region", "")
        
        if not any(matches_region(meta_region, filter_region) 
                   for filter_region in f.region):
            return False

    # Province filter WITH NORMALIZATION
    if f.province:
        meta_province = meta.get("province_norm") or meta.get("province", "")
        
        if not any(matches_province(meta_province, filter_province) 
                   for filter_province in f.province):
            return False

    # Organization type filter WITH SYNONYM MATCHING
    # meta might be "PRC", filter has ["EMPRESA", "PRC"]
    if f.organization_type:
        meta_org_type = meta.get("organization_type", "")
        
        # Check if metadata matches ANY synonym in filter list
        if not any(matches_org_type(meta_org_type, filter_type) 
                   for filter_type in f.organization_type):
            return False

    # Organizations filter (name + type + location)
    # Multiple orgs = ALL must participate (AND logic per prompt)
    if f.organisations:
        project_id = meta.get("project_id")
        if not project_id:
            return False
        
        # Get participants for this project
        _, participants = load_metadata()
        project_orgs = [p for p in participants 
                        if p.get('project_id') == project_id]
        
        if not project_orgs:
            return False  # No participants = can't match
        
        # Check each filter org (ALL must match)
        for filter_org in f.organisations:
            org_name = filter_org.get('name')
            org_type = filter_org.get('type')
            org_location = filter_org.get('location')
            org_location_level = filter_org.get('location_level')
            
            # Find matching participant
            match_found = False
            for p in project_orgs:
                # Check name if specified (using ROR normalization)
                name_match = True
                if org_name:
                    p_name = p.get('organization_name', '')
                    name_match = matches_organization(p_name, org_name)
                
                # Check type if specified (using synonym matching)
                type_match = True
                if org_type:
                    p_type = p.get('organization_type', '')
                    type_match = matches_org_type(p_type, org_type)
                
                # Check location if specified (using normalization)
                location_match = True
                if org_location:
                    if org_location_level == 'region':
                        p_region = p.get('region_norm') or p.get('region', '')
                        location_match = matches_region(p_region, org_location)
                    elif org_location_level == 'province':
                        p_province = p.get('province_norm') or p.get('province', '')
                        location_match = matches_province(p_province, org_location)
                    elif org_location_level == 'country':
                        p_country = p.get('country', '')
                        location_match = _norm(p_country) == _norm(org_location)
                    else:
                        # No level specified, try all
                        p_region = p.get('region_norm') or p.get('region', '')
                        p_province = p.get('province_norm') or p.get('province', '')
                        p_country = p.get('country', '')
                        location_match = (
                            matches_region(p_region, org_location) or
                            matches_province(p_province, org_location) or
                            _norm(p_country) == _norm(org_location)
                        )
                
                # All specified criteria must match
                if name_match and type_match and location_match:
                    match_found = True
                    break
            
            if not match_found:
                return False  # This org requirement not met

    return True

@app.on_event("startup")
async def startup_event():
    """Load metadata on startup"""
    logger.info("Loading projects and participants metadata...")
    load_metadata()
    logger.info("Startup complete")
   
@app.get("/health")
def health():
    store = get_store()
    size = len(getattr(store, "id_map", []))
    exists = INDEX_PATH.exists() and META_PATH.exists()
    projects_loaded = len(_PROJECTS_METADATA) if _PROJECTS_METADATA else 0
    participants_loaded = len(_PARTICIPANTS_METADATA) if _PARTICIPANTS_METADATA else 0
    parser_loaded = _PARSER is not None
    
    return {
        "status": "ok",
        "index_exists": exists,
        "index_size": size,
        "projects_metadata_loaded": projects_loaded,
        "participants_metadata_loaded": participants_loaded,
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
    # Allow empty query only if filters are provided
    if not req.query.strip() and not req.filters:
        raise HTTPException(status_code=400, detail="Empty query without filters")
    
    query_text = req.query
    filters = req.filters
    feedback = None
    
    # Parse query if requested and query is not empty
    if req.use_parsing and req.query.strip():
        parser = get_parser()
        parse_result = parser.parse(req.query)
        
        if parse_result["success"]:
            parsed = parse_result["parsed"]
            
            # Extract feedback
            meta = parsed.get("meta", {})
            feedback = {
                "query_rewrite": parsed.get("query_rewrite"),
                "notes": meta.get("notes"),
                "parsed_json": parsed  # Include full parsed JSON
            }
            
            # Use semantic_query if available
            query_text = parsed.get("semantic_query") or req.query
            
            # Map parsed filters and merge with existing user filters
            parsed_filters = map_parsed_filters(parsed)
            if parsed_filters:
                if filters:
                    # Merge: user filters take precedence over parsed filters
                    merged = parsed_filters.dict()
                    user_dict = filters.dict()
                    for key, value in user_dict.items():
                        if value is not None:
                            merged[key] = value
                    filters = MetaFilters(**merged)
                else:
                    filters = parsed_filters
        else:
            logger.warning(f"Parsing failed: {parse_result.get('error')}")
            feedback = {"error": "Could not parse query, using direct search"}
    
    # -------- Query Expansion (Definition-based) -------------------
    expansion_data = None
    definition_vectors = []

    emb = get_embedder()

    if req.query_expansion and query_text.strip():
        expansion_data = expand_query_with_vectors(
            query=query_text,
            kb=KB,
            encoder=emb,
            languages=["en", "es", "ca"]
        )
        
        # Extract vectors for search
        definition_vectors = expansion_data["definition_vectors"]

    # Execute search
    store = get_store()
    total = len(getattr(store, "id_map", []))
    
    # Determine k for vector search
    if not query_text.strip():
        # Filter-only search: use configurable max to avoid HNSW index limits
        kk = min(settings.max_filter_only_retrieve, total)
        qv_base = np.zeros(emb.get_dim())
        logger.info(f"Filter-only search: retrieving {kk}/{total} documents for filtering")
    else:
        # Normal semantic search
        kk = min(req.k * max(1, req.k_factor * 2), total)
        qv_base = emb.encode([query_text])[0]
    
    # First search: base query
    hits_base = store.query(qv_base, k=kk)

    # --- Definition-expanded searches ---
    hits_expanded = []

    if definition_vectors:
        # Select ONLY the best definition using priority
        PRIORITY = ["en", "es", "ca"]
        best_def = pick_best_definition(definition_vectors, PRIORITY)

        if best_def:
            vec = best_def["vector"]
            h = store.query(vec, k=kk)

            # Annotate for explainability
            for item in h:
                item["expansion_language"] = best_def["language"]
                item["expansion_definition"] = best_def["definition"]

            hits_expanded.extend(h)
    
    # Combine: base + expanded
    hits = hits_base + hits_expanded
    
    # Enrich with metadata
    enriched_hits = [enrich_result_with_metadata(h) for h in hits]
    
    # Apply filters
    filtered = [h for h in enriched_hits if _passes_filters(h.get("metadata", {}), filters)]
    
    # Track if results were limited
    results_limited = False
    total_matching = len(filtered)
    
    if total_matching > settings.max_results_warning_threshold:
        results_limited = True
    
    # De-duplicate by project (keep highest score)
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
        "total_matching": total_matching,
        "filters": filters.dict() if filters else None,
        "results": results
    }
    
    # Add warning if many results found
    if results_limited:
        if not feedback:
            feedback = {}
        feedback["warning"] = (
            f"Found {total_matching} matching projects. "
            f"Showing top {len(results)}. "
            f"Consider adding more specific filters to narrow results."
        )
    
    # Add info about filter-only search limits
    if not query_text.strip() and total_matching == kk:
        if not feedback:
            feedback = {}
        feedback["info"] = (
            f"Filter-only search limited to {kk} documents for performance. "
            f"Use a semantic query for better ranking of large result sets."
        )
    
    if feedback:
        response["feedback"] = feedback

    # Add expansion data if requested
    if req.return_expansion_vectors and expansion_data:
        response["expansion"] = {
            "query_vector": expansion_data["query_vector"],
            "definition_vectors": expansion_data["definition_vectors"],
            "aliases": expansion_data["aliases"]
        }
    
    return response
