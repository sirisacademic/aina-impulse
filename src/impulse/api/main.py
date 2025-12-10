import os
import pandas as pd
import logging
import numpy as np
import traceback

from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.impulse.parser import QueryParser
from src.impulse.settings import settings
from src.impulse.embedding.embedder import Embedder
from src.impulse.vector_store.factory import build_store
from src.impulse.query_expansion.loader import load_kb
from src.impulse.query_expansion.expansion import (
    expand_query_with_vectors,
    ExpansionConfig,
    build_kb_index,
    get_active_centroids,
    get_expansion_summary
)
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

# API Key from environment
API_KEY = os.getenv("IMPULSE_API_KEY", "")

app = FastAPI(title="IMPULS-AINA API", version="1")

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

# Endpoints that don't require authentication
PUBLIC_ENDPOINTS = {"/health", "/docs", "/openapi.json", "/redoc"}

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    """Verify API key for protected endpoints"""
    # Skip auth for OPTIONS requests (CORS preflight)
    if request.method == "OPTIONS":
        return await call_next(request)
    
    # Skip auth for public endpoints
    if request.url.path in PUBLIC_ENDPOINTS:
        return await call_next(request)
    
    # Skip auth if no API key is configured (development mode)
    if not API_KEY:
        return await call_next(request)
    
    # Check API key
    provided_key = request.headers.get("X-API-Key")
    
    # TEMPORARY DEBUG - remove after testing
    logger.info(f"API_KEY configured: '{API_KEY[:4]}...'")
    logger.info(f"Provided key: '{provided_key[:4] if provided_key else None}...'")
    
    if provided_key != API_KEY:
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing API key"},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )
    
    return await call_next(request)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
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

# Load the knowledge base and build index
KB = load_kb(settings.KB_PATH)
KB_INDEX = build_kb_index(KB)
logger.info(f"Loaded KB with {len(KB)} concepts, {len(KB_INDEX)} indexed")


class Document(BaseModel):
    id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    metadata: Optional[Dict[str, Any]] = None


class AddRequest(BaseModel):
    documents: List[Document]


class MetaFilters(BaseModel):
    """Metadata filters with synonym support"""
    framework: Optional[List[str]] = None
    instrument: Optional[str] = None
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    country: Optional[List[str]] = None
    region: Optional[List[str]] = None
    province: Optional[List[str]] = None
    organization_type: Optional[List[str]] = None
    organisations: Optional[List[Dict[str, Optional[str]]]] = None


class ExpansionSettings(BaseModel):
    """User-controllable expansion settings"""
    enabled: bool = False
    alias_levels: List[int] = [1, 2]
    parent_levels: List[int] = []
    excluded_terms: List[str] = []  # NEW: Terms to exclude from search
    return_details: bool = False


class SearchRequest(BaseModel):
    query: str
    k: int = 5
    k_factor: int = 5
    filters: Optional[MetaFilters] = None
    use_parsing: bool = False
    expansion: Optional[ExpansionSettings] = None


class ParseRequest(BaseModel):
    query: str
    system_prompt: Optional[str] = None


class ParseResponse(BaseModel):
    success: bool
    parsed: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    raw_output: Optional[str] = None

# ============================================================================
# KB (Knowledge Base) Endpoints for Graph Exploration
# ============================================================================

class KBSearchResult(BaseModel):
    wikidata_id: str
    keyword: str
    label_en: Optional[str] = None
    label_es: Optional[str] = None
    label_ca: Optional[str] = None


class KBConceptDetail(BaseModel):
    wikidata_id: str
    keyword: str
    labels: Dict[str, str]  # lang -> label
    aliases: Dict[str, List[str]]  # lang -> list of aliases
    definition: Optional[str] = None
    parents: List[Dict[str, Any]]  # list of parent concepts
    children: List[Dict[str, Any]]  # list of child concepts

DATA_DIR = Path(settings.index_dir)
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = DATA_DIR / "vectors.hnsw"
META_PATH = DATA_DIR / "metadata.json"

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
    global _PROJECTS_METADATA, _PARTICIPANTS_METADATA
    
    if _PROJECTS_METADATA is not None:
        return _PROJECTS_METADATA, _PARTICIPANTS_METADATA
    
    _PROJECTS_METADATA = {}
    _PARTICIPANTS_METADATA = []
    
    try:
        if PROJECTS_PARQUET.exists():
            logger.info(f"Loading projects metadata from {PROJECTS_PARQUET}")
            projects_df = pd.read_parquet(PROJECTS_PARQUET)
            
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
        
        if PARTICIPANTS_PARQUET.exists():
            logger.info(f"Loading participants metadata from {PARTICIPANTS_PARQUET}")
            participants_df = pd.read_parquet(PARTICIPANTS_PARQUET)
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


def enrich_result_with_metadata(result: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich search result with project metadata from parquet files"""
    projects_meta, participants_meta = load_metadata()
    
    doc_id = result.get('id', '')
    project_id = doc_id.split('#')[0] if '#' in doc_id else doc_id
    
    if project_id in projects_meta:
        project_data = projects_meta[project_id]
        
        if 'metadata' not in result:
            result['metadata'] = {}
        
        result['title'] = project_data.get('title', '')
        result['abstract'] = project_data.get('abstract', '')
        
        for key, value in project_data.items():
            if key not in ('title', 'abstract'):
                result['metadata'][key] = value
    
    project_participants = [p for p in participants_meta 
                           if p.get('project_id') == project_id]
    result['participants'] = project_participants
    
    return result


def map_parsed_filters(parsed: Dict[str, Any]) -> Optional[MetaFilters]:
    """Convert parsed JSON filters to API filter format WITH synonym expansion."""
    filters_data = parsed.get("filters", {})
    mapped = {}
    
    if filters_data.get("programme"):
        frameworks = normalize_framework(filters_data["programme"])
        if frameworks:
            mapped["framework"] = frameworks
    
    year_val = filters_data.get("year")
    if year_val:
        year_str = str(year_val).strip()
        
        if '-' in year_str and not year_str.startswith(('<', '>', '=')):
            parts = year_str.split('-')
            try:
                mapped["year_from"] = int(parts[0])
                mapped["year_to"] = int(parts[1])
            except (ValueError, IndexError):
                pass
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
        else:
            try:
                y = int(year_str)
                mapped["year_from"] = y
                mapped["year_to"] = y
            except ValueError:
                pass
    
    if filters_data.get("instrument"):
        mapped["instrument"] = filters_data["instrument"]
    
    if filters_data.get("location"):
        loc = filters_data["location"]
        
        country = loc.get("country")
        if country:
            mapped["country"] = [country] if isinstance(country, str) else country
        
        region = loc.get("region")
        if region:
            regions = normalize_region(region)
            if regions:
                mapped["region"] = regions
        
        province = loc.get("province")
        if province:
            provinces = normalize_province(province)
            if provinces:
                mapped["province"] = provinces
    
    if filters_data.get("organization_type"):
        org_types = normalize_org_type(filters_data["organization_type"])
        if org_types:
            mapped["organization_type"] = org_types
    
    if parsed.get("organisations"):
        orgs = parsed["organisations"]
        if not isinstance(orgs, list):
            orgs = [orgs]
        mapped["organisations"] = orgs
    
    return MetaFilters(**mapped) if mapped else None


def _norm(x):
    return (str(x) if x is not None else "").strip().lower()


def _passes_filters(meta: Dict[str, Any], f: Optional[MetaFilters]) -> bool:
    """Check if metadata passes filters WITH SYNONYM MATCHING."""
    if not f:
        return True

    if f.framework:
        meta_framework = meta.get("framework_name", "")
        if not any(matches_framework(meta_framework, filter_fw) 
                   for filter_fw in f.framework):
            return False

    if f.instrument:
        meta_instrument = meta.get("instrument_name", "").lower()
        if f.instrument.lower() not in meta_instrument:
            return False

    year_val = meta.get("year_numeric") or meta.get("year")
    try:
        y = int(float(year_val)) if year_val else None
    except (ValueError, TypeError):
        y = None

    if f.year_from is not None and (y is None or y < f.year_from):
        return False
    if f.year_to is not None and (y is None or y > f.year_to):
        return False

    if f.country:
        meta_country = meta.get("country", "").strip().upper()
        if not any(c.strip().upper() == meta_country for c in f.country):
            return False

    if f.region:
        meta_region = meta.get("region_norm") or meta.get("region", "")
        if not any(matches_region(meta_region, filter_region) 
                   for filter_region in f.region):
            return False

    if f.province:
        meta_province = meta.get("province_norm") or meta.get("province", "")
        if not any(matches_province(meta_province, filter_province) 
                   for filter_province in f.province):
            return False

    if f.organization_type:
        meta_org_type = meta.get("organization_type", "")
        if not any(matches_org_type(meta_org_type, filter_type) 
                   for filter_type in f.organization_type):
            return False

    if f.organisations:
        project_id = meta.get("project_id")
        if not project_id:
            return False
        
        _, participants = load_metadata()
        project_orgs = [p for p in participants 
                        if p.get('project_id') == project_id]
        
        if not project_orgs:
            return False
        
        for filter_org in f.organisations:
            org_name = filter_org.get('name')
            org_type = filter_org.get('type')
            org_location = filter_org.get('location')
            org_location_level = filter_org.get('location_level')
            
            match_found = False
            for p in project_orgs:
                name_match = True
                if org_name:
                    p_name = p.get('organization_name', '')
                    name_match = matches_organization(p_name, org_name)
                
                type_match = True
                if org_type:
                    p_type = p.get('organization_type', '')
                    type_match = matches_org_type(p_type, org_type)
                
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
                        p_region = p.get('region_norm') or p.get('region', '')
                        p_province = p.get('province_norm') or p.get('province', '')
                        p_country = p.get('country', '')
                        location_match = (
                            matches_region(p_region, org_location) or
                            matches_province(p_province, org_location) or
                            _norm(p_country) == _norm(org_location)
                        )
                
                if name_match and type_match and location_match:
                    match_found = True
                    break
            
            if not match_found:
                return False

    return True


def search_with_expansion(
    query_vec: np.ndarray,
    expansion_result: Dict,
    store,
    k: int,
    k_factor: int,
    alias_levels: List[int] = [1, 2],
    parent_levels: List[int] = [],
    excluded_terms: List[str] = []  # NEW parameter
) -> List[Dict]:
    """
    Execute search using query vector + expansion centroid vectors.
    
    Strategy: 
    - Search with each vector separately
    - Merge results, keeping highest score per document
    - Weight original query higher than expansions
    """
    total_docs = len(getattr(store, "id_map", []))
    kk = min(k * max(1, k_factor * 2), total_docs)
    
    # Weight for expansion matches (slightly lower than direct query)
    EXPANSION_WEIGHT = 0.9
    
    # Convert excluded_terms to lowercase set for comparison
    excluded_set = {term.lower() for term in excluded_terms}
    
    # Track best score per document
    doc_scores: Dict[str, Dict] = {}
    
    # 1. Search with original query (weight = 1.0)
    hits_base = store.query(query_vec, k=kk)
    for hit in hits_base:
        doc_id = hit["id"]
        doc_scores[doc_id] = {
            "id": doc_id,
            "score": hit["score"],
            "metadata": hit.get("metadata", {}),
            "matched_by": ["query"]
        }
    
    # 2. Search with expansion centroids
    centroids = get_active_centroids(
        expansion_result,
        alias_levels=alias_levels,
        parent_levels=parent_levels
    )
    
    for centroid_info in centroids:
        # NEW: Skip excluded terms
        if centroid_info["representative"].lower() in excluded_set:
            continue
            
        centroid_vec = np.array(centroid_info["centroid"])
        hits = store.query(centroid_vec, k=kk)
        
        for hit in hits:
            doc_id = hit["id"]
            weighted_score = hit["score"] * EXPANSION_WEIGHT
            
            if doc_id in doc_scores:
                # Update if better score, always track matched_by
                if weighted_score > doc_scores[doc_id]["score"]:
                    doc_scores[doc_id]["score"] = weighted_score
                if centroid_info["representative"] not in doc_scores[doc_id]["matched_by"]:
                    doc_scores[doc_id]["matched_by"].append(centroid_info["representative"])
            else:
                doc_scores[doc_id] = {
                    "id": doc_id,
                    "score": weighted_score,
                    "metadata": hit.get("metadata", {}),
                    "matched_by": [centroid_info["representative"]]
                }
    
    # Convert to list and sort
    results = list(doc_scores.values())
    results.sort(key=lambda x: x["score"], reverse=True)
    
    return results


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
        "parser_loaded": parser_loaded,
        "kb_concepts": len(KB),
        "kb_indexed": len(KB_INDEX),
        "auth_enabled": bool(API_KEY)  # NEW: indicates if auth is active
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
    query_language = None
    
    # Parse query if requested
    if req.use_parsing and req.query.strip():
        parser = get_parser()
        parse_result = parser.parse(req.query)
        
        if parse_result["success"]:
            parsed = parse_result["parsed"]
            meta = parsed.get("meta", {})
            
            feedback = {
                "query_rewrite": parsed.get("query_rewrite"),
                "notes": meta.get("notes"),
                "parsed_json": parsed
            }
            
            query_text = parsed.get("semantic_query") or req.query
            query_language = meta.get("lang")
            
            parsed_filters = map_parsed_filters(parsed)
            if parsed_filters:
                if filters:
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
    
    # Get embedder and store
    emb = get_embedder()
    store = get_store()
    total = len(getattr(store, "id_map", []))
    
    # Handle empty query (filter-only search)
    if not query_text.strip():
        kk = min(settings.max_filter_only_retrieve, total)
        query_vec = np.zeros(emb.get_dim())
        logger.info(f"Filter-only search: retrieving {kk}/{total} documents")
        hits = store.query(query_vec, k=kk)
    else:
        # Embed query
        query_vec = emb.encode([query_text])[0]
        
        # Query Expansion (v3)
        expansion_result = None
        expansion_summary = None
        
        expansion_enabled = (
            req.expansion is not None and 
            req.expansion.enabled and 
            query_text.strip()
        )
        
        if expansion_enabled:
            config = ExpansionConfig(
                use_aliases=True,
                use_parents=bool(req.expansion.parent_levels),
                languages=["en", "es", "ca"]
            )
            
            expansion_result = expand_query_with_vectors(
                query=query_text,
                kb=KB,
                encoder=emb,
                config=config,
                kb_index=KB_INDEX,
                query_language=query_language
            )
            
            if req.expansion.return_details:
                expansion_summary = get_expansion_summary(expansion_result)
        
        # Execute search
        if expansion_result:
            hits = search_with_expansion(
                query_vec=query_vec,
                expansion_result=expansion_result,
                store=store,
                k=req.k,
                k_factor=req.k_factor,
                alias_levels=req.expansion.alias_levels,
                parent_levels=req.expansion.parent_levels,
                excluded_terms=req.expansion.excluded_terms  # NEW: pass excluded terms
            )
        else:
            kk = min(req.k * max(1, req.k_factor * 2), total)
            hits = store.query(query_vec, k=kk)
    
    # Enrich with metadata
    enriched_hits = [enrich_result_with_metadata(h) for h in hits]
    
    # Apply filters
    filtered = [h for h in enriched_hits if _passes_filters(h.get("metadata", {}), filters)]
    
    # Track totals
    total_matching = len(filtered)
    results_limited = total_matching > settings.max_results_warning_threshold
    
    # Deduplicate by project
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
    
    # Build response
    response = {
        "query": req.query,
        "query_used": query_text,
        "k": req.k,
        "returned": len(results),
        "total_matching": total_matching,
        "filters": filters.dict() if filters else {},
        "results": results
    }
    
    # Add expansion info if requested
    if expansion_enabled and expansion_summary:
        response["expansion"] = expansion_summary
    
    # Add feedback/warnings
    if results_limited:
        if not feedback:
            feedback = {}
        feedback["warning"] = (
            f"Found {total_matching} matching projects. "
            f"Showing top {len(results)}. "
            f"Consider adding more specific filters."
        )
    
    if not query_text.strip() and total_matching == kk:
        if not feedback:
            feedback = {}
        feedback["info"] = (
            f"Filter-only search limited to {kk} documents. "
            f"Use a semantic query for better ranking."
        )
    
    if feedback:
        response["feedback"] = feedback
    
    return response
    
@app.get("/kb/search", response_model=List[KBSearchResult])
def kb_search(q: str, limit: int = 10):
    """
    Search for concepts in the knowledge base.
    Returns top matching concepts by keyword/label.
    """
    if not q or len(q.strip()) < 2:
        return []
    
    query_lower = q.strip().lower()
    results = []
    seen_ids = set()
    
    for concept in KB:
        wikidata_id = concept.get("wikidata_id", "")
        if wikidata_id in seen_ids:
            continue
            
        keyword = concept.get("keyword", "")
        langs = concept.get("languages", {})
        
        # Check keyword match
        score = 0
        if query_lower == keyword.lower():
            score = 100  # Exact match
        elif keyword.lower().startswith(query_lower):
            score = 80  # Prefix match
        elif query_lower in keyword.lower():
            score = 60  # Contains match
        
        # Check label matches
        for lang in ["en", "es", "ca"]:
            label = langs.get(lang, {}).get("label", "")
            if label:
                if query_lower == label.lower():
                    score = max(score, 95)
                elif label.lower().startswith(query_lower):
                    score = max(score, 75)
                elif query_lower in label.lower():
                    score = max(score, 55)
        
        # Check aliases
        for lang in ["en", "es", "ca"]:
            aliases = langs.get(lang, {}).get("also_known_as", [])
            for alias in aliases:
                if alias and query_lower in alias.lower():
                    score = max(score, 50)
                    break
        
        if score > 0:
            seen_ids.add(wikidata_id)
            results.append({
                "wikidata_id": wikidata_id,
                "keyword": keyword,
                "label_en": langs.get("en", {}).get("label"),
                "label_es": langs.get("es", {}).get("label"),
                "label_ca": langs.get("ca", {}).get("label"),
                "_score": score
            })
    
    # Sort by score and limit
    results.sort(key=lambda x: x["_score"], reverse=True)
    
    # Remove internal score before returning
    for r in results:
        del r["_score"]
    
    return results[:limit]


@app.get("/kb/concept/{wikidata_id}")
def kb_concept(wikidata_id: str):
    """
    Get detailed information about a concept including parents and children.
    """
    # Find the concept
    concept = None
    for c in KB:
        if c.get("wikidata_id") == wikidata_id:
            concept = c
            break
    
    if not concept:
        raise HTTPException(status_code=404, detail=f"Concept not found: {wikidata_id}")
    
    langs = concept.get("languages", {})
    
    # Build labels dict
    labels = {}
    for lang in ["en", "es", "ca"]:
        label = langs.get(lang, {}).get("label")
        if label:
            labels[lang] = label
    
    # Build aliases dict
    aliases = {}
    for lang in ["en", "es", "ca"]:
        aka = langs.get(lang, {}).get("also_known_as", [])
        if aka:
            aliases[lang] = aka
    
    # Get definition (prefer English)
    definition = None
    for lang in ["en", "es", "ca"]:
        defn = langs.get(lang, {}).get("definition")
        if defn:
            definition = defn
            break
    
    # Get parents (from subclass_of)
    parents = []
    for parent_ref in concept.get("subclass_of", []):
        parent_id = parent_ref.get("id", "")
        parent_label = parent_ref.get("label", "")
        
        # Try to get more info from KB_INDEX
        if parent_id in KB_INDEX:
            parent_concept = KB_INDEX[parent_id]
            parent_langs = parent_concept.get("languages", {})
            parents.append({
                "wikidata_id": parent_id,
                "keyword": parent_concept.get("keyword", parent_label),
                "label_en": parent_langs.get("en", {}).get("label"),
                "label_es": parent_langs.get("es", {}).get("label"),
                "label_ca": parent_langs.get("ca", {}).get("label"),
                "in_kb": True
            })
        else:
            parents.append({
                "wikidata_id": parent_id,
                "keyword": parent_label,
                "label_en": parent_label,
                "label_es": None,
                "label_ca": None,
                "in_kb": False
            })
    
    # Find children (concepts that have this as parent)
    children = []
    for c in KB:
        for parent_ref in c.get("subclass_of", []):
            if parent_ref.get("id") == wikidata_id:
                c_langs = c.get("languages", {})
                children.append({
                    "wikidata_id": c.get("wikidata_id"),
                    "keyword": c.get("keyword"),
                    "label_en": c_langs.get("en", {}).get("label"),
                    "label_es": c_langs.get("es", {}).get("label"),
                    "label_ca": c_langs.get("ca", {}).get("label"),
                    "in_kb": True
                })
                break
    
    return {
        "wikidata_id": wikidata_id,
        "keyword": concept.get("keyword", ""),
        "labels": labels,
        "aliases": aliases,
        "definition": definition,
        "parents": parents,
        "children": children
    }
    

