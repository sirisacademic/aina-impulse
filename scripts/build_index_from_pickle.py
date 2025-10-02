
#!/usr/bin/env python
"""Build the IMPULSE vector index from a RIS3CAT pickle (grouped per project, no topic tagging)."""
import argparse, re
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# --- path bootstrap (keeps imports working when run from scripts/) ---
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.impulse.settings import settings
from src.impulse.embedding.embedder import Embedder
from src.impulse.vector_store.factory import build_store

KEEP_COLS = [
    "projectId", "projectTitle", "projectAbstract",
    "ecRef",
    "organizationId", "organizationName", "organizationCountry",
    "organizationNuts2Id", "organizationNuts2Name",
    "organizationNuts3Id", "organizationNuts3Name",
    "frameworkName", "instrumentName", "startingDate", "startingYear",
    "sdgName", "RIS3CAT Àmbit Sectorial Líder",
    "RIS3CAT Tecnologia Facilitadora Transversal",
]

# If selected, these fields are included as context when generating the embeddings.
CONTEXT_FIELDS = ["sdgName", "RIS3CAT Àmbit Sectorial Líder", "RIS3CAT Tecnologia Facilitadora Transversal"]

def clean_str(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()

def norm(s):
    return (str(s) if s is not None else "").strip().lower()

def normalize_framework(name: str) -> str:
    n = norm(name)
    if any(k in n for k in ["h2020", "horizon 2020", "horitzó 2020", "horizo 2020"]):
        return "H2020"
    if "horizon europe" in n or n == "he":
        return "Horizon Europe"
    return name or ""

def to_year(y):
    if pd.isna(y):
        return None
    try:
        yi = int(float(y))
        if 1900 <= yi <= 2100:
            return yi
    except Exception:
        pass
    m = re.search(r"\b(19|20)\d{2}\b", str(y))
    return int(m.group(0)) if m else None

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 100) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        slice_ = text[start:end]
        last_period = slice_.rfind(". ")
        if last_period > 300:
            end = start + last_period + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return [c for c in chunks if c]

def chunk_text_by_sentences(text: str, 
                           sentences_per_chunk: int = 6,
                           overlap_sentences: int = 1,
                           min_sentence_length: int = 10) -> List[str]:
    """
    Split text into chunks based on number of sentences.
    
    Args:
        text: The text to chunk (typically title + abstract)
        sentences_per_chunk: Number of sentences per chunk (default 6)
        overlap_sentences: Number of sentences to overlap between chunks (default 1)
        min_sentence_length: Minimum characters for a valid sentence (default 10)
    
    Returns:
        List of text chunks
    """
    from nltk.tokenize import sent_tokenize
    
    if not text or len(text.strip()) == 0:
        return []
    
    # Tokenize into sentences
    sentences = sent_tokenize(text)
    
    # Filter out very short "sentences" (often just numbers or artifacts)
    sentences = [s for s in sentences if len(s.strip()) >= min_sentence_length]
    
    if not sentences:
        return [text] if text.strip() else []
    
    # If short enough, return as single chunk
    if len(sentences) <= sentences_per_chunk:
        return [' '.join(sentences)]
    
    chunks = []
    start_idx = 0
    
    while start_idx < len(sentences):
        end_idx = min(start_idx + sentences_per_chunk, len(sentences))
        
        # Join sentences for this chunk
        chunk_sentences = sentences[start_idx:end_idx]
        chunk_text = ' '.join(chunk_sentences)
        
        # Only add non-empty chunks
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
        
        # Move forward with overlap
        if end_idx >= len(sentences):
            break
            
        # Ensure we make progress even with overlap
        start_idx = max(start_idx + 1, end_idx - overlap_sentences)
    
    return chunks if chunks else [text]

def chunk_project_text(title: str, abstract: str,
                       sentences_per_chunk: int = 6,
                       overlap_sentences: int = 1) -> List[str]:
    """
    Chunk project text, ensuring title is included for context.
    
    Args:
        title: Project title
        abstract: Project abstract
        sentences_per_chunk: Number of sentences per chunk
        overlap_sentences: Overlap between chunks
    
    Returns:
        List of text chunks with title context
    """
    from nltk.tokenize import sent_tokenize
    
    # Prepare text
    title = title.strip() if title else ""
    abstract = abstract.strip() if abstract else ""
    
    if not title and not abstract:
        return []
    
    # If only title or very short abstract, return as is
    if not abstract or len(abstract) < 100:
        return [f"{title}\n\n{abstract}".strip()] if title else [abstract]
    
    # For longer abstracts, chunk them but keep title for context
    abstract_sentences = sent_tokenize(abstract)
    
    # If abstract is short enough, return title + abstract as single chunk
    if len(abstract_sentences) <= sentences_per_chunk - 1:  # -1 to account for title
        return [f"{title}\n\n{abstract}".strip()]
    
    # Chunk the abstract and prepend title to each chunk
    chunks = []
    start_idx = 0
    
    while start_idx < len(abstract_sentences):
        end_idx = min(start_idx + sentences_per_chunk, len(abstract_sentences))
        
        # Get sentences for this chunk
        chunk_sentences = abstract_sentences[start_idx:end_idx]
        chunk_abstract = ' '.join(chunk_sentences)
        
        # Prepend title for context
        if title:
            chunk_text = f"{title}\n\n{chunk_abstract}"
        else:
            chunk_text = chunk_abstract
            
        chunks.append(chunk_text.strip())
        
        # Move forward with overlap
        if end_idx >= len(abstract_sentences):
            break
        start_idx = max(start_idx + 1, end_idx - overlap_sentences)
    
    return chunks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to RIS3CAT pickle (DataFrame).")
    ap.add_argument("--index-dir", default=settings.index_dir, help="Directory for vectors and metadata.")
    ap.add_argument("--backend", default=settings.vector_backend, help="Vector backend (WP1: hnsw).")
    ap.add_argument("--batch-size", type=int, default=256, help="Embedding batch size.")
    ap.add_argument("--max-chars", type=int, default=1200, help="Max characters per chunk.")
    ap.add_argument("--overlap", type=int, default=100, help="Overlap characters between chunks.")
    ap.add_argument("--use-sentence-chunking", action="store_true", 
                    help="Use sentence-based chunking instead of character-based.")
    ap.add_argument("--sentences-per-chunk", type=int, default=6,
                    help="Number of sentences per chunk (if using sentence chunking).")
    ap.add_argument("--overlap-sentences", type=int, default=1,
                    help="Number of overlapping sentences between chunks.")
    ap.add_argument("--include-context", action="store_true", help="Append context fields to text.")
    ap.add_argument("--limit", type=int, default=0, help="Only process first N rows (for dry runs).")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input pickle not found: {input_path}")

    df = pd.read_pickle(input_path)
    cols = [c for c in KEEP_COLS if c in df.columns]
    df = df[cols].copy()

    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "vectors.hnsw"
    meta_path = index_dir / "metadata.json"
    sidecar_dir = index_dir.parent / "meta"
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    # Sidecar tables: normalized framework/year + dates; no topic flags
    proj_cols = [
        "projectId", "ecRef", "projectTitle", "projectAbstract",
        "frameworkName", "instrumentName", "startingDate", "startingYear",
        "sdgName", "RIS3CAT Àmbit Sectorial Líder", "RIS3CAT Tecnologia Facilitadora Transversal"
    ]
    proj_df = df[[c for c in proj_cols if c in df.columns]].drop_duplicates(subset=["projectId"]).copy()

    # Normalize framework and year (dtype-safe)
    proj_df["framework_norm"] = proj_df.get("frameworkName", "").map(normalize_framework)

    # Start with numeric startingYear as nullable Int64
    proj_df["year_norm"] = pd.to_numeric(proj_df.get("startingYear", ""), errors="coerce").astype("Int64")

    # If missing, try startingDate -> year (also Int64)
    if "startingDate" in proj_df.columns:
        sd_year = pd.to_datetime(proj_df["startingDate"], errors="coerce").dt.year.astype("Int64")
        fill_mask = proj_df["year_norm"].isna() & sd_year.notna()
        proj_df.loc[fill_mask, "year_norm"] = sd_year[fill_mask]
        # keep original date as string (optional)
        proj_df["startingDate"] = pd.to_datetime(proj_df["startingDate"], errors="coerce").astype("datetime64[ns]").astype(str)

    proj_df.to_parquet(sidecar_dir / "projects.parquet", index=False)

    org_cols = [
        "projectId", "organizationId", "organizationName", "organizationCountry",
        "organizationNuts2Id", "organizationNuts2Name", "organizationNuts3Id", "organizationNuts3Name",
    ]
    org_cols = [c for c in org_cols if c in df.columns]
    org_df = df[org_cols].dropna(subset=["projectId"]).drop_duplicates().copy()
    org_df.to_parquet(sidecar_dir / "project_orgs.parquet", index=False)

    print(f"Saved sidecar tables in {sidecar_dir}")

    # Vector index
    emb = Embedder(model_name=settings.embedder_model_name, device=None)
    store = build_store(
        backend=args.backend,
        index_path=str(index_path),
        meta_path=str(meta_path),
        space=settings.hnsw_space,
        M=settings.hnsw_m,
        ef_construct=settings.hnsw_ef_construct,
        ef_query=settings.hnsw_ef_query,
    )

    if index_path.exists() and meta_path.exists():
        store.load()
        existing_ids = set(store.id_map)  # type: ignore[attr-defined]
    else:
        store.init(dim=emb.get_dim())
        existing_ids = set()

    grouped = df.groupby("projectId", sort=False)

    def first_nonempty(series):
        for v in series:
            s = clean_str(v)
            if s:
                return s
        return ""

    docs: List[Dict[str, Any]] = []
    for pid, g in tqdm(grouped, desc="Preparing project docs"):
        title = first_nonempty(g["projectTitle"]) if "projectTitle" in g else ""
        abstract = first_nonempty(g["projectAbstract"]) if "projectAbstract" in g else ""
        if not title and not abstract:
            continue

        row0 = g.iloc[0]
        meta = {
            "projectId": clean_str(pid),
            "ecRef": clean_str(row0.get("ecRef")),
            "frameworkName": clean_str(row0.get("frameworkName")),
            "instrumentName": clean_str(row0.get("instrumentName")),
            "startingYear": clean_str(row0.get("startingYear")),
            "sdgName": clean_str(row0.get("sdgName")),
            "RIS3CAT_Ambit": clean_str(row0.get("RIS3CAT Àmbit Sectorial Líder")),
            "RIS3CAT_TFT": clean_str(row0.get("RIS3CAT Tecnologia Facilitadora Transversal")),
        }

        parts = [p for p in [title, abstract] if p]
        '''
        if args.include_context:
            ctx_bits = []
            for k in ["frameworkName", "instrumentName", "startingYear", "sdgName",
                      "RIS3CAT Àmbit Sectorial Líder", "RIS3CAT Tecnologia Facilitadora Transversal"]:
                val = clean_str(row0.get(k))
                if val:
                    ctx_bits.append(f"{k}: {val}")
            if ctx_bits:
                parts.append("\n\n" + " | ".join(ctx_bits))
        '''
        if args.include_context:
            ctx_bits = []
            for k in CONTEXT_FIELDS:
                val = clean_str(row0.get(k))
                if val:
                    ctx_bits.append(val)
            if ctx_bits:
                parts.append("Categories: " + ", ".join(ctx_bits))

        full_text = "\n\n".join(parts).strip()
        
        # Chunk text into sentences to get better embeddings.
        if args.use_sentence_chunking:  # Add this argument to argparse
            chunks = chunk_project_text(
                title=title,
                abstract=abstract,
                sentences_per_chunk=args.sentences_per_chunk,  # Default: 6
                overlap_sentences=args.overlap_sentences        # Default: 1
            )
        else:
            # Fallback to character-based chunking
            chunks = chunk_text(full_text, max_chars=args.max_chars, overlap=args.overlap)
        
        for i, ch in enumerate(chunks):
            did = f"{pid}#c{i+1}" if len(chunks) > 1 else str(pid)
            if did in existing_ids:
                continue
            docs.append({"id": did, "text": ch, "metadata": meta})

    if not docs:
        print("Nothing to add (all docs already indexed).")
        return

    B = args.batch_size
    for i in tqdm(range(0, len(docs), B), desc="Indexing"):
        batch = docs[i:i+B]
        texts = [d["text"] for d in batch]
        ids = [d["id"] for d in batch]
        metas = [d["metadata"] for d in batch]
        vecs = emb.encode(texts)
        store.add(vecs, ids, metas)

    store.save()
    print(f"Indexed {len(docs)} documents into {index_dir}")

if __name__ == "__main__":
    main()
