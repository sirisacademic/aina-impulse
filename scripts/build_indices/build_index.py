#!/usr/bin/env python
"""
Build IMPULSE vector index from RIS3CAT parquet files.

Updated for new data structure:
- Reads from data/ris3cat/project_db.parquet + participant_db.parquet
- Applies geographic normalization (region, province)
- Saves normalized metadata to data/meta/ for API use
- Creates vector index in data/index/
"""
import argparse
import re
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

# Path bootstrap
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2] if '__file__' in globals() else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"PROJECT_ROOT={PROJECT_ROOT}")

from src.impulse.settings import settings
from src.impulse.embedding.embedder import Embedder
from src.impulse.vector_store.factory import build_store
from src.impulse.normalization import normalize_region, normalize_province

# Context fields to include in embeddings (if requested)
CONTEXT_FIELDS = ["framework_name", "instrument_name"]


def clean_str(x: Any) -> str:
    """Clean string value, handling NaN/None"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()


def to_year(y) -> int:
    """Extract year from various formats"""
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
    """Split text into overlapping chunks"""
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        slice_ = text[start:end]
        
        # Try to break at sentence boundary
        last_period = slice_.rfind(". ")
        if last_period > max_chars // 2:
            end = start + last_period + 1
            slice_ = text[start:end]
        
        chunks.append(slice_.strip())
        start = max(start + 1, end - overlap)
    
    return chunks


def chunk_project_text(
    title: str,
    abstract: str,
    sentences_per_chunk: int = 6,
    overlap_sentences: int = 1
) -> List[str]:
    """Chunk text by sentences with overlap"""
    from nltk.tokenize import sent_tokenize
    
    # Tokenize abstract into sentences
    sentences = sent_tokenize(abstract) if abstract else []
    
    if not sentences:
        return [title] if title else []
    
    # If few sentences, return title + abstract
    if len(sentences) <= sentences_per_chunk:
        parts = [title, abstract] if title else [abstract]
        return ["\n\n".join(parts)]
    
    # Create overlapping chunks
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk - overlap_sentences):
        chunk_sentences = sentences[i:i + sentences_per_chunk]
        chunk_text = " ".join(chunk_sentences)
        
        # Add title to first chunk
        if i == 0 and title:
            chunk_text = f"{title}\n\n{chunk_text}"
        
        chunks.append(chunk_text.strip())
        
        if i + sentences_per_chunk >= len(sentences):
            break
    
    return chunks


def load_and_process_data(
    data_dir: Path,
    projects_file: str = "project_db.parquet",
    participants_file: str = "participant_db.parquet"
) -> pd.DataFrame:
    """
    Load and process RIS3CAT data from parquet files.
    
    Returns DataFrame with one row per project with normalized columns.
    """
    print(f"Loading data from {data_dir}...")
    
    # Load projects
    projects = pd.read_parquet(data_dir / projects_file)
    print(f"  Loaded {len(projects)} project records from {projects_file}")
    
    # Load participants
    participants = pd.read_parquet(data_dir / participants_file)
    print(f"  Loaded {len(participants)} participant records from {participants_file}")
    
    # Apply geographic normalization to participants
    print("  Normalizing geographic data...")
    if 'region' in participants.columns:
        participants['region_norm'] = participants['region'].apply(
            lambda x: normalize_region(x) if pd.notna(x) else ""
        )
    
    if 'province' in participants.columns:
        participants['province_norm'] = participants['province'].apply(
            lambda x: normalize_province(x) if pd.notna(x) else ""
        )
    
    # For indexing, we only need project-level data
    # We'll save participant data separately for API use
    
    # Add year as numeric
    projects['year_numeric'] = projects['year'].apply(to_year)
    
    # Deduplicate projects (in case of any duplicates)
    projects_unique = projects.drop_duplicates(subset=['project_id'], keep='first')
    
    if len(projects_unique) < len(projects):
        print(f"  Removed {len(projects) - len(projects_unique)} duplicate projects")
    
    print(f"  Final: {len(projects_unique)} unique projects")
    
    return projects_unique, participants


def save_metadata(projects: pd.DataFrame, participants: pd.DataFrame, output_dir: Path):
    """Save processed metadata for API use"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving metadata to {output_dir}...")
    
    # Save projects
    projects.to_parquet(output_dir / "projects.parquet", index=False)
    print(f"  Saved projects metadata")
    
    # Save participants with normalized columns
    participants.to_parquet(output_dir / "participants.parquet", index=False)
    print(f"  Saved participants metadata")


def build_index(
    projects: pd.DataFrame,
    index_dir: Path,
    include_context: bool = False,
    use_sentence_chunking: bool = True,
    sentences_per_chunk: int = 6,
    overlap_sentences: int = 1,
    max_chars: int = 1200,
    overlap_chars: int = 100,
    batch_size: int = 32
):
    """Build vector index from projects"""
    
    print(f"Building index in {index_dir}...")
    index_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize embedder and store
    emb = Embedder(model_name=settings.embedder_model_name, device=None)
    
    index_path = index_dir / "vectors.hnsw"
    meta_path = index_dir / "metadata.json"
    
    store = build_store(
        backend=settings.vector_backend,
        index_path=str(index_path),
        meta_path=str(meta_path),
        space=settings.hnsw_space,
        M=settings.hnsw_m,
        ef_construct=settings.hnsw_ef_construct,
        ef_query=settings.hnsw_ef_query,
    )
    
    # Load existing index if present
    existing_ids = set()
    if index_path.exists() and meta_path.exists():
        print("  Loading existing index...")
        store.load()
        # id_map is a list, not a dict
        existing_ids = set(getattr(store, "id_map", []))
        print(f"  Found {len(existing_ids)} existing documents")
    else:
        store.init(dim=emb.get_dim())
    
    # Process projects
    docs = []
    
    for _, row in tqdm(projects.iterrows(), total=len(projects), desc="Processing projects"):
        project_id = clean_str(row['project_id'])
        if not project_id:
            continue
        
        title = clean_str(row.get('title', ''))
        abstract = clean_str(row.get('abstract', ''))
        
        if not title and not abstract:
            continue
        
        # Build metadata for this project
        meta = {
            "project_id": project_id,
            "ec_ref": clean_str(row.get('ec_ref', '')),
            "framework_name": clean_str(row.get('framework_name', '')),
            "instrument_name": clean_str(row.get('instrument_name', '')),
            "year": clean_str(row.get('year', '')),
            "year_numeric": row.get('year_numeric'),
        }
        
        # Build text for embedding
        parts = [p for p in [title, abstract] if p]
        
        if include_context:
            ctx_bits = []
            for field in CONTEXT_FIELDS:
                val = clean_str(row.get(field, ''))
                if val:
                    ctx_bits.append(val)
            if ctx_bits:
                parts.append("Categories: " + ", ".join(ctx_bits))
        
        full_text = "\n\n".join(parts).strip()
        
        # Chunk text
        if use_sentence_chunking and abstract:
            chunks = chunk_project_text(
                title=title,
                abstract=abstract,
                sentences_per_chunk=sentences_per_chunk,
                overlap_sentences=overlap_sentences
            )
        else:
            chunks = chunk_text(full_text, max_chars=max_chars, overlap=overlap_chars)
        
        # Create documents
        for i, chunk in enumerate(chunks):
            doc_id = f"{project_id}#c{i+1}" if len(chunks) > 1 else str(project_id)
            
            if doc_id in existing_ids:
                continue
            
            docs.append({
                "id": doc_id,
                "text": chunk,
                "metadata": meta
            })
    
    if not docs:
        print("  No new documents to index")
        return
    
    print(f"  Indexing {len(docs)} documents...")
    
    # Add documents in batches
    for i in tqdm(range(0, len(docs), batch_size), desc="Adding to index"):
        batch = docs[i:i+batch_size]
        texts = [d["text"] for d in batch]
        ids = [d["id"] for d in batch]
        metas = [d["metadata"] for d in batch]
        
        vecs = emb.encode(texts)
        store.add(vecs, ids, metas)
    
    # Save index
    print("  Saving index...")
    store.save()
    print(f"  âœ“ Indexed {len(docs)} documents")


def main():
    parser = argparse.ArgumentParser(description="Build IMPULSE vector index")
    
    # Input/output paths
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/ris3cat"),
        help="Directory with parquet files"
    )
    parser.add_argument(
        "--projects-file",
        type=str,
        default="project_db.parquet",
        help="Projects parquet filename (default: project_db.parquet)"
    )
    parser.add_argument(
        "--participants-file",
        type=str,
        default="participant_db.parquet",
        help="Participants parquet filename (default: participant_db.parquet)"
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/index"),
        help="Output directory for vector index"
    )
    parser.add_argument(
        "--meta-dir",
        type=Path,
        default=Path("data/meta"),
        help="Output directory for processed metadata"
    )
    
    # Indexing options
    parser.add_argument(
        "--include-context",
        action="store_true",
        help="Include framework/instrument as context in embeddings"
    )
    parser.add_argument(
        "--use-sentence-chunking",
        action="store_true",
        default=True,
        help="Use sentence-based chunking (default: True)"
    )
    parser.add_argument(
        "--sentences-per-chunk",
        type=int,
        default=6,
        help="Sentences per chunk (default: 6)"
    )
    parser.add_argument(
        "--overlap-sentences",
        type=int,
        default=1,
        help="Sentence overlap between chunks (default: 1)"
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1200,
        help="Max characters per chunk if not using sentence chunking"
    )
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=100,
        help="Character overlap between chunks"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding (default: 32)"
    )
    
    args = parser.parse_args()
    
    # Load and process data
    projects, participants = load_and_process_data(
        args.data_dir,
        args.projects_file,
        args.participants_file
    )
    
    # Save metadata for API
    save_metadata(projects, participants, args.meta_dir)
    
    # Build vector index
    build_index(
        projects=projects,
        index_dir=args.index_dir,
        include_context=args.include_context,
        use_sentence_chunking=args.use_sentence_chunking,
        sentences_per_chunk=args.sentences_per_chunk,
        overlap_sentences=args.overlap_sentences,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
        batch_size=args.batch_size
    )
    
    print("\nâœ“ Index building complete!")
    print(f"  Index: {args.index_dir}")
    print(f"  Metadata: {args.meta_dir}")


if __name__ == "__main__":
    main()
