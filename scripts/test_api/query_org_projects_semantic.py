
#!/usr/bin/env python
"""Query-time semantic + metadata intersection for org counts by topic."""
import argparse
from pathlib import Path
import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"Added project root: {PROJECT_ROOT} to the path.")

from src.impulse.settings import settings
from src.impulse.embedding.embedder import Embedder
from src.impulse.vector_store.factory import build_store

def norm(s): return (str(s) if s is not None else "").strip().lower()

def load_store():
    data_dir = Path(settings.index_dir)
    index_path = data_dir / "vectors.hnsw"
    meta_path  = data_dir / "metadata.json"
    emb = Embedder(model_name=settings.embedder_model_name, device=None)
    store = build_store(
        backend=settings.vector_backend,
        index_path=str(index_path),
        meta_path=str(meta_path),
        space=settings.hnsw_space,
        M=settings.hnsw_m,
        ef_construct=settings.hnsw_ef_construct,
        ef_query=settings.hnsw_ef_query,
    )
    store.load()
    return emb, store

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True, help="Organization name (exact, case-insensitive).")
    ap.add_argument("--framework", default="H2020", help="Normalized framework label (e.g., H2020).")
    ap.add_argument("--start", type=int, required=True, help="Start year inclusive")
    ap.add_argument("--end", type=int, required=True, help="End year inclusive")
    ap.add_argument("--topic-query", required=True, help="Semantic topic query text")
    ap.add_argument("--k", type=int, default=5000, help="How many semantic candidates to retrieve")
    ap.add_argument("--min-score", type=float, default=0.0, help="Minimum semantic score to accept (0..1)")
    ap.add_argument("--meta-dir", default="data/meta", help="Directory with projects.parquet and project_orgs.parquet")
    args = ap.parse_args()

    # Metadata filters
    meta_dir = Path(args.meta_dir)
    projects = pd.read_parquet(meta_dir / "projects.parquet")
    project_orgs = pd.read_parquet(meta_dir / "project_orgs.parquet")

    projects["projectId"] = projects["projectId"].astype(str)
    project_orgs["projectId"] = project_orgs["projectId"].astype(str)
    projects["year_norm"] = pd.to_numeric(projects["year_norm"], errors="coerce")

    fw_mask = projects["framework_norm"].map(norm) == norm(args.framework)
    yr_mask = (projects["year_norm"] >= args.start) & (projects["year_norm"] <= args.end)
    proj_meta = projects[fw_mask & yr_mask][["projectId"]].drop_duplicates()

    project_orgs["organizationName_norm"] = project_orgs["organizationName"].map(norm)
    org_hits = project_orgs[project_orgs["organizationName_norm"] == norm(args.org)][["projectId"]].drop_duplicates()

    meta_set = set(proj_meta["projectId"]).intersection(set(org_hits["projectId"]))
    if not meta_set:
        print("Organization:", args.org)
        print("Framework:", args.framework, "| Years:", f"{args.start}-{args.end}")
        print("Topic query:", repr(args.topic_query))
        print("Unique projects: 0")
        return

    # Semantic candidates
    emb, store = load_store()
    qv = emb.encode([args.topic_query])[0]
    hits = store.query(qv, k=args.k)

    def to_project_id(doc_id: str) -> str:
        return doc_id.split("#", 1)[0]

    sem_meta = set()
    for h in hits:
        if h.get("score", 0.0) < args.min_score:
            continue
        pid = to_project_id(h["id"])
        if pid in meta_set:
            sem_meta.add(pid)

    print("Organization:", args.org)
    print("Framework:", args.framework, "| Years:", f"{args.start}-{args.end}")
    print("Topic query:", repr(args.topic_query))
    print("Unique projects:", len(sem_meta))
    for pid in list(sem_meta)[:20]:
        print(pid)

if __name__ == "__main__":
    main()
