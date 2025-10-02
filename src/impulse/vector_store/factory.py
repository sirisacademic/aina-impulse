
from __future__ import annotations
from typing import Any
from src.impulse.vector_store.hnsw_store import HnswStore

def build_store(backend: str, **kwargs: Any):
    b = backend.lower().strip()
    if b == "hnsw":
        return HnswStore(**kwargs)
    raise ValueError(f"Unknown vector backend: {backend}")
