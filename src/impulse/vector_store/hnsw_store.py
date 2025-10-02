
from __future__ import annotations
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
import numpy as np
import hnswlib
from src.impulse.vector_store.base import VectorStore

class HnswStore(VectorStore):
    def __init__(
        self,
        index_path: str,
        meta_path: str,
        space: str = "cosine",
        M: int = 16,
        ef_construct: int = 200,
        ef_query: int = 50,
    ):
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.space = space
        self.M = M
        self.ef_construct = ef_construct
        self.ef_query = ef_query

        self.index: Optional[hnswlib.Index] = None
        self.dim: Optional[int] = None
        self.id_map: List[str] = []
        self.meta_map: Dict[str, Dict[str, Any]] = {}

    def init(self, dim: int):
        self.dim = dim
        self.index = hnswlib.Index(space=self.space, dim=dim)
        self.index.init_index(max_elements=1, ef_construction=self.ef_construct, M=self.M)
        self.index.set_ef(self.ef_query)

    def _ensure_capacity(self, needed: int):
        assert self.index is not None
        cur = self.index.get_max_elements()
        if needed > cur:
            self.index.resize_index(needed)

    def add(self, vectors: np.ndarray, ids: List[str], metas: Optional[List[Dict[str, Any]]] = None):
        assert self.index is not None and self.dim is not None
        assert vectors.ndim == 2 and vectors.shape[1] == self.dim
        if metas is None:
            metas = [{} for _ in ids]

        start = len(self.id_map)
        new_total = start + len(ids)
        self._ensure_capacity(new_total)

        labels = np.arange(start, new_total)
        self.index.add_items(vectors, labels)

        for i, doc_id in enumerate(ids):
            self.id_map.append(doc_id)
            self.meta_map[doc_id] = metas[i] if metas[i] is not None else {}

    def query(self, query_vec: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        assert self.index is not None
        labels, distances = self.index.knn_query(query_vec, k=k)
        labels = labels[0]
        distances = distances[0]

        out = []
        for lab, dist in zip(labels, distances):
            if lab < 0 or lab >= len(self.id_map):
                continue
            doc_id = self.id_map[int(lab)]
            score = 1.0 - float(dist) if self.space == "cosine" else -float(dist)
            out.append({
                "id": doc_id,
                "score": score,
                "metadata": self.meta_map.get(doc_id, {})
            })
        return out

    def save(self):
        assert self.index is not None
        self.index.save_index(str(self.index_path))
        payload = {
            "dim": self.dim,
            "ids": self.id_map,
            "metadata": self.meta_map,
            "space": self.space,
            "M": self.M,
            "ef_construct": self.ef_construct,
            "ef_query": self.ef_query,
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def load(self):
        with open(self.meta_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.dim = int(payload["dim"])
        self.id_map = list(payload["ids"])
        self.meta_map = dict(payload["metadata"])
        self.space = payload.get("space", "cosine")
        self.M = int(payload.get("M", 16))
        self.ef_construct = int(payload.get("ef_construct", 200))
        self.ef_query = int(payload.get("ef_query", 50))

        # Create index object but do NOT init_index() before load
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        max_elems = max(1, len(self.id_map))
        # Directly load; this allocates correctly
        self.index.load_index(str(self.index_path), max_elements=max_elems)
        # Set query ef after loading
        self.index.set_ef(self.ef_query)

