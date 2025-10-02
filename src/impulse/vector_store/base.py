
from __future__ import annotations
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import numpy as np

class VectorStore(ABC):
    @abstractmethod
    def init(self, dim: int) -> None: ...
    @abstractmethod
    def add(self, vectors: np.ndarray, ids: List[str], metas: Optional[List[Dict[str, Any]]] = None) -> None: ...
    @abstractmethod
    def query(self, query_vec: np.ndarray, k: int = 5) -> List[Dict[str, Any]]: ...
    @abstractmethod
    def save(self) -> None: ...
    @abstractmethod
    def load(self) -> None: ...
