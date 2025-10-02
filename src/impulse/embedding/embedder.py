
from typing import List, Optional
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return emb

    def get_dim(self) -> int:
        vec = self.encode(["dim probe"])[0]
        return int(vec.shape[0])
