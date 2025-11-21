import json
import numpy as np
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"Added project root: {PROJECT_ROOT} to the path.")

from src.impulse.query_expansion.loader import load_kb
from src.impulse.query_expansion.expansion import expand_query_with_vectors
from src.impulse.embedding.embedder import Embedder
from src.impulse.settings import settings

_EMBEDDER = None


# CHANGE THIS if needed
KB_PATH = "data/kb/wikidata_kb.jsonl"

def get_embedder() -> Embedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = Embedder(model_name=settings.embedder_model_name, device=None)
    return _EMBEDDER

def pretty(v):
    return json.dumps(v, indent=2, ensure_ascii=False)

def main():
    print("Loading KB...")
    kb = load_kb(KB_PATH)
    print(f"Loaded {len(kb)} KB records")

    emb = get_embedder()

    # DEMO QUERY
    query = "machine learning"
    print(f"\n=== TEST QUERY: '{query}' ===\n")

    result = expand_query_with_vectors(
        query=query,
        kb=kb,
        encoder=emb,
        languages=["en", "es", "ca", "it"]
    )

    # ---- Display Summary ----
    #print("\n=== QUERY VECTOR ===")
    #print(np.array(result["query_vector"]).shape)

    print("\n=== DEFINITION VECTORS ===")
    for d in result["definition_vectors"]:
        print(f"\nLanguage: {d['language']}")
        print(f"Definition: {d['definition']}")
        print(f"Vector shape: {np.array(d['vector']).shape}")

    print("\n=== ALIASES (PER LANGUAGE) ===")
    print(pretty(result["aliases"]))

    # You can also dump full result
    # print("\n=== FULL RESULT ===")
    # print(pretty(result))


if __name__ == "__main__":
    main()
