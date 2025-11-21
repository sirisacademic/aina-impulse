import json
from pathlib import Path

def load_kb(path: str):
    kb = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"KB not found at: {path}")

    with p.open("r", encoding="utf-8") as fin:
        for i, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                kb.append(obj)
            except json.JSONDecodeError as e:
                print("\n❌ JSON ERROR in KB file")
                print(f"   → File: {path}")
                print(f"   → Line number: {i}")
                print(f"   → Line content: {line[:500]}")
                print(f"   → Error: {e}\n")
                raise e

    return kb