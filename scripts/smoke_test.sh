#!/usr/bin/env bash
set -euo pipefail

# --- config you can tweak ---
PORT=${PORT:-8010}
PICKLE=${PICKLE:-data/R3C_data.pkl}
INDEX_DIR=${INDEX_DIR:-data/index}
K=${K:-5}
K_FACTOR=${K_FACTOR:-5}
ORG=${ORG:-"APOLLON SOLAR"}
FRAMEWORK=${FRAMEWORK:-H2020}
Y1=${Y1:-2016}
Y2=${Y2:-2020}
TOPIC_QUERY=${TOPIC_QUERY:-"sustainable energy and hydrogen storage"}
# ----------------------------

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Ensure imports find src/*
export PYTHONPATH="${PYTHONPATH:-$ROOT}"

echo "== Smoke test from: $ROOT =="
echo "Python: $(python -V)"
echo "Using pickle: $PICKLE"
echo

# 0) Quick checks
command -v curl >/dev/null 2>&1 || { echo "ERROR: curl not found"; exit 2; }

if [ ! -f "$PICKLE" ]; then
  echo "ERROR: pickle not found at $PICKLE"
  exit 2
fi

# 1) Build a small index if missing
if [ ! -f "$INDEX_DIR/metadata.json" ] || [ ! -f "$INDEX_DIR/vectors.hnsw" ]; then
  echo "== Building small index (first 1000 rows) =="
  python scripts/build_index_from_pickle.py \
    --input "$PICKLE" \
    --index-dir "$INDEX_DIR" \
    --backend hnsw \
    --batch-size 256 \
    --include-context \
    --limit 1000
else
  echo "== Index already present; skipping build =="
fi

# 2) Start API on a safer port (8010) and wait for /health
echo "== Starting API on port $PORT =="
python run_api.py --port "$PORT" --scan-limit 0 >/tmp/impulse_api.log 2>&1 &
API_PID=$!
trap 'echo "Stopping API (pid $API_PID)"; kill $API_PID 2>/dev/null || true' EXIT

echo -n "Waiting for /health "
for i in {1..40}; do
  if curl -s "http://127.0.0.1:${PORT}/health" >/dev/null; then
    echo "ok"
    break
  fi
  echo -n "."
  sleep 0.25
done

# Health payload (optional pretty with jq)
echo "== /health =="
if command -v jq >/dev/null 2>&1; then
  curl -s "http://127.0.0.1:${PORT}/health" | jq .
else
  curl -s "http://127.0.0.1:${PORT}/health"
fi
echo

# 3) Run analytics helpers (metadata + semantic)
echo "== query_org_projects_semantic.py =="
python scripts/query_org_projects_semantic.py \
  --org "$ORG" \
  --framework "$FRAMEWORK" \
  --start "$Y1" --end "$Y2" \
  --topic-query "$TOPIC_QUERY" \
  --k 5000 --min-score 0.55 || true
echo

echo "== query_org_projects.py (metadata only) =="
python scripts/query_org_projects.py \
  --org "$ORG" \
  --framework "$FRAMEWORK" \
  --start "$Y1" --end "$Y2" || true
echo

# 4) Query the running API (semantic + metadata)
echo "== query_api_search.py (hits API) =="
python scripts/query_api_search.py \
  --host "http://127.0.0.1" \
  --port "$PORT" \
  --query "hydrogen storage for residential buildings" \
  --k "$K" \
  --k-factor "$K_FACTOR" \
  --framework "$FRAMEWORK" \
  --year-from "$Y1" \
  --year-to "$Y2" \
  --pretty || true
echo

echo "== Smoke test complete =="

