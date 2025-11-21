#!/bin/bash
# Build IMPULSE vector index from RIS3CAT parquet files

set -euo pipefail

# Detect project root (works from project root or scripts/build_indices)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$SCRIPT_DIR" == */scripts/build_indices ]]; then
    # Running from scripts/build_indices
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
elif [[ -d "$SCRIPT_DIR/scripts/build_indices" ]]; then
    # Running from project root
    PROJECT_ROOT="$SCRIPT_DIR"
else
    echo "Error: Cannot determine project root"
    exit 1
fi

cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run index building with configurable filenames
python scripts/build_indices/build_index.py \
    --data-dir data/ris3cat \
    --projects-file project_db.parquet \
    --participants-file participant_db.parquet \
    --index-dir data/index \
    --meta-dir data/meta \
    --use-sentence-chunking \
    --sentences-per-chunk 6 \
    --overlap-sentences 1 \
    --batch-size 32 \
    "$@"
