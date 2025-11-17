#!/usr/bin/env bash
set -euo pipefail

# Always work from project root (two levels up from this script)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Create venv if missing
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  echo "Created virtual environment .venv"
else
  echo "Using existing virtual environment .venv"
fi

# Activate venv
source .venv/bin/activate

# Install/upgrade dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"

# Prepare data dirs and .env
mkdir -p data/index
[ -f .env ] || cp .env.example .env

echo
echo "Setup complete. Next steps:"
echo "  source .venv/bin/activate"
echo "  python scripts/build_indices/build_index_from_pickle.py --input data/R3C_data.pkl --index-dir data/index"
