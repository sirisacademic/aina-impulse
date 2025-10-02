#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Create venv if missing
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  echo "Created virtual environment .venv"
else
  echo "Using existing virtual environment .venv"
fi

# Activate venv
# shellcheck disable=SC1091
source .venv/bin/activate

# Install deps
python -m pip install --upgrade pip
pip install -r requirements.txt

# Prepare data dirs and .env
mkdir -p data/index
[ -f .env ] || cp .env.example .env

echo
echo "Setup complete. Next steps:"
echo "  source .venv/bin/activate"
echo "  python run_api.py"

