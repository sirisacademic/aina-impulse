#!/bin/bash
# Train with specific model configuration
# Usage: ./train.sh [model_key] [additional_args]
# Example: ./train.sh qwen2.5-7b
# Example: ./train.sh gemma-7b --epochs 3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"  # Work from project root

MODEL_KEY=${1:-7b-tools}
shift  # Remove model_key from args

"${SCRIPT_DIR}/run_llm_training_pipeline.sh" \
    --model "$MODEL_KEY" \
    --skip-prep \
    --skip-validate \
    "$@"
