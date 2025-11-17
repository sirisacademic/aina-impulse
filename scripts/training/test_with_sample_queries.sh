#!/bin/bash
# Test models with built-in sample queries
# Usage: ./test_with_sample_queries.sh [model_key]
# Example: ./test_with_sample_queries.sh ministral-3b
# Or test all: ./test_with_sample_queries.sh all

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROMPT_FILE="${PROJECT_ROOT}/data/training/salamandra_inference_prompt.txt"

MODEL_KEY=${1:-7b-tools}

"${SCRIPT_DIR}/eval_models.sh" "$MODEL_KEY1" --prompt-file "$PROMPT_FILE"

