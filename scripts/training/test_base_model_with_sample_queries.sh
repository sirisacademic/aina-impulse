#!/bin/bash
# Test base models with built-in sample queries
# Usage: ./test_base_model_with_sample_queries.sh [model_key] [--verbose] [--debug]
# Example: ./test_base_model_with_sample_queries.sh qwen2.5-7b
# Or test all: ./test_base_model_with_sample_queries.sh all
# With verbose: ./test_base_model_with_sample_queries.sh ministral-3b --verbose
# With debug: ./test_base_model_with_sample_queries.sh mistral-7b --debug

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROMPT_FILE="${PROJECT_ROOT}/data/training/salamandra_inference_prompt.txt"

MODEL_KEY=${1:-7b-tools}
VERBOSE=""
DEBUG=""

# Check for --verbose and --debug flags
for arg in "$@"; do
    if [[ "$arg" == "--verbose" ]]; then
        VERBOSE="--verbose"
    fi
    if [[ "$arg" == "--debug" ]]; then
        DEBUG="--debug"
    fi
done

# Wrapper around eval_base_models.sh (no test file = sample queries)
"${SCRIPT_DIR}/eval_base_models.sh" "$MODEL_KEY" $VERBOSE $DEBUG --prompt-file "$PROMPT_FILE"
