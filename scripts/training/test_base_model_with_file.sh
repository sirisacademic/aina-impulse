#!/bin/bash
# Test base models with specific test file
# Usage: ./test_base_model_with_file.sh [model_key] [test_file] [--verbose] [--debug]
# Example: ./test_base_model_with_file.sh qwen2.5-7b data/test/custom.json
# Or test all: ./test_base_model_with_file.sh all data/test/impulse_test.json
# With verbose: ./test_base_model_with_file.sh ministral-3b data/test/impulse_test.json --verbose
# With debug: ./test_base_model_with_file.sh mistral-7b data/test/impulse_test.json --debug

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROMPT_FILE="${PROJECT_ROOT}/data/training/salamandra_inference_prompt.txt"

TEST_FILE=${2:-data/test/impulse_test.json}
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

if [[ ! -f "$TEST_FILE" ]]; then
    echo "Error: Test file not found: $TEST_FILE"
    exit 1
fi

# Wrapper around eval_base_models.sh
"${SCRIPT_DIR}/eval_base_models.sh" "$1" --test-file "$TEST_FILE" $VERBOSE $DEBUG --prompt-file "$PROMPT_FILE"
