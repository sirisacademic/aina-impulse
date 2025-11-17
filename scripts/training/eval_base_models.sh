#!/bin/bash
# Base model evaluation script (without fine-tuning)
# Usage:
#   ./eval_base_models.sh --list                   # List available base models
#   ./eval_base_models.sh [model_key]              # Single model, sample queries
#   ./eval_base_models.sh [model_key] --test-file <file>
#   ./eval_base_models.sh [model_key] --verbose    # With detailed output
#   ./eval_base_models.sh [model_key] --debug      # Show model responses
#   ./eval_base_models.sh all                      # All models, sample queries
#   ./eval_base_models.sh all --test-file <file>   # All models, test file
#   ./eval_base_models.sh all --verbose --debug    # All models with full debugging

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
PROMPT_FILE="${PROJECT_ROOT}/data/training/salamandra_finetuning_prompt.txt"
EQUIV_DIR="${PROJECT_ROOT}/data/test/equivalences"
TEST_FILE=""
SKIP_MODELS=""
BATCH_MODE=false
VERBOSE_FLAG=""
DEBUG_FLAG=""

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Base model configurations: model_path|quantize
declare -A BASE_MODELS=(
    ["7b-tools"]="langtech-innovation/7b-tools-v3|4bit"
    ["qwen2.5-7b"]="Qwen/Qwen2.5-7B-Instruct|4bit"
    ["mistral-7b"]="mistralai/Mistral-7B-Instruct-v0.3|4bit"
    ["salamandra-2b"]="BSC-LT/salamandra-2b-instruct|none"
    ["qwen2.5-3b"]="Qwen/Qwen2.5-3B-Instruct|8bit"
    ["ministral-3b"]="ministral/Ministral-3b-instruct|none"
)

# Load eval config for base models
load_eval_config() {
    local model_key=$1
    
    if [[ -z "${BASE_MODELS[$model_key]}" ]]; then
        echo "Error: Unknown model key '$model_key'"
        echo "Available: ${!BASE_MODELS[@]}"
        return 1
    fi
    
    IFS='|' read -r EVAL_MODEL_PATH EVAL_QUANTIZE <<< "${BASE_MODELS[$model_key]}"
    export EVAL_MODEL_PATH EVAL_QUANTIZE
}

# Get all eval model keys
get_all_eval_keys() {
    echo "${!BASE_MODELS[@]}"
}

# Parse arguments
MODEL_KEY=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --test-file) TEST_FILE="$2"; shift 2 ;;
        --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
        --skip) SKIP_MODELS="$SKIP_MODELS $2"; shift 2 ;;
        --verbose) VERBOSE_FLAG="--verbose"; shift ;;
        --debug) DEBUG_FLAG="--debug"; shift ;;
        --list)
            echo "Available base model keys:"
            for key in $(get_all_eval_keys | tr ' ' '\n' | sort); do
                load_eval_config "$key" 2>/dev/null
                echo "  ✓ $key - $EVAL_MODEL_PATH (quantize: $EVAL_QUANTIZE)"
            done
            exit 0
            ;;
        *) 
            if [[ -z "$MODEL_KEY" ]]; then
                MODEL_KEY="$1"
            fi
            shift
            ;;
    esac
done

MODEL_KEY=${MODEL_KEY:-7b-tools}

# Determine output base directory based on test mode
if [[ -n "$TEST_FILE" ]]; then
    RESULTS_BASE="${PROJECT_ROOT}/results/base_models/test_file"
else
    RESULTS_BASE="${PROJECT_ROOT}/results/base_models/sample_queries"
fi

# Determine if running in batch mode
if [[ "$MODEL_KEY" == "all" ]]; then
    BATCH_MODE=true
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    SUMMARY_FILE="${RESULTS_BASE}/batch_summary_${TIMESTAMP}.txt"
    REPORT_FILE="${RESULTS_BASE}/comparative_report_${TIMESTAMP}.txt"
    
    mkdir -p "$RESULTS_BASE"
    
    echo "IMPULSE Base Model Evaluation - $(date)" > "$SUMMARY_FILE"
    echo "========================================" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    
    TOTAL_MODELS=0
    SUCCESSFUL_EVALS=0
    FAILED_EVALS=0
    RESULT_DIRS=()
fi

test_model() {
    local key=$1
    
    load_eval_config "$key" || return 1
    
    # Determine output base directory based on test mode
    if [[ -n "$TEST_FILE" ]]; then
        RESULTS_BASE="${PROJECT_ROOT}/results/base_models/test_file"
    else
        RESULTS_BASE="${PROJECT_ROOT}/results/base_models/sample_queries"
    fi
    
    # Extract model name from path (remove organization prefix)
    MODEL_NAME=$(basename "$EVAL_MODEL_PATH")
    
    # Same structure for single and batch mode
    OUTPUT_DIR="${RESULTS_BASE}/${MODEL_NAME}/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$OUTPUT_DIR"
    
    if [[ "$BATCH_MODE" == true ]]; then
        print_header "Evaluating: $key"
        TOTAL_MODELS=$((TOTAL_MODELS + 1))
        RESULT_DIRS+=("$OUTPUT_DIR")
    else
        echo ""
        echo "Testing $key..."
    fi
    
    echo "  Model: $EVAL_MODEL_PATH"
    echo "  Quantization: $EVAL_QUANTIZE"
    if [[ -n "$TEST_FILE" ]]; then
        echo "  Test file: $TEST_FILE"
    else
        echo "  Test mode: Sample queries"
    fi
    echo "  Output: $OUTPUT_DIR"
    echo ""
    
    # Build command
    EVAL_CMD="python ${PROJECT_ROOT}/scripts/training/validate_base_model.py \
        $EVAL_MODEL_PATH \
        --prompt-file $PROMPT_FILE \
        --quantize $EVAL_QUANTIZE \
        --max-new-tokens 1024 \
        --temperature 0 \
        --output-dir $OUTPUT_DIR \
        --equiv-dir $EQUIV_DIR \
        $VERBOSE_FLAG \
        $DEBUG_FLAG"
    
    if [[ -f "$TEST_FILE" ]]; then
        EVAL_CMD="$EVAL_CMD --test-file $TEST_FILE"
    fi
    
    # Run evaluation
    START_TIME=$(date +%s)
    
    if eval $EVAL_CMD; then
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        
        print_success "Evaluation complete (${ELAPSED}s)"
        
        if [[ "$BATCH_MODE" == true ]]; then
            SUCCESSFUL_EVALS=$((SUCCESSFUL_EVALS + 1))
            
            # Extract accuracy
            STATS_FILE=$(ls -t "$OUTPUT_DIR"/validation_stats_*.json 2>/dev/null | head -n 1)
            if [[ -f "$STATS_FILE" ]]; then
                ACCURACY=$(python3 -c "import json; data=json.load(open('$STATS_FILE')); print(f\"{data['all_correct']}/{data['total']} ({data['all_correct']/data['total']*100:.1f}%)\")" 2>/dev/null || echo "N/A")
                echo "[$key] SUCCESS - Accuracy: $ACCURACY - Time: ${ELAPSED}s - Output: $OUTPUT_DIR" >> "$SUMMARY_FILE"
            else
                echo "[$key] SUCCESS - Time: ${ELAPSED}s - Output: $OUTPUT_DIR" >> "$SUMMARY_FILE"
            fi
        fi
    else
        print_error "Evaluation failed"
        if [[ "$BATCH_MODE" == true ]]; then
            FAILED_EVALS=$((FAILED_EVALS + 1))
            echo "[$key] FAILED - Output: $OUTPUT_DIR" >> "$SUMMARY_FILE"
        fi
    fi
    
    if [[ "$BATCH_MODE" == true ]]; then
        echo "" >> "$SUMMARY_FILE"
    fi
}

# Main execution
if [[ "$MODEL_KEY" == "all" ]]; then
    print_header "IMPULSE Base Model Batch Evaluation"
    echo "Test mode: $([ -n "$TEST_FILE" ] && echo "$TEST_FILE" || echo "Sample queries")"
    echo "Summary: $SUMMARY_FILE"
    echo ""
    
    for key in $(get_all_eval_keys | tr ' ' '\n' | sort); do
        if [[ -n "$SKIP_MODELS" ]] && [[ "$SKIP_MODELS" =~ "$key" ]]; then
            print_warning "Skipping $key (--skip)"
            continue
        fi
        test_model "$key"
    done
    
    # Print summary
    print_header "Evaluation Summary"
    echo "Total: $TOTAL_MODELS | Success: $SUCCESSFUL_EVALS | Failed: $FAILED_EVALS"
    echo "Summary: $SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    echo "========================================" >> "$SUMMARY_FILE"
    echo "Total: $TOTAL_MODELS | Success: $SUCCESSFUL_EVALS | Failed: $FAILED_EVALS" >> "$SUMMARY_FILE"
    
    # Generate comparative report
    if command -v python3 &> /dev/null && [[ $SUCCESSFUL_EVALS -gt 0 ]]; then
        print_header "Generating Comparative Report"
        
        python3 ${PROJECT_ROOT}/scripts/training/generate_comparative_report.py \
            "${RESULT_DIRS[@]}" \
            --output "$REPORT_FILE"
        
        if [[ $? -eq 0 ]]; then
            echo "Report: $REPORT_FILE"
        else
            print_error "Report generation failed"
        fi
    fi
    
    print_success "Done"
else
    # Single model mode
    test_model "$MODEL_KEY"
fi
