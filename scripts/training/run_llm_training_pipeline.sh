#!/bin/bash
# Enhanced training pipeline for IMPULSE Salamandra

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"  # Work from project root

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

###########################################
##                  MODEL                ##
###########################################

# Load model configuration
# Usage: ./run_llm_training_pipeline.sh --model <model_key>
# Example: ./run_llm_training_pipeline.sh --model 7b-tools
# Available models: 7b-tools, qwen2.5-7b, llama-3.1-8b, mistral-7b, gemma-7b,
#                   salamandra-2b, qwen2.5-3b, ministral-3b, phi-4-mini

# Default model (change this or use --model flag)
MODEL_KEY="7b-tools"

# Parse --model argument early
for ((i=1; i<=$#; i++)); do
    if [[ "${!i}" == "--model" ]]; then
        next_i=$((i+1))
        MODEL_KEY="${!next_i}"
        break
    fi
done

# Load model configuration
source "${SCRIPT_DIR}/model_configs.sh"

load_training_config "$MODEL_KEY" || exit 1

###########################################

DATA_DIR="data/training/batches"
PROMPT_FILE="data/training/salamandra_finetuning_prompt.txt"

# Training parameters
EPOCHS=5
LEARNING_RATE=1e-4
BATCH_SIZE=1
GRADIENT_ACCUMULATION=16
MAX_LENGTH=4096
EARLY_STOPPING_PATIENCE=3
SKIP_PERPLEXITY=true
EVAL_ACCUMULATION_STEPS=4

# Validation parameters
TEST_FILE=""
MAX_NEW_TOKENS=1024
TEMPERATURE=0
EVAL_MODEL_PATH=""

# Functions
print_header() {
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}\n"
}

print_error() {
    echo -e "${RED}✗ Error: $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

check_file() {
    if [ ! -f "$1" ]; then
        print_error "File not found: $1"
        exit 1
    fi
}

check_dir() {
    if [ ! -d "$1" ]; then
        print_error "Directory not found: $1"
        exit 1
    fi
}

# Main pipeline
main() {
    print_header "IMPULSE Salamandra Training Pipeline"

    echo "Model configuration: $MODEL_KEY"
    echo "MODEL_NAME=$MODEL_NAME"
    echo "OUTPUT_DIR=$OUTPUT_DIR"
    echo "MERGED_DIR=$MERGED_DIR"
    echo "QUANTIZE=$QUANTIZE"
    echo

    # Parse arguments
    SKIP_PREP=false
    SKIP_TRAIN=false
    SKIP_VALIDATE=false
    SKIP_MERGE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model) shift 2 ;;  # Already handled
            --skip-prep) SKIP_PREP=true; shift ;;
            --skip-train) SKIP_TRAIN=true; shift ;;
            --skip-validate) SKIP_VALIDATE=true; shift ;;
            --skip-merge) SKIP_MERGE=true; shift ;;
            --epochs) EPOCHS="$2"; shift 2 ;;
            --batch-size) BATCH_SIZE="$2"; shift 2 ;;
            --quantize) QUANTIZE="$2"; shift 2 ;;
            --eval-model) EVAL_MODEL_PATH="$2"; shift 2 ;;
            --test-file) TEST_FILE="$2"; shift 2 ;;
            --skip-perplexity) SKIP_PERPLEXITY=true; shift ;;
            --enable-perplexity) SKIP_PERPLEXITY=false; shift ;;
            --eval-accumulation-steps) EVAL_ACCUMULATION_STEPS="$2"; shift 2 ;;
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    done
    
    # Verify GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
        echo "GPU memory available: ${GPU_MEM} MB"
        if [ "$GPU_MEM" -lt 10000 ]; then
            print_error "Low GPU memory. Consider using --quantize 4bit"
        fi
    else
        print_error "No GPU detected"
        exit 1
    fi
    
    # Step 1: Data Preparation
    if [ "$SKIP_PREP" = false ]; then
        print_header "Step 1: Preparing Training Data"
        
        check_dir "$DATA_DIR"
        check_file "$PROMPT_FILE"
        
        python scripts/training/prepare_training_data.py \
            --input-dir "$DATA_DIR" \
            --output-file "$TRAINING_DATA" \
            --prompt-file "$PROMPT_FILE" \
            --validate \
            --shuffle \
            --seed 42
        
        if [ $? -eq 0 ]; then
            print_success "Data preparation complete"
        else
            print_error "Data preparation failed"
            exit 1
        fi
    else
        echo "Skipping data preparation"
        check_file "$TRAINING_DATA"
    fi
    
    # Step 2: Training
    if [ "$SKIP_TRAIN" = false ]; then
        print_header "Step 2: Fine-tuning Model"
        
        check_file "$TRAINING_DATA"
        
        echo "Configuration:"
        echo "  Model: $MODEL_NAME"
        echo "  Epochs: $EPOCHS"
        echo "  Batch size: $BATCH_SIZE"
        echo "  Gradient accumulation: $GRADIENT_ACCUMULATION"
        echo "  Learning rate: $LEARNING_RATE"
        echo "  Quantization: $QUANTIZE"
        echo "  Early stopping patience: $EARLY_STOPPING_PATIENCE"
        echo "  Skip perplexity: $SKIP_PERPLEXITY"
        echo "  Eval accumulation steps: $EVAL_ACCUMULATION_STEPS"
        echo ""
        
        TRAIN_CMD="python scripts/training/train.py \
            --model \"$MODEL_NAME\" \
            --data \"$TRAINING_DATA\" \
            --output-dir \"$OUTPUT_DIR\" \
            --epochs $EPOCHS \
            --batch-size $BATCH_SIZE \
            --gradient-accumulation-steps $GRADIENT_ACCUMULATION \
            --learning-rate $LEARNING_RATE \
            --max-length $MAX_LENGTH \
            --quantize $QUANTIZE \
            --early-stopping-patience $EARLY_STOPPING_PATIENCE"
        
        if [ "$SKIP_PERPLEXITY" = true ]; then
            TRAIN_CMD="$TRAIN_CMD --skip-perplexity"
        fi
        
        if [ -n "$EVAL_ACCUMULATION_STEPS" ]; then
            TRAIN_CMD="$TRAIN_CMD --eval-accumulation-steps $EVAL_ACCUMULATION_STEPS"
        fi
        
        eval $TRAIN_CMD
        
        if [ $? -eq 0 ]; then
            print_success "Training complete"
        else
            print_error "Training failed"
            exit 1
        fi
    else
        echo "Skipping training"
        check_dir "$OUTPUT_DIR"
    fi
    
    # Step 3: Validation
    if [ "$SKIP_VALIDATE" = false ]; then
        print_header "Step 3: Validating Model"
        
        check_dir "$OUTPUT_DIR"
        check_file "$PROMPT_FILE"
        
        MODEL_TO_VALIDATE="${EVAL_MODEL_PATH:-$OUTPUT_DIR}"
        OUTPUT_DIR_RESULTS="results/$MODEL_TO_VALIDATE"
        
        echo "Configuration:"
        echo "  Model path: $MODEL_TO_VALIDATE"
        echo "  Max new tokens: $MAX_NEW_TOKENS"
        echo "  Temperature: $TEMPERATURE"
        echo "  Quantize: $QUANTIZE"
        
        if [ -f "$TEST_FILE" ]; then
            echo "  Test file: $TEST_FILE"
            RESULTS_DIR="$OUTPUT_DIR_RESULTS/test_file"
            TEST_ARGS="--test-file $TEST_FILE"
        else
            echo "  Using default test queries"
            RESULTS_DIR="$OUTPUT_DIR_RESULTS/default_queries"
            TEST_ARGS=""
        fi
        echo ""

        echo "  Results directory: $RESULTS_DIR"
        mkdir -p $RESULTS_DIR
        
        python scripts/training/validate.py \
            --model-path "$MODEL_TO_VALIDATE" \
            --prompt-file "$PROMPT_FILE" \
            --quantize $QUANTIZE \
            --max-new-tokens $MAX_NEW_TOKENS \
            --temperature $TEMPERATURE \
            --verbose \
            --output-dir "$RESULTS_DIR" \
            $TEST_ARGS
        
        if [ $? -eq 0 ]; then
            print_success "Validation complete"
            echo "Results saved to: $RESULTS_DIR"
        else
            print_error "Validation failed"
        fi
    else
        echo "Skipping validation"
    fi
    
    # Step 4: Merge
    if [ "$SKIP_MERGE" = false ]; then
        print_header "Step 4: Merging LoRA Adapters (Optional)"
        
        read -p "Merge LoRA adapters for production? (y/n) " -n 1 -r
        echo
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python scripts/training/merge_lora.py \
                --adapter-path "$OUTPUT_DIR" \
                --output-path "$MERGED_DIR" \
                --force
            
            if [ $? -eq 0 ]; then
                print_success "Model merged successfully"
                echo "Merged model: $MERGED_DIR"
            else
                print_error "Merge failed"
                exit 1
            fi
        fi
        
        # Merge averaged model if exists
        AVERAGED_DIR="$OUTPUT_DIR/averaged_model"
        if [ -d "$AVERAGED_DIR" ]; then
            print_header "Merging Averaged Model"
        
            AVERAGED_MERGED_DIR="${MERGED_DIR}-averaged"
        
            python scripts/training/merge_lora.py \
                --adapter-path "$AVERAGED_DIR" \
                --output-path "$AVERAGED_MERGED_DIR" \
                --force
        
            if [ $? -eq 0 ]; then
                print_success "Averaged model merged: $AVERAGED_MERGED_DIR"
            fi
        fi
    fi
    
    # Summary
    print_header "Pipeline Complete!"
    
    echo "Summary:"
    print_success "Training data: $TRAINING_DATA"
    print_success "Fine-tuned model: $OUTPUT_DIR"
    
    if [ -d "$MERGED_DIR" ]; then
        print_success "Merged model: $MERGED_DIR"
    fi
    
    echo ""
    echo "Next steps:"
    echo "1. Review validation results in $RESULTS_DIR"
    echo "2. Test with custom queries:"
    echo "   python scripts/training/validate.py \\"
    echo "     --model-path $OUTPUT_DIR \\"
    echo "     --prompt-file $PROMPT_FILE"
    echo ""
}

main "$@"
