#!/bin/bash
# Centralized model configurations for IMPULSE training pipeline
# Source this file to access model configs

# Model configurations: base_model|output_dir|merged_dir|training_data|quantize
declare -A TRAINING_CONFIGS=(
    ["7b-tools"]="langtech-innovation/7b-tools-v3|models/impulse-7b-tools-v3-ft|models/impulse-7b-tools-v3-merged|data/training/impulse_training_salamandra.jsonl|4bit"
    ["qwen2.5-7b"]="Qwen/Qwen2.5-7B-Instruct|models/impulse-qwen2.5-7b-instruct-ft|models/impulse-qwen2.5-7b-instruct-merged|data/training/impulse_training_salamandra.jsonl|4bit"
    #["llama-3.1-8b"]="meta-llama/Meta-Llama-3.1-8B-Instruct|models/impulse-meta-llama-3.1-8b-instruct-ft|models/impulse-meta-llama-3.1-8b-instruct-merged|data/training/impulse_training_salamandra.jsonl|4bit"
    ["mistral-7b"]="mistralai/Mistral-7B-Instruct-v0.3|models/impulse-mistral-7b-instruct-v0.3-ft|models/impulse-mistral-7b-instruct-v0.3-merged|data/training/impulse_training_salamandra.jsonl|4bit"
    #["gemma-7b"]="google/gemma-7b-it|models/impulse-gemma-7b-it-ft|models/impulse-gemma-7b-it-merged|data/training/impulse_training_gemma.jsonl|4bit"
    ["salamandra-2b"]="BSC-LT/salamandra-2b-instruct|models/impulse-salamandra-2b-instruct-ft|models/impulse-salamandra-2b-instruct-merged|data/training/impulse_training_salamandra.jsonl|none"
    ["qwen2.5-3b"]="Qwen/Qwen2.5-3B-Instruct|models/impulse-qwen2.5-3b-instruct-ft|models/impulse-qwen2.5-3b-instruct-merged|data/training/impulse_training_salamandra.jsonl|8bit"
    ["ministral-3b"]="ministral/Ministral-3b-instruct|models/impulse-ministral-3b-instruct-ft|models/impulse-ministral-3b-instruct-merged|data/training/impulse_training_salamandra.jsonl|none"
    #["phi-4-mini"]="microsoft/Phi-4-mini-instruct|models/impulse-phi-4-mini-instruct-ft|models/impulse-phi-4-mini-instruct-merged|data/training/impulse_training_salamandra.jsonl|none"
)

# Evaluation configurations: merged_model_path|quantize
declare -A EVAL_CONFIGS=(
    ["7b-tools"]="models/impulse-7b-tools-v3-merged|4bit"
    ["qwen2.5-7b"]="models/impulse-qwen2.5-7b-instruct-merged|4bit"
    #["gemma-7b"]="models/impulse-gemma-7b-it-merged|4bit"
    ["mistral-7b"]="models/impulse-mistral-7b-instruct-v0.3-merged|4bit"
    ["salamandra-2b"]="models/impulse-salamandra-2b-instruct-merged|none"
    ["qwen2.5-3b"]="models/impulse-qwen2.5-3b-instruct-merged|8bit"
    ["ministral-3b"]="models/impulse-ministral-3b-instruct-merged|none"
)

# Load training configuration
load_training_config() {
    local model_key=$1
    
    if [[ -z "${TRAINING_CONFIGS[$model_key]}" ]]; then
        echo "Error: Unknown model key '$model_key'"
        echo "Available: ${!TRAINING_CONFIGS[@]}"
        return 1
    fi
    
    IFS='|' read -r MODEL_NAME OUTPUT_DIR MERGED_DIR TRAINING_DATA QUANTIZE <<< "${TRAINING_CONFIGS[$model_key]}"
    export MODEL_NAME OUTPUT_DIR MERGED_DIR TRAINING_DATA QUANTIZE
    
    echo "Loaded training config: $model_key"
    echo "  MODEL_NAME=$MODEL_NAME"
    echo "  OUTPUT_DIR=$OUTPUT_DIR"
    echo "  MERGED_DIR=$MERGED_DIR"
    echo "  QUANTIZE=$QUANTIZE"
}

# Load evaluation configuration
load_eval_config() {
    local model_key=$1
    
    if [[ -z "${EVAL_CONFIGS[$model_key]}" ]]; then
        echo "Error: Unknown model key '$model_key'"
        echo "Available: ${!EVAL_CONFIGS[@]}"
        return 1
    fi
    
    IFS='|' read -r EVAL_MODEL_PATH EVAL_QUANTIZE <<< "${EVAL_CONFIGS[$model_key]}"
    export EVAL_MODEL_PATH EVAL_QUANTIZE
}

# Get all eval model keys
get_all_eval_keys() {
    echo "${!EVAL_CONFIGS[@]}"
}

# Check if model exists
model_exists() {
    local model_key=$1
    IFS='|' read -r model_path _ <<< "${EVAL_CONFIGS[$model_key]}"
    [[ -d "$model_path" ]]
}


