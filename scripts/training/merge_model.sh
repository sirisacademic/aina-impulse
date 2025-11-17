#!/bin/bash
# Merge LoRA adapters for specific model
# Usage: ./merge_model.sh [model_key]
# Example: ./merge_model.sh gemma-7b

MODEL_KEY=${1:-7b-tools}

./run_llm_training_pipeline.sh \
    --model "$MODEL_KEY" \
    --skip-prep \
    --skip-train \
    --skip-validate
