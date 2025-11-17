# For BSC/Salamandra
python scripts/training/prepare_training_data.py \
    --input-dir data/training/batches \
    --output-file data/training/impulse_training_salamandra.jsonl \
    --prompt-file data/training/salamandra_finetuning_prompt.txt \
    --model BSC-LT/salamandra-7b-instruct

# For Gemma
python scripts/training/prepare_training_data.py \
    --input-dir data/training/batches \
    --output-file data/training/impulse_training_gemma.jsonl \
    --prompt-file data/training/salamandra_finetuning_prompt.txt \
    --model google/gemma-7b-it
    
# Inspect output
echo "Sample Salamandra"
head -n 1 data/training/impulse_training_salamandra.jsonl | jq '.messages[].role'
echo
echo "Sample Gemma"
head -n 1 data/training/impulse_training_gemma.jsonl | jq '.messages[].role'
