# Complete Script Reference

## All Evaluation Scripts Overview

### Fine-Tuned Model Evaluation

| Script | Purpose | Usage |
|--------|---------|-------|
| **eval_models.sh** | Main evaluation script | `./eval_models.sh all` |
| **validate.py** | Core validation logic | Called by eval_models.sh |
| **test_with_file.sh** | Wrapper for test file | `./test_with_file.sh qwen2.5-7b data/test/test.json` |
| **test_with_sample_queries.sh** | Wrapper for samples | `./test_with_sample_queries.sh qwen2.5-7b` |

### Base Model Evaluation

| Script | Purpose | Usage |
|--------|---------|-------|
| **eval_base_models.sh** | Main evaluation script | `./eval_base_models.sh all` |
| **validate_base_model.py** | Core validation logic | Called by eval_base_models.sh |
| **test_base_model_with_file.sh** | Wrapper for test file | `./test_base_model_with_file.sh qwen2.5-7b data/test/test.json` |
| **test_base_model_with_sample_queries.sh** | Wrapper for samples | `./test_base_model_with_sample_queries.sh qwen2.5-7b` |

### Supporting Scripts

| Script | Purpose |
|--------|---------|
| **generate_comparative_report.py** | Generate comparison reports |
| **model_configs.sh** | Fine-tuned model configurations |
| **model_config.py** | Python model configurations |

## Quick Start Examples

### Fine-Tuned Models

```bash
# List models
./eval_models.sh --list

# Single model
./eval_models.sh qwen2.5-7b

# All models with test file
./eval_models.sh all --test-file data/test/impulse_test.json

# Using wrapper
./test_with_file.sh all data/test/impulse_test.json
```

### Base Models

```bash
# List models
./eval_base_models.sh --list

# Single model
./eval_base_models.sh qwen2.5-7b

# All models with test file
./eval_base_models.sh all --test-file data/test/impulse_test.json

# Using wrapper
./test_base_model_with_file.sh all data/test/impulse_test.json
```

## Complete Comparison Workflow

```bash
# Step 1: Evaluate base models
./eval_base_models.sh all --test-file data/test/impulse_test.json

# Step 2: Evaluate fine-tuned models
./eval_models.sh all --test-file data/test/impulse_test.json

# Step 3: Compare reports
echo "=== BASE MODELS ==="
cat results/base_models/test_file/batch_summary_*.txt

echo -e "\n=== FINE-TUNED MODELS ==="
cat results/test_file/batch_summary_*.txt
```

## Directory Structure

```
scripts/training/
├── Fine-Tuned Evaluation
│   ├── eval_models.sh                     ← Main script
│   ├── validate.py                        ← Core validation
│   ├── test_with_file.sh                  ← Wrapper (file)
│   └── test_with_sample_queries.sh        ← Wrapper (samples)
│
├── Base Model Evaluation
│   ├── eval_base_models.sh                ← Main script
│   ├── validate_base_model.py             ← Core validation
│   ├── test_base_model_with_file.sh       ← Wrapper (file)
│   └── test_base_model_with_sample_queries.sh  ← Wrapper (samples)
│
├── Support
│   ├── generate_comparative_report.py     ← Report generation
│   ├── model_configs.sh                   ← Shell configs
│   └── model_config.py                    ← Python configs
│
└── Training
    ├── train.py
    ├── prepare_training_data.py
    └── merge_lora.py
```

## Results Organization

```
results/
├── Fine-Tuned Models
│   ├── sample_queries/
│   │   ├── [model]/[timestamp]/
│   │   ├── batch_summary_*.txt
│   │   └── comparative_report_*.txt
│   └── test_file/
│       ├── [model]/[timestamp]/
│       ├── batch_summary_*.txt
│       └── comparative_report_*.txt
│
└── Base Models
    ├── base_models/
        ├── sample_queries/
        │   ├── [model]/[timestamp]/
        │   ├── batch_summary_*.txt
        │   └── comparative_report_*.txt
        └── test_file/
            ├── [model]/[timestamp]/
            ├── batch_summary_*.txt
            └── comparative_report_*.txt
```

## Common Flags (All Scripts)

| Flag | Description | Example |
|------|-------------|---------|
| `--list` | List available models | Any eval script |
| `--test-file <file>` | Use custom test file | `--test-file data/test/custom.json` |
| `--verbose` | Show detailed output | `--verbose` |
| `--debug` | Show model responses | `--debug` |
| `--skip <model>` | Skip model in batch | `--skip ministral-3b` (main scripts only) |

## Script Selection Guide

### Use Main Scripts When:
- Running batch evaluations (`all`)
- Need fine control over options
- Want to skip specific models
- Doing production evaluations

### Use Wrappers When:
- Quick single model testing
- Simpler command syntax preferred
- Don't need advanced options
- Rapid development/debugging

## Equivalence

These are equivalent:

```bash
# Main script approach
./eval_models.sh qwen2.5-7b --test-file data/test/test.json

# Wrapper approach
./test_with_file.sh qwen2.5-7b data/test/test.json
```

```bash
# Main script approach
./eval_base_models.sh ministral-3b

# Wrapper approach
./test_base_model_with_sample_queries.sh ministral-3b
```

## Recommendations

### For Development
```bash
# Use wrappers for quick tests
./test_base_model_with_sample_queries.sh qwen2.5-3b --verbose
```

### For Production
```bash
# Use main scripts for comprehensive evaluation
./eval_models.sh all --test-file data/test/impulse_test.json --verbose
./eval_base_models.sh all --test-file data/test/impulse_test.json --verbose
```

### For Comparison
```bash
# Use both main scripts with same test file
TEST_FILE="data/test/impulse_test.json"
./eval_base_models.sh all --test-file $TEST_FILE
./eval_models.sh all --test-file $TEST_FILE
```

## Migration from Old Names

If you see references to old scripts:

| Old Name | New Name |
|----------|----------|
| `test_base_model.sh` | `eval_base_models.sh` |
| Any script without clear naming | See table above |

## Summary

**8 total evaluation scripts:**
- 4 for fine-tuned models (1 main + 1 core + 2 wrappers)
- 4 for base models (1 main + 1 core + 2 wrappers)

**Consistent naming pattern:**
- Main scripts: `eval_*.sh`
- Core logic: `validate*.py`
- Wrappers: `test_*_with_*.sh`

**All scripts support:**
- Single model or batch evaluation
- Sample queries or custom test files
- Verbose and debug output
- Automated report generation
