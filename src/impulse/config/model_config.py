#!/usr/bin/env python3
"""
Centralized model configuration for IMPULSE training pipeline
"""

MODEL_CONFIGS = {
    "BSC-LT/salamandra-7b-instruct": {
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],
        "supports_8bit": True,
        "chat_template": "auto",
        "merge_system_into_user": False,
    },
    "langtech-innovation/7b-tools-v3": {
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],
        "supports_8bit": True,
        "chat_template": "auto",
        "merge_system_into_user": False,
    },
    "meta-llama/Llama-2-7b-chat-hf": {
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        "supports_8bit": True,
        "chat_template": "auto",
        "merge_system_into_user": False,
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        "supports_8bit": True,
        "chat_template": "auto",
        "merge_system_into_user": False,
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        "supports_8bit": True,
        "chat_template": "auto",
        "merge_system_into_user": False,
    },
    "google/gemma-7b-it": {
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "supports_8bit": False,
        "chat_template": "auto",
        "merge_system_into_user": True,  # Gemma doesn't support system role
    },
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        "supports_8bit": True,
        "chat_template": "auto",
        "merge_system_into_user": False,
    },
    "BSC-LT/salamandra-2b-instruct": {
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        "supports_8bit": True,
        "chat_template": "auto",
        "merge_system_into_user": False,
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        "supports_8bit": True,
        "chat_template": "auto",
        "merge_system_into_user": False,
    },
    "ministral/Ministral-3b-instruct": {
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        "supports_8bit": True,
        "chat_template": "auto",
        "merge_system_into_user": False,
    },
    "microsoft/Phi-4-mini-instruct": {
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        "supports_8bit": True,
        "chat_template": "auto",
        "merge_system_into_user": False,
    },
    "google/gemma-2-2b-it": {
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "supports_8bit": True,
        "chat_template": "auto",
        "merge_system_into_user": False,  # gemma-2 supports system role
    }
}

def get_model_config(model_name: str) -> dict:
    """Get model-specific configuration with fallback to defaults"""
    # Exact match
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]
    
    # Partial match (for local paths)
    for key in MODEL_CONFIGS:
        if key.split('/')[-1] in model_name:
            return MODEL_CONFIGS[key]
    
    # Extract base model from fine-tuned naming convention
    # Pattern: impulse-{base-model-name}-{suffix}
    if "impulse-" in model_name.lower():
        import re
        # Remove paths, impulse prefix, and suffixes
        base_name = model_name.split('/')[-1]
        base_name = re.sub(r'^impulse-', '', base_name, flags=re.IGNORECASE)
        base_name = re.sub(r'(-merged|--averaged|-v\d+).*$', '', base_name)
        base_name_lower = base_name.lower()
        
        # Try matching against config keys (org/model format)
        for config_key in MODEL_CONFIGS:
            # Extract just the model name part (after /)
            model_part = config_key.split('/')[-1].lower()
            # Check if our base_name contains the model part
            if model_part in base_name_lower or base_name_lower in model_part:
                print(f"✓ Mapped {model_name} → {config_key}")
                return MODEL_CONFIGS[config_key]
    
    # Conservative defaults
    print(f"WARNING: No specific config for {model_name}, using defaults")
    return {
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "supports_8bit": True,
        "chat_template": "auto",
        "merge_system_into_user": False,
    }
