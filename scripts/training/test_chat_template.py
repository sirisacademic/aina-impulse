#!/usr/bin/env python3
"""
Quick test to verify chat template works with a model
"""

from transformers import AutoTokenizer

def test_chat_template(model_name: str):
    """Test if chat template works"""
    print(f"Testing: {model_name}")
    print("=" * 60)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Test messages WITHOUT system role (for Gemma compatibility)
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        # Try to apply template
        result = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        print("✓ Chat template works!\n")
        print("Output preview:")
        print("-" * 60)
        print(result[:500])
        print("-" * 60)
        
        # Also test with merged system prompt (how training will use it)
        print("\nWith system prompt merged into user:")
        print("-" * 60)
        system_prompt = "You are a helpful assistant."
        messages_merged = [
            {"role": "user", "content": f"{system_prompt}\n\nHello!"}
        ]
        result_merged = tokenizer.apply_chat_template(
            messages_merged,
            tokenize=False,
            add_generation_prompt=False
        )
        print(result_merged[:500])
        print("-" * 60)
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}\n")
        return False

if __name__ == "__main__":
    import sys
    
    model = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-7b-it"
    test_chat_template(model)
