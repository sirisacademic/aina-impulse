"""Query parser using fine-tuned model"""
import json
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.impulse.config.model_config import get_model_config
from src.impulse.settings import settings

logger = logging.getLogger(__name__)


class QueryParser:
    """Singleton query parser for converting natural language to structured JSON"""
    _instance = None
    _model = None
    _tokenizer = None
    _model_config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._model is None:
            self._load_model()
    
    def _load_model(self):
        """Load model from settings"""
        model_path = settings.query_parser_model
        quantize = settings.query_parser_quantize
        
        logger.info(f"Loading query parser from {model_path}")
        
        # Detect LoRA vs full model
        adapter_config_path = Path(model_path) / "adapter_config.json"
        is_lora = adapter_config_path.exists()
        
        if is_lora:
            logger.info("Detected LoRA adapter")
            with open(adapter_config_path, 'r') as f:
                config = json.load(f)
                base_model = config.get("base_model_name_or_path")
            
            if not base_model:
                raise ValueError("Base model not found in adapter config")
            
            logger.info(f"Base model: {base_model}")
        else:
            logger.info("Loading full merged model")
            base_model = model_path
        
        # Get model-specific config
        self._model_config = get_model_config(base_model)
        logger.info(f"Model config: {self._model_config}")
        
        # Quantization config
        quantization_config = None
        if quantize == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif quantize == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        
        # Load model
        if is_lora:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map="auto",
                trust_remote_code=True,
                quantization_config=quantization_config
            )
            model = PeftModel.from_pretrained(model, model_path)
            tokenizer = AutoTokenizer.from_pretrained(base_model)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True,
                quantization_config=quantization_config
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        self._model = model
        self._tokenizer = tokenizer
        
        logger.info("Query parser loaded successfully")
    
    def parse(self, query: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Parse natural language query to structured JSON"""
        
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()
        
        # Handle models that don't support system role
        if self._model_config.get("merge_system_into_user", False):
            messages = [
                {"role": "user", "content": f"{system_prompt}\n\n{query}"}
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        
        input_text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self._tokenizer(input_text, return_tensors="pt", padding=True).to(self._model.device)
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id
            )
        
        response = self._tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Extract and validate JSON
        try:
            start = response.find('{')
            end = response.rfind('}')
            
            if start == -1 or end == -1:
                raise ValueError("No JSON found in output")
            
            json_str = response[start:end+1]
            parsed = json.loads(json_str)
            
            return {
                "success": True,
                "parsed": parsed,
                "raw_output": response
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return {
                "success": False,
                "error": f"Invalid JSON: {str(e)}",
                "raw_output": response
            }
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return {
                "success": False,
                "error": str(e),
                "raw_output": response if response else None
            }
    
    def _get_default_system_prompt(self) -> str:
        """Load system prompt from settings"""
        prompt_file = Path(settings.query_parser_prompt)
        
        if prompt_file.exists():
            return prompt_file.read_text(encoding='utf-8')
        
        logger.warning(f"Prompt file not found: {prompt_file}, using fallback")
        
        # Fallback prompt
        return """Convert natural language queries into structured JSON for R&D project search.

Output only valid JSON with this schema:
{
  "doc_type": "projects",
  "filters": {
    "programme": null,
    "funding_level": null,
    "year": null,
    "location": null,
    "location_level": null
  },
  "organisations": [],
  "semantic_query": null,
  "query_rewrite": ""
}"""


