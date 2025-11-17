#!/usr/bin/env python3
"""
Complete validation pipeline for IMPULSE LLM query parser
Enhanced with multilingual equivalences and language-aware validation
"""

import json
import torch
import csv
import argparse
import time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from typing import Dict, List, Any, Optional
from collections import defaultdict
from datetime import datetime

# Import shared model configuration
from src.impulse.config.model_config import get_model_config

# ============================================================================
# LOAD EQUIVALENCE DICTIONARIES FROM JSON
# ============================================================================

def load_equivalences(test_file_path: str, equiv_dir: Optional[str] = None) -> Dict:
    """Load equivalence dictionaries from JSON files"""
    if equiv_dir is None:
        # Default: equivalences dir next to test file
        equiv_dir = Path(test_file_path).parent / "equivalences"
    else:
        equiv_dir = Path(equiv_dir)
    
    if not equiv_dir.exists():
        print(f"Warning: Equivalences directory not found: {equiv_dir}")
        return {
            'semantic': [],
            'org_names': [],
            'org_types': [],
            'programmes': [],
            'locations': []
        }
    
    equivalences = {}
    
    # Load each equivalence file
    files = {
        'semantic': 'semantic_equivalences.json',
        'org_names': 'org_equivalences.json',
        'org_types': 'org_type_equivalences.json',
        'programmes': 'programme_equivalences.json',
        'locations': 'location_equivalences.json'
    }
    
    for key, filename in files.items():
        filepath = equiv_dir / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Convert lists to sets for faster lookup, lowercase for comparison
                equivalences[key] = [
                    set(term.lower().strip() for term in equiv_set)
                    for equiv_set in data['equivalence_sets']
                ]
        else:
            print(f"Warning: {filename} not found")
            equivalences[key] = []
    
    return equivalences

# Global equivalence sets (loaded at startup)
EQUIVALENCES = {}

# ============================================================================
# SAMPLE QUERIES
# ============================================================================

SAMPLE_QUERIES = [
    {"meta": {"id": "SAMPLE_001", "lang": "EN", "original_query": "biodiversity"},
     "filters": {"programme": None, "funding_level": None, "year": None, "location": None, "location_level": None},
     "organisations": [], "semantic_query": "biodiversity",
     "query_rewrite": "List of projects on biodiversity"},
    
    {"meta": {"id": "SAMPLE_002", "lang": "CA", "original_query": "projectes SIFECAT"},
     "filters": {"programme": "SIFECAT", "funding_level": None, "year": None, "location": None, "location_level": None},
     "organisations": [], "semantic_query": None,
     "query_rewrite": "Llista de projectes SIFECAT"},
    
    {"meta": {"id": "SAMPLE_003", "lang": "EN", "original_query": "projects by University of Lleida about animal husbandry"},
     "filters": {"programme": None, "funding_level": None, "year": None, "location": None, "location_level": None},
     "organisations": [{"type": "university", "name": "Universitat de Lleida", "location": None, "location_level": None}],
     "semantic_query": "animal husbandry",
     "query_rewrite": "List of projects by Universitat de Lleida on animal husbandry"},
    
    {"meta": {"id": "SAMPLE_004", "lang": "CA", "original_query": "quins projectes de la UB sobre transició ecològica?"},
     "filters": {"programme": None, "funding_level": None, "year": None, "location": None, "location_level": None},
     "organisations": [{"type": "university", "name": "Universitat de Barcelona", "location": None, "location_level": None}],
     "semantic_query": "transició ecològica",
     "query_rewrite": "Llista de projectes de la Universitat de Barcelona sobre transició ecològica"},
    
    {"meta": {"id": "SAMPLE_005", "lang": "EN", "original_query": "MSCA projects in 2023 on social sciences and humanities"},
     "filters": {"programme": "MSCA", "funding_level": None, "year": "2023", "location": None, "location_level": None},
     "organisations": [], "semantic_query": "social sciences and humanities",
     "query_rewrite": "List of MSCA projects in 2023 on social sciences and humanities"},
    
    {"meta": {"id": "SAMPLE_006", "lang": "ES", "original_query": "proyectos sobre energías renovables con participación de instituciones eslovenas"},
     "filters": {"programme": None, "funding_level": None, "year": None, "location": None, "location_level": None},
     "organisations": [{"type": None, "name": None, "location": "Eslovenia", "location_level": "country"}],
     "semantic_query": "energías renovables",
     "query_rewrite": "Lista de proyectos con organizaciones eslovenas sobre energías renovables"},
    
    {"meta": {"id": "SAMPLE_007", "lang": "EN", "original_query": "ERC grants since 2020 on climate change with research centers from Catalunya"},
     "filters": {"programme": "ERC", "funding_level": None, "year": ">=2020", "location": None, "location_level": None},
     "organisations": [{"type": "research_center", "name": None, "location": "Catalunya", "location_level": "region"}],
     "semantic_query": "climate change",
     "query_rewrite": "List of ERC projects from 2020 by research centers in Catalunya on climate change"},
    
    {"meta": {"id": "SAMPLE_008", "lang": "CA", "original_query": "projectes H2020 a Espanya entre 2018 i 2020 sobre IA en salut amb col·laboració universitat-empresa"},
     "filters": {"programme": "Horizon 2020", "funding_level": "european", "year": "2018-2020", "location": "Espanya", "location_level": "country"},
     "organisations": [{"type": "university", "name": None, "location": None, "location_level": None},
                      {"type": "company", "name": None, "location": None, "location_level": None}],
     "semantic_query": "IA en salut",
     "query_rewrite": "Llista de projectes Horizon 2020 europeus del 2018 al 2020 a Espanya sobre IA en salut amb participació d'universitats i empreses"},
    
    {"meta": {"id": "SAMPLE_009", "lang": "ES", "original_query": "proyectos del CSIC"},
     "filters": {"programme": None, "funding_level": None, "year": None, "location": None, "location_level": None},
     "organisations": [{"type": None, "name": "CSIC", "location": None, "location_level": None}],
     "semantic_query": None,
     "query_rewrite": "Lista de proyectos del CSIC"},
    
    {"meta": {"id": "SAMPLE_010", "lang": "EN", "original_query": "regional funding about forestry"},
     "filters": {"programme": None, "funding_level": "regional", "year": None, "location": None, "location_level": None},
     "organisations": [], "semantic_query": "forestry",
     "query_rewrite": "List of regional projects on forestry"}
]

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_path: str, base_model: str = None, quantize: str = None):
    """Load model and tokenizer"""
    print(f"Loading model from {model_path}...")
    
    adapter_config_path = Path(model_path) / "adapter_config.json"
    is_lora = adapter_config_path.exists()
    
    if is_lora:
        print("  Type: LoRA adapter")
        if not base_model:
            with open(adapter_config_path, 'r') as f:
                config = json.load(f)
                base_model = config.get("base_model_name_or_path")
                
                if not base_model:
                    model_config = get_model_config(model_path)
                    base_model = model_config.get("default_base", "BSC-LT/salamandra-7b-instruct")
        
        print(f"  Base model: {base_model}")
        
        model_config = get_model_config(base_model)
        if quantize == "8bit" and not model_config["supports_8bit"]:
            print(f"    Warning: {base_model} may not support 8-bit quantization")
    else:
        print("  Type: Full model")
        base_model = model_path
    
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
    
    return model, tokenizer, base_model

# ============================================================================
# INFERENCE
# ============================================================================

def generate_prediction(model, tokenizer, system_prompt: str, query: str, args, model_config: dict) -> str:
    """Generate prediction for a single query"""
    # Handle models that don't support system role (like Gemma)
    if model_config.get("merge_system_into_user", False):
        messages = [
            {"role": "user", "content": f"{system_prompt}\n\n{query}"}
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature if args.temperature > 0 else None,
            do_sample=args.temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

"""
def parse_json_response(response: str) -> Dict:
    #Extract and parse JSON from model response
    start = response.find('{')
    end = response.rfind('}')
    
    if start == -1 or end == -1:
        return None
    
    json_str = response[start:end+1]
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None
"""

def parse_json_response(response: str) -> Dict:
    """Extract and parse JSON from model response"""
    start = response.find('{')
    end = response.rfind('}')
    
    if start == -1 or end == -1:
        return get_empty_parse()
    
    json_str = response[start:end+1]
    
    try:
        parsed = json.loads(json_str)
        
        # Ensure all required keys exist
        if 'filters' not in parsed:
            parsed['filters'] = {}
        if 'organisations' not in parsed:
            parsed['organisations'] = []
        if 'semantic_query' not in parsed:
            parsed['semantic_query'] = None
        if 'query_rewrite' not in parsed:
            parsed['query_rewrite'] = ""
            
        return parsed
    except json.JSONDecodeError:
        return get_empty_parse()

def get_empty_parse() -> dict:
    """Return empty valid parse structure matching expected schema"""
    return {
        "doc_type": "projects",
        "filters": {},
        "organisations": [],
        "semantic_query": None,
        "query_rewrite": ""
    }

# ============================================================================
# NORMALIZATION WITH EQUIVALENCES
# ============================================================================

def normalize_text(text: str) -> str:
    if not text:
        return ""
    return text.lower().strip()

def find_canonical_form(text: str, equivalence_sets: List[set]) -> Optional[str]:
    """Find canonical form for a text using equivalence sets"""
    if not text:
        return None
    
    text_norm = normalize_text(text)
    
    for equiv_set in equivalence_sets:
        if text_norm in equiv_set:
            # Return shortest form as canonical
            return min(equiv_set, key=len)
    
    return text_norm
   
def normalize_with_equivalence(value, equiv_key):
    """Normalize text using appropriate equivalence dictionary"""
    if value is None:
        return None
    if isinstance(value, dict):
        # If it's a dict, it might be a complex filter - skip normalization
        return value
    return find_canonical_form(value, EQUIVALENCES.get(equiv_key, []))

def semantic_contains_concepts(query: str, filter_value: str, equiv_key: str) -> bool:
    """Check if semantic query contains concepts from filter value"""
    if not query or not filter_value:
        return False
    
    query_norm = normalize_text(query)
    filter_norm = normalize_text(filter_value)
    
    # Direct substring match
    if filter_norm in query_norm:
        return True
    
    # Check equivalences
    filter_canonical = normalize_with_equivalence(filter_value, equiv_key)
    if filter_canonical and filter_canonical in query_norm:
        return True
    
    return False

# ============================================================================
# VALIDATION WITH LANGUAGE AWARENESS
# ============================================================================

def compare_filters(pred: Dict, gold: Dict) -> Dict[str, bool]:
    """Compare filter values with equivalence matching"""
    results = {}
    
    pred_prog = normalize_with_equivalence(pred.get('programme'), 'programmes')
    gold_prog = normalize_with_equivalence(gold.get('programme'), 'programmes')
    results['programme'] = (pred_prog == gold_prog)
    
    results['funding_level'] = (pred.get('funding_level') == gold.get('funding_level'))
    results['year'] = (str(pred.get('year')) == str(gold.get('year')))

    # Before normalizing, extract string from dict if needed
    pred_loc = pred.get('location')
    if isinstance(pred_loc, dict):
        pred_loc = pred_loc.get('value') or pred_loc.get('name') or str(pred_loc)
    pred_loc = normalize_with_equivalence(pred_loc, 'locations')

    gold_loc = gold.get('location')
    if isinstance(gold_loc, dict):
        gold_loc = gold_loc.get('value') or gold_loc.get('name') or str(gold_loc)
    gold_loc = normalize_with_equivalence(gold_loc, 'locations')

    results['location'] = (pred_loc == gold_loc)    
    results['location_level'] = (pred.get('location_level') == gold.get('location_level'))
    
    return results

def compare_organisations(pred: List[Dict], gold: List[Dict]) -> Dict[str, bool]:
    """Compare organisations with equivalence matching"""
    if len(pred) != len(gold):
        return {'exact': False, 'relaxed': False}
    
    exact_match = True
    relaxed_match = True
    
    for p_org, g_org in zip(pred, gold):
        # Type comparison with equivalence
        p_type = normalize_with_equivalence(p_org.get('type'), 'org_types')
        g_type = normalize_with_equivalence(g_org.get('type'), 'org_types')
        
        if p_type != g_type:
            exact_match = False
            # For relaxed, allow type mismatch if name is correct
        
        # Name comparison with equivalence
        p_name = normalize_with_equivalence(p_org.get('name'), 'org_names')
        g_name = normalize_with_equivalence(g_org.get('name'), 'org_names')
        
        if p_name != g_name:
            exact_match = False
            relaxed_match = False
        
        # Location comparison
        p_loc = normalize_with_equivalence(p_org.get('location'), 'locations')
        g_loc = normalize_with_equivalence(g_org.get('location'), 'locations')
        
        if p_loc != g_loc:
            exact_match = False
        
        if p_org.get('location_level') != g_org.get('location_level'):
            exact_match = False
    
    return {'exact': exact_match, 'relaxed': relaxed_match}

def compare_semantic_query(pred: str, gold: str) -> bool:
    """Compare semantic queries with concept-level equivalence"""
    if not pred and not gold:
        return True
    if not pred or not gold:
        return False
    
    pred_norm = normalize_text(pred)
    gold_norm = normalize_text(gold)
    
    # Direct match
    if pred_norm == gold_norm:
        return True
    
    # Check if they share canonical concepts
    pred_canonical = normalize_with_equivalence(pred, 'semantic')
    gold_canonical = normalize_with_equivalence(gold, 'semantic')
    
    if pred_canonical and gold_canonical and pred_canonical == gold_canonical:
        return True
    
    # Check for concept overlap (for multi-word queries)
    pred_words = set(pred_norm.split())
    gold_words = set(gold_norm.split())
    
    # Find canonical forms for all words
    pred_concepts = set()
    for word in pred_words:
        canonical = find_canonical_form(word, EQUIVALENCES.get('semantic', []))
        if canonical:
            pred_concepts.add(canonical)
    
    gold_concepts = set()
    for word in gold_words:
        canonical = find_canonical_form(word, EQUIVALENCES.get('semantic', []))
        if canonical:
            gold_concepts.add(canonical)
    
    # Check overlap
    if pred_concepts and gold_concepts:
        overlap = pred_concepts & gold_concepts
        if len(overlap) / max(len(pred_concepts), len(gold_concepts)) > 0.5:
            return True
    
    return False

def detect_language(text: str) -> Optional[str]:
    """Simple language detection based on common words/patterns"""
    if not text:
        return None
    
    text_lower = text.lower()
    
    # Catalan indicators
    ca_indicators = ['projectes', 'sobre', 'de la', "d'", 'a catalunya', 'amb', 'des de']
    # Spanish indicators  
    es_indicators = ['proyectos', 'de la', 'del', 'con', 'desde', 'en españa']
    # English indicators
    en_indicators = ['projects', 'about', 'from', 'with', 'in', 'by']
    
    ca_score = sum(1 for ind in ca_indicators if ind in text_lower)
    es_score = sum(1 for ind in es_indicators if ind in text_lower)
    en_score = sum(1 for ind in en_indicators if ind in text_lower)
    
    if ca_score > es_score and ca_score > en_score:
        return 'CA'
    elif es_score > ca_score and es_score > en_score:
        return 'ES'
    elif en_score > 0:
        return 'EN'
    
    return None

def validate_prediction(pred: Dict, gold: Dict) -> Dict:
    """Validate prediction with language awareness"""
    if pred is None:
        return {
            'test_id': gold['meta']['id'],
            'valid_json': False,
            'language_correct': False,
            'filters': {k: False for k in ['programme', 'funding_level', 'year', 'location', 'location_level']},
            'organisations_exact': False,
            'organisations_relaxed': False,
            'semantic_query': False,
            'all_correct_strict': False,
            'all_correct_relaxed': False
        }
    
    # Check language match
    gold_lang = gold['meta']['lang']
    pred_lang_semantic = detect_language(pred.get('semantic_query', '')) if pred.get('semantic_query') else None
    pred_lang_rewrite = detect_language(pred.get('query_rewrite', '')) if pred.get('query_rewrite') else None
    
    # Consider language correct if either semantic_query or query_rewrite matches
    language_correct = (pred_lang_semantic == gold_lang) or (pred_lang_rewrite == gold_lang)
    
    filter_results = compare_filters(pred.get('filters', {}), gold.get('filters', {}))
    org_results = compare_organisations(pred.get('organisations', []), gold.get('organisations', []))
    semantic_match = compare_semantic_query(pred.get('semantic_query'), gold.get('semantic_query'))
    
    all_filters_correct = all(filter_results.values())
    
    # Strict: everything correct including language
    all_correct_strict = (
        all_filters_correct and
        org_results['exact'] and
        semantic_match and
        language_correct
    )
    
    # Relaxed: content correct, language can differ
    all_correct_relaxed = (
        all_filters_correct and
        org_results['relaxed'] and
        semantic_match
    )
    
    return {
        'test_id': gold['meta']['id'],
        'valid_json': True,
        'language_correct': language_correct,
        'predicted_language': pred_lang_semantic or pred_lang_rewrite,
        'expected_language': gold_lang,
        'filters': filter_results,
        'organisations_exact': org_results['exact'],
        'organisations_relaxed': org_results['relaxed'],
        'semantic_query': semantic_match,
        'all_correct_strict': all_correct_strict,
        'all_correct_relaxed': all_correct_relaxed
    }

# ============================================================================
# TESTING
# ============================================================================

def test_model(model, tokenizer, test_data: List[Dict], system_prompt: str, args, model_config: dict) -> Dict:
    """Run full test suite"""
    results = []
    predictions = []
    
    print(f"\nRunning {len(test_data)} tests...")
    start_time = time.time()
    
    for i, test_case in enumerate(test_data):
        query = test_case['meta']['original_query']
        test_id = test_case['meta']['id']
        
        if args.verbose:
            print(f"\n[{i+1}/{len(test_data)}] {test_id}: {query}")
        else:
            print(f"\rProgress: {i+1}/{len(test_data)} ({(i+1)/len(test_data)*100:.0f}%)", end='', flush=True)
        
        response = generate_prediction(model, tokenizer, system_prompt, query, args, model_config)
        pred_json = parse_json_response(response)
        
        # Debug mode
        if args.debug:
            print(f"\n{'='*80}")
            print(f"[DEBUG] Test {i+1}: {test_id}")
            print(f"Query: {query}")
            print(f"\nModel Response:")
            print(f"{'-'*80}")
            print(response)
            print(f"{'-'*80}")
            if pred_json:
                print(f"✓ Valid JSON parsed")
            else:
                print(f"✗ Failed to parse JSON")
            print(f"{'='*80}\n")
        
        validation = validate_prediction(pred_json, test_case)
        results.append(validation)
        
        predictions.append({
            'test_id': test_id,
            'query': query,
            'response': response,
            'parsed': pred_json,
            'gold': test_case,
            'validation': validation
        })
        
        if args.verbose and validation['valid_json']:
            print(f"  Valid JSON: [OK]")
            print(f"  Language match: {'[OK]' if validation['language_correct'] else '[X]'}")
            print(f"  Strict correct: {'[OK]' if validation['all_correct_strict'] else '[X]'}")
            print(f"  Relaxed correct: {'[OK]' if validation['all_correct_relaxed'] else '[X]'}")
    
    if not args.verbose:
        print()
    
    elapsed = time.time() - start_time
    
    stats = compute_statistics(results, predictions, test_data)
    stats['elapsed_time'] = elapsed
    stats['predictions'] = predictions
    
    return stats

def compute_statistics(results: List[Dict], predictions: List[Dict], test_data: List[Dict]) -> Dict:
    """Compute detailed statistics with strict/relaxed metrics"""
    total = len(results)
    valid_json = sum(1 for r in results if r['valid_json'])
    language_correct = sum(1 for r in results if r.get('language_correct', False))
    all_correct_strict = sum(1 for r in results if r['all_correct_strict'])
    all_correct_relaxed = sum(1 for r in results if r['all_correct_relaxed'])
    
    stats = {
        'total': total,
        'valid_json': valid_json,
        'language_correct': language_correct,
        'all_correct_strict': all_correct_strict,
        'all_correct_relaxed': all_correct_relaxed,
        'filters': {k: sum(1 for r in results if r['filters'][k]) for k in ['programme', 'funding_level', 'year', 'location', 'location_level']},
        'organisations_exact': sum(1 for r in results if r['organisations_exact']),
        'organisations_relaxed': sum(1 for r in results if r['organisations_relaxed']),
        'semantic_query': sum(1 for r in results if r['semantic_query']),
    }
    
    # By language
    by_lang = defaultdict(lambda: {'total': 0, 'valid_json': 0, 'language_correct': 0, 'strict': 0, 'relaxed': 0})
    for pred, test in zip(predictions, test_data):
        lang = test['meta']['lang']
        by_lang[lang]['total'] += 1
        if pred['validation']['valid_json']:
            by_lang[lang]['valid_json'] += 1
        if pred['validation'].get('language_correct'):
            by_lang[lang]['language_correct'] += 1
        if pred['validation']['all_correct_strict']:
            by_lang[lang]['strict'] += 1
        if pred['validation']['all_correct_relaxed']:
            by_lang[lang]['relaxed'] += 1
    
    stats['by_language'] = dict(by_lang)
    
    return stats

# ============================================================================
# ERROR ANALYSIS
# ============================================================================

def analyze_errors_detailed(predictions: List[Dict]) -> Dict:
    """Enhanced error analysis with language mismatch tracking"""
    errors = {
        'invalid_json': [],
        'language_mismatch': [],
        'critical': defaultdict(list),
        'moderate': defaultdict(list),
        'minor': defaultdict(list)
    }
    
    for pred in predictions:
        test_id = pred['test_id']
        query = pred['query']
        val = pred['validation']
        parsed = pred.get('parsed') or pred.get('predicted')
        response = pred.get('response', '')
        gold = pred.get('gold', {})

        # Ensure parsed has complete structure
        if not parsed or not isinstance(parsed, dict):
            parsed = {
                'filters': {},
                'organisations': [],
                'semantic_query': None,
                'query_rewrite': ''
            }
        
        # Ensure nested structures exist
        if 'filters' not in parsed:
            parsed['filters'] = {}
        if 'organisations' not in parsed:
            parsed['organisations'] = []
        if 'semantic_query' not in parsed:
            parsed['semantic_query'] = None
        if 'query_rewrite' not in parsed:
            parsed['query_rewrite'] = ''
        
        # Ensure gold has filters
        if 'filters' not in gold:
            gold['filters'] = {}
        if 'organisations' not in gold:
            gold['organisations'] = []

        if not val.get('valid_json', False):
            errors['invalid_json'].append({
                'test_id': test_id,
                'query': query,
                'response': response[:200] if response else 'N/A'
            })
            continue
        
        # Language mismatch (but content might be correct)
        if not val.get('language_correct', True) and val.get('all_correct_relaxed', False):
            errors['language_mismatch'].append({
                'test_id': test_id,
                'query': query,
                'predicted_lang': val.get('predicted_language'),
                'expected_lang': val.get('expected_language'),
                'semantic_query': parsed.get('semantic_query'),
                'query_rewrite': parsed.get('query_rewrite')
            })
        
        # Critical errors
        if not val.get('filters', {}).get('programme', True) and gold['filters'].get('programme'):
            errors['critical']['wrong_programme'].append({
                'test_id': test_id,
                'query': query,
                'predicted': parsed['filters'].get('programme'),
                'expected': gold['filters']['programme']
            })
        
        if not val.get('organisations_relaxed', True) and gold['organisations']:
            errors['critical']['wrong_organisations'].append({
                'test_id': test_id,
                'query': query,
                'predicted': parsed.get('organisations'),
                'expected': gold['organisations']
            })
        
        # Moderate errors
        if not val.get('filters', {}).get('year', True) and gold['filters'].get('year'):
            errors['moderate']['wrong_year'].append({
                'test_id': test_id,
                'query': query,
                'predicted': parsed['filters'].get('year'),
                'expected': gold['filters']['year']
            })
        
        if not val.get('semantic_query', True) and gold.get('semantic_query'):
            errors['moderate']['wrong_semantic_query'].append({
                'test_id': test_id,
                'query': query,
                'predicted': parsed.get('semantic_query'),
                'expected': gold['semantic_query']
            })
        
        # Check for filter leakage into semantic
        if parsed.get('semantic_query') and gold.get('filters'):
            for filter_type, filter_value in gold['filters'].items():
                if filter_value and semantic_contains_concepts(
                    parsed['semantic_query'], str(filter_value), 
                    'locations' if filter_type == 'location' else 'programmes'
                ):
                    errors['moderate']['extra_filter_in_semantic'].append({
                        'test_id': test_id,
                        'query': query,
                        'filter_type': filter_type,
                        'leaked_value': filter_value,
                        'semantic_query': parsed['semantic_query']
                    })
        
        # Minor errors
        if val.get('organisations_relaxed', False) and not val.get('organisations_exact', True):
            errors['minor']['org_type_mismatch'].append({
                'test_id': test_id,
                'query': query,
                'predicted': parsed.get('organisations'),
                'expected': gold['organisations']
            })
        
        if not val.get('filters', {}).get('funding_level', True) and gold['filters'].get('funding_level'):
            errors['minor']['wrong_funding_level'].append({
                'test_id': test_id,
                'query': query,
                'predicted': parsed['filters'].get('funding_level'),
                'expected': gold['filters']['funding_level']
            })
    
    return errors

def save_error_analysis_detailed(errors: Dict, filename: str):
    """Save enhanced error analysis with language tracking"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("IMPULSE LLM VALIDATION - DETAILED ERROR ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        # Invalid JSON
        if errors['invalid_json']:
            f.write(f"[!!!] INVALID JSON ({len(errors['invalid_json'])} cases)\n")
            f.write("="*80 + "\n")
            for err in errors['invalid_json']:
                f.write(f"\nTest: {err['test_id']}\n")
                f.write(f"Query: {err['query']}\n")
                f.write(f"Response preview: {err['response']}\n")
        
        # Language mismatch
        if errors['language_mismatch']:
            f.write(f"\n\n[!] LANGUAGE MISMATCH - Content Correct ({len(errors['language_mismatch'])} cases)\n")
            f.write("="*80 + "\n")
            for err in errors['language_mismatch']:
                f.write(f"\nTest: {err['test_id']}\n")
                f.write(f"Query: {err['query']}\n")
                f.write(f"Expected language: {err['expected_lang']}\n")
                f.write(f"Predicted language: {err['predicted_lang']}\n")
                f.write(f"Semantic query: {err['semantic_query']}\n")
                f.write(f"Query rewrite: {err['query_rewrite']}\n")
        
        # Critical errors
        f.write(f"\n\n{'='*80}\n")
        f.write("[!!!] CRITICAL ERRORS (High Impact)\n")
        f.write("="*80 + "\n")
        
        for error_type in ['wrong_programme', 'wrong_organisations']:
            if errors['critical'][error_type]:
                f.write(f"\n=> {error_type.replace('_', ' ').title()} ({len(errors['critical'][error_type])} cases)\n")
                f.write("-" * 80 + "\n")
                for err in errors['critical'][error_type]:
                    f.write(f"\nTest: {err['test_id']}\n")
                    f.write(f"Query: {err['query']}\n")
                    f.write(f"Predicted: {err['predicted']}\n")
                    f.write(f"Expected: {err['expected']}\n")
        
        # Moderate errors
        f.write(f"\n\n{'='*80}\n")
        f.write("[!!] MODERATE ERRORS (Content Issues)\n")
        f.write("="*80 + "\n")
        
        for error_type in ['extra_filter_in_semantic', 'wrong_year', 'wrong_semantic_query']:
            if errors['moderate'][error_type]:
                f.write(f"\n=> {error_type.replace('_', ' ').title()} ({len(errors['moderate'][error_type])} cases)\n")
                f.write("-" * 80 + "\n")
                for err in errors['moderate'][error_type]:
                    f.write(f"\nTest: {err['test_id']}\n")
                    f.write(f"Query: {err['query']}\n")
                    if 'filter_type' in err:
                        f.write(f"Filter type: {err['filter_type']}\n")
                        f.write(f"Leaked value: {err['leaked_value']}\n")
                        f.write(f"Semantic: {err['semantic_query']}\n")
                    else:
                        f.write(f"Predicted: {err['predicted']}\n")
                        f.write(f"Expected: {err['expected']}\n")
        
        # Minor errors
        f.write(f"\n\n{'='*80}\n")
        f.write("[!] MINOR ERRORS (Low Impact)\n")
        f.write("="*80 + "\n")
        
        for error_type in ['org_type_mismatch', 'wrong_funding_level']:
            if errors['minor'][error_type]:
                f.write(f"\n=> {error_type.replace('_', ' ').title()} ({len(errors['minor'][error_type])} cases)\n")
                f.write("-" * 80 + "\n")
                for err in errors['minor'][error_type]:
                    f.write(f"\nTest: {err['test_id']}\n")
                    f.write(f"Query: {err['query']}\n")
                    f.write(f"Predicted: {err['predicted']}\n")
                    f.write(f"Expected: {err['expected']}\n")

def print_error_summary_detailed(errors: Dict):
    """Print error summary with language tracking"""
    print("\nError Analysis by Severity:")
    print("="*60)
    
    if errors['invalid_json']:
        print(f"\n[!] Invalid JSON: {len(errors['invalid_json'])}")
    
    if errors['language_mismatch']:
        print(f"\n[!] Language Mismatch (content correct): {len(errors['language_mismatch'])}")
    
    critical_total = sum(len(v) for v in errors['critical'].values())
    if critical_total > 0:
        print(f"\n[!!!] CRITICAL ({critical_total} total):")
        for error_type, error_list in errors['critical'].items():
            if error_list:
                print(f"  • {error_type.replace('_', ' ').title()}: {len(error_list)}")
    
    moderate_total = sum(len(v) for v in errors['moderate'].values())
    if moderate_total > 0:
        print(f"\n[!!] MODERATE ({moderate_total} total):")
        for error_type, error_list in errors['moderate'].items():
            if error_list:
                print(f"  • {error_type.replace('_', ' ').title()}: {len(error_list)}")
    
    minor_total = sum(len(v) for v in errors['minor'].values())
    if minor_total > 0:
        print(f"\n[!] MINOR ({minor_total} total):")
        for error_type, error_list in errors['minor'].items():
            if error_list:
                print(f"  • {error_type.replace('_', ' ').title()}: {len(error_list)}")
    
    print("="*60)

# ============================================================================
# OUTPUT
# ============================================================================

def save_detailed_json(predictions: List[Dict], filename: str):
    """Save detailed predictions with validation"""
    output = []
    for pred in predictions:
        output.append({
            'test_id': pred['test_id'],
            'query': pred['query'],
            'validation': pred['validation'],
            'predicted': pred['parsed'],
            'gold': pred['gold']
        })
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

def save_table_tsv(predictions: List[Dict], filename: str):
    """Save enhanced validation table with language and strict/relaxed metrics"""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['test_id', 'valid_json', 'language_correct', 'programme', 'funding_level', 'year', 
                        'location', 'location_level', 'orgs_exact', 'orgs_relaxed', 'semantic_query', 
                        'all_correct_strict', 'all_correct_relaxed'])
        for pred in predictions:
            v = pred['validation']
            writer.writerow([
                pred['test_id'],
                '1' if v['valid_json'] else '0',
                '1' if v.get('language_correct', False) else '0',
                '1' if v['filters']['programme'] else '0',
                '1' if v['filters']['funding_level'] else '0',
                '1' if v['filters']['year'] else '0',
                '1' if v['filters']['location'] else '0',
                '1' if v['filters']['location_level'] else '0',
                '1' if v['organisations_exact'] else '0',
                '1' if v['organisations_relaxed'] else '0',
                '1' if v['semantic_query'] else '0',
                '1' if v['all_correct_strict'] else '0',
                '1' if v['all_correct_relaxed'] else '0'
            ])

def print_results(stats: Dict):
    """Print summary with strict/relaxed metrics"""
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    
    print(f"\nOverall:")
    print(f"  Total tests: {stats['total']}")
    print(f"  Valid JSON: {stats['valid_json']}/{stats['total']} ({stats['valid_json']/stats['total']*100:.1f}%)")
    print(f"  Language correct: {stats['language_correct']}/{stats['total']} ({stats['language_correct']/stats['total']*100:.1f}%)")
    print(f"  All correct (strict): {stats['all_correct_strict']}/{stats['total']} ({stats['all_correct_strict']/stats['total']*100:.1f}%)")
    print(f"  All correct (relaxed): {stats['all_correct_relaxed']}/{stats['total']} ({stats['all_correct_relaxed']/stats['total']*100:.1f}%)")
    print(f"  Time: {stats['elapsed_time']:.1f}s")
    
    if stats['valid_json'] > 0:
        print(f"\nComponent Accuracy:")
        for field, count in stats['filters'].items():
            print(f"  {field}: {count}/{stats['total']} ({count/stats['total']*100:.1f}%)")
        print(f"  organisations (exact): {stats['organisations_exact']}/{stats['total']} ({stats['organisations_exact']/stats['total']*100:.1f}%)")
        print(f"  organisations (relaxed): {stats['organisations_relaxed']}/{stats['total']} ({stats['organisations_relaxed']/stats['total']*100:.1f}%)")
        print(f"  semantic_query: {stats['semantic_query']}/{stats['total']} ({stats['semantic_query']/stats['total']*100:.1f}%)")
    
    print(f"\nBy Language:")
    for lang, lang_stats in sorted(stats['by_language'].items()):
        print(f"  {lang}:")
        print(f"    Language correct: {lang_stats['language_correct']}/{lang_stats['total']} ({lang_stats['language_correct']/lang_stats['total']*100:.1f}%)")
        print(f"    Strict: {lang_stats['strict']}/{lang_stats['total']} ({lang_stats['strict']/lang_stats['total']*100:.1f}%)")
        print(f"    Relaxed: {lang_stats['relaxed']}/{lang_stats['total']} ({lang_stats['relaxed']/lang_stats['total']*100:.1f}%)")
    
    print("="*80)

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Complete IMPULSE validation pipeline with multilingual support")
    parser.add_argument("--model-path", required=True, help="Path to model or LoRA adapter")
    parser.add_argument("--base-model", default=None, help="Base model (for LoRA)")
    parser.add_argument("--prompt-file", required=True, help="System prompt file")
    parser.add_argument("--test-file", default=None, help="Test JSON (default: sample queries)")
    parser.add_argument("--equiv-dir", default=None, help="Equivalences directory (default: test_file/../equivalences)")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--temperature", type=float, default=0, help="Generation temperature")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max tokens")
    parser.add_argument("--quantize", choices=["none", "8bit", "4bit"], default="none")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Show model responses")
    args = parser.parse_args()
    
    # Load equivalences
    test_file = args.test_file if args.test_file else "dummy"
    global EQUIVALENCES
    EQUIVALENCES = load_equivalences(test_file, args.equiv_dir)
    print(f"[OK] Loaded equivalences:")
    for key, sets in EQUIVALENCES.items():
        print(f"  {key}: {len(sets)} equivalence sets")
    
    # Load prompt
    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        system_prompt = f.read().strip()
    print(f"[OK] Loaded prompt ({len(system_prompt)} chars)")
    
    # Load model
    model, tokenizer, base_model = load_model(args.model_path, args.base_model,
                                   None if args.quantize == "none" else args.quantize)
    
    # Get model config
    model_config = get_model_config(args.base_model or base_model)
    print(f"[OK] Model config: merge_system={model_config.get('merge_system_into_user', False)}")
    
    # Load test data
    if args.test_file:
        with open(args.test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        print(f"[OK] Loaded {len(test_data)} test cases")
    else:
        test_data = SAMPLE_QUERIES
        print(f"[OK] Using {len(test_data)} sample queries")
    
    # Run tests
    results = test_model(model, tokenizer, test_data, system_prompt, args, model_config)

    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save outputs
    save_detailed_json(results['predictions'], output_dir / f'validation_detailed_{current_datetime}.json')
    save_table_tsv(results['predictions'], output_dir / f'validation_table_{current_datetime}.tsv')
    
    # Detailed error analysis
    errors = analyze_errors_detailed(results['predictions'])
    save_error_analysis_detailed(errors, output_dir / f'error_analysis_{current_datetime}.txt')
    
    # Add error counts to stats
    with open(output_dir / f'validation_stats_{current_datetime}.json', 'w', encoding='utf-8') as f:
        stats_out = {k: v for k, v in results.items() if k != 'predictions'}
        stats_out['error_counts'] = {
            'invalid_json': len(errors['invalid_json']),
            'language_mismatch': len(errors['language_mismatch']),
            'critical': {k: len(v) for k, v in errors['critical'].items()},
            'moderate': {k: len(v) for k, v in errors['moderate'].items()},
            'minor': {k: len(v) for k, v in errors['minor'].items()}
        }
        json.dump(stats_out, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Results saved to {output_dir}/")
    print_results(results)
    print_error_summary_detailed(errors)
    
    return 0 if results['all_correct_relaxed'] / results['total'] >= 0.5 else 1

if __name__ == "__main__":
    exit(main())
