# IMPULSE Salamandra Fine-tuning Guide

## 📋 Tabla de Contenidos
1. [Introducción](#introducción)
2. [Requisitos](#requisitos)
3. [Instalación](#instalación)
4. [Preparación de Datos](#preparación-de-datos)
5. [Entrenamiento](#entrenamiento)
6. [Validación](#validación)
7. [Despliegue](#despliegue)
8. [FAQ](#faq)
9. [Troubleshooting](#troubleshooting)

## 🎯 Introducción

Este proyecto implementa el fine-tuning del modelo Salamandra para el parser de queries del proyecto IMPULSE. El sistema convierte consultas en lenguaje natural (Catalán, Español, Inglés) a formato JSON estructurado para búsquedas en bases de datos de I+D.

### Características principales:
- ✅ Fine-tuning eficiente con **LoRA** (solo ~1% de parámetros)
- ✅ Soporte multilingüe (CA, ES, EN)
- ✅ Formato ChatML compatible con Salamandra
- ✅ Validación automática de JSON y esquema
- ✅ Scripts simplificados y bien documentados

### ¿Por qué LoRA?

LoRA (Low-Rank Adaptation) permite:
- **Eficiencia**: Solo entrena ~50MB vs 14GB del modelo completo
- **Velocidad**: 3-5x más rápido que fine-tuning completo
- **Flexibilidad**: Puedes cambiar/combinar adaptadores sin recargar el modelo base
- **Memoria**: Requiere ~24GB VRAM vs ~40GB+ para full fine-tuning

## 📦 Requisitos

### Hardware
- GPU con mínimo 24GB VRAM (RTX 3090, A10, T4x2, etc.)
- 32GB RAM sistema
- 50GB espacio en disco

### Software
```bash
# Python 3.8+
python --version

# CUDA 11.8+ (para GPU)
nvidia-smi
```

### Dependencias Python
```bash
# requirements.txt
torch>=2.0.0
transformers>=4.36.0
peft>=0.7.0
datasets>=2.14.0
accelerate>=0.25.0
bitsandbytes>=0.41.0  # Opcional, para cuantización
wandb  # Opcional, para logging
```

## 🔧 Instalación

### 1. Clonar repositorio
```bash
git clone https://github.com/tu-org/impulse-salamandra
cd impulse-salamandra
```

### 2. Crear entorno virtual
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate  # Windows
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Verificar instalación
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
```

## 📊 Preparación de Datos

### Estructura de datos esperada

Los datos deben estar en formato JSON con la siguiente estructura:

```json
{
  "doc_type": "projects",
  "filters": {
    "programme": "Horizon 2020",
    "funding_level": null,
    "year": 2023,
    "location": "Barcelona",
    "location_level": "nuts3"
  },
  "organisations": [
    {
      "type": "university",
      "name": null,
      "location": "France",
      "location_level": "country"
    }
  ],
  "semantic_query": "renewable energy",
  "metric": "list",
  "metric_params": {
    "k": null,
    "field": null,
    "group_by": null
  },
  "query_rewrite": "List of projects...",
  "meta": {
    "id": "TRAIN_001",
    "lang": "EN",
    "original_query": "projects in Barcelona with French universities on renewable energy",
    "notes": null
  }
}
```

### Preparar datos de entrenamiento

```bash
# Estructura de carpetas esperada
data/
├── training/
│   ├── batches/
│   │   ├── batch1.json
│   │   ├── batch2.json
│   │   └── ...
│   └── salamandra_finetuning_prompt.txt  # Opcional
└── test/
    └── test_set.json
```

### Ejecutar preparación

```bash
python scripts/prepare_training_data.py \
    --input-dir data/training/batches \
    --output-file data/training/impulse_training.jsonl \
    --prompt-file data/training/salamandra_finetuning_prompt.txt \
    --validate \
    --shuffle
```

**Opciones:**
- `--validate`: Valida estructura JSON
- `--shuffle`: Mezcla ejemplos aleatoriamente
- `--seed`: Semilla para reproducibilidad

## 🚀 Entrenamiento

### Configuración básica

```bash
python scripts/train_simple.py \
    --model BSC-LT/salamandra-7b-instruct \
    --data data/training/impulse_training.jsonl \
    --output-dir ./models/impulse-salamandra-ft \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-4
```

### Configuración avanzada

```bash
python scripts/train_simple.py \
    --model BSC-LT/salamandra-7b-instruct \
    --data data/training/impulse_training.jsonl \
    --output-dir ./models/impulse-salamandra-ft \
    --epochs 5 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 2e-4 \
    --max-length 2048 \
    --warmup-steps 100 \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --eval-split 0.1 \
    --use-wandb
```

### Parámetros importantes

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--epochs` | 3 | Número de epochs |
| `--batch-size` | 4 | Tamaño de batch por GPU |
| `--learning-rate` | 2e-4 | Tasa de aprendizaje |
| `--lora-r` | 16 | Rango de matrices LoRA |
| `--lora-alpha` | 32 | Factor de escalado LoRA |
| `--max-length` | 2048 | Longitud máxima de secuencia |

### Monitoreo con Weights & Biases

```bash
# Login en W&B
wandb login

# Entrenar con logging
python scripts/train_simple.py \
    --model BSC-LT/salamandra-7b-instruct \
    --data data/training/impulse_training.jsonl \
    --use-wandb
```

## ✅ Validación

### Validación básica

```bash
python scripts/validate_simple.py \
    --model-path ./models/impulse-salamandra-ft
```

### Validación con archivo de test

```bash
python scripts/validate_simple.py \
    --model-path ./models/impulse-salamandra-ft \
    --test-file data/test/test_set.json \
    --verbose \
    --save-results results/validation_results.json
```

### Interpretación de resultados

La validación reporta:
- **JSON validity**: % de respuestas con JSON válido
- **Schema compliance**: % que cumple el esquema
- **By language**: Rendimiento por idioma
- **By type**: Rendimiento por tipo de query

Ejemplo de salida:
```
TEST RESULTS SUMMARY
================================================================================
Overall:
  Total tests: 20
  Passed: 18 (90.0%)
  Failed: 2 (10.0%)

By language:
  CA: 6/7 (85.7%)
  ES: 6/6 (100.0%)
  EN: 6/7 (85.7%)
```

## 🚢 Despliegue

### Opción 1: Usar con adaptadores LoRA (recomendado para desarrollo)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Cargar modelo base
base_model = AutoModelForCausalLM.from_pretrained(
    "BSC-LT/salamandra-7b-instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Cargar adaptador LoRA
model = PeftModel.from_pretrained(
    base_model, 
    "./models/impulse-salamandra-ft"
)

# Usar para inferencia
tokenizer = AutoTokenizer.from_pretrained("BSC-LT/salamandra-7b-instruct")
```

### Opción 2: Fusionar para producción

```bash
# Fusionar adaptadores en modelo completo
python scripts/merge_lora.py \
    --adapter-path ./models/impulse-salamandra-ft \
    --output-path ./models/impulse-salamandra-merged
```

Luego usar directamente:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "./models/impulse-salamandra-merged",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./models/impulse-salamandra-merged")
```

### Integración con API

```python
# Ejemplo de endpoint FastAPI
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    language: str = "CA"

@app.post("/parse")
async def parse_query(request: QueryRequest):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": request.query}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extraer JSON
    json_str = response.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()
    
    return json.loads(json_str)
```

## ❓ FAQ

### ¿Cuánto tiempo toma el entrenamiento?

| GPU | Batch Size | Tiempo/Epoch | Total (3 epochs) |
|-----|------------|--------------|------------------|
| RTX 3090 (24GB) | 2 | ~45 min | ~2.5 horas |
| A100 (40GB) | 4 | ~30 min | ~1.5 horas |
| V100 (32GB) | 4 | ~40 min | ~2 horas |

### ¿Cómo ajustar para mi GPU?

**GPU con menos memoria:**
```bash
# Para 16GB VRAM
--batch-size 1 --gradient-accumulation-steps 16

# Para 24GB VRAM
--batch-size 2 --gradient-accumulation-steps 8
```

### ¿Qué modelo base usar?

1. **BSC-LT/salamandra-7b-instruct** - Recomendado, ya ajustado para instrucciones
2. **langtech-innovation/7b-tools-v3** - Mejor para output JSON estructurado
3. **BSC-LT/salamandra-7b** - Base sin instrucciones (requiere más epochs)

### ¿Cómo mejorar la calidad?

1. **Más datos**: Mínimo 500-1000 ejemplos diversos
2. **Más epochs**: Probar 5-10 epochs con early stopping
3. **Ajustar LoRA**: Aumentar r=32, alpha=64 para más capacidad
4. **Data augmentation**: Parafrasear queries, añadir variaciones

## 🔧 Troubleshooting

### Error: CUDA out of memory

```bash
# Reducir batch size
--batch-size 1

# Usar gradient accumulation
--gradient-accumulation-steps 16

# Reducir max_length
--max-length 1024

# Usar 8-bit (requiere bitsandbytes)
pip install bitsandbytes
# Modificar script para load_in_8bit=True
```

### Error: JSON decode error en validación

Causas comunes:
1. Temperatura muy alta (usar 0.1-0.3)
2. Pocos datos de entrenamiento
3. Formato incorrecto en datos

Solución:
```python
# Ajustar generación
temperature=0.1  # Más determinístico
top_p=0.9
repetition_penalty=1.1
```

### El modelo no sigue el formato

1. Verificar que el prompt incluye ejemplos
2. Aumentar epochs de entrenamiento
3. Verificar formato ChatML correcto
4. Añadir más ejemplos diversos al training

### Pérdida no baja / Overfitting

```bash
# Ajustar learning rate
--learning-rate 5e-5  # Más bajo

# Más regularización
--lora-dropout 0.1

# Early stopping manual
# Revisar métricas cada epoch
```

## 📚 Referencias

- [Salamandra Model Card](https://huggingface.co/BSC-LT/salamandra-7b-instruct)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [ChatML Format](https://github.com/openai/openai-python/blob/main/chatml.md)

## 📝 Licencia

Apache 2.0 (heredada del modelo Salamandra)

## 🤝 Contribuciones

Para contribuir:
1. Fork el repositorio
2. Crear branch para feature (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Abrir Pull Request

## 📧 Contacto

Para preguntas sobre el proyecto IMPULSE:
- Email: [tu-email@organizacion.com]
- Issues: [GitHub Issues](https://github.com/tu-org/impulse-salamandra/issues)

---

**Última actualización**: Enero 2025
**Versión**: 1.0.0
