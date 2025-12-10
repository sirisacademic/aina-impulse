# IMPULS - Maintenance Guide

## Operations and maintenance manual for system administrators

---

## 1. System Overview

### 1.1 Deployed Components

| Component | Port | Description |
|-----------|------|-------------|
| **REST API** | 8000 | FastAPI backend with search endpoints |
| **Web UI** | 8080 | Static web interface (nginx) |

### 1.2 File Locations

```
/opt/impuls/
├── .venv/                    # Python virtual environment
├── src/impulse/              # Source code
├── data/
│   ├── index/                # HNSW vector index
│   ├── kb/                   # Knowledge base (wikidata_kb.jsonl)
│   ├── ris3cat/              # Project data (parquet)
│   └── normalization/        # Normalization tables
├── html/                     # Web interface
├── logs/                     # Application logs
├── .env                      # Configuration
└── scripts/                  # Utility scripts
```

### 1.3 Systemd Services

```bash
# API service
/etc/systemd/system/impuls-api.service

# Nginx (UI)
/etc/nginx/sites-enabled/impuls
```

---

## 2. Routine Operations

### 2.1 Check System Status

```bash
# API health check
curl http://localhost:8000/health

# Expected response:
{
  "status": "ok",
  "index_size": 26941,
  "projects_metadata_loaded": 26941,
  "kb_concepts": 4265,
  "kb_indexed": 4265,
  "parser_loaded": true,
  "auth_enabled": true
}

# Service status
sudo systemctl status impuls-api
sudo systemctl status nginx
```

### 2.2 Restart Services

```bash
# Restart API
sudo systemctl restart impuls-api

# Restart nginx
sudo systemctl restart nginx

# Restart both
sudo systemctl restart impuls-api nginx
```

### 2.3 View Logs

```bash
# API logs (systemd)
sudo journalctl -u impuls-api -f

# API logs (file)
tail -f /opt/impuls/logs/impuls.log

# Nginx logs
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log
```

### 2.4 Resource Monitoring

```bash
# Memory usage (Salamandra model uses ~4GB with quantization)
free -h

# GPU usage
nvidia-smi

# API processes
ps aux | grep gunicorn

# Disk space
df -h /opt/impuls
```

---

## 3. Maintenance Tasks

### 3.1 Update Code

```bash
cd /opt/impuls
source .venv/bin/activate

# Backup configuration
cp .env .env.backup

# Pull updates
git pull origin main

# Reinstall dependencies if changed
pip install -r requirements.txt

# Restart service
sudo systemctl restart impuls-api
```

### 3.2 Update Project Index

When new project data is available:

```bash
cd /opt/impuls
source .venv/bin/activate

# 1. Copy new data
cp /path/to/new/project_db.parquet data/ris3cat/
cp /path/to/new/participant_db.parquet data/ris3cat/

# 2. Rebuild index
python scripts/build_indices/build_index.py \
    --input data/ris3cat/project_db.parquet \
    --index-dir data/index

# 3. Restart API to load new index
sudo systemctl restart impuls-api

# 4. Verify
curl http://localhost:8000/health | jq .index_size
```

### 3.3 Update Knowledge Base

```bash
# 1. Copy new KB
cp /path/to/new/wikidata_kb.jsonl data/kb/

# 2. Restart API
sudo systemctl restart impuls-api

# 3. Verify
curl http://localhost:8000/health | jq .kb_concepts
```

### 3.4 Log Rotation

Configured in `/etc/logrotate.d/impuls`:

```bash
# Force manual rotation
sudo logrotate -f /etc/logrotate.d/impuls

# Test configuration
sudo logrotate -d /etc/logrotate.d/impuls
```

### 3.5 Backup

```bash
# Daily backup script
/opt/impuls/scripts/backup.sh

# Manual backup
tar -czf impuls-backup-$(date +%Y%m%d).tar.gz \
    /opt/impuls/data \
    /opt/impuls/.env \
    /opt/impuls/html
```

---

## 4. Troubleshooting

### 4.1 API Not Responding

```bash
# 1. Check if service is running
sudo systemctl status impuls-api

# 2. If down, check logs
sudo journalctl -u impuls-api -n 100

# 3. Common errors:
#    - "CUDA out of memory" → Restart or enable quantization
#    - "FileNotFoundError" → Verify paths in .env
#    - "Connection refused" → Port busy or firewall

# 4. Restart
sudo systemctl restart impuls-api
```

### 4.2 GPU Memory Errors

```bash
# Check current usage
nvidia-smi

# Kill orphan processes if full
sudo fuser -v /dev/nvidia*

# Enable 4-bit quantization (reduces 14GB to 3.5GB)
# In .env:
QUERY_PARSER_QUANTIZE=true

# Restart
sudo systemctl restart impuls-api
```

### 4.3 Slow Searches

```bash
# Check if parser is loaded
curl http://localhost:8000/health | jq .parser_loaded

# If false, model loads on first query
# First query may take 30-60s

# Monitor CPU/GPU during search
htop  # or nvidia-smi -l 1
```

### 4.4 Incorrect Results

1. **Verify index version**:
   ```bash
   ls -la data/index/
   ```

2. **Verify KB loaded**:
   ```bash
   curl http://localhost:8000/health | jq .kb_concepts
   # Should be 4265
   ```

3. **Test simple query**:
   ```bash
   curl -X POST http://localhost:8000/search \
     -H "Content-Type: application/json" \
     -H "X-API-Key: YOUR_KEY" \
     -d '{"query": "machine learning", "k": 5}'
   ```

---

## 5. Configuration Reference

### 5.1 Environment Variables (.env)

```bash
# API
API_HOST=0.0.0.0
API_PORT=8000
API_KEY=your-secure-key-here
LOG_LEVEL=INFO

# Models
EMBEDDER_MODEL_NAME=nicolauduran45/mRoBERTA_retrieval-scientific_domain
QUERY_PARSER_MODEL=SIRIS-Lab/impuls-salamandra-7b-query-parser
QUERY_PARSER_QUANTIZE=true
QUERY_PARSER_ENABLED=true
QUERY_PARSER_PROMPT=/opt/impuls/data/training/salamandra_inference_prompt.txt

# Data
INDEX_DIR=/opt/impuls/data/index
KB_PATH=/opt/impuls/data/kb/wikidata_kb.jsonl
PROJECTS_METADATA_PATH=/opt/impuls/data/ris3cat/project_db.parquet
PARTICIPANTS_METADATA_PATH=/opt/impuls/data/ris3cat/participant_db.parquet

# HNSW
HNSW_M=16
HNSW_EF_CONSTRUCTION=200
HNSW_EF_SEARCH=100
```

### 5.2 Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 16 GB | 32 GB |
| GPU VRAM | 8 GB (4-bit) | 16 GB |
| Disk | 20 GB | 50 GB SSD |

---

## 6. Maintenance Checklist

### Daily
- [ ] Verify `/health` returns OK
- [ ] Review logs for errors

### Weekly
- [ ] Check disk space
- [ ] Review memory/GPU usage
- [ ] Verify backups

### Monthly
- [ ] Update security dependencies
- [ ] Review usage metrics
- [ ] Rotate API keys if needed

### Quarterly
- [ ] Update project data (if available)
- [ ] Review and update documentation
- [ ] Test backup recovery

---

## 7. Support and Escalation

### Resources

- **Repository**: https://github.com/sirisacademic/aina-impulse
- **Issues**: https://github.com/sirisacademic/aina-impulse/issues
- **HuggingFace Models**: https://huggingface.co/SIRIS-Lab

---

*Last updated: December 2024*
