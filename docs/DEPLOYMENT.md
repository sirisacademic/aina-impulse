# IMPULS Deployment Guide

This guide covers deploying IMPULS in development and production environments.

## System Requirements

### Minimum (API without Query Parser)
- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 10GB
- **GPU**: Not required

### Recommended (Full System with Query Parser)
- **CPU**: 8+ cores
- **RAM**: 32GB
- **Storage**: 50GB SSD
- **GPU**: NVIDIA with 8GB+ VRAM (for query parser)
  - With 4-bit quantization: 8GB VRAM
  - Without quantization: 16GB+ VRAM

### Production Server (Current Deployment)
- **Server**: Hetzner dedicated server
- **CPU**: AMD Ryzen 9 (16 cores)
- **RAM**: 64GB
- **GPU**: NVIDIA RTX 4090 (24GB)
- **Storage**: 1TB NVMe SSD

## Environment Configuration

### Development (.env.dev)

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8001
API_KEY=dev-api-key-change-in-production
LOG_LEVEL=DEBUG

# Models
EMBEDDER_MODEL_NAME=langtech-innovation/mRoBERTA_retrieval
QUERY_PARSER_MODEL=BSC-LT/salamandra-7b-instruct-tools
QUERY_PARSER_QUANTIZE=true
QUERY_PARSER_ENABLED=true

# Data Paths
INDEX_DIR=data/index
KB_PATH=data/kb/wikidata_kb.jsonl
PROJECTS_METADATA_PATH=data/ris3cat/project_db.parquet
PARTICIPANTS_METADATA_PATH=data/ris3cat/participant_db.parquet

# HNSW Index Parameters
HNSW_M=16
HNSW_EF_CONSTRUCTION=200
HNSW_EF_SEARCH=100

# CORS (for local UI development)
CORS_ORIGINS=http://localhost:8080,http://127.0.0.1:8080
```

### Production (.env.prod)

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_KEY=your-secure-production-key
LOG_LEVEL=INFO

# Models
EMBEDDER_MODEL_NAME=langtech-innovation/mRoBERTA_retrieval
QUERY_PARSER_MODEL=BSC-LT/salamandra-7b-instruct-tools
QUERY_PARSER_QUANTIZE=true
QUERY_PARSER_ENABLED=true

# Data Paths
INDEX_DIR=/opt/impuls/data/index
KB_PATH=/opt/impuls/data/kb/wikidata_kb.jsonl
PROJECTS_METADATA_PATH=/opt/impuls/data/ris3cat/project_db.parquet
PARTICIPANTS_METADATA_PATH=/opt/impuls/data/ris3cat/participant_db.parquet

# HNSW Index Parameters
HNSW_M=16
HNSW_EF_CONSTRUCTION=200
HNSW_EF_SEARCH=100

# CORS
CORS_ORIGINS=http://impuls-aina.sirisacademic.com:8080

# Worker Configuration
WORKERS=4
TIMEOUT=120
```

## Deployment Options

### Option 1: Direct Deployment

#### 1. Setup Server

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10+
sudo apt install python3.10 python3.10-venv python3-pip -y

# Install NVIDIA drivers (if using GPU)
sudo apt install nvidia-driver-535 -y

# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-1 -y
```

#### 2. Clone and Setup Application

```bash
# Create application directory
sudo mkdir -p /opt/impuls
sudo chown $USER:$USER /opt/impuls
cd /opt/impuls

# Clone repository
git clone https://github.com/sirisacademic/aina-impulse.git .

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install PyTorch with CUDA (if using GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

#### 3. Prepare Data

```bash
# Create data directories
mkdir -p data/{index,kb,ris3cat,normalization}

# Copy data files (from your data source)
cp /path/to/project_db.parquet data/ris3cat/
cp /path/to/participant_db.parquet data/ris3cat/
cp /path/to/wikidata_kb.jsonl data/kb/

# Build vector index
python scripts/build_indices/build_index.py \
  --input data/ris3cat/project_db.parquet \
  --index-dir data/index
```

#### 4. Configure Environment

```bash
# Copy production environment
cp .env.prod .env

# Edit configuration
nano .env
# Set API_KEY and other production values
```

#### 5. Run with Gunicorn (Production)

```bash
# Install gunicorn
pip install gunicorn uvicorn[standard]

# Run API
gunicorn src.impulse.api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --access-logfile /var/log/impuls/access.log \
  --error-logfile /var/log/impuls/error.log
```

### Option 2: Systemd Service

Create `/etc/systemd/system/impuls-api.service`:

```ini
[Unit]
Description=IMPULS API Service
After=network.target

[Service]
Type=simple
User=impuls
Group=impuls
WorkingDirectory=/opt/impuls
Environment="PATH=/opt/impuls/.venv/bin"
EnvironmentFile=/opt/impuls/.env
ExecStart=/opt/impuls/.venv/bin/gunicorn src.impulse.api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable impuls-api
sudo systemctl start impuls-api
sudo systemctl status impuls-api
```

### Option 3: Docker Deployment

#### Dockerfile

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt')"

# Expose port
EXPOSE 8000

# Run application
CMD ["gunicorn", "src.impulse.api.main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120"]
```

#### docker-compose.yml

```yaml
version: '3.8'

services:
  impuls-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./.env:/app/.env
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  impuls-ui:
    image: nginx:alpine
    ports:
      - "8080:80"
    volumes:
      - ./html:/usr/share/nginx/html:ro
    restart: unless-stopped
```

Build and run:

```bash
docker-compose up -d --build
```

## Web UI Deployment

### Nginx Configuration

Create `/etc/nginx/sites-available/impuls`:

```nginx
server {
    listen 8080;
    server_name impuls-aina.sirisacademic.com;

    root /opt/impuls/html;
    index impuls_ui.html;

    location / {
        try_files $uri $uri/ =404;
    }

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 7d;
        add_header Cache-Control "public, immutable";
    }
}
```

Enable site:

```bash
sudo ln -s /etc/nginx/sites-available/impuls /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Update UI Configuration

Edit `html/impuls_ui.html` and update:

```javascript
const API_URL = 'http://impuls-aina.sirisacademic.com:8000';
const API_KEY = 'your-production-api-key';
```

## Monitoring

### Health Check Script

Create `/opt/impuls/scripts/health_check.sh`:

```bash
#!/bin/bash

API_URL="http://localhost:8000"
SLACK_WEBHOOK="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

response=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/health")

if [ "$response" != "200" ]; then
    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"⚠️ IMPULS API is down! HTTP status: '"$response"'"}' \
        "$SLACK_WEBHOOK"
fi
```

Add to crontab:

```bash
*/5 * * * * /opt/impuls/scripts/health_check.sh
```

### Log Rotation

Create `/etc/logrotate.d/impuls`:

```
/var/log/impuls/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 impuls impuls
    sharedscripts
    postrotate
        systemctl reload impuls-api > /dev/null 2>&1 || true
    endscript
}
```

## Backup Strategy

### Data Backup Script

```bash
#!/bin/bash
# /opt/impuls/scripts/backup.sh

BACKUP_DIR="/backup/impuls"
DATE=$(date +%Y%m%d)

mkdir -p "$BACKUP_DIR"

# Backup index and data
tar -czf "$BACKUP_DIR/impuls-data-$DATE.tar.gz" \
    /opt/impuls/data/index \
    /opt/impuls/data/kb \
    /opt/impuls/data/ris3cat

# Keep only last 7 days
find "$BACKUP_DIR" -name "impuls-data-*.tar.gz" -mtime +7 -delete
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```bash
# Enable 4-bit quantization
QUERY_PARSER_QUANTIZE=true

# Or disable parser if not needed
QUERY_PARSER_ENABLED=false
```

#### 2. Slow Startup

First startup downloads models (~5-10GB). Subsequent starts are faster as models are cached in `~/.cache/huggingface/`.

#### 3. API Timeout

Increase timeout for complex queries:

```bash
# In gunicorn
--timeout 180

# In nginx (if using reverse proxy)
proxy_read_timeout 180s;
```

#### 4. Permission Errors

```bash
sudo chown -R impuls:impuls /opt/impuls
sudo chmod -R 755 /opt/impuls
```

### Logs

```bash
# Application logs
tail -f /var/log/impuls/error.log

# Systemd logs
journalctl -u impuls-api -f

# Docker logs
docker-compose logs -f impuls-api
```

## Security Checklist

- [ ] Change default API key
- [ ] Enable HTTPS (use reverse proxy with SSL)
- [ ] Restrict CORS origins
- [ ] Set up firewall (only allow 8000, 8080)
- [ ] Use non-root user for service
- [ ] Keep dependencies updated
- [ ] Enable log monitoring
- [ ] Regular backups

## Current Production URLs

| Service | URL |
|---------|-----|
| Web UI | http://impuls-aina.sirisacademic.com:8080/impuls_ui.html |
| API | http://impuls-aina.sirisacademic.com:8000 |
| API Docs | http://impuls-aina.sirisacademic.com:8000/docs |
