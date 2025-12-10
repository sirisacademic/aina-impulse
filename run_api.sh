#!/usr/bin/env bash
set -euo pipefail

# Configuration (NOTE: Overriden by values in .env file)
API_PORT=${API_PORT:-8000}
HTML_PORT=${HTML_PORT:-8080}
API_HOST=${API_HOST:-0.0.0.0}
HTML_FILE=${HTML_FILE:-impuls_ui.html}
INIT_JS_FILE=${INIT_JS_FILE:-impuls_config.js}

# Load .env to override defaults
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Determine browser-accessible host
if [ "$API_HOST" = "0.0.0.0" ] || [ "$API_HOST" = "::" ]; then
    BROWSER_HOST="impuls-aina.sirisacademic.com"  # or use: $(hostname -I | awk '{print $1}')
else
    BROWSER_HOST="$API_HOST"
fi

mkdir -p logs

# Function to check if port is in use
port_in_use() {
    lsof -ti ":$1" >/dev/null 2>&1
}

# Function to stop process on port
stop_port() {
    local port="$1"
    local name="$2"
    if port_in_use "$port"; then
        echo "⚠ Port $port in use, stopping existing $name..."
        local pid=$(lsof -ti ":$port")
        kill $pid 2>/dev/null || true
        sleep 1
        # Force kill if still running
        if port_in_use "$port"; then
            kill -9 $(lsof -ti ":$port") 2>/dev/null || true
            sleep 1
        fi
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    local url="$1"
    local name="$2"
    local max_attempts="${3:-30}"
    
    echo -n "Waiting for $name..."
    for i in $(seq 1 $max_attempts); do
        if curl -s "$url" >/dev/null 2>&1; then
            echo " ready"
            return 0
        fi
        echo -n "."
        sleep 1
    done
    echo " timeout!"
    return 1
}

echo "Starting IMPULS services..."

# Stop any existing services on our ports
stop_port "$API_PORT" "API"
stop_port "$HTML_PORT" "HTML server"

# Update API URL in HTML
sed -i "s|const API_URL = '.*';|const API_URL = 'http://${BROWSER_HOST}:${API_PORT}';|" "html/js/${INIT_JS_FILE}"

# Start API server
nohup python run_api.py --host "$API_HOST" --port "$API_PORT" > logs/api.log 2>&1 &
API_PID=$!
echo "$API_PID" > logs/api.pid
echo "API: http://${BROWSER_HOST}:${API_PORT} (PID: $API_PID)"

# Wait for API to be ready
if ! wait_for_service "http://localhost:${API_PORT}/health" "API"; then
    echo "ERROR: API failed to start. Check logs/api.log"
    exit 1
fi

# Verify HTML has correct API URL
INIT_JS_API_URL=$(grep -oP "const API_URL = '\K[^']*" "html/js/${INIT_JS_FILE}" | head -n1 || echo "NOT_FOUND")
EXPECTED_URL="http://${BROWSER_HOST}:${API_PORT}"

if [ "$INIT_JS_API_URL" != "$EXPECTED_URL" ]; then
    echo "WARNING: HTML API_URL mismatch!"
    echo "  Expected: $EXPECTED_URL"
    echo "  Found:    $INIT_JS_API_URL"
    echo "  Fix with: sed -i \"s|const API_URL = '.*';|const API_URL = '$EXPECTED_URL';|\" html/js/${INIT_JS_FILE}"
    exit 1
else
    echo "✓ HTML API_URL matches: $EXPECTED_URL"
fi

# Start HTML server
nohup python -m http.server "$HTML_PORT" --directory html > logs/html.log 2>&1 &
HTML_PID=$!
echo "$HTML_PID" > logs/html.pid
echo "Web UI: http://${BROWSER_HOST}:${HTML_PORT}/${HTML_FILE} (PID: $HTML_PID)"

# Verify HTML server started
sleep 1
if ! port_in_use "$HTML_PORT"; then
    echo "ERROR: HTML server failed to start. Check logs/html.log"
    exit 1
fi

echo ""
echo "Services running:"
echo "  API docs: http://${BROWSER_HOST}:${API_PORT}/docs"
echo "  Web UI:   http://${BROWSER_HOST}:${HTML_PORT}/${HTML_FILE}"
echo "  Logs:     logs/"
echo "  Stop:     ./stop_api.sh"
