#!/usr/bin/env bash
set -euo pipefail

# Configuration (NOTE: Overriden by values in .env file - see below!!)
API_PORT=${API_PORT:-8000}
HTML_PORT=${HTML_PORT:-8080}
API_HOST=${API_HOST:-0.0.0.0}
HTML_FILE=${HTML_FILE:-impuls_ui.html}

# Determine browser-accessible host
if [ "$API_HOST" = "0.0.0.0" ] || [ "$API_HOST" = "::" ]; then
    BROWSER_HOST="65.21.33.94"  # or use: $(hostname -I | awk '{print $1}')
else
    BROWSER_HOST="$API_HOST"
fi

# Load .env to override defaults
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

mkdir -p logs

# Update API URL in HTML
sed -i "s|const API_URL = '.*';|const API_URL = 'http://${BROWSER_HOST}:${API_PORT}';|" "html/${HTML_FILE}"

echo "Starting IMPULS services..."

# Start API server
nohup python run_api.py --host "$API_HOST" --port "$API_PORT" > logs/api.log 2>&1 &
API_PID=$!
echo "$API_PID" > logs/api.pid
echo "API: http://${BROWSER_HOST}:${API_PORT} (PID: $API_PID)"

# Wait for API to be ready
echo -n "Waiting for API..."
for i in {1..30}; do
    if curl -s "http://localhost:${API_PORT}/health" >/dev/null 2>&1; then
        echo " ready"
        break
    fi
    echo -n "."
    sleep 1
done

# Verify HTML has correct API URL
HTML_API_URL=$(grep -oP "const API_URL = '\K[^']*" "html/${HTML_FILE}" | head -n1 || echo "NOT_FOUND")

EXPECTED_URL="http://${BROWSER_HOST}:${API_PORT}"

if [ "$HTML_API_URL" != "$EXPECTED_URL" ]; then
    echo "WARNING: HTML API_URL mismatch!"
    echo "  Expected: $EXPECTED_URL"
    echo "  Found:    $HTML_API_URL"
    echo "  Fix with: sed -i \"s|const API_URL = '.*';|const API_URL = '$EXPECTED_URL';|\" html/${HTML_FILE}"
    exit 1
else
    echo "✓ HTML API_URL matches: $EXPECTED_URL"
fi

# Start HTML server
nohup python -m http.server "$HTML_PORT" --directory html > logs/html.log 2>&1 &
HTML_PID=$!
echo "$HTML_PID" > logs/html.pid
echo "Web UI: http://${BROWSER_HOST}:${HTML_PORT}/${HTML_FILE} (PID: $HTML_PID)"

echo ""
echo "Services running:"
echo "  API docs: http://${BROWSER_HOST}:${API_PORT}/docs"
echo "  Web UI:   http://${BROWSER_HOST}:${HTML_PORT}/${HTML_FILE}"
echo "  Logs:     logs/"
echo "  Stop:     kill \$(cat logs/*.pid)"
