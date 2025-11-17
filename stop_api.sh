#!/usr/bin/env bash
set -euo pipefail

echo "Stopping IMPULS services..."

if [ -f logs/api.pid ]; then
    API_PID=$(cat logs/api.pid)
    if kill "$API_PID" 2>/dev/null; then
        echo "✓ API stopped (PID: $API_PID)"
    else
        echo "✗ API process not found"
    fi
    rm logs/api.pid
fi

if [ -f logs/html.pid ]; then
    HTML_PID=$(cat logs/html.pid)
    if kill "$HTML_PID" 2>/dev/null; then
        echo "✓ HTML server stopped (PID: $HTML_PID)"
    else
        echo "✗ HTML server process not found"
    fi
    rm logs/html.pid
fi

echo "Services stopped"

