#!/usr/bin/env bash
set -euo pipefail

# Configuration
API_PORT=${API_PORT:-8000}
HTML_PORT=${HTML_PORT:-8080}

# Load .env to get port overrides
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "Stopping IMPULS services..."

# Function to stop a service by PID file and/or port
stop_service() {
    local name="$1"
    local pid_file="$2"
    local port="$3"
    local process_pattern="$4"
    
    local stopped=false
    
    # Try PID file first
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null && echo "✓ $name stopped (PID: $pid)" && stopped=true
        fi
        rm -f "$pid_file"
    fi
    
    # If not stopped, try to find by port
    if [ "$stopped" = false ] && [ -n "$port" ]; then
        local port_pid=$(lsof -ti ":$port" 2>/dev/null || true)
        if [ -n "$port_pid" ]; then
            kill $port_pid 2>/dev/null && echo "✓ $name stopped (port $port, PID: $port_pid)" && stopped=true
        fi
    fi
    
    # If still not stopped, try by process pattern
    if [ "$stopped" = false ] && [ -n "$process_pattern" ]; then
        local pattern_pids=$(pgrep -f "$process_pattern" 2>/dev/null || true)
        if [ -n "$pattern_pids" ]; then
            echo "$pattern_pids" | xargs kill 2>/dev/null && echo "✓ $name stopped (by pattern)" && stopped=true
        fi
    fi
    
    if [ "$stopped" = false ]; then
        echo "○ $name was not running"
    fi
}

# Stop API server
stop_service "API" "logs/api.pid" "$API_PORT" "uvicorn.*impulse"

# Stop HTML server
stop_service "HTML server" "logs/html.pid" "$HTML_PORT" "http.server.*$HTML_PORT"

# Clean up any stale PID files
rm -f logs/*.pid 2>/dev/null || true

echo "Services stopped"
