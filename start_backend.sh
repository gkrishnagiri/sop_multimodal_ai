#!/usr/bin/env bash

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"

cd "$PROJECT_ROOT"

export UV_LINK_MODE=copy

echo "Starting FastAPI backend on port 8015..."
echo "Project root: $PROJECT_ROOT"
echo "Backend directory: $BACKEND_DIR"
echo "Auto-reload: disabled"
echo "Backend URL: http://127.0.0.1:8015"
echo "Swagger UI: http://127.0.0.1:8015/docs"

uv --project backend run uvicorn backend.main:app \
  --host 127.0.0.1 \
  --port 8015