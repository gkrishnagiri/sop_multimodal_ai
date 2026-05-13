#!/bin/bash

#!/bin/bash

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"

cd "$PROJECT_ROOT"

export UV_LINK_MODE=copy

echo "Starting FastAPI backend on port 8015..."
echo "Watching: $BACKEND_DIR"

uv --project backend run uvicorn backend.main:app \
  --host 127.0.0.1 \
  --port 8015 \
  --reload \
  --reload-dir "$BACKEND_DIR"