#!/usr/bin/env bash

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

echo "Starting SOP Multimodal AI frontend..."
echo "Project root: $PROJECT_ROOT"
echo "Frontend directory: $FRONTEND_DIR"

if [ ! -d "$FRONTEND_DIR" ]; then
  echo "ERROR: frontend directory not found: $FRONTEND_DIR"
  exit 1
fi

cd "$FRONTEND_DIR"

if [ ! -d "node_modules" ]; then
  echo "node_modules not found. Running npm install..."
  npm install
fi

echo "Starting frontend on http://localhost:3015"
npm run dev