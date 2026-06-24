#!/usr/bin/env bash
#
# dev.sh — start the Nabla backend (FastAPI/uvicorn) and frontend (Vite) together.
# Press Ctrl+C once to stop both cleanly.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT/backend"
FRONTEND_DIR="$ROOT/frontend"

# ── Pre-flight checks ──────────────────────────────────────────────────────
if [ ! -f "$BACKEND_DIR/.venv/bin/activate" ]; then
  echo "Error: backend virtualenv not found at backend/.venv"
  echo "Create it with:"
  echo "  python3 -m venv backend/.venv"
  echo "  backend/.venv/bin/pip install -r backend/requirements.txt"
  exit 1
fi

if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
  echo "Error: frontend dependencies not installed."
  echo "Install them with:  (cd frontend && npm install)"
  exit 1
fi

backend_pid=""
frontend_pid=""
cleaned=0

cleanup() {
  [ "$cleaned" -eq 1 ] && return   # guard against the INT+EXIT double-fire
  cleaned=1
  echo ""
  echo "Stopping Nabla dev servers..."
  # SIGTERM each server and its children (e.g. uvicorn's reload worker).
  for pid in "$frontend_pid" "$backend_pid"; do
    [ -n "$pid" ] || continue
    pkill -TERM -P "$pid" 2>/dev/null
    kill -TERM "$pid" 2>/dev/null
  done
  # Grace period, then force-kill anything still standing.
  sleep 1
  for pid in "$frontend_pid" "$backend_pid"; do
    [ -n "$pid" ] || continue
    pkill -KILL -P "$pid" 2>/dev/null
    kill -KILL "$pid" 2>/dev/null
  done
  wait 2>/dev/null
  echo "Done."
}

# Ctrl+C (INT) / TERM / normal exit all trigger a single cleanup.
trap cleanup INT TERM EXIT

# ── Backend: activate venv, run uvicorn with autoreload ────────────────────
# `exec` so the backgrounded subshell *becomes* uvicorn, making $! its real PID.
( cd "$BACKEND_DIR" && source .venv/bin/activate && exec uvicorn app.main:app --reload --reload-dir app --port 8000 ) &
backend_pid=$!
echo "▶ backend   uvicorn  →  http://localhost:8000   (pid $backend_pid)"

# ── Frontend: Vite dev server ──────────────────────────────────────────────
# Run the vite binary directly (not via npm) so signals reach it without an
# npm wrapper process orphaning the child.
( cd "$FRONTEND_DIR" && exec node_modules/.bin/vite ) &
frontend_pid=$!
echo "▶ frontend  vite     →  http://localhost:5173   (pid $frontend_pid)"

echo ""
echo "Both servers running. Press Ctrl+C to stop them."

# Block until the servers exit (or Ctrl+C fires the trap).
wait
