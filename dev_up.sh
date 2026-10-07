#!/usr/bin/env bash
# ==============================================================================
# dev_up.sh - Predictive Geofencing Development Environment Launcher
# Launches FastAPI backend and React/Vite frontend with interleaved color logs.
# ==============================================================================

set -m  # Enable job monitor mode so background pipelines get distinct PGIDs

# Color definitions
CYAN="\033[0;36m"
MAGENTA="\033[0;35m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
BOLD="\033[1m"
NC="\033[0m" # No Color

# Determine script directory (repo root)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR" || exit 1

echo -e "${BOLD}--- Initializing Predictive Geofencing Development Environment ---${NC}"

# ------------------------------------------------------------------------------
# 1. Environment & Prerequisite Checks
# ------------------------------------------------------------------------------

# Locate backend virtualenv
VENV_DIR=""
if [ -d "$ROOT_DIR/backend/.venv" ]; then
    VENV_DIR="$ROOT_DIR/backend/.venv"
elif [ -d "$ROOT_DIR/backend/venv" ]; then
    VENV_DIR="$ROOT_DIR/backend/venv"
elif [ -d "$ROOT_DIR/.venv" ]; then
    VENV_DIR="$ROOT_DIR/.venv"
elif [ -d "$ROOT_DIR/venv" ]; then
    VENV_DIR="$ROOT_DIR/venv"
fi

if [ -z "$VENV_DIR" ]; then
    echo -e "${RED}[ERROR]${NC} Backend virtual environment not found!"
    echo "Expected a virtualenv at 'backend/.venv' (recommended) or 'backend/venv'."
    echo ""
    echo "Please set up the backend environment first:"
    echo "    cd backend"
    echo "    python -m venv .venv"
    echo "    source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows"
    echo "    pip install -r requirements.txt"
    echo "    cd .."
    exit 1
fi

# Detect activation script (POSIX bin/activate vs Windows Scripts/activate)
ACTIVATE_SCRIPT=""
if [ -f "$VENV_DIR/bin/activate" ]; then
    ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
    ACTIVATE_SCRIPT="$VENV_DIR/Scripts/activate"
else
    echo -e "${RED}[ERROR]${NC} Activation script not found inside virtualenv at $VENV_DIR."
    echo "Please recreate your virtual environment:"
    echo "    cd backend && python -m venv .venv && cd .."
    exit 1
fi

# Activate virtualenv
# shellcheck source=/dev/null
source "$ACTIVATE_SCRIPT"

# Validate backend dependencies
if ! python -c "import fastapi, uvicorn" 2>/dev/null; then
    echo -e "${RED}[ERROR]${NC} Required dependencies (fastapi/uvicorn) missing in $VENV_DIR."
    echo "Please install backend dependencies:"
    echo "    cd backend && pip install -r requirements.txt && cd .."
    exit 1
fi

# Validate frontend dependencies
if [ ! -d "$ROOT_DIR/frontend/node_modules" ]; then
    echo -e "${RED}[ERROR]${NC} Frontend dependencies not found in frontend/node_modules."
    echo "Please install frontend dependencies first:"
    echo "    cd frontend && npm install && cd .."
    exit 1
fi

# ------------------------------------------------------------------------------
# 2. Process Cleanup Handler
# ------------------------------------------------------------------------------

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
    # Disable trap during shutdown to avoid recursion
    trap - SIGINT SIGTERM EXIT
    echo ""
    echo -e "${YELLOW}Shutting down background services...${NC}"

    # Layer 1: Kill process groups created via set -m
    if [ -n "$BACKEND_PID" ]; then
        kill -TERM -"$BACKEND_PID" 2>/dev/null || kill "$BACKEND_PID" 2>/dev/null || true
    fi
    if [ -n "$FRONTEND_PID" ]; then
        kill -TERM -"$FRONTEND_PID" 2>/dev/null || kill "$FRONTEND_PID" 2>/dev/null || true
    fi

    sleep 0.5

    # Layer 2: POSIX fallback with pkill if available
    if command -v pkill >/dev/null 2>&1; then
        pkill -f "uvicorn.*app.main:app" 2>/dev/null || true
        pkill -f "vite" 2>/dev/null || true
    fi

    # Layer 3: Windows fallback via netstat + taskkill for listening ports
    if command -v netstat >/dev/null 2>&1 && command -v taskkill >/dev/null 2>&1; then
        for port in 8000 5173; do
            pids=$(netstat -ano 2>/dev/null | grep ":$port " | grep "LISTENING" | awk '{print $NF}' | sort -u)
            for p in $pids; do
                if [ -n "$p" ] && [ "$p" != "0" ]; then
                    taskkill //F //PID "$p" 2>/dev/null || true
                fi
            done
        done
    fi

    echo -e "${GREEN}Both services stopped cleanly.${NC}"
}

trap cleanup SIGINT SIGTERM EXIT

# ------------------------------------------------------------------------------
# 3. Log Stream Formatters
# ------------------------------------------------------------------------------

prefix_backend() {
    while IFS= read -r line || [ -n "$line" ]; do
        echo -e "${CYAN}[BACKEND]${NC}  $line"
    done
}

prefix_frontend() {
    while IFS= read -r line || [ -n "$line" ]; do
        echo -e "${MAGENTA}[FRONTEND]${NC} $line"
    done
}

# ------------------------------------------------------------------------------
# 4. Launch Services in Background
# ------------------------------------------------------------------------------

echo -e "${YELLOW}Starting backend service (FastAPI on port 8000)...${NC}"
(
    cd "$ROOT_DIR/backend" || exit 1
    python -m uvicorn app.main:app --reload --port 8000 2>&1 | prefix_backend
) &
BACKEND_PID=$!

echo -e "${YELLOW}Starting frontend service (Vite on port 5173)...${NC}"
(
    cd "$ROOT_DIR/frontend" || exit 1
    npm run dev 2>&1 | prefix_frontend
) &
FRONTEND_PID=$!

# ------------------------------------------------------------------------------
# 5. Active Health Polling
# ------------------------------------------------------------------------------

echo -e "${YELLOW}Waiting for backend health check (http://localhost:8000/health)...${NC}"
BACKEND_READY=false
for _ in $(seq 1 30); do
    if curl -s -f http://localhost:8000/health >/dev/null 2>&1; then
        BACKEND_READY=true
        break
    fi
    sleep 0.5
done

if [ "$BACKEND_READY" = false ]; then
    echo -e "${RED}[ERROR] Backend failed to become healthy within 15 seconds.${NC}"
    exit 1
fi
echo -e "${GREEN}[BACKEND READY]${NC} Health check passed."

echo -e "${YELLOW}Waiting for frontend dev server (http://localhost:5173/)...${NC}"
FRONTEND_READY=false
for _ in $(seq 1 30); do
    if curl -s -f http://localhost:5173/ >/dev/null 2>&1; then
        FRONTEND_READY=true
        break
    fi
    sleep 0.5
done

if [ "$FRONTEND_READY" = false ]; then
    echo -e "${RED}[ERROR] Frontend failed to respond within 15 seconds.${NC}"
    exit 1
fi
echo -e "${GREEN}[FRONTEND READY]${NC} Vite dev server responsive."

# ------------------------------------------------------------------------------
# 6. Operational Summary Banner
# ------------------------------------------------------------------------------

echo ""
echo -e "${GREEN}========================================================================${NC}"
echo -e "${BOLD}  Predictive Geofencing Development Environment Ready!${NC}"
echo -e "${GREEN}========================================================================${NC}"
echo -e "  ${BOLD}Backend API:${NC}      http://localhost:8000  (Docs: http://localhost:8000/docs)"
echo -e "  ${BOLD}Frontend App:${NC}     http://localhost:5173"
echo -e "------------------------------------------------------------------------"
echo -e "  ${YELLOW}NOTE:${NC} Run the trajectory simulator separately in another terminal:"
echo -e "        python backend/scripts/trajectory_simulator.py --user-id sim_01 --pattern lapping ..."
echo -e "------------------------------------------------------------------------"
echo -e "  Press ${BOLD}Ctrl+C${NC} to stop both services."
echo -e "${GREEN}========================================================================${NC}"
echo ""

# Keep alive to stream logs until interrupted
wait
