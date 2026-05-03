#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}  Dapr Deep Research — Full Setup${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# ------------------------------------------------------------------
# Prerequisites
# ------------------------------------------------------------------
echo -e "${YELLOW}[1/6] Checking prerequisites...${NC}"

command -v docker >/dev/null 2>&1 || { echo -e "${RED}Error: docker is required${NC}"; exit 1; }
echo -e "  ${GREEN}✓${NC} docker"

command -v uv >/dev/null 2>&1 || { echo -e "${RED}Error: uv is required (install: curl -LsSf https://astral.sh/uv/install.sh | sh)${NC}"; exit 1; }
echo -e "  ${GREEN}✓${NC} uv"

if command -v dapr >/dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} dapr CLI"
else
    echo -e "  ${YELLOW}⚠  dapr CLI not found (install: https://docs.dapr.io/getting-started/install-dapr-cli/)${NC}"
    echo -e "  ${YELLOW}   The 'dapr run' commands will not work. No-infra commands (run, mission, chat) still work.${NC}"
fi

# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[2/6] Setting up environment...${NC}"

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "  ${GREEN}✓${NC} Created .env from .env.example"
        echo -e "  ${YELLOW}⚠  Edit .env and set DEEPSEEK_API_KEY before running research commands${NC}"
    else
        echo -e "  ${YELLOW}⚠  No .env.example found — create .env manually with DEEPSEEK_API_KEY${NC}"
    fi
else
    echo -e "  ${GREEN}✓${NC} .env already exists"
fi

# ------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[3/6] Installing Python dependencies...${NC}"
uv sync
echo -e "  ${GREEN}✓${NC} uv sync complete"

# ------------------------------------------------------------------
# Docker infrastructure
# ------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[4/6] Starting Docker infrastructure...${NC}"

if docker info >/dev/null 2>&1; then
    docker compose -f lab/10_dapr_deep_research/docker-compose.yml up -d 2>&1 || true
    echo -e "  ${GREEN}✓${NC} Crawl4AI started (port 11235)"

    # Check if dapr redis/zipkin are already running (from dapr init)
    if docker ps --format '{{.Names}}' | grep -q dapr_redis; then
        echo -e "  ${GREEN}✓${NC} Dapr Redis already running"
    fi
else
    echo -e "  ${YELLOW}⚠  Docker not running — start Docker and re-run for full infrastructure${NC}"
fi

# ------------------------------------------------------------------
# Dapr init
# ------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[5/6] Initializing Dapr...${NC}"

if command -v dapr >/dev/null 2>&1; then
    if docker ps --format '{{.Names}}' | grep -q dapr_placement; then
        echo -e "  ${GREEN}✓${NC} Dapr already initialized"
    else
        dapr init 2>&1 || echo -e "  ${YELLOW}⚠  dapr init failed (run manually: dapr init)${NC}"
    fi
else
    echo -e "  ${YELLOW}⚠  dapr CLI not found — skip dapr init${NC}"
fi

# ------------------------------------------------------------------
# Verify
# ------------------------------------------------------------------
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}  Setup Complete${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Print infrastructure status
echo -e "Infrastructure:"
for svc in crawl4ai dapr_placement dapr_redis dapr_zipkin; do
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "$svc"; then
        echo -e "  ${GREEN}✓${NC} $svc running"
    else
        echo -e "  ${YELLOW}  $svc not running${NC}"
    fi
done

echo ""
echo -e "Commands:"
echo -e "  ${CYAN}No infrastructure needed:${NC}"
echo -e "    uv run python -m lab.10_dapr_deep_research --help"
echo -e "    uv run python -m lab.10_dapr_deep_research --query \"<topic>\" run"
echo -e "    uv run python -m lab.10_dapr_deep_research --query \"<topic>\" --iterations 5 mission"
echo -e "    uv run python -m lab.10_dapr_deep_research chat"
echo ""
echo -e "  ${CYAN}With Dapr sidecar (dapr run):${NC}"
echo -e "    dapr run -f lab/10_dapr_deep_research/dapr-multi-app-run.yaml"
echo ""
echo -e "  ${CYAN}Distillation (requires Dapr + Ollama):${NC}"
echo -e "    ollama pull gemma4"
echo -e "    uv run python -m lab.10_dapr_deep_research distill"
echo ""
echo -e "  ${CYAN}Quick start:${NC}"
echo -e "    uv run python -m lab.10_dapr_deep_research --query \"DSPy optimization patterns\" mission"
