# 09 — Super Deep Research: Self-Evolving Agentic Platform

Multi-agent research platform with LSE meta-optimization, autonomous discovery, persistent knowledge graph, and collective intelligence via OpenRouter MCP.

## Architecture

```
mcp_servers.json
  ├── crawl4ai (SSE)     →  deep web scraping
  ├── fetch (stdio)       →  URL fetching
  ├── openrouter (stdio)  →  100+ models, ensemble reasoning, consensus
  └── filesystem (stdio)  →  artifact persistence

ResearchFrontier (UCB priority queue)
  └── Orchestrator (LSE loop)
        ├── Explorer      →  discovers directions (search tools)
        ├── DeepReader    →  deep content analysis (fetch tools)
        ├── Synthesizer   →  cross-source synthesis
        ├── Critic        →  gap identification, quality evaluation
        └── SkillAuthor   →  Trace2Skill parallel consolidation

Memory:
  ├── knowledge_graph.json   →  findings + typed relationships
  ├── skills/                →  reusable DSPy demonstrations
  ├── logs/                  →  execution trajectories
  └── frontier.json          →  research state across runs

Evolution:
  ├── LSE          →  improvement-based reward trains orchestrator
  ├── SelfDistill  →  hindsight distribution from execution feedback
  └── Trace2Skill  →  parallel patch proposal + conflict-free merge
```

## Prerequisites

```bash
# Crawl4AI
docker compose -f lab/09-super-deep-research/docker-compose.yml up -d

# OpenRouter MCP (requires Node.js 16+)
npx @physics91/openrouter-mcp init

# Set API keys in project root .env
# OPENROUTER_API_KEY=sk-or-...
# DEEPSEEK_API_KEY=...
```

## Running

```bash
python -m lab.09-super-deep-research.main
```

## MCP Servers

| Server | Transport | Tools |
|--------|-----------|-------|
| `crawl4ai` | SSE | `md`, `html`, `crawl`, `screenshot` |
| `fetch` | stdio | `fetch` |
| `openrouter` | stdio | `chat`, `model_list`, `consensus`, `ensemble`, `usage_stats` |
| `filesystem` | stdio | `read`, `write`, `search` |

## Key Patterns

- **LSE**: orchestrator strategy improves across runs via r = quality(c₁) − quality(c₀)
- **ResearchFrontier**: UCB-based topic selection for autonomous exploration
- **CORAL heartbeat**: periodic reflection + consolidation + stagnation redirection
- **Trace2Skill**: trajectories → parallel patches → conflict-free skill
- **Self-distillation**: SDPO-style conditioning on execution history
- **OpenRouter collective intelligence**: ensemble reasoning, cross-model validation
