# experiments

Lab / showroom for AI agent experiments with **DSPy**, **MCP**, and **dapr-agents**.

## Lab Structure

```
lab/
├── 01-basics/                     # Signatures, Predict, ChainOfThought
├── 02-react-tools/                # ReAct agent loop with tools
├── 03-rag-pipeline/               # RAG + BootstrapFewShot optimization
├── 04-optimizers/                 # MIPROv2, GEPA, BetterTogether
├── 05-rlm/                        # Recursive Language Model (REPL)
├── 06-advanced/                   # MultiChainComparison, Streaming, Async
├── 07-generative-feedback-loops/  # GFL optimizers comparison
├── 08-rlm-mcp/                    # RLM + MCP tools + BAMLAdapter
├── 09_super_deep_research/        # Multi-agent research platform
│                                  # (RLM agents + LSE + Knowledge Graph)
├── 10_dapr_deep_research/         # Dapr-backed durable research
│                                  # (DurableAgent + StateStore + Pub/Sub)
├── 99-sandbox/                    # Scratch space
└── shared/                        # Shared config utilities
```

## Quick Start

```bash
uv sync
cp .env.example .env   # fill in DEEPSEEK_API_KEY
```

## Running

Each sub-project is self-contained:

```bash
# Simple DSPy examples
python lab/01-basics/main.py

# MCP + RLM research agent (requires Crawl4AI Docker)
docker compose -f lab/08-rlm-mcp/docker-compose.yml up -d
python lab/08-rlm-mcp/main.py

# Self-evolving research platform
python -m lab.09_super_deep_research.cli --chat

# Dapr-backed distributed research (requires dapr init)
dapr run -f lab/10_dapr_deep_research/dapr-multi-app-run.yaml
```
