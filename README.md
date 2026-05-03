# experiments

Lab / showroom for AI agent experiments with **DSPy**.

## Lab Structure

```
lab/
├── 01-basics/           # Signatures, Predict, ChainOfThought, custom modules
├── 02-react-tools/      # ReAct agent loop with tools + PythonInterpreter
├── 03-rag-pipeline/     # RAG with ColBERTv2 + BootstrapFewShot optimization
├── 04-optimizers/       # MIPROv2, GEPA, BootstrapFewShot, BetterTogether
├── 05-rlm/              # Recursive Language Model (REPL-based exploration)
├── 06-advanced/         # MultiChainComparison, Parallel, Ensemble, Streaming, Async
├── 07-generative-feedback-loops/  # BootstrapFewShot, MIPROv2, GEPA, BetterTogether
├── 08-rlm-mcp/          # RLM + MCP tools + BAMLAdapter
├── 99-sandbox/          # Scratch space
└── shared/              # Shared config utilities
```

## Setup

```bash
uv sync
```

Copy `.env` and fill in `DEEPSEEK_API_KEY` (or swap to another provider).

## Running

Each sub-project is self-contained:

```bash
python lab/01-basics/main.py
```
