# 10 — Dapr Deep Research: Durable Agentic Research Platform

Multi-agent research platform combining **dapr-agents** (durable workflows, stateful execution) with **DSPy** (optimization, RLMs, GFL patterns).

## Architecture

```mermaid
flowchart TB
    O[ResearchWorkflow]
    O -->|call_agent| E[Explorer]
    O -->|call_agent| D[DeepReader]
    O -->|call_agent| S[Synthesizer]
    O -->|call_agent| C[Critic]

    E --> R[(Dapr State / Redis)]
    D --> R
    S --> R
    C --> R
```

Each agent is a `DurableAgent` subclass with a full DSPy pipeline inside:

| Agent | DSPy Modules |
|---|---|
| Explorer | `dspy.RLM` (discovery) + `dspy.ChainOfThought` (hypothesis gen) + `dspy.BestOfN` (top-k) + `BootstrapFewShot` (compile) |
| DeepReader | `dspy.RLM` (content extraction) + `dspy.ChainOfThought` (cross-validation) + `BootstrapFewShot` (compile) |
| Synthesizer | `dspy.RLM` (synthesis) + `dspy.ChainOfThought(SynthesizeAcrossSources)` + `BootstrapFewShot` (compile) |
| Critic | `dspy.RLM` (2-pass) + `dspy.Refine` (iterative improvement) + `dspy.MultiChainComparison` (3-chain compare) + `BootstrapFewShot` (compile) |
| Orchestrator | `dspy.ChainOfThought(SelectAgent)` + `dspy.ChainOfThought(ComputeConfidenceDelta)` + `BootstrapFewShot` (compile) |

All agents wrapped in `@workflow_entry` for durable execution with `DaprChatClient`,
`StateStoreService`, and automatic retry.

## DSPy + Dapr Integration

| Component | DSPy Implementation | Dapr Role |
|-----------|-------------------|-----------|
| Quality eval | `dspy.ChainOfThought(QualityEvaluation)` + `BootstrapFewShot` (compile) | State persisted in Redis |
| Pattern extraction | `dspy.ChainOfThought(ExtractPatterns)` + `BootstrapFewShot` (compile) | State persisted in Redis |
| Agent dispatch | `dspy.ChainOfThought(SelectAgent)` | `call_agent()` cross-app invocation |
| Agent reasoning | `dspy.RLM` + `dspy.CoT` + `dspy.BestOfN` + `dspy.Refine` + `dspy.MultiChainComparison` | `DurableAgent` shell + `@workflow_entry` |
| Agent optimization | `BootstrapFewShot.compile()` on all agents | `DaprFrontier` persistent state |
| Structured output | `BAMLAdapter` for Pydantic models | — |
| Confidence deltas | `dspy.ChainOfThought(ComputeConfidenceDelta)` per agent result | — |
| Saturation | `dspy.ChainOfThought(AssessDirectionSaturation)` per direction | — |
| Frontier | `ResearchDirection.ucb_score` (pure math) + DSPy saturation check | `DaprFrontier` via `StateStoreService` |
| Metrics | `dspy.Evaluate` | Workflow step checkpointing |

## References

- **LSE** (Chen et al., 2026): [Learning to Self-Evolve](https://arxiv.org/abs/2603.18620) — improvement-based reward `r = R̄(c₁) − R̄(c₀)` evaluated via `dspy.ChainOfThought`
- **Trace2Skill** (Ni et al., 2026): [Distill Trajectory-Local Lessons into Transferable Agent Skills](https://arxiv.org/abs/2603.25158) — parallel multi-agent patch proposal via `dspy.ChainOfThought`

## Configuration

All configuration is in the project root `.env` file:

```bash
# Required
DEEPSEEK_API_KEY="sk-..."

# LLM model selection
LLM_MODEL="deepseek/deepseek-v4-flash"        # Teacher / default LM
STUDENT_LLM_MODEL="ollama_chat/gemma4"        # Student LM for distillation
LLM_TEMPERATURE=0.3

# Infrastructure
CRAWL4AI_URL="http://localhost:11235/mcp/sse"
DAPR_REDIS_HOST="localhost:6379"
DAPR_STATE_STORE="research-state"
DAPR_PUBSUB="research-pubsub"
```

Copy `.env.example` from the project root to get started.

## Prerequisites

```bash
# Dapr
dapr init

# Crawl4AI
docker compose -f lab/10_dapr_deep_research/docker-compose.yml up -d

# Install deps
uv sync

# Optional — student model for distillation
ollama pull gemma4
```

## Running

### Multi-app run (all 5 agents at once, from project root):

```bash
dapr run -f lab/10_dapr_deep_research/dapr-multi-app-run.yaml
```

Launches orchestrator (8000), explorer (8001), deepreader (8002), synthesizer (8003), critic (8004) with shared Redis state store and pub/sub.

### Individual agents (separate terminals, from project root):

```bash
dapr run --app-id orchestrator --app-protocol grpc --app-port 8000 \
    --resources-path lab/10_dapr_deep_research/resources -- \
    python -m lab.10_dapr_deep_research --mode orchestrator

dapr run --app-id explorer-agent --app-protocol grpc --app-port 8001 \
    --resources-path lab/10_dapr_deep_research/resources -- \
    python -m lab.10_dapr_deep_research --mode explorer
```

### Programmatic (no Dapr sidecar, single process):

```bash
python -m lab.10_dapr_deep_research --mode run
```

## CLI Reference

```text
usage: python -m lab.10_dapr_deep_research [-h] [--mode {orchestrator,explorer,deepreader,synthesizer,critic,run,distill}]

Dapr Deep Research — multi-agent research platform

options:
  -h, --help            show this help message and exit
  --mode {orchestrator,explorer,deepreader,synthesizer,critic,run,distill}
                        orchestrator  — LSE-driven research orchestrator (port 8000)
                        explorer     — direction discovery agent (port 8001)
                        deepreader   — content extraction agent (port 8002)
                        synthesizer  — cross-source synthesis agent (port 8003)
                        critic       — quality evaluation agent (port 8004)
                        run          — single-process demo (no Dapr sidecar)
                        distill      — teacher/student compilation for all DSPy programs
```

### Teacher/Student distillation:

```bash
python -m lab.10_dapr_deep_research --mode distill
```

Compiles every DSPy program (`ChainOfThought`, `Refine`, etc.) using
`BootstrapFewShot` — teacher (DeepSeek) generates demonstrations, student
(Gemma 4 via Ollama) learns from them. After compilation, all internal
modules use the student LM at inference time.

## Key Features

- **Durable workflows**: Research survives process crashes — Dapr Workflows checkpoint after each iteration
- **Stateful frontier**: `DaprFrontier` uses Redis-backed state store, not JSON files
- **Multi-agent dispatch**: `call_agent()` for cross-agent workflow orchestration
- **DSPy-driven confidence**: Hardcoded confidence deltas (0.3, 0.2, 0.15) replaced with `ComputeConfidenceDelta` signature — delta adapts to finding quality
- **DSPy-driven saturation**: Static 0.95 threshold replaced with `AssessDirectionSaturation` — per-direction assessment
- **MultiChainComparison**: CriticAgent compares 3 critique chains via `dspy.MultiChainComparison` before refinement
- **Universal compilation**: Every DSPy program (`DeepReader`, `Synthesizer`, `Critic`, `LSE`, `Trace2Skill`, `Orchestrator`) has a `compile()` method ready for `BootstrapFewShot`
- **LSE meta-optimization**: Improvement-based reward trains the orchestrator across runs
- **Pub/sub coordination**: `research-pubsub` topic for agent broadcasts
- **Parallel tool execution**: `ToolExecutionMode.PARALLEL` for MCP tool calls
