# 14 — Durable Meta-Agent: DSPy + Dapr Production Framework

**The production-hardening experiment.** Lab 14 takes the meta-agent substrate from Lab 13 (Autonomous Software Factory) and wraps it in Dapr durability — without changing a single DSPy module. The same BestOfN task analysis, RLM/ReAct/CodeAct/ChainOfThought agent generation, MultiChainComparison selection, Refine self-adaptation, GFLPipeline optimization, LSE evolution, and Trace2Skill consolidation all remain the core reasoning engine. Dapr adds crash-resistant workflows, state persistence, observability, hot-reload, and secrets management.

## Architecture

```mermaid
flowchart TB
    subgraph DAPR["Dapr Durability Layer"]
        WFE["@workflow_entry<br/>checkpointing"]
        OBS["AgentObservabilityConfig<br/>Zipkin spans"]
        STS["StateStoreService<br/>Redis persistence"]
        RTR["WorkflowRetryPolicy<br/>exponential backoff"]
    end

    subgraph DSPY["DSPy Reasoning Engine (unchanged)"]
        AG["AgentGenerator<br/>BestOfN → RLM/ReAct/CodeAct/CoT"]
        MC["MultiChainComparison<br/>agent selection"]
        RF["Refine<br/>prompt self-adaptation"]
        GF["GFL Pipeline<br/>BootstrapFewShot → MIPROv2 → GEPA"]
        LS["LSE Optimizer<br/>improvement-based reward"]
        T2["Trace2Skill<br/>parallel consolidation"]
    end

    subgraph MCP["MCP Tool Layer (unchanged)"]
        BR["MCPBridge"]
        DSP["get_dspy_tools()<br/>→ dspy.RLM/ReAct"]
        DAP["get_agent_tools()<br/>→ AgentTool"]
    end

    DAPR --> DSPY
    DSPY --> MCP
    BR --> DSP
    BR --> DAP
```

| Layer | Technology | Role |
|-------|-----------|------|
| Reasoning engine | **DSPy** — modules, signatures, adapters, optimizers | All ML logic: `dspy.RLM` (recursive code+tools), `dspy.ReAct` (tool loop), `dspy.CodeAct` (code-only), `dspy.ChainOfThought` (reasoning), `BAMLAdapter` (structured output via Pydantic) |
| Tool integration | **MCP** — Model Context Protocol | Tool discovery, auth, health checks, dual-format bridge |
| Durability | **Dapr** — Distributed Application Runtime | Workflow checkpointing, state persistence, observability, retry, secrets |

## What Changed From Lab 13

| Aspect | Lab 13 | Lab 14 |
|--------|--------|--------|
| Meta-agent loop | Single process — crash loses everything | DurableMetaAgent with @workflow_entry — resumes from last checkpoint |
| Agent registry | AgentStack — in-memory dict | DaprAgentStack — Redis-backed, survives restarts |
| Frontier | InMemoryFrontier — volatile UCB | Optional DaprFrontier — Redis-backed + batch saturation cache |
| LSE runs | In-memory list — lost on crash | Optional DaprLSEOptimizer — persisted via StateStoreService |
| Observability | print() statements | AgentObservabilityConfig — Zipkin spans for every iteration |
| Secrets | .env file | Optional secretstore component via Dapr |
| Hot-reload | Restart required | RuntimeSubscriptionConfig — swap LLM at runtime |
| Generated agents | Raw DSPy modules | Optional GeneratedDurableAgent wrapper |

## New Files

```
dapr/
├── __init__.py              # Dapr layer exports
├── wrappers.py              # GeneratedDurableAgent, wrap_module()
├── frontier.py              # DaprFrontier (ported from Lab 10)
├── lse.py                   # DaprLSEOptimizer
├── resources/               # Dapr component YAMLs (state, pubsub, secrets, config)
├── multi-app-run.yaml
├── swarm-multi-app-run.yaml # 4-app swarm: coordinator + 3 workers
core/
└── durable_meta_agent.py    # DurableMetaAgent — Dapr wrapper around DSPy MetaAgent
swarm/
├── coordinator.py           # SwarmCoordinator — dispatches tasks, owns frontier
├── worker.py                # SwarmMetaAgent — subscribes to tasks, publishes findings
├── messages.py              # Pydantic A2A message models (task, discovery, heartbeat, inquiry)
```

## CLI Commands

Pure DSPy commands — no Dapr needed:

```text
generate       Analyze task and generate agents onto the stack
run            Full pipeline: generate -> run stack -> LSE -> consolidate
gfl            Run GFL pipeline (BootstrapFewShot, MIPROv2, GEPA)
stack          Inspect the current agent stack
list-servers   List all configured MCP servers
health         Check connectivity of all configured MCP servers
```

New Dapr commands:

```text
dapr-orchestrator  Start DurableMetaAgent as a Dapr service (requires Dapr sidecar)
dapr-wrap          Show how agents would be wrapped for Dapr deployment
```

Swarm commands (multi-agent coordination):

```text
swarm-coordinator  Start SwarmCoordinator — dispatches tasks to workers via call_agent()
swarm-worker       Start SwarmMetaAgent — subscribes to swarm.tasks, publishes findings
swarm              Run full swarm in-process (coordinator + workers) for testing
```

## Running

### Pure DSPy mode (no Dapr needed)

```bash
uv run python -m lab.14_durable_meta_agent --query "Research topic" --iterations 10 run
```

### Dapr mode (requires Dapr sidecar + Redis)

```bash
# Terminal 1: infrastructure
docker compose -f lab/08-rlm-mcp/docker-compose.yml up -d
redis-server &> /dev/null &

# Terminal 2: start DurableMetaAgent
dapr run --app-id durable-meta-agent --app-protocol grpc --app-port 8000 \
  --resources-path lab/14_durable_meta_agent/dapr/resources -- \
  uv run python -m lab.14_durable_meta_agent \
  --query "Research topic" --iterations 10 \
  dapr-orchestrator --tracing --dapr-frontier --dapr-lse

# Or use multi-app runner:
dapr run -f lab/14_durable_meta_agent/dapr/multi-app-run.yaml
```

## The Dual-Path Pattern

Every subsystem follows the same ABC pattern inherited from Lab 10 — in-memory for dev, Dapr for production:

```mermaid
flowchart LR
    subgraph ABC["ResearchFrontier ABC"]
        NF["next_action()<br/>absorb_findings()<br/>seed_from_query()"]
    end

    subgraph IM["InMemoryFrontier<br/>(dev, no infra)"]
        IM1["in-memory dict<br/>UCB selection"]
    end

    subgraph DP["DaprFrontier<br/>(production)"]
        DP1["Redis-backed<br/>dirty-flag persistence<br/>batch saturation"]
    end

    ABC --> IM
    ABC --> DP
```

Same pattern applies to AgentStack/DaprAgentStack and LSEOptimizer/DaprLSEOptimizer.

Swap without changing any DSPy code:

```python
meta = MetaAgent(
    llm=lm,
    generator=generator,
    frontier=InMemoryFrontier(),   # or DaprFrontier()
    stack=AgentStack(),            # or DaprAgentStack()
    lse=LSEOptimizer(),            # or DaprLSEOptimizer()
)
```

## Key Design Decisions

### DSPy Is the Engine, Dapr Is the Chassis

DSPy is configured with `BAMLAdapter` (`dspy.adapters.baml_adapter.BAMLAdapter`) for structured output parsing. This adapter enables Pydantic models (e.g., `ExplorationResult`, `DeepReadResult`, `SynthesisReport`, `Critique` in `agents/research_agents.py`) as first-class output types in DSPy signatures. All DSPy modules throughout the lab benefit from type-validated, schema-enforced outputs — no raw JSON parsing, no prompt-instructed formatting.

The `AgentGenerator` selects the DSPy module type based on the agent's needs:

| Condition | Module | Capability |
|-----------|--------|------------|
| `use_code=True` + tools | `dspy.RLM` | Full REPL agent — runs Python, calls MCP tools, sub-LLM queries |
| Has tools (no code) | `dspy.ReAct` | Tool-using agent with thought-action-observation loop |
| `use_code=True` only | `dspy.CodeAct` | Code-capable agent without tool dependencies |
| Neither | `dspy.ChainOfThought` | Plain CoT with dynamically-created signature class via `type()` |

DSPy is NOT replaced. Dapr is NOT an alternative to DSPy. **DSPy handles all reasoning. Dapr handles all infrastructure.** The `GeneratedDurableAgent` wraps a DSPy module without modifying it:

```python
# DSPy module — unchanged, this is the core engine
dspy_module = dspy.RLM("task: str -> result: str", tools=dspy_tools)

# Dapr durability shell
durable_agent = GeneratedDurableAgent(
    dspy_module=dspy_module,
    name="my-agent", role="assistant",
    tools=agent_tools,        # AgentTool format from MCPBridge
    llm_component="llm-provider",
    enable_tracing=True,
)
```

The `DurableMetaAgent` uses single-phase init with a config dataclass:

```python
from lab.14_durable_meta_agent.core.durable_meta_agent import (
    DurableMetaAgent, DurableMetaConfig,
)

agent = DurableMetaAgent(
    generator=generator,
    tool_defs=tool_defs,
    config=DurableMetaConfig(
        enable_tracing=True,
        use_dapr_frontier=True,
        use_dapr_lse=True,
        max_iterations_per_segment=20,  # Continue-as-New every 20 iterations
    ),
)
```

### DRY Iteration Loop: `run_stack_iter()`

The `DurableMetaAgent` does NOT duplicate the iteration loop. `MetaAgent.run_stack_iter()` is a generator that yields `(iteration, direction, entry, prediction, quality, state)` per iteration. `run_stack()` wraps it for result collection. `DurableMetaAgent.run_research()` consumes it directly for checkpointing:

```python
# The single source of truth for the research loop
for iteration, direction, entry, prediction, quality, state in meta.run_stack_iter(
    query, max_iterations
):
    yield ctx.set_state("last_completed_iteration", iteration)
```

### Dirty-Flag Persistence

`DaprFrontier` uses a dirty flag to avoid writing to Redis on every mutation. Calls to `seed_*()` and `absorb_findings()` set `_dirty = True`. The actual `_save()` happens on the next `next_action()` or `saturated()` call via `_flush()`. This batches writes at the natural polling boundary.

### Failed Commands Removed

The original Lab 13 CLI had 6 commands (`optimize`, `distill`, and incorrectly-wired `generate`/`run`/`stack`/`gfl`) that referenced non-existent methods on `MetaAgent`/`GFLPipeline`. These were removed or fixed. The remaining commands now call the correct APIs: `generate_agents()`, `run_stack()`, `snapshot()`.

### Continue-as-New Workflow History

Long-running `DurableMetaAgent` workflows (50+ iterations) accumulate execution history in the Dapr state store, which degrades performance over time. The Continue-as-New pattern restarts the workflow after a configurable number of iterations, resetting the history while preserved state (frontier, LSE runs, agent stack) lives in Redis.

Enable via `DurableMetaConfig`:

```python
config = DurableMetaConfig(max_iterations_per_segment=20)
```

After every 20 iterations, the workflow spawns a new `run_research` instance with `ctx.call_workflow()`, passing current state. The old workflow terminates cleanly. The new one resumes from the last checkpoint.

### Delta-Update State Keys

`DaprAgentStack` uses per-entry state store keys instead of saving the full agent list on every mutation:

```
{key}:meta          →  ordered list of agent names (small, O(1) write)
{key}:entries:{name}  →  individual agent entry (one per agent, O(1) write)
```

This means `push()` is a single-entry write regardless of how many agents exist. `record_run()` and `record_failure()` only update that agent's key. Full-state saves only happen on `pop()` (rare). Same dirty-flag pattern as `DaprFrontier` — mutations are cheap, persistence is batched.

## Swarm Mode: Multi-Agent Coordination

Lab 14 supports running a **swarm of meta agents** that coordinate via Dapr pub/sub and A2A (Agent-to-Agent) protocol.

### Architecture

```mermaid
flowchart TB
    subgraph REDIS["Dapr State / Redis"]
        FR["DaprFrontier<br/>research directions"]
        LS["LSE runs<br/>quality history"]
        AS["DaprAgentStack<br/>agent registry"]
    end

    COORD["SwarmCoordinator<br/>port 8000"]
    COORD --> FR
    COORD --> LS
    COORD --> AS

    COORD -->|call_agent| W1
    COORD -->|call_agent| W2
    COORD -->|call_agent| W3
    COORD -->|publish| PS["Dapr Pub/Sub<br/>swarm.tasks"]

    subgraph W1["SwarmMetaAgent A<br/>port 8001 · research"]
        AG1["AgentGenerator<br/>DSPy sub-agents"]
        RC1["run_stack()<br/>frontier exploration"]
    end

    subgraph W2["SwarmMetaAgent B<br/>port 8002 · verification"]
        AG2["AgentGenerator<br/>DSPy sub-agents"]
        RC2["run_stack()<br/>frontier exploration"]
    end

    subgraph W3["SwarmMetaAgent C<br/>port 8003 · synthesis"]
        AG3["AgentGenerator<br/>DSPy sub-agents"]
        RC3["run_stack()<br/>frontier exploration"]
    end

    W1 -->|publish discovery| PS
    W2 -->|publish discovery| PS
    W3 -->|publish discovery| PS
    W1 -->|heartbeat| PS
    W2 -->|heartbeat| PS
    W3 -->|heartbeat| PS
    PS -->|collect| COORD
```

### Message Protocol

| Message | Topic | Source | Description |
|---------|-------|--------|-------------|
| `SwarmTask` | `swarm.tasks` | Coordinator | Research direction assigned to a worker |
| `SwarmDiscovery` | `swarm.discoveries` | Worker | Findings published after executing a task |
| `SwarmHeartbeat` | `swarm.heartbeat` | Worker | Liveness signal (alive/busy/error, load, task counts) |
| `SwarmInquiry` | `swarm.inquiry` | Any agent | A2A question to another agent |
| `SwarmResponse` | `swarm.response` | Any agent | A2A answer with correlation_id matching |

### A2A Protocol Flow

```mermaid
sequenceDiagram
    participant C as SwarmCoordinator
    participant PS as Dapr Pub/Sub
    participant W1 as SwarmMetaAgent A
    participant W2 as SwarmMetaAgent B

    C->>FR: frontier.next_action()
    C->>PS: publish swarm.tasks
    PS->>W1: @message_router on_task()
    Note over W1: run_stack(direction)
    W1->>PS: publish swarm.discoveries
    PS->>C: collect discovery
    C->>FR: frontier.absorb_findings()

    W1->>PS: publish swarm.inquiry
    PS->>W2: @message_router on_inquiry()
    W2->>PS: publish swarm.response
    PS->>W1: correlation_id matched

    W1->>PS: publish swarm.heartbeat (30s)
    W2->>PS: publish swarm.heartbeat (30s)
    C->>PS: subscribe swarm.heartbeat
    Note over C: timeout 90s → reassign
```

### Key Design Decisions

1. **Coordinator owns the frontier** — Only the SwarmCoordinator calls `next_action()` and `absorb_findings()`. Workers are stateless task executors. This avoids distributed locking entirely.
2. **Workers are DurableMetaAgents** — Each worker inherits the full DSPy pipeline: AgentGenerator, GFL optimization, LSE evolution, Trace2Skill consolidation. No changes to DSPy internals.
3. **A2A via pub/sub** — Agents discover each other via Dapr AgentRegistry and communicate through topic-routed messages with correlation_ids for request/response matching.
4. **Heartbeat-based failure detection** — Workers publish liveness every 30s. The coordinator marks a worker offline after 90s of silence and reassigns its tasks.

### Running the Swarm

```bash
# Production: Separate Dapr apps
dapr run -f lab/14_durable_meta_agent/dapr/swarm-multi-app-run.yaml

# Development: In-process swarm
uv run python -m lab.14_durable_meta_agent \
  --query "Research topic" --iterations 10 swarm --workers 3

# Manual: Start coordinator + workers individually
dapr run --app-id swarm-coordinator --app-protocol grpc --app-port 8000 ... swarm-coordinator
dapr run --app-id swarm-worker-0 --app-protocol grpc --app-port 8001 ... swarm-worker --worker-id swarm-worker-0
dapr run --app-id swarm-worker-1 --app-protocol grpc --app-port 8002 ... swarm-worker --worker-id swarm-worker-1
```

## Real-World Example Projects

### 1. Continuous Vulnerability Research & Automated Patching

A swarm that monitors CVE feeds, researches exploits, generates verified patches using Z3 formal verification, sandbox-tests them via E2B, deploys fixes via Terraform, and logs the full audit trail to MLflow.

```mermaid
flowchart LR
    subgraph Coordinator
        FR["DaprFrontier<br/>CVE queue"]
    end
    subgraph W1["Researcher Agent"]
        CV["crawl4ai + fetch<br/>CVE details & PoC"]
        AR["ArXiv + Exa<br/>patch literature"]
    end
    subgraph W2["Verifier Agent"]
        Z3["Z3 SMT Solver<br/>constraint verification"]
        E2["E2B sandbox<br/>runtime validation"]
    end
    subgraph W3["Deployer Agent"]
        TF["Terraform<br/>IaC deployment"]
        ML["MLflow + Git<br/>audit trail"]
    end

    FR -->|task: CVE-2026-XXXX| W1
    W1 -->|findings| FR
    FR -->|task: verify patch| W2
    W2 -->|UNSAT proof| FR
    FR -->|task: deploy| W3
    W3 -->|commit + deploy| FR
```

```bash
uv run python -m lab.14_durable_meta_agent \
  --query "Monitor NVD feed for critical RCE vulnerabilities in PostgreSQL extensions. For each CVE, research exploit vectors via crawl4ai and Exa, generate a verified patch with Z3 proof of correctness, sandbox-test with E2B, deploy via Terraform, and log the complete audit trail to MLflow and git." \
  --iterations 50 swarm --workers 4
```

**Capabilities demonstrated**: Multi-phase research (discovery → deep-read → verify → deploy), formal proof generation, sandboxed execution, immutable audit trail, crash survival across 50+ iterations.

---

### 2. Competitive Intelligence Platform

A swarm of domain-specialized meta agents that continuously monitor competitor products, SEC filings, hiring patterns, patent filings, and social media sentiment. Each worker publishes structured discoveries to pub/sub; the coordinator consolidates them into a living knowledge graph.

```mermaid
flowchart LR
    subgraph Coordinator
        CON["Consolidation<br/>FalkorDB KG"]
    end
    subgraph P1["Product Agent"]
        CW["crawl4ai<br/>changelogs, pricing"]
        GH["GitHub MCP<br/>commit activity"]
    end
    subgraph P2["Legal Agent"]
        AR["ArXiv + Exa<br/>patent filings"]
        SEC["SEC EDGAR<br/>financial disclosures"]
    end
    subgraph P3["Talent Agent"]
        LI["LinkedIn<br/>hiring patterns"]
        SO["social media<br/>sentiment analysis"]
    end
    subgraph P4["Signal Agent"]
        ST["sequential-thinking<br/>threat assessment"]
        OR["OpenRouter consensus<br/>multi-model scoring"]
    end

    P1 & P2 & P3 -->|publish discovery| CON
    CON -->|cross-reference| P4
    P4 -->|risk score| CON
```

```bash
uv run python -m lab.14_durable_meta_agent \
  --query "Launch competitive intelligence swarm monitoring top 5 competitors. Product tracker: scrape changelogs and pricing pages via crawl4ai every 6 hours. Legal tracker: search USPTO patent filings and SEC EDGAR disclosures via Exa. Talent tracker: analyze LinkedIn hiring patterns and social media sentiment. Signal aggregator: use sequential-thinking + OpenRouter consensus for cross-domain threat assessment. Store all findings in FalkorDB knowledge graph. Publish weekly consolidated brief to filesystem." \
  --iterations 100 swarm --workers 5
```

**Capabilities demonstrated**: Heterogeneous worker domains, continuous long-running operation (100+ iterations), pub/sub discovery fan-in, knowledge graph persistence, multi-model consensus for signal assessment.

---

### 3. Self-Healing Production Infrastructure

A swarm that monitors production telemetry (MLflow metrics), detects anomaly patterns, researches root causes via log analysis with sequential-thinking, generates fix candidates using RLM code agents, proofs correctness with Z3, sandbox-rolls out to canary via E2B, and deploys to production via Terraform — all without human intervention.

```mermaid
flowchart LR
    subgraph Coordinator
        FR2["DaprFrontier<br/>anomaly queue"]
        HB["heartbeat monitor<br/>worker health"]
    end
    subgraph M1["Monitor Agent"]
        MLF["MLflow<br/>metric drift"]
        ST2["sequential-thinking<br/>root cause"]
    end
    subgraph M2["Fix Agent"]
        RLM["dspy.RLM<br/>code generation"]
        Z32["Z3 solver<br/>correctness proof"]
    end
    subgraph M3["Rollout Agent"]
        E2B2["E2B sandbox<br/>canary test"]
        TF2["Terraform<br/>prod deploy"]
    end
    subgraph M4["Verify Agent"]
        SNK["Snyk security<br/>SAST scan"]
        PG["Postgres<br/>canary metrics"]
    end

    MLF -->|latency spike| FR2
    FR2 -->|diagnose| M1
    M1 -->|root cause| FR2
    FR2 -->|generate fix| M2
    M2 -->|patch + proof| FR2
    FR2 -->|canary 5%| M3
    M3 -->|canary metrics| M4
    M4 -->|verified| FR2
    FR2 -->|rollout 100%| M3
```

```bash
uv run python -m lab.14_durable_meta_agent \
  --query "Deploy self-healing infrastructure monitor. Watch MLflow for latency p99 > 500ms or error rate > 1%. On anomaly: use sequential-thinking to decompose root cause from log patterns. Generate fix via RLM code agent. Prove fix correctness with Z3 SMT solver. Roll out to 5% canary via E2B sandbox. Validate canary metrics in Postgres. Full rollout via Terraform if metrics improve. Rollback automatically if degradation detected. Log every decision to MLflow." \
  --iterations 200 swarm --workers 4
```

**Capabilities demonstrated**: Closed-loop observability → diagnosis → fix → verify → deploy → monitor, automatic rollback on degradation, formal proof of fix correctness, canary-based staged rollout, crash survival across 200+ iterations.

---

## Research Foundation

All DSPy research foundations from Lab 13 apply unchanged:
- **GFL Pipeline** — BootstrapFewShot, MIPROv2, GEPA, Sequential optimizers
- **LSE** — Learning to Self-Evolve with improvement-based reward
- **Trace2Skill** — Parallel skill consolidation from execution trajectories

## Prerequisites

| Dependency | Installation |
|-----------|-------------|
| Python 3.11+ | `uv sync` |
| Dapr CLI | `dapr init` |
| Redis | `redis-server` or Docker |
| Crawl4AI | `docker compose -f lab/08-rlm-mcp/docker-compose.yml up -d` |

Set API keys in `.env` from the project root.
