# 12 — Formal Evolution: Verified Multi-Model Consensus

**Extension of lab 11** adding 6 MCP servers for formal verification (Z3, Lean4),
multi-model consensus (OpenRouter), research (arXiv), filesystem access, and
Git operations — all auto-discovered by the meta-agent via the MCP bridge.

Zero code changes. Add any MCP server to `config/mcp_servers.json` and the agent
finds its tools automatically.

## MCP Servers

| Server | Transport | Enabled | Tools | Purpose |
|--------|-----------|---------|-------|---------|
| `crawl4ai` | SSE | ✅ | `md`, `html`, `crawl`, `screenshot` | Deep web scraping |
| `fetch` | stdio | ✅ | `fetch` | URL fetching |
| `openrouter` | stdio | ✅ | `chat_completion`, `model_list`, `consensus`, `ensemble`, `usage_stats` | 100+ models for cross-validation |
| `z3-solver` | stdio | ✅ | `solve_constraint_problem`, `simple_constraint_solver`, `analyze_relationships`, `simple_relationship_analyzer` | SMT solving, property verification |
| `arxiv` | stdio | ✅ | `search_papers`, `download_paper`, `read_paper`, `list_papers`, `semantic_search`, `citation_graph` | Paper search & analysis |
| `lean-lsp` | stdio | ❌ | `lean_goal`, `lean_build`, `lean_diagnostic_messages`, `lean_run_code`, `lean_leansearch`, 20+ more | Theorem proving |
| `filesystem` | stdio | ❌ | `read_file`, `write_file`, `edit_file`, `search_files`, `directory_tree`, `list_directory` | Local file access |
| `git` | stdio | ❌ | `git_status`, `git_diff`, `git_log`, `git_commit`, `git_branch`, `git_checkout` | Git operations |

Toggle any server via `"enabled": false` — disabled servers are skipped at startup.

## Architecture

```
12_formal_evolution/
├── cli.py                    # Same CLI: generate, run, optimize, gfl, stack, distill
├── meta/                     # Agent generation + meta-agent loop
├── evolution/                # GFL pipeline (BootstrapFewShot, MIPROv2, GEPA)
├── memory/                   # InMemoryFrontier + NoopStore
├── mcp/                      # MCPClient + MCPBridge (from lab 11)
├── z3_mcp/                   # Cloned Z3 MCP server (javergar/z3_mcp)
└── config/
    └── mcp_servers.json      # All 8 MCP servers configured
```

## How It Works

No new code. The meta-agent workflow is unchanged from lab 11:

1. **Analyze**: `BestOfN` samples 3 task analyses, picks best by agent count
2. **Generate**: Creates RLM/ReAct/CodeAct/CoT agents with relevant tools
3. **Run**: Agents discover all connected MCP server tools via the bridge
4. **Evolve**: GFL pipeline optimizes prompts (BootstrapFewShot → MIPROv2 → GEPA)

The MCP bridge (`mcp/bridge.py`) converts any connected MCP server's tools into
DSPy-compatible callables. When the meta-agent generates agents, it passes
these tools to RLM and ReAct modules — they call Z3, Lean4, OpenRouter, arXiv,
filesystem, or Git tools as naturally as they'd call a fetch tool.

## Real-World Workflows

The meta-agent autonomously orchestrates its available MCP tools based on task analysis.
Here are three concrete workflows the zero-code config enables:

### 1. Bulletproof Fintech Auditor

Verify a multi-tier rewards algorithm for correctness before deployment.

```
User query: "Design a rewards algorithm with tiered payout rates"
    → BestOfN analysis: requires constraint solving, multi-model review
    → OpenRouter consensus: Claude 3.5 drafts logic, Llama 3.1 405B stress-tests
    → Z3 bounded model checking: finds counter-examples (e.g. payout > deposit)
    → Agent iterates until Z3 returns UNSAT (no violation possible)
    → GFL optimizes the final prompt with discovered edge cases
```

**What the agent does**: spots an "off-by-one" or floating-point error before you see it.
The agent's log shows: *"Z3 returned SAT on constraint payout > deposit — counter-example
found. Rewriting bounds guard."*

```bash
uv run python -m lab.12_formal_evolution \
  --query "Design a rewards algorithm with tiered payout rates and verify no payout exceeds deposit" \
  --iterations 10 run
```

### 2. Formal Scientific Researcher

Go from arXiv paper to provably correct code in one pipeline.

```
User query: "Implement the optimization algorithm from the latest distributed consensus paper"
    → ArXiv MCP: searches, downloads, reads paper
    → RLM agent extracts core theorem and formal spec
    → Lean-LSP MCP: verifies the mathematical proof step by step
    → Once Lean confirms "no goals" (proof complete), agent distills the verified
      logic into a student model via BootstrapFewShot
    → Skill consolidator saves the verified pattern for reuse
```

**What the agent does**: closes the loop from research discovery to formal verification.
No "hallucinated" logic — Lean4 mathematically guarantees correctness.

```bash
uv run python -m lab.12_formal_evolution \
  --query "Search arXiv for latest distributed consensus protocol, implement the optimization algorithm, and verify with Lean4" \
  --iterations 15 run
```

### 3. Zero-Trust Security Auditor

Generate and prove IAM policies safe against privilege escalation.

```
User query: "Create an IAM policy for multi-tenant cloud storage"
    → OpenRouter: spawns red team (GPT-4o) and blue team (Claude 3.5) in debate
    → Cross-model validation surfaces edge cases
    → Z3 relationship analysis: models users, roles, resources as SMT constraints
    → Agent queries: "Is there any path from unauthenticated user to Admin bucket?"
    → Z3 returns SAT → agent rewrites policy → re-verifies → UNSAT (safe)
    → Policy saved to filesystem via filesystem MCP
```

**What the agent does**: red/blue team debate + symbolic execution. A single logic
error in a cloud policy can breach $M data — Z3 proves no escalation path exists.

```bash
uv run python -m lab.12_formal_evolution \
  --query "Create an IAM policy for multi-tenant cloud storage and verify no privilege escalation path exists" \
  --iterations 12 run
```

### 4. Multi-Task Chain: Distributed Systems Audit

Parallel research + formal verification + code generation across 5 MCP servers.

```
User query: "Audit a distributed key-value store for data integrity under partition"
    ┌─ Parallel Phase 1 (discovery) ─────────────────────────────┐
    │  arXiv MCP:       search "distributed consensus + partition tolerance" │
    │  crawl4ai:         scrape latest blog posts on Raft/Paxos            │
    │  fetch:            grab SPEC benchmarks for KV stores                │
    └──────────────────────────────┬──────────────────────────────────────┘
                                   ↓
    ┌─ Parallel Phase 2 (analysis) ─────────────────────────────┐
    │  OpenRouter (Claude):  formal spec of the quorum logic     │
    │  OpenRouter (GPT-4o):  identify edge cases in replication │
    │  OpenRouter (Llama):   draft invariants for data integrity │
    └──────────────────────────────┬──────────────────────────────────────┘
                                   ↓
    ┌─ Sequential Phase 3 (verify) ────────────────────────────┐
    │  Z3: model quorum intersection → prove no split-brain     │
    │  Z3: verify read-after-write consistency invariants       │
    │  if Z3 SAT → feed counter-example back → re-generate      │
    └──────────────────────────────┬──────────────────────────────────────┘
                                   ↓
    ┌─ Phase 4 (produce) ──────────────────────────────────────┐
    │  GFL pipeline: optimize the audit agent with new invariants│
    │  filesystem MCP: write audit report to disk                │
    │  git MCP: commit report to repo                            │
    └────────────────────────────────────────────────────────────┘
```

**What the agent does**: two parallel discovery phases (3 servers concurrently),
cross-model consensus analysis, sequential Z3 verification loop with
counter-example feedback, then writes + git-commits the final audit report.

```bash
uv run python -m lab.12_formal_evolution \
  --query "Audit a distributed KV store for data integrity under network partition. Search papers, scrape blogs, cross-validate with 3 models, prove quorum safety with Z3, and write the audit report" \
  --iterations 25 run
```

### 5. Full R&D Lifecycle (combined)

```
arxiv (discovery) → crawl4ai (deep read) → openrouter (consensus) →
z3-solver (verify) → gfl (optimize) → distill (compress)
```

All connected. All zero-code. The agent routes automatically.

```bash
uv run python -m lab.12_formal_evolution \
  --query "Research the latest advances in vector optimization, build a consensus-backed implementation, verify it with Z3, and distill to a student model" \
  --iterations 20 run
```

## Prerequisites

```bash
# Z3 MCP — already cloned to lab/12_formal_evolution/z3_mcp/
uv sync --directory lab/12_formal_evolution/z3_mcp

# Lean LSP MCP (optional, requires Lean toolchain)
uvx lean-lsp-mcp

# OpenRouter MCP (requires Node.js 16+)
npx @physics91/openrouter-mcp init

# arXiv — works out of the box via uvx
# Filesystem — requires npx (Node.js)
# Git — works out of the box via uvx

# Set API keys in .env
OPENROUTER_API_KEY=sk-or-...
DEEPSEEK_API_KEY=...
```

## Running

```bash
# Same commands as lab 11 — agents auto-discover all tools
uv run python -m lab.12_formal_evolution --query "Research Z3 constraint solving" run
uv run python -m lab.12_formal_evolution --query "Search arXiv for agent papers" run
uv run python -m lab.12_formal_evolution --query "Verify sorting algorithm with Z3" run
```

> Full MCP server reference: [`docs/12-formal-evolution.md`](../../docs/12-formal-evolution.md)

## What Changes vs Lab 11

| Aspect | Lab 11 | Lab 12 |
|--------|--------|--------|
| MCP servers | crawl4ai, fetch, openrouter (disabled) | 8 servers: +z3-solver✅, arxiv✅, openrouter✅, lean-lsp❌, filesystem❌, git❌ |
| Agent capabilities | Web research, content analysis | + constraint solving, paper research, multi-model consensus, file ops, git |
| Tool count | ~6 tools | ~40+ tools across all MCP servers |
| Local dependencies | — | z3_mcp/ cloned in-tree |
| Code changes | None | None — just config |
