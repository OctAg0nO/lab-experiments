# 09: Super Deep Research. Multi-Agent Research with UCB Frontier and LSE Evolution

> **Source:** `lab/09_super_deep_research/` (16 files across 5 packages)
> **Concepts:** `dspy.RLM` agents, UCB exploration/exploitation frontier, LSE meta-optimization, knowledge graph memory, Trace2Skill consolidation, MCP tool bridge, interactive REPL CLI.

## Purpose

A self-evolving multi-agent research platform. Given a user query, it dispatches specialized DSPy RLM agents (explorer, deep reader, synthesizer, critic) in a loop governed by a UCB priority queue. Findings flow into a knowledge graph, trajectories get consolidated into reusable skills, and an LSE optimizer measures research quality deltas across iterations.

---

## Package Map

```
09_super_deep_research/
+-- __init__.py
+-- __main__.py              # Entry: delegates to cli.main()
+-- agents.py                # Pydantic models + RLM agent factories
+-- frontier.py               # UCB priority queue for research directions
+-- orchestrator.py           # Main research loop with LSE evolution
+-- cli.py                    # CLI commands, REPL, argparse entry point
+-- config/
|   +-- mcp_servers.json      # MCP server definitions
+-- memory/
|   +-- __init__.py
|   +-- store.py              # MemoryStore: skills + logs + graph
|   +-- knowledge_graph.py    # KnowledgeGraph: directed findings graph
+-- evolution/
|   +-- __init__.py
|   +-- lse.py                # LSEOptimizer: meta-optimizer
|   +-- trace2skill.py         # SkillConsolidator: trajectory→skill
|   +-- self_distill.py        # SelfDistill: SDPO-style self-distillation
+-- mcp/
    +-- __init__.py
    +-- client.py              # MCPClient: async→sync MCP bridge
```

## Setup

```python
import dspy
lm = dspy.LM("deepseek/deepseek-v4-flash")
dspy.configure(lm=lm)
```

Requires a `config/mcp_servers.json` with MCP server definitions (stdio or SSE). MCP tools provide the search, fetch, crawl, and extraction capabilities that the agents use.

---

## 1. agents.py: Pydantic Models and Agent Factories

### Pydantic Output Models

Six structured output models that DSPy RLM agents produce. All inherit from `pydantic.BaseModel` and use `Field` for descriptions.

#### FoundDirection

```python
class FoundDirection(BaseModel):
    topic: str
    relevance: str
    seed_query: str
```

| Field | Type | Description |
|-------|------|-------------|
| `topic` | `str` | Research topic discovered |
| `relevance` | `str` | Why this topic matters |
| `seed_query` | `str` | Search query to explore the topic further |

#### ExplorationResult

```python
class ExplorationResult(BaseModel):
    directions: list[FoundDirection]
```

| Field | Type | Description |
|-------|------|-------------|
| `directions` | `list[FoundDirection]` | Discovered research directions |

#### ExtractedFinding

```python
class ExtractedFinding(BaseModel):
    claim: str
    evidence: str
    source: str
    confidence: str
```

| Field | Type | Description |
|-------|------|-------------|
| `claim` | `str` | Main claim or finding |
| `evidence` | `str` | Supporting evidence from the source |
| `source` | `str` | Source URL |
| `confidence` | `str` | One of `"high"`, `"medium"`, `"low"` |

#### DeepReadResult

```python
class DeepReadResult(BaseModel):
    findings: list[ExtractedFinding]
    summary: str
```

| Field | Type | Description |
|-------|------|-------------|
| `findings` | `list[ExtractedFinding]` | Extracted findings from deep reading |
| `summary` | `str` | Content summary |

#### SynthesisReport

```python
class SynthesisReport(BaseModel):
    synthesis: str
    key_insights: list[str]
    gaps: list[str]
```

| Field | Type | Description |
|-------|------|-------------|
| `synthesis` | `str` | Cross-source synthesis |
| `key_insights` | `list[str]` | Key insights across sources |
| `gaps` | `list[str]` | Identified knowledge gaps |

#### Critique

```python
class Critique(BaseModel):
    strengths: list[str]
    weaknesses: list[str]
    follow_ups: list[str]
```

| Field | Type | Description |
|-------|------|-------------|
| `strengths` | `list[str]` | What the research did well |
| `weaknesses` | `list[str]` | What needs improvement |
| `follow_ups` | `list[str]` | Recommended next directions |

### Agent Factory Functions

Each factory creates a `dspy.RLM` with a specific signature, tool set, and iteration budget.

#### create_explorer

```python
def create_explorer(tools: list, lm: dspy.LM) -> dspy.RLM
```

Creates an RLM with the signature `"task: str -> result: ExplorationResult"`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `tools` | `list` | MCP tool functions for search and discovery |
| `lm` | `dspy.LM` | Language model instance |

**Limits:** max 8 iterations, max 12 LLM calls.

The explorer receives a research task and produces an `ExplorationResult` containing discovered research directions, each with a seed query for further exploration.

#### create_deep_reader

```python
def create_deep_reader(tools: list, lm: dspy.LM) -> dspy.RLM
```

Creates an RLM with the signature `"topic: str, url: str -> result: DeepReadResult"`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `tools` | `list` | MCP tool functions for fetch, crawl, and extraction |
| `lm` | `dspy.LM` | Language model instance |

**Limits:** max 10 iterations, max 16 LLM calls.

The deep reader takes a topic and URL, fetches and analyzes the content, and produces structured findings with claims, evidence, and confidence ratings.

#### create_synthesizer

```python
def create_synthesizer(tools: list, lm: dspy.LM) -> dspy.RLM
```

Creates an RLM with the signature `"task: str, findings: str -> result: SynthesisReport"`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `tools` | `list` | All available MCP tool functions |
| `lm` | `dspy.LM` | Language model instance |

**Limits:** max 8 iterations, max 12 LLM calls.

The synthesizer takes a task description and accumulated findings text, produces a cross-source synthesis, key insights, and identifies knowledge gaps.

#### create_critic

```python
def create_critic(lm: dspy.LM) -> dspy.RLM
```

Creates an RLM with the signature `"research_summary: str -> result: Critique"`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `lm` | `dspy.LM` | Language model instance |

**Limits:** max 6 iterations, max 8 LLM calls. No tools (pure reasoning).

The critic evaluates the research summary, identifies strengths and weaknesses, and suggests follow-up directions. Used as a quality gate every 2 iterations and when stagnation is detected.

### Usage Example

```python
from lab.nine_super_deep_research.agents import (
    create_explorer, create_deep_reader, create_synthesizer, create_critic
)

explorer = create_explorer(search_tools, lm)
result = explorer(task="Recent advances in DSPy optimization")
# result.result -> ExplorationResult with directions
```

---

## 2. frontier.py: UCB Research Frontier

The frontier replaces a fixed list of URLs with a dynamic priority queue that selects what to explore next based on Upper Confidence Bound.

### ResearchDirection

```python
@dataclass
class ResearchDirection:
    topic: str
    confidence: float = 0.0
    exploration_depth: int = 0
    source_count: int = 0
    last_updated: str = ""
    parent_topic: str | None = None
    seed_query: str = ""
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `topic` | `str` | (required) | Research topic |
| `confidence` | `float` | `0.0` | How well understood (0.0 to 1.0) |
| `exploration_depth` | `int` | `0` | Times explored |
| `source_count` | `int` | `0` | Unique sources consulted |
| `last_updated` | `str` | `""` | ISO timestamp |
| `parent_topic` | `str \| None` | `None` | Parent topic this was derived from |
| `seed_query` | `str` | `""` | Initial search query |

#### ucb_score

```python
def ucb_score(self, total_explorations: int, exploration_constant: float = 1.4) -> float
```

Computes the UCB1 score: `confidence + C * sqrt(log(N) / (depth + 1))`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `total_explorations` | `int` | - | Total explorations across all directions |
| `exploration_constant` | `float` | `1.4` | UCB exploration constant (C) |

**Returns:** `float` (infinity if `exploration_depth == 0`, guaranteeing unexplored topics are selected first).

#### to_dict / from_dict

```python
def to_dict(self) -> dict
@classmethod
def from_dict(cls, d: dict) -> ResearchDirection
```

Serialize/deserialize for JSON persistence.

### ResearchFrontier

```python
class ResearchFrontier:
    def __init__(self, persist_path: str | Path | None = None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `persist_path` | `str \| Path \| None` | `None` | Path to JSON file for persistence across runs |

The frontier maintains a list of `ResearchDirection` objects and a running counter of total explorations.

#### seed_from_query

```python
def seed_from_query(self, query: str)
```

Creates the initial research direction from a user query. Confidence starts at 0.0, depth at 0. Persisted immediately.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | User's research query |

#### seed_from_directions

```python
def seed_from_directions(self, topics: list[str], parent: str | None = None)
```

Seeds multiple sub-directions from a broader topic. Deduplicates by topic name.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `topics` | `list[str]` | - | Topics to add as new directions |
| `parent` | `str \| None` | `None` | Parent topic name |

#### next_action

```python
def next_action(self) -> ResearchDirection | None
```

Selects the highest-UCB direction for exploration. Filters to active directions (confidence < 0.95).

**Returns:** `ResearchDirection` or `None` if all directions are saturated.

#### absorb_findings

```python
def absorb_findings(self, topic: str, confidence_delta: float, sources: int, follow_ups: list[str])
```

Updates a direction with new findings and spawns follow-up directions.

| Parameter | Type | Description |
|-----------|------|-------------|
| `topic` | `str` | Topic to update |
| `confidence_delta` | `float` | Amount to increase confidence (capped at 1.0) |
| `sources` | `int` | Number of new sources to add |
| `follow_ups` | `list[str]` | New directions to spawn |

#### total_explorations

```python
@property
def total_explorations(self) -> int
```

Returns the running count of all explorations performed.

#### saturated

```python
def saturated(self) -> bool
```

Returns `True` when all active directions have confidence >= 0.95.

#### summary

```python
def summary(self) -> str
```

Returns a human-readable string: `"{N} active, {M} explored, {T} total explorations"`.

### Usage Example

```python
frontier = ResearchFrontier(persist_path="/tmp/frontier.json")
frontier.seed_from_query("DSPy optimizer comparison")

while (direction := frontier.next_action()) is not None:
    print(f"Exploring: {direction.topic}")
    # ... do research ...
    frontier.absorb_findings(
        topic=direction.topic,
        confidence_delta=0.3,
        sources=2,
        follow_ups=["sub-topic A", "sub-topic B"],
    )

print(frontier.summary())
```

---

## 3. orchestrator.py: Research Loop with LSE Evolution

### _default_quality_fn

```python
def _default_quality_fn(state: dict) -> float
```

Measures research quality as a weighted composite: `coverage * 0.4 + depth * 0.4 + novelty * 0.2`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | `dict` | Must contain `num_directions`, `num_findings`, `frontier_saturation` |

**Formula:**
- `coverage = min(1.0, num_directions / 10.0)`
- `depth = min(1.0, num_findings / max(1, num_directions) / 3.0)`
- `novelty = frontier_saturation` (fraction of directions with confidence >= 0.95)

Returns `0.0` when `num_directions == 0`.

### ResearchOrchestrator

```python
class ResearchOrchestrator:
    def __init__(
        self,
        mcp_client: MCPClient,
        tool_defs: list[dict],
        lm: dspy.LM,
        memory: MemoryStore,
        frontier: ResearchFrontier,
        max_iterations: int = 6,
    )
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mcp_client` | `MCPClient` | - | Connected MCP client for tool calls |
| `tool_defs` | `list[dict]` | - | Tool definitions from `MCPClient.connect_all()` |
| `lm` | `dspy.LM` | - | Language model |
| `memory` | `MemoryStore` | - | Persistent memory store |
| `frontier` | `ResearchFrontier` | - | UCB frontier for direction selection |
| `max_iterations` | `int` | `6` | Maximum research iterations |

On construction, the orchestrator:

1. Builds tool subsets per agent (fetch tools, search tools, all tools).
2. Creates the four RLM agents via the factories in `agents.py`.
3. Initializes an `LSEOptimizer` (using `_default_quality_fn`) and a `SkillConsolidator`.
4. Initializes state tracking: iteration counter, trajectories list, findings text list.

#### run

```python
def run(self, user_query: str) -> dict
```

The main research loop. Flow per iteration:

1. Seed the frontier with the user query (first iteration only, via `seed_from_query`).
2. Select the next direction from the frontier via UCB (`next_action`).
3. Dispatch the appropriate agent based on exploration state:
   - `exploration_depth == 0` → `_explore()`
   - `confidence < 0.6` → `_deep_read()`
   - otherwise → `_synthesize()`
4. Run critic review every 2 iterations (if findings exist).
5. Run heartbeat (reflection + consolidation) every 3 iterations.
6. Record the run in the LSE optimizer with the current state.
7. If the frontier is saturated (no next action), stop early.

After the loop, consolidates all trajectories into a skill and saves it to both the `SkillConsolidator` and `MemoryStore`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_query` | `str` | The research query |

**Returns:** A report dict with:

| Key | Type | Description |
|-----|------|-------------|
| `iterations` | `int` | Number of iterations completed |
| `frontier` | `str` | Frontier summary string |
| `memory` | `str` | Memory summary string |
| `findings_count` | `int` | Number of accumulated findings |
| `trajectories_count` | `int` | Number of logged trajectories |
| `improvement_trend` | `list[float]` | LSE quality deltas between iterations |
| `best_strategy` | `str \| None` | Strategy ID with highest LSE quality |

#### Agent Dispatch Methods

These are called by `run()` internally.

```python
def _explore(self, topic: str)
```

Dispatches the explorer RLM. On success:
- Seeds sub-directions from the explorer's output into the frontier.
- Absorbs findings (confidence delta 0.3, 1 source).
- Adds a finding to the knowledge graph with category `"direction"`.
- Logs the trajectory.

```python
def _deep_read(self, topic: str)
```

Fetches content via the `fetch` MCP tool (if available), then dispatches the deep reader RLM. On success:
- Adds each finding to the knowledge graph with category `"finding"`.
- Absorbs findings (confidence delta 0.2, count of findings).
- Logs the trajectory.

```python
def _synthesize(self, topic: str)
```

Dispatches the synthesizer RLM. On success:
- Seeds identified gaps as new frontier directions.
- Absorbs findings (confidence delta 0.15, 0 sources, gaps as follow-ups).
- Logs the trajectory.

```python
def _critique(self)
```

Dispatches the critic RLM on the last 3 findings. On success:
- Seeds follow-up suggestions as new frontier directions.
- Logs the trajectory.

#### _heartbeat

```python
def _heartbeat(self)
```

Prints frontier and memory summaries. Checks for stagnation (negative LSE trend over last 2 iterations) and triggers an additional critic review if detected. Consolidates the last 3 trajectories into a skill and saves it with the `heartbeat_{N}` name.

#### _report

```python
def _report(self) -> dict
```

Assembles the final report dict from current state (see `run()` return value).

### Usage Example

```python
from lab.nine_super_deep_research.orchestrator import ResearchOrchestrator

client = MCPClient("config/mcp_servers.json")
tool_defs = client.connect_all()
memory = MemoryStore(BASE_DIR / "memory")
frontier = ResearchFrontier(persist_path="/tmp/frontier.json")

orch = ResearchOrchestrator(
    mcp_client=client, tool_defs=tool_defs, lm=lm,
    memory=memory, frontier=frontier, max_iterations=8,
)
report = orch.run("Transformer attention mechanism efficiency")
print(report["improvement_trend"])
client.close()
```

---

## 4. cli.py: Command Line Interface

Entry point for running research from the terminal. Supports single queries, interactive REPL mode, and platform administration commands.

### CLI Commands

#### cmd_query

```python
def cmd_query(query: str, max_iterations: int)
```

Runs a single research query.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Research topic |
| `max_iterations` | `int` | Maximum research iterations |

Creates an LM, MemoryStore, and ResearchFrontier. Connects MCP, builds an orchestrator, runs it, and prints the report.

#### cmd_chat

```python
def cmd_chat(max_iterations: int)
```

Interactive research REPL. Loads existing memory and frontier. Accepts:

- Plain text → runs as a research query
- Slash commands:

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/status` | Show frontier + memory summaries |
| `/skills` | List accumulated skills |
| `/frontier` | List all frontier directions with confidence and depth |
| `/reset` | Reset all persisted memory and frontier |
| `/quit` | Exit REPL |

#### cmd_status

```python
def cmd_status()
```

Prints platform status: MCP server counts (enabled/disabled), memory summary, frontier summary.

#### cmd_list_servers

```python
def cmd_list_servers()
```

Prints a table of configured MCP servers: name, enabled status, transport type, description.

#### cmd_list_skills

```python
def cmd_list_skills()
```

Lists accumulated skills from memory with save timestamps.

#### cmd_list_frontier

```python
def cmd_list_frontier()
```

Prints all frontier directions sorted by UCB score descending, with confidence, depth, and UCB columns.

#### cmd_enable_server / cmd_disable_server

```python
def cmd_enable_server(name: str)
def cmd_disable_server(name: str)
```

Toggles the `enabled` flag on an MCP server in `config/mcp_servers.json`.

#### cmd_reset_memory

```python
def cmd_reset_memory()
```

Deletes the `memory/` directory and recreates it, clearing all persisted data.

### Argument Parser (main)

```python
def main()
```

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--query` | `-q` | `str` | - | Single research query |
| `--chat` | `-c` | flag | - | Interactive chat mode |
| `--iterations` | `-i` | `int` | `6` | Max research iterations |
| `--status` | `-s` | flag | - | Show platform status |
| `--list-servers` | - | flag | - | List MCP servers |
| `--enable-server` | - | `str` | - | Enable an MCP server |
| `--disable-server` | - | `str` | - | Disable an MCP server |
| `--list-skills` | - | flag | - | List accumulated skills |
| `--list-frontier` | - | flag | - | List research frontier |
| `--reset-memory` | - | flag | - | Reset memory |

### Usage Examples

```bash
# Single query
python -m lab.09_super_deep_research --query "DSPy optimization benchmarks"

# Interactive REPL
python -m lab.09_super_deep_research --chat

# Platform status
python -m lab.09_super_deep_research --status

# Manage MCP servers
python -m lab.09_super_deep_research --list-servers
python -m lab.09_super_deep_research --disable-server openrouter

# View frontier
python -m lab.09_super_deep_research --list-frontier

# Reset
python -m lab.09_super_deep_research --reset-memory
```

---

## 5. memory/: Persistence Layer

### MemoryStore

```python
class MemoryStore:
    def __init__(self, base_dir: str | Path)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `base_dir` | `str \| Path` | Root directory for all memory artifacts |

Creates subdirectories:
- `{base_dir}/skills/` skill JSON files
- `{base_dir}/logs/` execution log JSON files
- `{base_dir}/graph.json` knowledge graph persistence
- `self.graph` → `KnowledgeGraph` instance

#### save_skill

```python
def save_skill(self, name: str, data: dict)
```

Saves a skill dict to `{skills_dir}/{name}.json`. Adds a `saved` timestamp.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Skill name (used as filename) |
| `data` | `dict` | Skill data to persist |

#### load_skills

```python
def load_skills(self) -> list[dict]
```

Loads all skills from the skills directory, sorted by filename in reverse (newest first).

**Returns:** `list[dict]`. Each dict includes any fields from the original `save_skill` call plus a `_file` key if a `_file` field was present, and the `saved` timestamp.

#### log_execution

```python
def log_execution(self, agent: str, topic: str, trajectory: list, result: dict | None)
```

Logs an agent execution to `{logs_dir}/{timestamp}_{agent}_{slug}.json`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `agent` | `str` | Agent name |
| `topic` | `str` | Topic explored |
| `trajectory` | `list` | Execution trajectory steps |
| `result` | `dict \| None` | Structured result data |

The topic is slugified for the filename (lowercased, non-alphanumeric replaced with underscores, truncated to 40 chars).

#### summary

```python
def summary(self) -> str
```

Returns `"{N} findings, {M} relationships, {K} skills, {L} logs"` combining graph stats and file counts.

### KnowledgeGraph

```python
class KnowledgeGraph:
    def __init__(self, persist_path: str | Path)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `persist_path` | `str \| Path` | Path to JSON file for persistence |

Internal state: `nodes: dict[str, dict]` and `edges: list[dict]`. Loaded from file on construction.

#### add_finding

```python
def add_finding(self, finding_id: str, content: str, source: str, category: str = "", metadata: dict | None = None)
```

Adds a node to the graph. No-op if `finding_id` already exists.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `finding_id` | `str` | - | Unique identifier |
| `content` | `str` | - | Finding text content |
| `source` | `str` | - | Source identifier |
| `category` | `str` | `""` | Category label |
| `metadata` | `dict \| None` | `None` | Additional metadata |

#### get_finding

```python
def get_finding(self, finding_id: str) -> dict[str, Any] | None
```

Returns the node dict or `None` if not found.

#### search

```python
def search(self, query: str) -> list[dict[str, Any]]
```

Simple keyword search across node content, category, and source. Returns results sorted by relevance score.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Keyword to search for |

Scoring:
- Content match: `len(query) / len(content)`
- Category match: `+0.3`
- Source match: `+0.2`

**Returns:** `list[dict]` with keys `node` and `score`.

#### relate

```python
def relate(self, source_id: str, target_id: str, relation: str)
```

Adds a directed edge between two findings.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source_id` | `str` | Source finding ID |
| `target_id` | `str` | Target finding ID |
| `relation` | `str` | Relationship type (e.g., `"supports"`, `"contradicts"`, `"extends"`, `"derived_from"`) |

#### get_related

```python
def get_related(self, finding_id: str) -> list[dict[str, Any]]
```

Returns all related findings with their relationship type. Outgoing edges use the original relation; incoming edges get an `inverse_` prefix.

**Returns:** `list[dict]` with keys `node` and `relation`.

#### summary

```python
def summary(self) -> str
```

Returns `"{N} findings, {M} relationships"`.

### Usage Example

```python
from lab.nine_super_deep_research.memory.store import MemoryStore

memory = MemoryStore("/tmp/memory")
memory.graph.add_finding("f1", "Attention is all you need", "arxiv", "paper")
memory.graph.add_finding("f2", "Transformer-XL extends attention", "arxiv", "paper")
memory.graph.relate("f1", "f2", "extends")
results = memory.graph.search("attention")
print(memory.summary())

memory.save_skill("exploration_pattern", {"patterns": [...]})
skills = memory.load_skills()
```

---

## 6. evolution/: Self-Evolution Components

### LSEOptimizer (Learning to Self-Evolve)

```python
class LSEOptimizer:
    def __init__(self, quality_fn: Callable[[Any], float])
```

Meta-optimizer that tracks research quality across iterations and measures improvement deltas.

| Parameter | Type | Description |
|-----------|------|-------------|
| `quality_fn` | `Callable[[Any], float]` | Function that evaluates a state dict and returns a quality score |

#### LSERun

```python
@dataclass
class LSERun:
    strategy_id: str
    quality_score: float
    strategy_description: str
    num_directions: int
    num_findings: int
```

Internal record of a single optimization step.

#### compute_improvement

```python
def compute_improvement(self, current_quality: float) -> float
```

Returns the quality delta from the last recorded run. Returns `0.0` if no runs exist.

#### record_run

```python
def record_run(self, strategy_id: str, state: Any, strategy_description: str)
```

Records a new run. Computes quality via `self.quality_fn(state)`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `strategy_id` | `str` | Identifier for this strategy/iteration |
| `state` | `Any` | State dict passed to `quality_fn` |
| `strategy_description` | `str` | Human-readable description |

#### best_strategy

```python
def best_strategy(self) -> str | None
```

Returns the `strategy_id` of the run with the highest quality score, or `None` if no runs exist.

#### improvement_trend

```python
def improvement_trend(self) -> list[float]
```

Returns the sequence of quality deltas between consecutive runs. Returns an empty list if fewer than 2 runs exist.

### SkillConsolidator (Trace2Skill)

```python
class SkillConsolidator:
    def __init__(self, persist_dir: str | Path)
```

Extracts reusable patterns from agent trajectories.

| Parameter | Type | Description |
|-----------|------|-------------|
| `persist_dir` | `str \| Path` | Directory for storing consolidated skill JSONs |

#### consolidate

```python
def consolidate(self, trajectories: list[dict]) -> dict
```

Extracts error patterns, success patterns, and demonstrations from a list of trajectories.

| Parameter | Type | Description |
|-----------|------|-------------|
| `trajectories` | `list[dict]` | Each dict should contain a `"trajectory"` key (list of steps) or be the list directly. Steps have `reasoning`, `output`, and `code` keys. |

**Returns:** A dict with:

| Key | Type | Description |
|-----|------|-------------|
| `error_patterns` | `list[dict]` | Up to 10 patterns from steps containing "error" or "fail" |
| `success_patterns` | `list[dict]` | Up to 10 patterns from steps with reasoning > 30 chars |
| `demonstrations` | `list[dict]` | Up to 5 final-step outputs as (reasoning, output) pairs |
| `n_trajectories` | `int` | Number of trajectories processed |

Each error pattern has `symptom` and `reasoning`. Each success pattern has `reasoning` and `code`.

#### save_skill

```python
def save_skill(self, name: str, skill: dict)
```

Saves a skill dict to `{persist_dir}/{name}.json`.

#### load_skills

```python
def load_skills(self) -> list[dict]
```

Loads all consolidated skills from the persist directory, newest first.

### SelfDistill (SDPO-Style)

```python
class SelfDistill:
    def __init__(self, agent: dspy.Module)
```

Self-distillation loop for RLM agents. Conditions on execution history to generate improved responses.

| Parameter | Type | Description |
|-----------|------|-------------|
| `agent` | `dspy.Module` | The agent module to self-distill |

#### reflect_and_distill

```python
def reflect_and_distill(self, task: str, result, trajectory: list | None) -> dict | None
```

Records an execution and returns the result. Currently a pass-through (stores history for future adaptation).

| Parameter | Type | Description |
|-----------|------|-------------|
| `task` | `str` | The task string |
| `result` | - | The agent's result |
| `trajectory` | `list \| None` | Execution trajectory |

**Returns:** The original `result` unchanged.

#### adaptation_context

```python
def adaptation_context(self) -> str
```

Returns a formatted string of the last 3 executions (task + outcome) for use as few-shot context in subsequent agent calls. Returns `""` if no history exists.

### Usage Examples

```python
from lab.nine_super_deep_research.evolution.lse import LSEOptimizer
from lab.nine_super_deep_research.evolution.trace2skill import SkillConsolidator
from lab.nine_super_deep_research.evolution.self_distill import SelfDistill

# LSE: track quality progression
lse = LSEOptimizer(quality_fn=lambda s: s.get("quality", 0))
lse.record_run("iter_1", {"num_directions": 3, "num_findings": 5, "frontier_saturation": 0.2}, "initial")
lse.record_run("iter_2", {"num_directions": 5, "num_findings": 12, "frontier_saturation": 0.4}, "deep_read")
print(lse.improvement_trend())  # [quality_delta]
print(lse.best_strategy())      # "iter_2" if its quality was higher

# SkillConsolidator: extract patterns from trajectories
consolidator = SkillConsolidator("/tmp/consolidated")
trajectories = [
    {"trajectory": [{"reasoning": "...", "output": "result", "code": ""}]}
]
skill = consolidator.consolidate(trajectories)
consolidator.save_skill("my_skill", skill)

# SelfDistill: reflect on past executions
distill = SelfDistill(agent=my_agent)
distill.reflect_and_distill("research topic", result, trajectory)
ctx = distill.adaptation_context()
```

---

## 7. MCP Client: Async-to-Sync MCP Bridge

### MCPClient

```python
class MCPClient:
    def __init__(self, config_path: str)
```

Connects to MCP servers (stdio and SSE) on a background event loop thread. All async operations are bridged to synchronous calls.

| Parameter | Type | Description |
|-----------|------|-------------|
| `config_path` | `str` | Path to JSON config file with `mcpServers` key |

Config format:

```json
{
  "mcpServers": {
    "server-name": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-fetch"],
      "enabled": true
    },
    "sse-server": {
      "type": "sse",
      "url": "http://localhost:8000/sse",
      "enabled": true
    }
  }
}
```

Internal state:
- Creates a new `asyncio` event loop on a daemon thread.
- Maintains a dict of `_ServerCtx` objects (session + close coroutine).

#### connect_all

```python
def connect_all(self) -> list[dict]
```

Connects to all enabled MCP servers. For each server:
- stdio: uses `stdio_client` with `StdioServerParameters`
- SSE: uses `sse_client` with the configured URL

Skips servers with `"enabled": false`. Prints connection errors per server without failing the whole batch.

**Returns:** `list[dict]`. Aggregated tool definitions across all servers:

```python
[
    {"server": "server-name", "name": "tool_name",
     "description": "...", "inputSchema": {...}},
    ...
]
```

#### call_tool

```python
def call_tool(self, server: str, tool_name: str, arguments: dict) -> str
```

Calls a tool on a specific server. Returns concatenated text content from the result.

| Parameter | Type | Description |
|-----------|------|-------------|
| `server` | `str` | Server name |
| `tool_name` | `str` | Tool name |
| `arguments` | `dict` | Tool arguments |

**Returns:** `str`. Concatenated text content from all result parts.

#### close

```python
def close(self)
```

Cleanup: closes all server sessions, stops the event loop, joins the background thread.

#### find_tool

```python
def find_tool(self, tool_defs: list[dict], server: str, name: str) -> dict | None
```

Looks up a tool definition by server and name.

#### build_tool_fns

```python
def build_tool_fns(self, tool_defs: list[dict]) -> list
```

Wraps MCP tool definitions into callable functions for DSPy RLM. Each wrapper:

- Sets `fn.__name__` to the tool name (DSPy uses this for tool routing).
- Sets `fn.__doc__` to the tool description.
- Calls `self.call_tool(server, name, kwargs)` when invoked.

**Returns:** `list` of callable functions, one per tool definition.

### Usage Example

```python
from lab.nine_super_deep_research.mcp.client import MCPClient

client = MCPClient("config/mcp_servers.json")
tool_defs = client.connect_all()

# Call a tool directly
result = client.call_tool("fetch-server", "fetch", {"url": "https://example.com"})

# Build DSPy-compatible functions
tool_fns = client.build_tool_fns(tool_defs)

client.close()
```

---

## Architecture Flow

```
User Query
    |
    v
ResearchOrchestrator.run()
    |
    +--[loop]--+-- frontier.next_action() → ResearchDirection
    |           |
    |           +-- Agent dispatch:
    |           |     depth==0  → Explorer (search, discover directions)
    |           |     conf<0.6  → DeepReader (fetch, extract findings)
    |           |     otherwise → Synthesizer (cross-source synthesis)
    |           |
    |           +-- absorb_findings() → update UCB scores + spawn sub-directions
    |           |
    |           +-- Every 2 iters → Critic (quality eval + follow-ups)
    |           |
    |           +-- Every 3 iters → Heartbeat (reflect, consolidate, stagnation check)
    |           |
    |           +-- LSEOptimizer.record_run() (quality delta tracking)
    |
    +-- SkillConsolidator.consolidate() → save final skill
    |
    v
Report dict (iterations, frontier, memory, findings, trajectories, trend)
```

### Component Responsibilities

| Component | Role |
|-----------|------|
| `ResearchFrontier` | Decides WHAT to explore next (UCB-driven) |
| `Explorer` (RLM) | Discovers NEW research directions |
| `DeepReader` (RLM) | Extracts structured findings FROM a source |
| `Synthesizer` (RLM) | Cross-references findings, finds gaps |
| `Critic` (RLM) | Evaluates quality, suggests follow-ups |
| `KnowledgeGraph` | Stores findings + typed relationships |
| `MemoryStore` | Persists skills, logs, and graph |
| `LSEOptimizer` | Tracks research quality improvement over iterations |
| `SkillConsolidator` | Extracts reusable patterns from agent trajectories |
| `MCPClient` | Provides search, fetch, crawl tools to agents |

### Key Design Decisions

- **UCB over greedy selection.** Unseen topics get infinite UCB score, guaranteeing exploration before exploitation. The exploration constant `C = 1.4` balances novelty vs. depth.
- **Agent dispatch by confidence.** Topics at different confidence levels get different agents: explorer for new ground, deep reader for deeper analysis, synthesizer for mature topics.
- **Critic as stagnation breaker.** Beyond its regular review cycle, the critic activates when LSE trend goes negative for 2 consecutive iterations, ensuring the system self-corrects.
- **Skills as compressed experience.** Both heartbeat and final consolidation turn agent trajectories into reusable patterns, so the platform learns from past research sessions.
- **MCP decoupling.** Tools are discovered dynamically from MCP server config, so the platform works with any set of tools (search, fetch, crawl, code execution) without code changes.
