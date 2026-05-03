# 10 — Dapr Deep Research: API Reference

> Multi-agent research platform combining **dapr-agents** (DurableAgent, workflows, StateStoreService) with **DSPy 3.2** (RLM, ChainOfThought, BestOfN, Refine, MultiChainComparison, BootstrapFewShot, BAMLAdapter).
>
> Source: `lab/10_dapr_deep_research/` (16 files across 6 packages + `lab/shared/research.py`)

## Package Structure

```
lab/10_dapr_deep_research/
├── __init__.py
├── __main__.py                          # Entry: delegates to cli.main()
├── cli.py                              # CLI with dynamic commands, _create_agents factory
├── config/
│   └── mcp_servers.json                # MCP server definitions (crawl4ai, fetch, openrouter)
├── agents/
│   ├── __init__.py
│   └── research_agents.py              # Pydantic models, DSPy signatures, 4 agent classes
├── evolution/
│   ├── __init__.py
│   ├── lse.py                          # LSEOptimizer — quality evaluation + improvement tracking
│   └── trace2skill.py                  # SkillConsolidator — trajectory pattern extraction
├── memory/
│   ├── __init__.py
│   ├── frontier.py                     # InMemoryFrontier — dict-backed UCB frontier (no Dapr)
│   ├── dapr_frontier.py                # DaprFrontier — Redis-backed, batch saturation + cache
│   ├── noop_store.py                   # NoopStore — in-memory StateStoreService subclass
│   └── skills/                         # Saved skill files (from SkillConsolidator)
├── orchestrator/
│   ├── __init__.py
│   └── workflow.py                     # ResearchWorkflow — LSE-driven orchestration loop
├── mcp/
│   ├── __init__.py
│   ├── client.py                       # MCPClient — async-to-sync MCP transport bridge
│   └── bridge.py                       # MCPBridge — dual-format tool provider
├── resources/
│   ├── llm-provider.yaml               # Dapr conversation.openai → api.deepseek.com
│   ├── state-store.yaml                # Dapr state.redis at localhost:6379
│   ├── pubsub.yaml                     # Dapr pubsub.redis at localhost:6379
│   └── agent-registry.yaml             # Dapr state.redis for agent registry
├── dapr-multi-app-run.yaml             # 5-agent Dapr multi-app run config
├── docker-compose.yml                  # Crawl4AI container
├── README.md
└── [shared] lab/shared/research.py      # ResearchDirection, ResearchFrontier ABC, constants
```

---

## agents/research_agents.py

### Pydantic Models

Structured output models used by `dspy.RLM` for typed extraction. All inherit from `pydantic.BaseModel`.

#### `FoundDirection`

A single research direction discovered during exploration.

| Field | Type | Description |
|-------|------|-------------|
| `topic` | `str` | Research topic discovered |
| `relevance` | `str` | Why this matters |
| `seed_query` | `str` | Search query to explore further |

#### `ExplorationResult`

Container for multiple discovered directions.

| Field | Type | Description |
|-------|------|-------------|
| `directions` | `list[FoundDirection]` | Discovered research directions |

#### `ExtractedFinding`

A single claim extracted from a source during deep reading.

| Field | Type | Description |
|-------|------|-------------|
| `claim` | `str` | Main claim or finding |
| `evidence` | `str` | Supporting evidence |
| `source` | `str` | Source URL |
| `confidence` | `str` | Confidence level: `high`, `medium`, or `low` |

#### `DeepReadResult`

Container for all findings from a deep read operation.

| Field | Type | Description |
|-------|------|-------------|
| `findings` | `list[ExtractedFinding]` | Extracted findings |
| `summary` | `str` | Content summary |

#### `SynthesisReport`

Cross-source synthesis output.

| Field | Type | Description |
|-------|------|-------------|
| `synthesis` | `str` | Cross-source synthesis |
| `key_insights` | `list[str]` | Key insights |
| `gaps` | `list[str]` | Knowledge gaps |

#### `Critique`

Quality critique of research output.

| Field | Type | Description |
|-------|------|-------------|
| `strengths` | `list[str]` | Strengths |
| `weaknesses` | `list[str]` | Weaknesses |
| `follow_ups` | `list[str]` | Next directions |

---

### DSPy Signatures

Ten DSPy signatures. Each is a `dspy.Signature` subclass with typed `InputField` and `OutputField` annotations. Used by `dspy.ChainOfThought`, `dspy.RLM`, `dspy.BestOfN`, `dspy.MultiChainComparison`, and `dspy.Refine` modules.

#### `GenerateHypotheses`

Generate diverse research hypotheses from a topic.

- **Input**: `topic: str`
- **Output**: `hypotheses: list[str]` — diverse hypotheses to explore

Used by: `ExplorerAgent._hypothesis_gen` (CoT), `ExplorerAgent._hypothesis_best` (BestOfN)

#### `CrossValidateFindings`

Cross-validate findings from multiple sources for consistency.

- **Input**: `findings_summary: str`
- **Output**:
  - `validated_claims: list[str]` — claims supported by multiple sources
  - `contradictions: list[str]` — conflicting information found

Used by: `DeepReaderAgent._cross_validator` (CoT)

#### `SynthesizeAcrossSources`

Synthesize findings from multiple sources into a coherent report.

- **Input**: `task: str`
- **Output**:
  - `synthesis: str` — cross-source synthesis
  - `key_insights: list[str]` — key insights
  - `gaps: list[str]` — knowledge gaps

Used by: `SynthesizerAgent._synthesizer` (CoT)

#### `SelectAgent`

Select the best agent for a research task based on frontier state.

- **Input**:
  - `exploration_depth: int` — how many times explored (0 = new)
  - `confidence: float` — current confidence 0-1
  - `topic: str`
- **Output**: `selected_agent: str` — one of `explorer`, `deepreader`, or `synthesizer`

Used by: `ResearchWorkflow._agent_selector` (CoT)

#### `ComputeConfidenceDelta`

Determine confidence increase from research findings.

- **Input**:
  - `topic: str`
  - `agent_type: str` — `explorer`, `deepreader`, or `synthesizer`
  - `num_findings: int` — number of findings collected
  - `findings_summary: str` — key findings summary
  - `exploration_depth: int` — times explored
- **Output**:
  - `confidence_delta: float` — confidence increase 0.0-0.5
  - `reasoning: str` — why this delta

Used by: `ResearchWorkflow._conf_delta` (CoT)

#### `AssessSaturation`

Assess whether continued exploration of a direction is still valuable.

- **Input**:
  - `topic: str`
  - `confidence: float`
  - `exploration_depth: int`
  - `source_count: int`
- **Output**:
  - `is_saturated: bool` — whether saturated
  - `reasoning: str` — why

Used by: `DaprFrontier._saturation` (CoT)

#### `CritiqueReasoning`

Critique research findings and identify gaps.

- **Input**: `research_summary: str`
- **Output**: `critique: str` — critical analysis

Used by: `CriticAgent._comparison` (MultiChainComparison)

#### `QualityEvaluation`

Evaluate research iteration quality based on coverage, depth, and novelty.

- **Input**:
  - `num_directions: int` — number of active research directions
  - `num_findings: int` — number of findings collected
  - `frontier_saturation: float` — fraction of directions at high confidence (0-1)
- **Output**:
  - `quality_score: float` — research quality from 0.0 to 1.0
  - `explanation: str` — why this score was assigned

Used by: `LSEOptimizer._evaluator` (CoT)

#### `ExtractPatterns`

Extract reusable reasoning patterns from an execution trajectory.

- **Input**: `trajectory_context: str` — execution steps with reasoning, code, and output
- **Output**:
  - `error_patterns: str` — what went wrong and why
  - `success_patterns: str` — effective reasoning patterns to reuse
  - `improvement_suggestion: str` — how to improve next attempt

Used by: `SkillConsolidator._extractor` (CoT)

#### `AssessDirectionSaturation`

Determine if a research direction is saturated.

- **Input**:
  - `topic: str`
  - `confidence: float`
  - `exploration_depth: int`
  - `source_count: int`
- **Output**:
  - `is_saturated: bool` — whether saturated
  - `reasoning: str` — why

Used by: `DaprFrontier._saturation` (CoT)

---

### Helper

#### `_rlm_factory(signature, max_iter, max_calls, tools)`

Factory function that constructs a `dspy.RLM` instance with consistent configuration.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `str` | — | String signature for the RLM (e.g. `"task: str -> result: ExplorationResult"`) |
| `max_iter` | `int` | `20` | Maximum iterations for the RLM loop |
| `max_calls` | `int` | `50` | Maximum LLM calls allowed |
| `tools` | `list` or `None` | `None` | List of tool functions available to the RLM |

**Returns**: `dspy.RLM` — configured RLM instance with `verbose=False`.

---

### Agent Classes

All agents inherit from `DurableAgent` (dapr-agents) and use `@workflow_entry` for durable workflow execution. Each wraps a DSPy pipeline that combines `dspy.RLM` (tool-equipped reasoning) with `dspy.ChainOfThought` (structured reasoning), and optionally `dspy.BestOfN`, `dspy.Refine`, or `dspy.MultiChainComparison`.

#### `ExplorerAgent`

**Role**: Research Explorer — discovers novel research directions and topics.

**Inherits**: `DurableAgent`

**DSPy Modules**:
- `_rlm`: `dspy.RLM("task: str -> result: ExplorationResult")` with search tools, 8 max iterations, 12 max LLM calls
- `_hypothesis_gen`: `dspy.ChainOfThought(GenerateHypotheses)`
- `_hypothesis_best`: `dspy.BestOfN(dspy.ChainOfThought(GenerateHypotheses), N=3)` — reward: count of hypotheses, threshold 0.5

**Dapr Configuration**:
- LLM: `DaprChatClient(component_name="llm-provider")`
- Tools: `bridge.get_agent_tools()` — MCP tools in dapr-agents `AgentTool` format
- State store: `StateStoreService(store_name="research-state")`
- Execution: `max_iterations=10`, `tool_execution_mode=ToolExecutionMode.PARALLEL`

---

##### `__init__(self, bridge: MCPBridge, **kwargs)`

Initialize the ExplorerAgent with a bridge to MCP tools.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `bridge` | `MCPBridge` | Bridge providing tools in both DSPy and dapr-agents formats |
| `**kwargs` | — | Additional keyword arguments forwarded to `DurableAgent.__init__()` |

**DSPy setup**: Filters dspy tools by name (`search`, `chat`, `model_list`), constructs RLM with search-capable tools, initializes ChainOfThought and BestOfN for hypothesis generation.

---

##### `compile(self, trainset: list[dspy.Example], student_lm: dspy.LM | None = None)`

Compile the hypothesis generator using `BootstrapFewShot`.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trainset` | `list[dspy.Example]` | — | Labeled examples for BootstrapFewShot |
| `student_lm` | `dspy.LM` or `None` | `None` | Student LM for distillation. If provided, a fresh `ChainOfThought(GenerateHypotheses)` is created with the student LM. Otherwise, self-compiles (teacher = student). |

**Behavior**:
1. Teacher = `self._hypothesis_gen` (current ChainOfThought)
2. If `student_lm` provided, creates a new `ChainOfThought(GenerateHypotheses)` with the student LM
3. Runs `BootstrapFewShot.compile()` with metric: `len(pred.hypotheses) > 0`, `max_bootstrapped_demos=4`, `max_labeled_demos=2`
4. If `student_lm` provided, sets the student LM on the compiled module
5. Replaces `self._hypothesis_gen` with the compiled module

---

##### `explore(self, ctx, input: dict) -> dict` ( `@workflow_entry`)

Execute the exploration workflow.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ctx` | `WorkflowContext` | Dapr workflow context (injected by `@workflow_entry`) |
| `input` | `dict` | Must contain key `"topic"` (str) |

**Workflow**:
1. Runs `self._rlm(task=input["topic"])` for tool-assisted discovery
2. Runs `self._hypothesis_gen(topic=input["topic"])` for structured hypothesis generation
3. Runs `self._hypothesis_best(topic=input["topic"])` for best-of-N hypothesis sampling
4. Aggregates directions from RLM result, hypotheses from CoT, and hypotheses from BestOfN
5. Deduplicates by topic via `set()`
6. Persists result to workflow state: `ctx.set_state("explorer_result", ...)`

**Returns**: `dict` with keys:
- `topic` (str) — the input topic
- `directions` (list[dict]) — each with key `"topic"` (str)

---

#### `DeepReaderAgent`

**Role**: Content Analyst — extracts structured findings from web content.

**Inherits**: `DurableAgent`

**DSPy Modules**:
- `_rlm`: `dspy.RLM("topic: str, url: str -> result: DeepReadResult")` with fetch tools, 10 max iterations, 16 max LLM calls
- `_cross_validator`: `dspy.ChainOfThought(CrossValidateFindings)`

**Dapr Configuration**:
- LLM: `DaprChatClient(component_name="llm-provider")`
- Tools: `bridge.get_agent_tools()`
- State store: `StateStoreService(store_name="research-state")`
- Execution: `max_iterations=10`, `tool_execution_mode=ToolExecutionMode.PARALLEL`

---

##### `__init__(self, bridge: MCPBridge, **kwargs)`

Initialize the DeepReaderAgent.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `bridge` | `MCPBridge` | Bridge for dual-format tool access |
| `**kwargs` | — | Additional keyword arguments for `DurableAgent` |

**DSPy setup**: Filters dspy tools by name (`fetch`, `md`, `crawl`), constructs RLM with fetch-capable tools, initializes ChainOfThought for cross-validation.

---

##### `compile(self, trainset: list[dspy.Example], student_lm: dspy.LM | None = None)`

Compile the cross-validator using `BootstrapFewShot`.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trainset` | `list[dspy.Example]` | — | Labeled examples |
| `student_lm` | `dspy.LM` or `None` | `None` | Student LM for distillation |

**Metric**: `hasattr(pred, "validated_claims") and len(pred.validated_claims) > 0`
**BootstrapFewShot**: `max_bootstrapped_demos=4`, `max_labeled_demos=2`

---

##### `deep_read(self, ctx, input: dict) -> dict` ( `@workflow_entry`)

Execute the deep reading workflow.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ctx` | `WorkflowContext` | Dapr workflow context |
| `input` | `dict` | Must contain `"topic"` (str). Optional `"url"` (str) overrides the URL used. |

**Workflow**:
1. Resolves URL: `input.get("url") or input["topic"]`
2. Runs `self._rlm(topic=input["topic"], url=url)` for tool-assisted content extraction
3. Runs `self._cross_validator(findings_summary=findings_text)` for cross-validation (skipped if no findings)
4. Persists to workflow state: `ctx.set_state("deepread_result", ...)`

**Returns**: `dict` with keys:
- `topic` (str)
- `findings` (list[dict]) — each with keys `claim`, `evidence`, `source`, `confidence`
- `summary` (str)
- `validated_claims` (list[str])
- `contradictions` (list[str])

---

#### `SynthesizerAgent`

**Role**: Research Synthesizer — synthesizes findings across sources.

**Inherits**: `DurableAgent`

**DSPy Modules**:
- `_rlm`: `dspy.RLM("task: str -> result: SynthesisReport")` with all dspy tools, 8 max iterations, 12 max LLM calls
- `_synthesizer`: `dspy.ChainOfThought(SynthesizeAcrossSources)`

**Dapr Configuration**:
- LLM: `DaprChatClient(component_name="llm-provider")`
- Tools: `bridge.get_agent_tools()`
- State store: `StateStoreService(store_name="research-state")`
- Execution: `max_iterations=8`, `tool_execution_mode=ToolExecutionMode.PARALLEL`

---

##### `__init__(self, bridge: MCPBridge, **kwargs)`

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `bridge` | `MCPBridge` | Bridge for dual-format tool access |
| `**kwargs` | — | Additional keyword arguments for `DurableAgent` |

**DSPy setup**: Constructs RLM with all available dspy tools. Initializes ChainOfThought for `SynthesizeAcrossSources`.

---

##### `compile(self, trainset: list[dspy.Example], student_lm: dspy.LM | None = None)`

Compile the synthesizer using `BootstrapFewShot`.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trainset` | `list[dspy.Example]` | — | Labeled examples |
| `student_lm` | `dspy.LM` or `None` | `None` | Student LM for distillation |

**Metric**: `hasattr(pred, "synthesis") and len(pred.synthesis) > 50`
**BootstrapFewShot**: `max_bootstrapped_demos=4`, `max_labeled_demos=2`

---

##### `synthesize(self, ctx, input: dict) -> dict` ( `@workflow_entry`)

Execute the synthesis workflow.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ctx` | `WorkflowContext` | Dapr workflow context |
| `input` | `dict` | Must contain `"topic"` (str) |

**Workflow**:
1. Runs `self._rlm(task=f"Synthesize: {input['topic']}")` for tool-assisted synthesis
2. Runs `self._synthesizer(task=input["topic"])` for ChainOfThought synthesis
3. Falls back from RLM result to CoT result if RLM yields no output
4. Persists to workflow state: `ctx.set_state("synthesis_result", ...)`

**Returns**: `dict` with keys:
- `topic` (str)
- `synthesis` (str)
- `insights` (list[str])
- `gaps` (list[str])

---

#### `CriticAgent`

**Role**: Research Critic — evaluates research quality and finds gaps.

**Inherits**: `DurableAgent`

**DSPy Modules** (4-stage pipeline):
1. `_rlm`: `dspy.RLM("research_summary: str -> result: Critique")` — first-pass critique (6 max iterations, 8 max LLM calls, no tools)
2. `_comparison`: `dspy.MultiChainComparison(CritiqueReasoning, n=3)` — compares 3 critique chains
3. `_refine`: `dspy.Refine(dspy.ChainOfThought("research_summary: str, critique: str -> improved_critique: str"), N=3)` — iterative refinement (reward: `len(pred.improved_critique) > 50`, threshold 0.5)
4. `_rlm_second`: `dspy.RLM("research_summary: str, refinement_guidance: str -> result: Critique")` — second-pass refined critique (4 max iterations, 6 max LLM calls)

**Dapr Configuration**:
- LLM: `DaprChatClient(component_name="llm-provider")`
- State store: `StateStoreService(store_name="research-state")`
- Execution: `max_iterations=6` (no tools, no parallel mode)

---

##### `__init__(self, **kwargs)`

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `**kwargs` | — | Additional keyword arguments for `DurableAgent` |

Note: CriticAgent does **not** accept an `MCPBridge` — it operates without external tools.

---

##### `compile(self, trainset: list[dspy.Example], student_lm: dspy.LM | None = None)`

Compile the refine module using `BootstrapFewShot`.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trainset` | `list[dspy.Example]` | — | Labeled examples |
| `student_lm` | `dspy.LM` or `None` | `None` | Student LM for distillation |

**Metric**: `hasattr(pred, "improved_critique") and len(pred.improved_critique) > 100`
**BootstrapFewShot**: `max_bootstrapped_demos=4`, `max_labeled_demos=2`
**Special**: When `student_lm` is provided, a fresh `dspy.Refine(ChainOfThought(...))` is created with the student LM.

---

##### `critique(self, ctx, input: dict) -> dict` ( `@workflow_entry`)

Execute the critique workflow (4-stage pipeline).

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ctx` | `WorkflowContext` | Dapr workflow context |
| `input` | `dict` | Must contain `"summary"` (str) — the research summary to critique |

**Workflow**:
1. **First pass**: `self._rlm(research_summary=summary)` — produces initial Critique
2. **MultiChainComparison**: `self._comparison(research_summary=summary)` — compares 3 reasoning chains
3. **Refine**: `self._refine(research_summary=summary, critique=refine_input)` — iteratively improves the critique
4. **Second pass**: `self._rlm_second(research_summary=summary, refinement_guidance=guidance)` — refined RLM critique using improvement guidance
5. Persists to workflow state: `ctx.set_state("critique_result", ...)`

**Returns**: `dict` with keys:
- `strengths` (list[str])
- `weaknesses` (list[str])
- `follow_ups` (list[str])
- `refined` (str) — the intermediate refinement guidance text

---

## evolution/lse.py

### `QualityEvaluation` (DSPy Signature)

Evaluate research iteration quality based on coverage, depth, and novelty.

- **Input**:
  - `num_directions: int` — number of active research directions
  - `num_findings: int` — number of findings collected
  - `frontier_saturation: float` — fraction of directions at high confidence (0-1)
- **Output**:
  - `quality_score: float` — research quality from 0.0 to 1.0
  - `explanation: str` — why this score was assigned

---

### `LSERun` (dataclass)

Record of a single LSE optimization run.

| Field | Type | Description |
|-------|------|-------------|
| `strategy_id` | `str` | Identifier for the strategy |
| `quality_score` | `float` | Evaluated quality score (0.0–1.0) |
| `strategy_description` | `str` | Description of what the strategy does |
| `num_directions` | `int` | Number of active research directions |
| `num_findings` | `int` | Number of findings collected |

---

### `LSEOptimizer`

Meta-optimizer that uses DSPy ChainOfThought to evaluate research quality. Tracks improvement across runs with the reward formula `r = quality(c1) - quality(c0)`. The quality evaluator is itself compilable via `BootstrapFewShot`.

---

#### `__init__(self)`

Initialize the LSE optimizer.

**Internal state**:
- `self.runs: list[LSERun]` — run history
- `self._evaluator: dspy.ChainOfThought[QualityEvaluation]` — quality evaluator module

---

#### `compile(self, trainset: list[dspy.Example], student_lm: dspy.LM | None = None)`

Compile the quality evaluator using `BootstrapFewShot`.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trainset` | `list[dspy.Example]` | — | Labeled examples |
| `student_lm` | `dspy.LM` or `None` | `None` | Student LM for distillation |

**Metric**: `hasattr(pred, "quality_score") and 0.0 <= pred.quality_score <= 1.0`
**BootstrapFewShot**: `max_bootstrapped_demos=4`, `max_labeled_demos=2`

---

#### `compute_improvement(self, current_quality: float) -> float`

Compute improvement relative to the last run.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `current_quality` | `float` | Current iteration's quality score |

**Returns**: `float` — difference from last run's quality score. Returns `0.0` if no prior runs exist.

---

#### `record_run(self, strategy_id: str, state: dict, strategy_description: str)`

Evaluate and record a research iteration.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `strategy_id` | `str` | Identifier for this run (e.g. `"iter_3"`) |
| `state` | `dict` | Must contain keys `num_directions` (int), `num_findings` (int), `frontier_saturation` (float) |
| `strategy_description` | `str` | Human-readable description of the strategy |

**Behavior**:
1. Calls `self._evaluator(num_directions=..., num_findings=..., frontier_saturation=...)`
2. Clamps the quality score to `[0.0, 1.0]`
3. Appends a new `LSERun` to `self.runs`

---

#### `best_strategy(self) -> str | None`

Return the strategy ID of the highest-quality run.

**Returns**: `str` or `None` — the `strategy_id` of the run with the maximum `quality_score`, or `None` if no runs exist.

---

#### `improvement_trend(self) -> list[float]`

Compute the sequence of quality deltas between consecutive runs.

**Returns**: `list[float]` — per-step improvements `[s1 - s0, s2 - s1, ...]`. Returns empty list if fewer than 2 runs exist.

---

## evolution/trace2skill.py

### `ExtractPatterns` (DSPy Signature)

Extract reusable reasoning patterns from an execution trajectory.

- **Input**: `trajectory_context: str` — execution steps with reasoning, code, and output
- **Output**:
  - `error_patterns: str` — what went wrong and why
  - `success_patterns: str` — effective reasoning patterns to reuse
  - `improvement_suggestion: str` — how to improve next attempt

---

### `SkillConsolidator`

Uses DSPy ChainOfThought to extract patterns from execution trajectories. Each trajectory is processed independently, matching the Trace2Skill paper's parallel sub-agent approach.

---

#### `__init__(self, persist_dir: str | Path)`

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `persist_dir` | `str` or `Path` | Directory for saving/loading extracted skill JSON files. Created if it does not exist. |

**Internal state**:
- `self._extractor: dspy.ChainOfThought[ExtractPatterns]` — pattern extraction module

---

#### `compile(self, trainset: list[dspy.Example], student_lm: dspy.LM | None = None)`

Compile the pattern extractor using `BootstrapFewShot`.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trainset` | `list[dspy.Example]` | — | Labeled examples |
| `student_lm` | `dspy.LM` or `None` | `None` | Student LM for distillation |

**Metric**: `hasattr(pred, "error_patterns") and len(pred.error_patterns) > 10`
**BootstrapFewShot**: `max_bootstrapped_demos=4`, `max_labeled_demos=2`

---

#### `consolidate(self, trajectories: list[dict]) -> dict`

Process a list of trajectories and extract patterns.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `trajectories` | `list[dict]` | List of trajectory dicts. Each can be a list of steps or a dict with key `"trajectory"`. |

**Processing**:
1. For each trajectory, builds text representation via `_build_trajectory_text()` (up to 8 steps, each with reasoning/code/output truncated)
2. Runs `self._extractor(trajectory_context=text)` per trajectory
3. Collects error patterns and success patterns (filtered to length > 10 chars)
4. Extracts up to 5 demonstrations from the last step of the first 5 trajectories

**Returns**: `dict` with keys:
- `error_patterns` (list[dict], max 10) — each `{"symptom": ..., "extracted_by": "dspy.CoT"}`
- `success_patterns` (list[dict], max 10) — each `{"pattern": ..., "extracted_by": "dspy.CoT"}`
- `demonstrations` (list[dict], max 5) — each `{"reasoning": ..., "output": ...}`
- `n_trajectories` (int)

---

#### `save_skill(self, name: str, skill: dict)`

Save extracted skill to a JSON file.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Filename stem (`.json` appended) |
| `skill` | `dict` | Skill data to serialize |

**Behavior**: Writes to `{persist_dir}/{name}.json` with `indent=2`.

---

#### `load_skills(self) -> list[dict]`

Load all saved skills from the persist directory.

**Returns**: `list[dict]` — skill dicts loaded from JSON files sorted by filename in reverse order (newest first based on filename).

---

## memory/frontier.py

### `InMemoryFrontier`

An in-memory `ResearchFrontier` implementation. No Dapr sidecar or Redis needed.

**Constructor:**
```python
frontier = InMemoryFrontier()
# self.directions: dict[str, ResearchDirection] = {}
# self.total_explorations: int = 0
```

All methods follow the `ResearchFrontier` ABC interface. Uses `dict` for O(1) lookups and `_active_count()` computed from actual data (no bug-prone increment-only cache).

---

## memory/dapr_frontier.py

### `AssessBatchSaturation` (DSPy Signature)

Assess saturation for multiple research directions in a single call (batch replaces N+1 per-direction calls).

- **Input**:
  - `directions_json: str` — JSON array of `{topic, confidence, exploration_depth, source_count}`
- **Output**:
  - `saturated_indices: list[int]` — indices of saturated directions

---

### `DaprFrontier`

ResearchFrontier persisted via Dapr `StateStoreService`. Survives process restarts by storing the frontier state in Redis through Dapr's state management building block.

---

#### `__init__(self, store_name: str = "research-state", key: str = "frontier")`

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_name` | `str` | `"research-state"` | Dapr state store component name |
| `key` | `str` | `"frontier"` | State store key for the frontier data |

**Internal state**:
- `self._store: StateStoreService` — Dapr state store connection
- `self._key: str` — persistence key
- `self.directions: list[ResearchDirection]` — all tracked directions
- `self._total_explorations: int` — accumulated exploration count
- `self.directions: dict[str, ResearchDirection]` — O(1) dict lookup, not O(n) list scan
- `self._saturation_batch: dspy.ChainOfThought[AssessBatchSaturation]` — batch saturation evaluator
- `self._saturation_cache: set[int] | None` — cached saturation indices, invalidated on mutations

**Init behavior**: Calls `self._load()` to restore state from Redis, sets up batch saturation with caching.

---

#### `compile(self, trainset: list[dspy.Example], student_lm: dspy.LM | None = None)`

Compile the saturation assessor using `BootstrapFewShot`.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trainset` | `list[dspy.Example]` | — | Labeled examples |
| `student_lm` | `dspy.LM` or `None` | `None` | Student LM for distillation |

**Metric**: `hasattr(pred, "is_saturated")`
**BootstrapFewShot**: `max_bootstrapped_demos=4`, `max_labeled_demos=2`

---

#### `_load(self)`

Restore frontier state from Dapr state store.

**Behavior**:
1. Calls `self._store.load(key=self._key)`
2. If data exists, deserializes `ResearchDirection` list and `total_explorations` counter

---

#### `_save(self)`

Persist frontier state to Dapr state store.

**Behavior**:
1. Serializes all directions via `to_dict()` and writes with `total_explorations` to `self._store.save(key=self._key, value=...)`

---

#### `total_explorations` (property)

**Returns**: `int` — total number of explorations across all directions.

---

#### `seed_from_query(self, query: str)`

Add an initial research direction from a query.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | The research query to add as a direction |

**Behavior**: Creates a new `ResearchDirection` with zero confidence and exploration. Persists to state store.

---

#### `seed_from_directions(self, topics: list[str], parent: str | None = None)`

Add multiple research directions, avoiding duplicates.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `topics` | `list[str]` | — | Topics to add as directions |
| `parent` | `str` or `None` | `None` | Parent topic to link new directions to |

**Behavior**: Skips topics already present in `self.directions` (checked by `topic` equality). Persists to state store.

---

#### `next_action(self) -> ResearchDirection | None`

Select the next direction to explore using UCB + saturation assessment.

**Behavior**:
1. For each direction, runs `self._saturation(topic=..., confidence=..., exploration_depth=..., source_count=...)`
2. Filters to non-saturated directions
3. Returns the candidate with the highest `ucb_score()`

**Returns**: `ResearchDirection` or `None` — the best unsaturated direction, or `None` if all are saturated or no directions exist.

---

#### `absorb_findings(self, topic: str, confidence_delta: float, sources: int, follow_ups: list[str])`

Update a direction with new findings and spawn follow-up directions.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `topic` | `str` | The topic to update |
| `confidence_delta` | `float` | Confidence increase (clamped to maintain max 1.0) |
| `sources` | `int` | Number of new sources to add to `source_count` |
| `follow_ups` | `list[str]` | Follow-up topics to seed as new directions |

**Behavior**:
1. Finds the matching direction by topic
2. Updates: `confidence = min(1.0, confidence + confidence_delta)`, increments `exploration_depth` and `source_count`, sets `last_updated`
3. Increments `_total_explorations`
4. For each follow-up not already a direction, creates a new `ResearchDirection` linked to the parent topic
5. Persists to state store

---

#### `saturated(self) -> bool`

Check whether all directions are saturated.

**Behavior**: Runs saturation assessment on every direction using `_saturation`. A direction is saturated if `pred.is_saturated` is `True`.

**Returns**: `bool` — `True` if every direction is saturated.

---

#### `summary(self) -> str`

Return a human-readable summary of frontier state.

**Returns**: `str` — formatted as `"{active} active, {explored} explored, {total_explorations} total explorations"`. A direction is "explored" if `confidence >= 0.9`, otherwise "active".

---

## orchestrator/workflow.py

### `ResearchWorkflow`

Orchestrator agent that runs the LSE-driven research loop. Each iteration selects the next direction from `DaprFrontier`, dispatches agents via `call_agent()`, absorbs findings, and tracks LSE improvement. Uses `yield` checkpoints for durable workflow state.

**Inherits**: `DurableAgent`

---

#### `__init__(self, frontier: DaprFrontier, **kwargs)`

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `frontier` | `DaprFrontier` | Shared frontier instance (Redis-backed) |
| `**kwargs` | — | Additional keyword arguments for `DurableAgent` |

**Internal state**:
- `self.frontier: DaprFrontier` — research frontier
- `self.lse: LSEOptimizer` — LSE meta-optimizer
- `self.all_findings: list[str]` — JSON-serialized findings from all iterations
- `self._agent_selector: dspy.ChainOfThought[SelectAgent]` — agent dispatch module
- `self._conf_delta: dspy.ChainOfThought[ComputeConfidenceDelta]` — confidence delta module
- `self._evaluate: dspy.Evaluate` — evaluation wrapper (configured with empty devset, no-op metric)

**Dapr Configuration**:
- LLM: `DaprChatClient(component_name="llm-provider")`
- State store: `StateStoreService(store_name="research-state")`
- Execution: `max_iterations=30`

---

#### `compile(self, trainset: list[dspy.Example], student_lm: dspy.LM | None = None)`

Compile BOTH `_agent_selector` and `_conf_delta` using `BootstrapFewShot`.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trainset` | `list[dspy.Example]` | — | Labeled examples (shared across both compilations) |
| `student_lm` | `dspy.LM` or `None` | `None` | Student LM for distillation |

**Behavior**:
1. Iterates over `"_agent_selector"` and `"_conf_delta"`
2. For each, if `student_lm` provided, creates a fresh `ChainOfThought` with the appropriate signature
3. Compiles via `BootstrapFewShot` with signature-specific metrics:
   - `_agent_selector`: `pred.selected_agent in ("explorer", "deepreader", "synthesizer")`
   - `_conf_delta`: `0.0 <= pred.confidence_delta <= 0.5`
4. `BootstrapFewShot` config: `max_bootstrapped_demos=4`, `max_labeled_demos=2`

---

#### `run_research(self, ctx, input: dict) -> dict` ( `@workflow_entry`, generator)

Execute the full LSE-driven research loop. This is a **generator coroutine** that yields at checkpoint points for durable execution.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ctx` | `WorkflowContext` | Dapr workflow context |
| `input` | `dict` | Must contain `"query"` (str). Optional `"max_iterations"` (int, default 6). |

**Workflow** (generator with `yield` checkpoints):

1. **Initialization**: Records `research_started_at`, seeds frontier from query
2. **Main loop** (up to `max_iterations`):
   a. Gets `next_action()` from frontier; breaks if `None`
   b. Calls `_agent_selector` to pick agent type
   c. **Explorer branch**: Calls `call_agent(ctx, "explore", ...)` on `explorer-agent` app. Seeds follow-up directions from results. Computes confidence delta via `_conf_delta`.
   d. **DeepReader branch**: Calls `call_agent(ctx, "deep_read", ...)` on `deepreader-agent` app. Absorbs findings with computed delta.
   e. **Synthesizer branch**: Calls `call_agent(ctx, "synthesize", ...)` on `synthesizer-agent` app. Seeds gaps as new directions.
   f. Every 3 iterations: yields heartbeat state (frontier summary, findings count)
   g. Records LSE run with frontier state
3. **Finalization**: Records completion timestamp, final iteration count, findings count

**Yields** (checkpoints):
- `ctx.set_state("research_started_at", ...)` — ISO timestamp
- `ctx.set_state("frontier_summary", ...)` — initial frontier state
- `ctx.set_state("current_iteration", int)` — each iteration
- `ctx.set_state("heartbeat_frontier", ...)` — every 3 iterations
- `ctx.set_state("heartbeat_findings_count", int)` — every 3 iterations
- `ctx.set_state(f"lse_iter_{iteration}", dict)` — each iteration
- `call_agent(...)` — cross-agent invocations
- `ctx.set_state("research_completed_at", ...)` — completion timestamp
- `ctx.set_state("final_iterations", int)`
- `ctx.set_state("final_findings_count", int)`

**Returns**: `dict` with keys:
- `iterations` (int) — total iterations executed
- `frontier` (str) — `frontier.summary()` text
- `findings_count` (int) — total findings collected
- `improvement_trend` (list[float]) — LSE improvement deltas

---

## memory/noop_store.py

### `NoopStore`

In-memory `StateStoreService` subclass. No Dapr sidecar needed — stores everything in a dict.

```python
store = NoopStore()
store.save(key="my-key", value={"data": 42})
result = store.load(key="my-key")  # {"data": 42}
```

Used in `_create_agents()` to let all 4 agent classes run without Dapr infrastructure.

---

## mcp/client.py

### `MCPClient`

Async-to-sync MCP transport bridge. Manages multiple MCP server connections (stdio and SSE) in a background event loop thread.

---

#### `__init__(self, config_path: str)`

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `config_path` | `str` | Path to JSON config file with `mcpServers` key |

**Config format** (`mcp_servers.json`):
```json
{
  "mcpServers": {
    "server-name": {
      "type": "stdio" | "sse",
      "command": "...",
      "args": ["..."],
      "url": "..." (SSE only),
      "enabled": true | false,
      "description": "...",
      "env": {...} (optional)
    }
  }
}
```

**Behavior**: Creates a daemon background thread running an `asyncio` event loop.

---

#### `connect_all(self) -> list[dict]`

Connect to all enabled MCP servers.

**Returns**: `list[dict]` — flattened list of tool definitions from all servers, each with keys:
- `server` (str) — server name
- `name` (str) — tool name
- `description` (str) — tool description
- `inputSchema` (dict) — JSON schema for tool parameters

**Behavior**: Iterates `mcpServers` config, skipping disabled servers. For each server, calls either `_connect_sse` or `_connect_stdio`.

---

#### `call_tool(self, server: str, tool_name: str, arguments: dict) -> str`

Call a tool on a connected MCP server.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `server` | `str` | Server name from config |
| `tool_name` | `str` | Tool name to invoke |
| `arguments` | `dict` | Tool arguments |

**Returns**: `str` — concatenated text content from all response parts (text and resource types).

---

#### `close(self)`

Clean up all connections and stop the event loop.

**Behavior**: Awaits all server close coroutines, stops the event loop, and joins the background thread.

---

### Internal: `_ServerCtx` (dataclass)

| Field | Type | Description |
|-------|------|-------------|
| `session` | `ClientSession` | MCP client session |
| `close_coro` | `Any` | Coroutine for async cleanup |

---

## mcp/bridge.py

### `MCPBridge`

Wraps `MCPClient` to produce tools in both DSPy RLM format and dapr-agents `AgentTool` format from a single set of tool definitions.

---

#### `__init__(self, client: MCPClient, tool_defs: list[dict])`

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `client` | `MCPClient` | Connected MCP client |
| `tool_defs` | `list[dict]` | Tool definitions from `MCPClient.connect_all()` |

---

#### `get_dspy_tools(self) -> list[callable]`

Return tool callables suitable for `dspy.RLM(tools=...)`.

**Returns**: `list[callable]` — functions with `__name__` set to the tool name and `__doc__` set to the tool description. Each function accepts `**kwargs` and delegates to `MCPClient.call_tool()`.

---

#### `get_agent_tools(self) -> list[AgentTool]`

Return `AgentTool` list suitable for `DurableAgent(tools=...)`.

**Returns**: `list[AgentTool]` — each with `name`, `description`, `func` (calls `MCPClient.call_tool`), and `args_model=None`.

---

## cli.py

### Global Configuration

```python
_TEACHER_LM = dspy.LM(get_lm_model())           # Default: deepseek/deepseek-v4-flash
dspy.configure(lm=_TEACHER_LM, adapter=BAMLAdapter())
```

All CLI commands use `dspy.LM` configured with the environment's `LLM_MODEL` and `BAMLAdapter` for structured Pydantic output parsing.

---

### `_get_bridge() -> MCPBridge`

Helper that creates an `MCPClient`, connects all servers, and wraps the result in an `MCPBridge`.

**Returns**: `MCPBridge`

---

### `cmd_orchestrator()`

Launch the research orchestrator as a Dapr service.

**Behavior**:
1. Creates `DaprFrontier()` (loads frontier from Redis)
2. Creates `ResearchWorkflow(frontier=frontier)`
3. Serves on `port=8000` via `AgentRunner`

---

### `cmd_explorer()`

Launch the ExplorerAgent as a Dapr service.

**Behavior**:
1. Creates MCP bridge
2. Creates `ExplorerAgent(bridge=bridge)`
3. Serves on `port=8001` via `AgentRunner`

---

### `cmd_deep_reader()`

Launch the DeepReaderAgent as a Dapr service.

**Behavior**:
1. Creates MCP bridge
2. Creates `DeepReaderAgent(bridge=bridge)`
3. Serves on `port=8002` via `AgentRunner`

---

### `cmd_synthesizer()`

Launch the SynthesizerAgent as a Dapr service.

**Behavior**:
1. Creates MCP bridge
2. Creates `SynthesizerAgent(bridge=bridge)`
3. Serves on `port=8003` via `AgentRunner`

---

### `cmd_critic()`

Launch the CriticAgent as a Dapr service.

**Behavior**:
1. Creates `CriticAgent()` (no MCP bridge needed)
2. Serves on `port=8004` via `AgentRunner`

---

### `cmd_run()`

Single-process programmatic demo (no Dapr sidecar required).

**Behavior**:
1. Connects MCP servers
2. Creates `DaprFrontier()` (loads frontier from Redis)
3. Seeds frontier with query: `"Research DSPy optimization patterns for LLM pipelines"`
4. Runs 3 iterations: `next_action()` → `absorb_findings(topic, 0.2, 1, [])` per iteration
5. Prints summary and closes MCP client

---

### `cmd_distill()`

Teacher-to-student distillation for all 8 DSPy programs.

**Behavior**:
1. Creates `teacher_lm` (`deepseek/deepseek-v4-flash`) and `student_lm` (`ollama_chat/gemma4`)
2. Instantiates all 8 compilable objects:
   - `ExplorerAgent(bridge)` — compiles `_hypothesis_gen`
   - `DeepReaderAgent(bridge)` — compiles `_cross_validator`
   - `SynthesizerAgent(bridge)` — compiles `_synthesizer`
   - `CriticAgent()` — compiles `_refine`
   - `ResearchWorkflow(frontier)` — compiles `_agent_selector` and `_conf_delta`
   - `LSEOptimizer()` — compiles `_evaluator`
   - `SkillConsolidator(...)` — compiles `_extractor`
   - `DaprFrontier()` — compiles `_saturation`
3. Calls `.compile(trainset, student_lm=student_lm)` on each (skipped if `trainset` is empty)

---

### `main()`

Entry point. Parses `--mode` argument and dispatches to the corresponding command.

**Arguments**

| Argument | Choices | Default | Description |
|----------|---------|---------|-------------|
| `--mode` | `orchestrator`, `explorer`, `deepreader`, `synthesizer`, `critic`, `run`, `distill` | `run` | Execution mode |

---

## __main__.py

```python
from .cli import main
main()
```

Delegates to `cli.main()`.

---

## Dapr Resources (YAML)

### `resources/llm-provider.yaml`

Dapr component of type `conversation.openai`, configured to use DeepSeek's API.

| Property | Value |
|----------|-------|
| `metadata.name` | `llm-provider` |
| `spec.type` | `conversation.openai` |
| `spec.metadata[0].name` | `endpoint` → `https://api.deepseek.com/v1` |
| `spec.metadata[1].name` | `model` → `deepseek-v4-flash` |

---

### `resources/state-store.yaml`

Dapr component of type `state.redis`.

| Property | Value |
|----------|-------|
| `metadata.name` | `research-state` |
| `spec.type` | `state.redis` |
| `spec.metadata[0].name` | `redisHost` → `localhost:6379` |
| `spec.metadata[1].name` | `redisPassword` → (empty) |

---

### `resources/pubsub.yaml`

Dapr component of type `pubsub.redis`.

| Property | Value |
|----------|-------|
| `metadata.name` | `research-pubsub` |
| `spec.type` | `pubsub.redis` |
| `spec.metadata[0].name` | `redisHost` → `localhost:6379` |
| `spec.metadata[1].name` | `redisPassword` → (empty) |

---

### `resources/agent-registry.yaml`

Dapr component of type `state.redis` for agent registry.

| Property | Value |
|----------|-------|
| `metadata.name` | `agent-registry` |
| `spec.type` | `state.redis` |
| `spec.metadata[0].name` | `redisHost` → `localhost:6379` |
| `spec.metadata[1].name` | `redisPassword` → (empty) |

---

### `dapr-multi-app-run.yaml`

Multi-app run configuration launching all 5 agents concurrently.

| App ID | Port | Command | Agent |
|--------|------|---------|-------|
| `orchestrator` | 8000 | `python -m lab.10_dapr_deep_research --mode orchestrator` | ResearchWorkflow |
| `explorer-agent` | 8001 | `python -m lab.10_dapr_deep_research --mode explorer` | ExplorerAgent |
| `deepreader-agent` | 8002 | `python -m lab.10_dapr_deep_research --mode deepreader` | DeepReaderAgent |
| `synthesizer-agent` | 8003 | `python -m lab.10_dapr_deep_research --mode synthesizer` | SynthesizerAgent |
| `critic-agent` | 8004 | `python -m lab.10_dapr_deep_research --mode critic` | CriticAgent |

All apps use `appProtocol: grpc` and share `resourcesPath: lab/10_dapr_deep_research/resources`.

---

### `docker-compose.yml`

Crawl4AI service for web content extraction.

| Property | Value |
|----------|-------|
| `services.crawl4ai.image` | `unclecode/crawl4ai:latest` |
| `services.crawl4ai.ports` | `11235:11235` |
| `services.crawl4ai.shm_size` | `1g` |
| `services.crawl4ai.restart` | `unless-stopped` |

---

## Module Dependency Graph

```
cli.py  ───→  agents/research_agents.py   ──→  mcp/bridge.py  ──→  mcp/client.py
         │                                     │
         ├──→  orchestrator/workflow.py  ──→  memory/dapr_frontier.py
         │         │                          │
         │         └──→  evolution/lse.py     └──→  resources/*.yaml
         │
         └──→  evolution/trace2skill.py
```

## Compile (Distillation) Targets

All 8 objects with `.compile()` methods that can be targeted for teacher/student distillation:

| # | Object | Compiled Module | Signature |
|---|--------|----------------|-----------|
| 1 | `ExplorerAgent` | `_hypothesis_gen` | `GenerateHypotheses` |
| 2 | `DeepReaderAgent` | `_cross_validator` | `CrossValidateFindings` |
| 3 | `SynthesizerAgent` | `_synthesizer` | `SynthesizeAcrossSources` |
| 4 | `CriticAgent` | `_refine` | `"research_summary, critique -> improved_critique"` |
| 5 | `ResearchWorkflow` | `_agent_selector` + `_conf_delta` | `SelectAgent` + `ComputeConfidenceDelta` |
| 6 | `LSEOptimizer` | `_evaluator` | `QualityEvaluation` |
| 7 | `SkillConsolidator` | `_extractor` | `ExtractPatterns` |
| 8 | `DaprFrontier` | `_saturation` | `AssessDirectionSaturation` |
