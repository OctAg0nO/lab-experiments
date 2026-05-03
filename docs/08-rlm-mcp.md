# 08 — Practical DSPy Patterns with MCP + RLM

> Source: `lab/08-rlm-mcp/main.py`, `lab/08-rlm-mcp/mcp_server.json`

End-to-end pipeline: scrape real web content via MCP (Crawl4AI), build a classification dataset, run 7 GFL optimization patterns, launch an RLM agent for open-ended research, and consolidate execution trajectories into persistent memory (Trace2Skill).

---

## Pipeline Overview

```
Crawl4AI MCP ──────→ Scrape URLs → chunk → label → dataset
                                                        ↓
DSPy program ───────→ 7 GFL optimization patterns
                                                        ↓
MCP tools ──────────→ RLM agent → task result
                                                        ↓
                    Trace2Skill → attempts/ notes/ skills/ (persisted)
```

---

## MCP Client Infrastructure

### `MCPClient(config_path)`

Async-to-sync bridge for MCP servers (stdio and SSE). Runs an asyncio event loop on a background daemon thread, exposing a synchronous API.

```python
from mcp import ClientSession, StdioServerParameters

client = MCPClient("mcp_server.json")
tool_defs = client.connect_all()
```

#### Constructor

| Param | Type | Description |
|-------|------|-------------|
| `config_path` | `str` | Path to a JSON file with `mcpServers` configuration |

#### `connect_all() -> list[dict]`

Connects to all enabled MCP servers defined in the config. Returns a flat list of tool definitions.

```python
tool_defs = client.connect_all()
# Each tool_def: {"server": str, "name": str, "description": str, "inputSchema": dict}
```

Returns `list[dict]` — one entry per tool across all servers.

Config format (`mcp_server.json`):

```json
{
  "mcpServers": {
    "crawl4ai": {
      "description": "Web crawling and content extraction",
      "enabled": true,
      "type": "sse",
      "url": "http://localhost:11235/mcp/sse"
    },
    "fetch": {
      "description": "Fetch URLs and extract markdown",
      "enabled": true,
      "type": "stdio",
      "command": "uvx",
      "args": ["mcp-server-fetch"]
    }
  }
}
```

| Config field | Type | Description |
|-------------|------|-------------|
| `type` | `"stdio"` or `"sse"` | Transport protocol |
| `command` | `str` | Binary to spawn (stdio only) |
| `args` | `list[str]` | Arguments (stdio only) |
| `url` | `str` | SSE endpoint URL (SSE only) |
| `env` | `dict` | Environment variables (stdio only) |
| `enabled` | `bool` | Set to `false` to skip this server |

#### `call_tool(server, tool_name, arguments) -> str`

Call an MCP tool and return string content. Handles text, resource, and unknown content types.

```python
content = client.call_tool("crawl4ai", "md", {"url": "https://dspy.ai"})
```

| Param | Type | Description |
|-------|------|-------------|
| `server` | `str` | Server name from the config |
| `tool_name` | `str` | Tool name (e.g., `"md"`, `"fetch"`) |
| `arguments` | `dict` | Tool arguments as a JSON-compatible dict |

Returns `str` — concatenated text content from all MCP response parts.

#### `find_tool(tool_defs, server, name) -> dict | None`

Look up a tool definition by server and name.

```python
md_tool = client.find_tool(tool_defs, "crawl4ai", "md")
if md_tool:
    print(f"Found: {md_tool['name']}")
```

| Param | Type | Description |
|-------|------|-------------|
| `tool_defs` | `list[dict]` | Output from `connect_all()` |
| `server` | `str` | Server name |
| `name` | `str` | Tool name |

Returns the matching tool dict, or `None`.

#### `close()`

Clean up all server sessions and stop the background event loop thread.

```python
client.close()
```

---

## CORAL-style Persistent Memory

### `MemoryManager(base_dir)`

Filesystem-backed persistent memory inspired by CORAL. Stores execution traces, reflection notes, and reusable skills across runs.

```python
memory = MemoryManager(Path("lab/08-rlm-mcp/memory"))
print(memory.summary())  # "Memory: 3 attempts, 2 notes, 1 skills"
```

#### Directory structure

```
memory/
├── attempts/     # Raw execution traces (JSON)
├── notes/        # Reflection notes (Markdown)
└── skills/       # Reusable few-shot demonstrations (JSON)
```

#### Constructor

| Param | Type | Description |
|-------|------|-------------|
| `base_dir` | `Path` | Root directory for all memory files. Creates subdirectories on init. |

#### `save_attempt(task, trajectory, report) -> str`

Save an RLM execution trace as an attempt JSON file. Returns the file stem (timestamp + slug).

```python
stem = memory.save_attempt(task, result.trajectory, report_dict)
# Creates memory/attempts/20260301_143012_research_dspy_and_python.json
```

| Param | Type | Description |
|-------|------|-------------|
| `task` | `str` | The task description |
| `trajectory` | `list` | List of execution steps |
| `report` | `dict \| None` | Final report or result dict |

Returns `str` — the file stem (e.g., `"20260301_143012_research_dspy_and_python"`).

The stem is derived from the task description (lowercased, slugified, truncated to 40 chars) with a timestamp prefix.

#### `save_note(stem, content)`

Write a reflection note in Markdown format.

```python
memory.save_note(stem, f"RLM completed {n_steps} iterations.\nTask: ...")
```

| Param | Type | Description |
|-------|------|-------------|
| `stem` | `str` | File stem (from `save_attempt`) |
| `content` | `str` | Markdown body content |

Creates `memory/notes/{stem}.md`.

#### `save_skill(name, signature, demonstrations)`

Persist a reusable DSPy skill (signature + few-shot demonstrations) as JSON.

```python
memory.save_skill("rlm_trajectory_skills", str(rlm.signature), demos)
```

| Param | Type | Description |
|-------|------|-------------|
| `name` | `str` | Skill name (used as filename: `{name}.json`) |
| `signature` | `str` | DSPy signature string |
| `demonstrations` | `list[dict]` | Few-shot examples extracted from trajectories |

#### `load_skills() -> list[dict]`

Load all persisted skills, sorted by recency (newest first).

```python
skills = memory.load_skills()
for s in skills:
    print(s["signature"])
```

Returns `list[dict]` — each dict has `"signature"` and `"demonstrations"` keys.

#### `summary() -> str`

Return a human-readable summary of stored memory.

```python
print(memory.summary())
# "Memory: 3 attempts, 2 notes, 1 skills"
```

---

## Data Collection Pipeline

### `SCRAPE_URLS`

```python
SCRAPE_URLS = [
    ("https://dspy.ai", "documentation"),
    ("https://docs.python.org/3/tutorial/index.html", "tutorial"),
    ("https://raw.githubusercontent.com/stanfordnlp/dspy/main/README.md", "documentation"),
    ("https://raw.githubusercontent.com/stanfordnlp/dspy/main/dspy/__init__.py", "source_code"),
]
```

### `chunk_text(text, max_chars=1200) -> list[str]`

Split text into paragraph-aligned chunks, each up to `max_chars` characters. Splits on double newlines, then further splits long paragraphs on sentence boundaries.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | `str` | required | Raw text to split |
| `max_chars` | `int` | `1200` | Maximum characters per chunk |

Returns `list[str]`.

### `build_dataset(client, tool_defs) -> list[tuple[str, str, str]]`

Scrape URLs via Crawl4AI MCP `md` tool, chunk each page, and return labeled records.

```python
records = build_dataset(client, tool_defs)
# Each record: (chunk_text, category, source_url)
```

Returns `list[tuple[str, str, str]]` — up to 6 chunks per URL, labeled by category.

---

## Task and Metrics

### `ClassifyContent(dspy.Signature)`

Classify a web content chunk by source category and extract key topics.

```python
class ClassifyContent(dspy.Signature):
    """Classify web content by source type and extract key topics."""
    chunk: str = dspy.InputField()
    category: str = dspy.OutputField()
    key_topics: list[str] = dspy.OutputField()
```

| Field | Direction | Type | Description |
|-------|-----------|------|-------------|
| `chunk` | Input | `str` | Text chunk from a scraped page |
| `category` | Output | `str` | One of: `documentation`, `tutorial`, `source_code` |
| `key_topics` | Output | `list[str]` | Extracted key topics from the chunk |

### `content_metric(example, prediction, trace=None)`

Standard 3-arg metric for BootstrapFewShot and MIPROv2.

```python
def content_metric(example, prediction, trace=None):
    del trace
    return example.category == prediction.category and len(prediction.key_topics) > 0
```

Returns `bool` — category match AND at least one key topic extracted.

### `gepa_content_metric(gold, pred, trace=None, pred_name=None, pred_trace=None)`

GEPA-compatible 5-arg metric returning a float.

```python
def gepa_content_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    del trace, pred_name, pred_trace
    return float(gold.category == pred.category and len(pred.key_topics) > 0)
```

Returns `float` — `1.0` for correct, `0.0` for incorrect.

### `eval_score(evaluator, program) -> float`

```python
def eval_score(evaluator, program) -> float:
    return evaluator(program).score / 100.0
```

---

## Pydantic Models for RLM

### `ScrapedContent(BaseModel)`

```python
class ScrapedContent(BaseModel):
    url: str = Field(description="Source URL")
    category: str = Field(description="Content category")
    summary: str = Field(description="Concise summary")
    key_topics: list[str] = Field(description="Key topics found")
```

### `ResearchReport(BaseModel)`

```python
class ResearchReport(BaseModel):
    findings: list[ScrapedContent] = Field(description="Content per source")
    synthesis: str = Field(description="Cross-source synthesis")
```

Usage with `dspy.RLM`:

```python
rlm = dspy.RLM(
    "task: str -> report: ResearchReport",
    tools=mcp_tools,
    max_iterations=10,
    max_llm_calls=16,
)
result = rlm(task="...")
print(result.report.synthesis)
```

---

## All 7 Optimization Patterns

### Shared setup

```python
base_prog = dspy.ChainOfThought(ClassifyContent)
evaluator = dspy.Evaluate(devset=devset, metric=content_metric, num_threads=4, display_progress=True)
```

### Pattern 1: BootstrapFewShot

```python
bs = dspy.BootstrapFewShot(
    metric=content_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=2,
)
bs_prog = bs.compile(dspy.ChainOfThought(ClassifyContent), trainset=trainset)
```

Cheapest optimizer. Good first try for any task.

### Pattern 2: MIPROv2 `auto="light"`

```python
mipro = dspy.MIPROv2(
    metric=content_metric,
    auto="light",
    num_threads=4,
)
mipro_prog = mipro.compile(dspy.ChainOfThought(ClassifyContent), trainset=trainset)
```

Jointly optimizes instructions and demonstrations via Bayesian search.

### Pattern 3: GEPA

```python
gepa = dspy.GEPA(
    metric=gepa_content_metric,       # Note: 5-arg metric!
    max_full_evals=2,
    reflection_lm=dspy.LM("deepseek/deepseek-v4-flash"),
    num_threads=4,
)
gepa_prog = gepa.compile(dspy.ChainOfThought(ClassifyContent), trainset=trainset[:8])
```

Reflective evolution: reads failure traces, mutates instructions, selects via Pareto frontier.

### Pattern 4: Sequential GEPA → BootstrapFewShot

```python
seq_prog = dspy.ChainOfThought(ClassifyContent)
seq_gepa = dspy.GEPA(
    metric=gepa_content_metric,
    max_full_evals=1,
    reflection_lm=dspy.LM("deepseek/deepseek-v4-flash"),
    num_threads=4,
)
seq_prog = seq_gepa.compile(seq_prog, trainset=trainset[:8])
seq_bs = dspy.BootstrapFewShot(
    metric=content_metric,
    max_bootstrapped_demos=3,
    max_labeled_demos=2,
)
seq_prog = seq_bs.compile(seq_prog, trainset=trainset)
```

Instructions first (GEPA), then demonstrations (BootstrapFewShot). Order matters because demonstrations are instruction-specific.

### Pattern 5: Ensemble (best of N)

```python
candidates = []
for i in range(4):
    p = dspy.BootstrapFewShot(
        metric=content_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=2,
    ).compile(dspy.ChainOfThought(ClassifyContent), trainset=trainset)
    s = eval_score(evaluator, p)
    candidates.append((s, p))

best_score = max(candidates, key=lambda x: x[0])[0]
```

Train N candidate programs with BootstrapFewShot (different random seeds), evaluate each on dev, pick the best. Simple ensemble strategy with no voting overhead.

### Pattern 6: Teacher/Student Distillation

```python
student_lm = dspy.LM("ollama_chat/gemma4")

teacher = dspy.ChainOfThought(ClassifyContent)    # uses default LM (DeepSeek)
alone = dspy.ChainOfThought(ClassifyContent)
alone.set_lm(student_lm)                           # Gemma 4, no demos

student = dspy.ChainOfThought(ClassifyContent)
student.set_lm(student_lm)

distilled = dspy.BootstrapFewShot(
    metric=content_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=2,
).compile(student, teacher=teacher, trainset=trainset)

distilled.set_lm(student_lm)  # IMPORTANT: re-set LM after compile
```

Key:
- Teacher uses default LM (DeepSeek). Generates demonstrations.
- Student uses `student_lm` (Gemma 4 via Ollama). Runs inference with teacher's demos.
- `set_lm()` must be called **after** `compile()` because compile may reset the LM.
- Compare `alone_score` (Gemma 4, no demos) vs `distilled_score` (Gemma 4 + teacher demos).

### Pattern 7: BetterTogether

```python
bt = dspy.BetterTogether(
    metric=content_metric,
    p=dspy.GEPA(
        metric=gepa_content_metric,
        max_full_evals=1,
        reflection_lm=dspy.LM("deepseek/deepseek-v4-flash"),
        num_threads=4,
    ),
    q=dspy.MIPROv2(
        metric=content_metric,
        auto="light",
        num_threads=4,
    ),
)
bt_student = dspy.ChainOfThought(ClassifyContent)
bt_prog = bt.compile(bt_student, trainset=trainset, valset=devset, strategy="p -> q")
```

| Param | Description |
|-------|-------------|
| `metric` | Evaluation metric (3-arg) |
| `p` | First optimizer (runs GEPA) |
| `q` | Second optimizer (runs MIPROv2) |
| `strategy="p -> q"` | Chain: GEPA output feeds into MIPROv2 |

BetterTogether chains multiple optimizers in a configurable sequence. `"p -> q"` means run `p` (GEPA) first, then pipe the result through `q` (MIPROv2).

---

## RLM Agent with MCP Tools

### Configuring RLM with BAML

```python
dspy.configure(adapter=BAMLAdapter())
```

`BAMLAdapter` enables Pydantic model support in `dspy.RLM` output signatures. Required when the output field type is a Pydantic `BaseModel` subclass (like `ResearchReport`).

### Wrapping MCP tools for RLM

```python
mcp_tools = []
for td in tool_defs:
    srv, tn, desc = td["server"], td["name"], td.get("description", "")
    def make_fn(srv=srv, tn=tn, desc=desc):
        def fn(**kwargs: Any) -> str:
            return client.call_tool(srv, tn, kwargs)
        fn.__name__ = tn
        fn.__doc__ = desc
        return fn
    mcp_tools.append(make_fn())
```

Each MCP tool is wrapped as a plain Python function that delegates to `client.call_tool()`. The function's `__name__` and `__doc__` are set from the tool definition so RLM can discover and describe them.

### Creating the RLM

```python
rlm = dspy.RLM(
    "task: str -> report: ResearchReport",
    tools=mcp_tools,
    max_iterations=10,
    max_llm_calls=16,
    verbose=False,
)
```

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| Signature string | `str` | required | `"task: str -> report: ResearchReport"` |
| `tools` | `list[Callable]` | `[]` | Tool functions the RLM can call |
| `max_iterations` | `int` | `10` | Max tool-use iterations |
| `max_llm_calls` | `int` | `20` | Max total LLM calls |
| `verbose` | `bool` | `False` | Print reasoning traces |

### Running the RLM

```python
result = rlm(task="Use the Crawl4AI MCP tools to fetch documentation pages from ...")

# Access structured output
if hasattr(result, "report") and result.report is not None:
    r = result.report
    for f in r.findings:
        print(f.url, f.category, f.summary[:200])
    print(r.synthesis)

# Access execution trajectory
if hasattr(result, "trajectory"):
    for step in result.trajectory:
        print(step.get("reasoning", "")[:200])
```

The RLM returns a prediction with `.report` (the `ResearchReport` Pydantic model) and `.trajectory` (list of execution steps).

---

## Trace2Skill Consolidation

After the RLM completes, the trajectory is saved to persistent memory and successful reasoning steps are extracted as reusable demonstrations.

```python
# Save raw trajectory
stem = memory.save_attempt(task, result.trajectory, report_dict)

# Save reflection note
memory.save_note(stem, f"RLM completed {n_steps} iterations.\n...")

# Extract demonstrations from trajectory
demos = []
for step in result.trajectory:
    reasoning = step.get("reasoning", "")
    if reasoning and len(reasoning) > 50:
        demos.append({"reasoning": reasoning[:500]})

# Save as persistent skill
if demos:
    memory.save_skill("rlm_trajectory_skills", str(rlm.signature), demos)

# Load skills from previous runs
all_skills = memory.load_skills()
```

On subsequent runs, previous skills are loaded at startup:

```python
if prev_skills := memory.load_skills():
    print(f"Loaded {len(prev_skills)} skill(s) from previous runs")
```

---

## Quick Reference

| API | Purpose |
|-----|---------|
| `MCPClient(config_path)` | Async-to-sync MCP bridge (stdio + SSE) |
| `.connect_all()` | Connect to all servers, return tool definitions |
| `.call_tool(server, name, args)` | Call MCP tool, return string |
| `.find_tool(defs, server, name)` | Look up tool by server and name |
| `MemoryManager(base_dir)` | CORAL-style persistent memory |
| `.save_attempt(task, traj, report)` | Save execution trace |
| `.save_note(stem, content)` | Save reflection note |
| `.save_skill(name, sig, demos)` | Save reusable skill |
| `.load_skills()` | Load all persisted skills |
| `.summary()` | Human-readable memory summary |
| `ClassifyContent` | Signature: `chunk -> category, key_topics` |
| `content_metric(...)` | 3-arg metric for BootstrapFewShot/MIPRO |
| `gepa_content_metric(...)` | 5-arg metric for GEPA |
| `ScrapedContent(BaseModel)` | URL, category, summary, key_topics |
| `ResearchReport(BaseModel)` | Findings list + synthesis |
| `BAMLAdapter()` | Pydantic model support for RLM |
| `dspy.BetterTogether(p=GEPA, q=MIPROv2)` | Chain multiple optimizers via `strategy="p -> q"` |
| `module.batch(examples, num_threads=4)` | Parallel candidate training |
| `module.set_lm(lm)` | Isolate model choice per module |
| `dspy.RLM(sig, tools=[...])` | REPL-based code-gen agent with tool use |

### All 7 patterns at a glance

| # | Pattern | Key class | When to use |
|---|---------|-----------|-------------|
| 1 | BootstrapFewShot | `dspy.BootstrapFewShot` | First try, simple tasks |
| 2 | MIPROv2 | `dspy.MIPROv2 auto="light"` | Production, needs better instructions |
| 3 | GEPA | `dspy.GEPA` | Hard tasks, reflective evolution |
| 4 | Sequential | GEPA + BootstrapFewShot | Full pipeline: instructions then demos |
| 5 | Ensemble | Train N, pick best | Production robustness |
| 6 | Distillation | BootstrapFewShot with `teacher=` | Model compression |
| 7 | BetterTogether | `dspy.BetterTogether` | Multi-optimizer chaining |
