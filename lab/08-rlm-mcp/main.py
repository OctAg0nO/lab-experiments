"""
Practical DSPy patterns with MCP tools and RLM agent.

Pipeline:
  1. Scrape real web content via Crawl4AI MCP → build dataset
  2. Run 7 GFL optimization patterns on the scraped data
  3. RLM agent solves open-ended tasks using MCP tools
  4. Consolidate execution trajectories into reusable skills (Trace2Skill)
  5. Persistent memory across runs (CORAL-style)

Three lines of DSPy:  define metric → define program → compile.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

from pydantic import BaseModel, Field

import dspy
from dspy.adapters.baml_adapter import BAMLAdapter

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

lm = dspy.LM("deepseek/deepseek-v4-flash")
dspy.configure(lm=lm)

# ===========================================================================
# MCP Client
# ===========================================================================

@dataclass
class _ServerCtx:
    session: ClientSession
    close_coro: Any


class MCPClient:
    """Connects to MCP servers (stdio or SSE) on a background event loop thread."""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = json.load(f)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._servers: dict[str, _ServerCtx] = {}

    def _run(self, coro) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def connect_all(self) -> list[dict]:
        all_tools: list[dict] = []
        for name, cfg in self.config.get("mcpServers", {}).items():
            if cfg.get("enabled", True) is False:
                print(f"  [-] {name}: disabled")
                continue
            transport = cfg.get("type", "stdio")
            if transport == "sse":
                tools = self._run(self._connect_sse(name, cfg["url"]))
            else:
                params = StdioServerParameters(
                    command=cfg["command"], args=cfg.get("args", []), env=cfg.get("env"),
                )
                tools = self._run(self._connect_stdio(name, params))
            all_tools.extend(tools)
        return all_tools

    async def _connect_stdio(self, name: str, params: StdioServerParameters) -> list[dict]:
        ctx = stdio_client(params)
        read, write = await ctx.__aenter__()
        session = await ClientSession(read, write).__aenter__()
        await session.initialize()
        async def _close():
            await session.__aexit__(None, None, None)
            await ctx.__aexit__(None, None, None)
        self._servers[name] = _ServerCtx(session=session, close_coro=_close())
        return await self._list_tools(name, session)

    async def _connect_sse(self, name: str, url: str) -> list[dict]:
        ctx = sse_client(url)
        read, write = await ctx.__aenter__()
        session = await ClientSession(read, write).__aenter__()
        await session.initialize()
        async def _close():
            await session.__aexit__(None, None, None)
            await ctx.__aexit__(None, None, None)
        self._servers[name] = _ServerCtx(session=session, close_coro=_close())
        return await self._list_tools(name, session)

    @staticmethod
    async def _list_tools(name: str, session: ClientSession) -> list[dict]:
        result = await session.list_tools()
        return [
            {"server": name, "name": t.name, "description": t.description, "inputSchema": t.inputSchema}
            for t in result.tools
        ]

    def call_tool(self, server: str, tool_name: str, arguments: dict) -> str:
        """Call MCP tool and return text content. Handles all MCP content types."""
        session = self._servers[server].session
        result = self._run(session.call_tool(tool_name, arguments=arguments))
        parts = []
        for c in result.content:
            if hasattr(c, "text") and c.text:
                parts.append(c.text)
            elif hasattr(c, "resource") and c.resource:
                parts.append(str(c.resource))
            else:
                parts.append(str(c))
        return "\n".join(parts)

    def close(self):
        async def _cleanup():
            for ctx in self._servers.values():
                await ctx.close_coro
        self._run(_cleanup())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()

    def find_tool(self, tool_defs: list[dict], server: str, name: str) -> dict | None:
        return next((t for t in tool_defs if t["server"] == server and t["name"] == name), None)


# ===========================================================================
# CORAL-style persistent memory — attempts, notes, skills across runs
# ===========================================================================

class MemoryManager:
    """Filesystem-backed persistent memory (CORAL-inspired).

    Directory structure::

        memory/
        ├── attempts/     # Raw execution traces (JSON)
        ├── notes/        # Reflection notes (Markdown)
        └── skills/       # Reusable few-shot examples (JSON)
    """

    def __init__(self, base_dir: Path):
        self.base = base_dir
        self.attempts_dir = base_dir / "attempts"
        self.notes_dir = base_dir / "notes"
        self.skills_dir = base_dir / "skills"
        for d in [self.attempts_dir, self.notes_dir, self.skills_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def save_attempt(self, task: str, trajectory: list, report: dict | None) -> str:
        """Save an RLM execution trace as an attempt. Returns the file stem."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = re.sub(r"[^a-z0-9]+", "_", task.lower())[:40].strip("_")
        stem = f"{timestamp}_{slug}"
        entry = {"task": task, "trajectory": trajectory, "report": report, "timestamp": timestamp}
        (self.attempts_dir / f"{stem}.json").write_text(json.dumps(entry, indent=2, default=str))
        return stem

    def save_note(self, stem: str, content: str):
        """Write a reflection note about an attempt."""
        (self.notes_dir / f"{stem}.md").write_text(f"# {stem}\n\n{content}\n")

    def save_skill(self, name: str, signature: str, demonstrations: list[dict]):
        """Persist a reusable DSPy skill (signature + demonstrations)."""
        (self.skills_dir / f"{name}.json").write_text(
            json.dumps({"signature": signature, "demonstrations": demonstrations}, indent=2)
        )

    def load_skills(self) -> list[dict]:
        """Load all persisted skills, sorted by recency."""
        skills = []
        for f in sorted(self.skills_dir.glob("*.json"), reverse=True):
            skills.append(json.loads(f.read_text()))
        return skills

    def summary(self) -> str:
        n_attempts = len(list(self.attempts_dir.glob("*.json")))
        n_notes = len(list(self.notes_dir.glob("*.md")))
        n_skills = len(list(self.skills_dir.glob("*.json")))
        return f"Memory: {n_attempts} attempts, {n_notes} notes, {n_skills} skills"


# ===========================================================================
# Data collection — scrape real web content via Crawl4AI MCP
# ===========================================================================

SCRAPE_URLS = [
    ("https://dspy.ai", "documentation"),
    ("https://docs.python.org/3/tutorial/index.html", "tutorial"),
    ("https://raw.githubusercontent.com/stanfordnlp/dspy/main/README.md", "documentation"),
    ("https://raw.githubusercontent.com/stanfordnlp/dspy/main/dspy/__init__.py", "source_code"),
]


def chunk_text(text: str, max_chars: int = 1200) -> list[str]:
    """Split text into paragraph-sized chunks, each up to max_chars."""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    chunks = []
    for p in paragraphs:
        p = p.strip()
        if len(p) < 60:
            continue
        while len(p) > max_chars:
            split_at = p.rfind(". ", 0, max_chars)
            if split_at == -1:
                split_at = max_chars
            chunks.append(p[:split_at + 1].strip())
            p = p[split_at + 1:].strip()
        if p:
            chunks.append(p)
    return chunks


def build_dataset(client: MCPClient, tool_defs: list[dict]) -> list[tuple[str, str, str]]:
    """Scrape URLs via Crawl4AI MCP ``md`` tool and return (chunk, category, url)."""
    md_tool = client.find_tool(tool_defs, "crawl4ai", "md")
    if not md_tool:
        raise RuntimeError("Crawl4AI not available. Cannot build dataset.")
    records: list[tuple[str, str, str]] = []
    for url, category in SCRAPE_URLS:
        print(f"  Scraping {url} ...")
        content = client.call_tool("crawl4ai", "md", {"url": url})
        chunks = chunk_text(content)
        for chunk in chunks[:6]:
            records.append((chunk, category, url))
        print(f"    → {len(chunks[:6])} chunks")
    return records


# ===========================================================================
# Task definition
# ===========================================================================

class ClassifyContent(dspy.Signature):
    """Classify web content by source type and extract key topics."""
    chunk: str = dspy.InputField()
    category: str = dspy.OutputField()
    key_topics: list[str] = dspy.OutputField()


def content_metric(example, prediction, trace=None):
    del trace
    return example.category == prediction.category and len(prediction.key_topics) > 0


def gepa_content_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    del trace, pred_name, pred_trace
    return float(gold.category == pred.category and len(pred.key_topics) > 0)


def eval_score(evaluator, program) -> float:
    return evaluator(program).score / 100.0


# ===========================================================================
# Pydantic models for RLM structured output
# ===========================================================================

class ScrapedContent(BaseModel):
    url: str = Field(description="Source URL")
    category: str = Field(description="Content category")
    summary: str = Field(description="Concise summary")
    key_topics: list[str] = Field(description="Key topics found")


class ResearchReport(BaseModel):
    findings: list[ScrapedContent] = Field(description="Content per source")
    synthesis: str = Field(description="Cross-source synthesis")


# ===========================================================================
# Main
# ===========================================================================

def main():
    config_path = Path(__file__).parent / "mcp_server.json"
    if not config_path.exists():
        raise FileNotFoundError(f"MCP config not found: {config_path}")

    memory = MemoryManager(Path(__file__).parent / "memory")
    print(f"  {memory.summary()}")

    if prev_skills := memory.load_skills():
        print(f"  Loaded {len(prev_skills)} skill(s) from previous runs")

    client = MCPClient(str(config_path))
    try:
        tool_defs = client.connect_all()
        if not tool_defs:
            raise RuntimeError("No MCP tools discovered.")

        print(f"Discovered {len(tool_defs)} MCP tool(s)")
        for td in tool_defs:
            print(f"  [{td['server']}] {td['name']}")

        # ---- 1. Scrape data ------------------------------------------------
        print("\n=== Data Collection: Scraping via Crawl4AI MCP ===")
        records = build_dataset(client, tool_defs)
        if len(records) < 4:
            raise RuntimeError(f"Only {len(records)} records scraped. Need at least 4.")

        print(f"Dataset: {len(records)} chunks across {len(set(r[1] for r in records))} categories")

        labelled = [
            dspy.Example(chunk=c, category=cat, key_topics=[url]).with_inputs("chunk")
            for c, cat, url in records
        ]

        split = int(len(labelled) * 0.8)
        trainset = labelled[:split]
        devset = labelled[split:]

        # ---- 2. GFL optimization patterns ---------------------------------
        print("\n" + "=" * 70)
        print("PRACTICAL DSPy PATTERNS")
        print("=" * 70)

        evaluator = dspy.Evaluate(devset=devset, metric=content_metric, num_threads=4, display_progress=True)

        def es(prog) -> float:
            return eval_score(evaluator, prog)

        base_prog = dspy.ChainOfThought(ClassifyContent)
        base_score = es(base_prog)
        print(f"Baseline accuracy: {base_score:.0%}\n")

        results = {}

        # Pattern 1 — BootstrapFewShot
        print("--- Pattern 1: BootstrapFewShot ---")
        bs = dspy.BootstrapFewShot(metric=content_metric, max_bootstrapped_demos=4, max_labeled_demos=2)
        bs_prog = bs.compile(dspy.ChainOfThought(ClassifyContent), trainset=trainset)
        bs_score = es(bs_prog)
        results["BootstrapFewShot"] = bs_score
        print(f"  Accuracy: {bs_score:.0%}   Δ {bs_score - base_score:+.0%}   demos: {len(bs_prog.demos)}")

        # Pattern 2 — MIPROv2
        print('\n--- Pattern 2: MIPROv2 auto="light" ---')
        mipro = dspy.MIPROv2(metric=content_metric, auto="light", num_threads=4)
        mipro_prog = mipro.compile(dspy.ChainOfThought(ClassifyContent), trainset=trainset)
        mipro_score = es(mipro_prog)
        results["MIPROv2"] = mipro_score
        print(f"  Accuracy: {mipro_score:.0%}   Δ {mipro_score - base_score:+.0%}")
        preds = mipro_prog.predictors()
        if preds:
            print(f"  Instruction: {preds[0].signature.instructions[:120]}...")

        # Pattern 3 — GEPA
        print("\n--- Pattern 3: GEPA ---")
        gepa = dspy.GEPA(
            metric=gepa_content_metric, max_full_evals=2,
            reflection_lm=dspy.LM("deepseek/deepseek-v4-flash"), num_threads=4,
        )
        gepa_prog = gepa.compile(dspy.ChainOfThought(ClassifyContent), trainset=trainset[:8])
        gepa_score = es(gepa_prog)
        results["GEPA"] = gepa_score
        print(f"  Accuracy: {gepa_score:.0%}   Δ {gepa_score - base_score:+.0%}")

        # Pattern 4 — Sequential
        print("\n--- Pattern 4: Sequential GEPA → BootstrapFewShot ---")
        seq_prog = dspy.ChainOfThought(ClassifyContent)
        seq_gepa = dspy.GEPA(
            metric=gepa_content_metric, max_full_evals=1,
            reflection_lm=dspy.LM("deepseek/deepseek-v4-flash"), num_threads=4,
        )
        seq_prog = seq_gepa.compile(seq_prog, trainset=trainset[:8])
        seq_bs = dspy.BootstrapFewShot(metric=content_metric, max_bootstrapped_demos=3, max_labeled_demos=2)
        seq_prog = seq_bs.compile(seq_prog, trainset=trainset)
        seq_score = es(seq_prog)
        results["Sequential"] = seq_score
        print(f"  Accuracy: {seq_score:.0%}   Δ {seq_score - base_score:+.0%}")

        # Pattern 5 — Ensemble
        print("\n--- Pattern 5: Ensemble (best of N) ---")
        candidates = []
        for i in range(4):
            p = dspy.BootstrapFewShot(metric=content_metric, max_bootstrapped_demos=3, max_labeled_demos=2
            ).compile(dspy.ChainOfThought(ClassifyContent), trainset=trainset)
            s = es(p)
            candidates.append((s, p))
            print(f"  Candidate {i+1}: {s:.0%}")
        best_score = max(candidates, key=lambda x: x[0])[0]
        results["Ensemble"] = best_score
        print(f"  Best: {best_score:.0%}   Δ {best_score - base_score:+.0%}")

        # Pattern 6 — Teacher/Student
        print("\n--- Pattern 6: Teacher/Student distillation ---")
        student_lm = dspy.LM("ollama_chat/gemma4")
        teacher = dspy.ChainOfThought(ClassifyContent)
        teacher.set_lm(lm)
        alone = dspy.ChainOfThought(ClassifyContent)
        alone.set_lm(student_lm)
        alone_score = es(alone)
        print(f"  Student alone (Gemma 4): {alone_score:.0%}")
        student = dspy.ChainOfThought(ClassifyContent)
        student.set_lm(student_lm)
        distilled = dspy.BootstrapFewShot(metric=content_metric, max_bootstrapped_demos=4, max_labeled_demos=2
        ).compile(student, teacher=teacher, trainset=trainset)
        distilled.set_lm(student_lm)
        dist_score = es(distilled)
        results["Distillation"] = dist_score
        print(f"  Distilled: {dist_score:.0%}   Δ {dist_score - alone_score:+.0%}")
        teacher.set_lm(lm)
        print(f"  Teacher ref: {es(teacher):.0%}")

        # Pattern 7 — BetterTogether
        print("\n--- Pattern 7: BetterTogether meta-optimization ---")
        bt = dspy.BetterTogether(
            metric=content_metric,
            p=dspy.GEPA(metric=gepa_content_metric, max_full_evals=1,
                        reflection_lm=dspy.LM("deepseek/deepseek-v4-flash"), num_threads=4),
            q=dspy.MIPROv2(metric=content_metric, auto="light", num_threads=4),
        )
        bt_student = dspy.ChainOfThought(ClassifyContent)
        bt_student.set_lm(lm)
        bt_prog = bt.compile(bt_student, trainset=trainset, valset=devset, strategy="p -> q")
        bt_score = es(bt_prog)
        results["BetterTogether"] = bt_score
        print(f"  Accuracy: {bt_score:.0%}   Δ {bt_score - base_score:+.0%}")

        # Summary — rank by improvement
        print("\n" + "=" * 70)
        print("OPTIMIZER RANKING (by improvement over baseline)")
        print("=" * 70)
        ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
        best_name, best_val = ranked[0]
        for name, score in ranked:
            delta = score - base_score
            marker = " ◀ BEST" if name == best_name else ""
            print(f"  {name:30s}  {score:.0%}  (Δ {delta:+.0%}){marker}")
        print(f"\n  Baseline: {base_score:.0%}")
        print(f"  Best gain: {best_val - base_score:+.0%} ({best_name})")

        # ---- 3. RLM agent with MCP tools ---------------------------------
        print("\n" + "=" * 70)
        print("RLM AGENT WITH MCP TOOLS")
        print("=" * 70)

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

        dspy.configure(adapter=BAMLAdapter())

        rlm = dspy.RLM(
            "task: str -> report: ResearchReport",
            tools=mcp_tools,
            max_iterations=10,
            max_llm_calls=16,
            verbose=False,
        )

        task = (
            "Use the Crawl4AI MCP tools to fetch documentation pages from "
            "https://dspy.ai and https://docs.python.org/3/tutorial/index.html. "
            "For each page, extract the category, key topics, and write a summary. "
            "Then synthesize the findings."
        )

        result = rlm(task=task)
        report_dict = None

        if hasattr(result, "report") and result.report is not None:
            r = result.report
            report_dict = r.model_dump()
            print(f"\nFindings: {len(r.findings)} source(s)")
            for i, f in enumerate(r.findings, 1):
                print(f"  [{i}] {f.url}")
                print(f"       Category: {f.category}")
                print(f"       Summary: {f.summary[:200]}")
                for kp in f.key_topics:
                    print(f"       • {kp}")
                print()
            print(f"Synthesis: {r.synthesis[:300]}")

        # ---- 4. Trace2Skill: consolidate trajectory -----------------------
        if hasattr(result, "trajectory"):
            n_steps = len(result.trajectory)
            print(f"\nRLM trajectory: {n_steps} iterations")

            stem = memory.save_attempt(task, result.trajectory, report_dict)
            memory.save_note(
                stem,
                f"RLM completed {n_steps} iterations.\n"
                f"Task: {task[:100]}...\n"
                f"Findings: {len(report_dict['findings']) if report_dict else 0} sources\n"
                f"Synthesis length: {len(report_dict.get('synthesis', '')) if report_dict else 0} chars",
            )

            # Extract successful few-shot demonstrations from trajectory
            demos = []
            for step in result.trajectory:
                reasoning = step.get("reasoning", "")
                if reasoning and len(reasoning) > 50:
                    demos.append({"reasoning": reasoning[:500]})

            if demos:
                memory.save_skill("rlm_trajectory_skills", str(rlm.signature), demos)
                print(f"  Saved {len(demos)} demo(s) to memory/skills/")

            print(f"  {memory.summary()}")

        # Load skills from previous runs for future reference
        all_skills = memory.load_skills()
        if len(all_skills) > 1:
            print(f"  Total skills accumulated: {len(all_skills)}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
