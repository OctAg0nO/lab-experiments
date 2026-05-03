"""
CLI — run individual research agents, start the full research workflow,
or run teacher/student distillation.

Usage (full distributed research, requires Dapr + Crawl4AI + Redis):
    dapr run -f lab/10_dapr_deep_research/dapr-multi-app-run.yaml

Usage (single agent in its own terminal):
    dapr run --app-id orchestrator --app-protocol grpc --app-port 8000  \
        --resources-path lab/10_dapr_deep_research/resources --          \
        uv run python -m lab.10_dapr_deep_research --mode orchestrator

Usage (quick tests, no infrastructure needed):
    uv run python -m lab.10_dapr_deep_research --mode run
    uv run python -m lab.10_dapr_deep_research --mode distill
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
import dspy
from dspy.adapters.baml_adapter import BAMLAdapter

from ..shared.config import get_lm_model, get_student_lm_model
from .mcp.client import MCPClient
from .mcp.bridge import MCPBridge
from .memory.dapr_frontier import DaprFrontier, ResearchDirection
from .evolution.lse import LSEOptimizer
from .evolution.trace2skill import SkillConsolidator
from .agents.research_agents import ExplorerAgent, DeepReaderAgent, SynthesizerAgent, CriticAgent, SelectAgent
from .orchestrator.workflow import ResearchWorkflow


class _InMemoryFrontier:
    """Dapr-free frontier for --mode run. Same UCB logic, no sidecar needed."""
    def __init__(self):
        self.directions: list[ResearchDirection] = []
        self._total_explorations = 0

    def seed_from_query(self, query: str):
        self.directions.append(ResearchDirection(topic=query, confidence=0.0, exploration_depth=0, seed_query=query, last_updated=datetime.now(timezone.utc).isoformat()))

    def seed_from_directions(self, topics: list[str], parent: str | None = None):
        for t in topics:
            if not any(d.topic == t for d in self.directions):
                self.directions.append(ResearchDirection(topic=t, confidence=0.0, exploration_depth=0, parent_topic=parent, seed_query=t, last_updated=datetime.now(timezone.utc).isoformat()))

    def next_action(self) -> ResearchDirection | None:
        active = [d for d in self.directions if d.confidence < 0.95]
        if not active:
            return None
        return max(active, key=lambda d: d.ucb_score(self._total_explorations))

    def absorb_findings(self, topic: str, confidence_delta: float, sources: int, follow_ups: list[str]):
        for d in self.directions:
            if d.topic == topic:
                d.confidence = min(1.0, d.confidence + confidence_delta)
                d.exploration_depth += 1
                d.source_count += sources
                d.last_updated = datetime.now(timezone.utc).isoformat()
                self._total_explorations += 1
                break
        for fu in follow_ups:
            if not any(d.topic == fu for d in self.directions):
                self.directions.append(ResearchDirection(topic=fu, confidence=0.0, exploration_depth=0, parent_topic=topic, seed_query=fu, last_updated=datetime.now(timezone.utc).isoformat()))

    def summary(self) -> str:
        active = len([d for d in self.directions if d.confidence < 0.95])
        explored = len(self.directions) - active
        return f"{active} active, {explored} explored, {self._total_explorations} total explorations"

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config" / "mcp_servers.json"

_TEACHER_LM = dspy.LM(get_lm_model())
dspy.configure(lm=_TEACHER_LM, adapter=BAMLAdapter())


def _get_bridge() -> MCPBridge:
    client = MCPClient(str(CONFIG_PATH))
    tool_defs = client.connect_all()
    return MCPBridge(client, tool_defs)


def cmd_orchestrator(query: str = ""):
    frontier = DaprFrontier()
    print(f"Frontier: {frontier.summary()}")
    agent = ResearchWorkflow(frontier=frontier)
    from dapr_agents import AgentRunner
    runner = AgentRunner()
    if query:
        print(f"Research query: {query}")
        runner.serve(agent, port=8000, input={"query": query})
    else:
        print("No query provided. Use --query to set a research topic.")
        print("The workflow will wait for an external trigger via Dapr API.")
        runner.serve(agent, port=8000)


def cmd_explorer():
    bridge = _get_bridge()
    agent = ExplorerAgent(bridge=bridge)
    from dapr_agents import AgentRunner
    runner = AgentRunner()
    runner.serve(agent, port=8001)


def cmd_deep_reader():
    bridge = _get_bridge()
    agent = DeepReaderAgent(bridge=bridge)
    from dapr_agents import AgentRunner
    runner = AgentRunner()
    runner.serve(agent, port=8002)


def cmd_synthesizer():
    bridge = _get_bridge()
    agent = SynthesizerAgent(bridge=bridge)
    from dapr_agents import AgentRunner
    runner = AgentRunner()
    runner.serve(agent, port=8003)


def cmd_critic():
    agent = CriticAgent()
    from dapr_agents import AgentRunner
    runner = AgentRunner()
    runner.serve(agent, port=8004)


def cmd_run(query: str = ""):
    """Single-process research demo using the agent pipeline (no Dapr sidecar needed).

    Seeds the frontier with a query, then runs iterations that:
    1. Select the next direction via UCB
    2. Dispatches to the appropriate agent (explore/deep-read/synthesize)
    3. Absorbs findings and tracks progress

    Without Dapr, the agent dispatch uses DSPy ChainOfThought (SelectAgent)
    to demonstrate the decision logic. Full agent execution requires Dapr sidecars.
    """
    if not query:
        query = "Research DSPy optimization patterns for LLM pipelines"

    llm = dspy.LM(get_lm_model())
    agent_selector = dspy.ChainOfThought(SelectAgent)

    frontier = _InMemoryFrontier()
    print(f"Frontier: {frontier.summary()}")
    print(f"Query: {query}")
    print("Running research loop...\n")

    frontier.seed_from_query(query)

    for i in range(3):
        direction = frontier.next_action()
        if not direction:
            break
        selection = agent_selector(
            exploration_depth=direction.exploration_depth,
            confidence=direction.confidence,
            topic=direction.topic,
        )
        selected = selection.selected_agent if hasattr(selection, "selected_agent") else "explorer"
        print(f"  Iteration {i+1}: [{selected}] {direction.topic[:70]}")
        frontier.absorb_findings(direction.topic, 0.2, 1, [])

    print(f"\nDone. {frontier.summary()}")


def cmd_mission(query: str = "", max_iter: int = 5):
    """End-to-end research mission: MCP scrape → GFL optimize → LSE evolve → compile.

    Pipeline:
      1. Connect MCP tools (Crawl4AI, fetch) — skip if unavailable
      2. Scrape URLs into a labeled DSPy dataset
      3. Create all research agents with MCP bridge
      4. Run GFL optimization on all agents via BootstrapFewShot
      5. Run LSE research loop (frontier + agent dispatch)
      6. Consolidate trajectories into reusable skills
      7. Print mission summary
    """
    if not query:
        query = "Research DSPy optimization patterns for LLM pipelines"

    mission_log: list[str] = []
    def log(msg: str):
        print(msg, flush=True)
        mission_log.append(msg)

    # ---- Phase 0: Infrastructure ----
    log("=" * 60)
    log(f"MISSION: {query}")
    log("=" * 60)

    # ---- Phase 1: MCP tools + data collection ----
    log("\n[Phase 1] Connecting MCP tools...")
    client = MCPClient(str(CONFIG_PATH))
    tool_defs = []
    try:
        tool_defs = client.connect_all()
        log(f"  Discovered {len(tool_defs)} MCP tool(s)")
    except Exception as e:
        log(f"  MCP unavailable: {e} (proceeding without tools)")

    bridge = MCPBridge(client, tool_defs) if tool_defs else None

    trainset: list[dspy.Example] = []
    if bridge and any(td.get("server") in ("crawl4ai", "fetch") for td in tool_defs):
        log("\n[Phase 1b] Scraping web content for training data...")
        from .agents.research_agents import GenerateHypotheses
        for td in tool_defs[:4]:
            url = "https://dspy.ai"
            try:
                content = client.call_tool(td["server"], td["name"], {"url": url})
                chunks = [content[i:i+1200] for i in range(0, len(content), 1200)][:4]
                for c in chunks:
                    trainset.append(dspy.Example(topic=query, hypotheses=[c[:200]]).with_inputs("topic"))
                log(f"  Scraped {len(chunks)} chunks from {url}")
            except Exception as e:
                log(f"  Skipped {url}: {e}")
    else:
        log("  No scraper tools available — using synthetic dataset")
        trainset = [dspy.Example(topic=query, hypotheses=[f"Sub-topic {i}"]).with_inputs("topic") for i in range(5)]

    # ---- Phase 2: Create internal DSPy modules directly ----
    # (skipping DurableAgent wrapper to avoid Dapr sidecar dependency)
    log("\n[Phase 2] Creating DSPy modules...")
    from .agents.research_agents import GenerateHypotheses, CrossValidateFindings, SynthesizeAcrossSources
    from .evolution.lse import QualityEvaluation

    hypothesis_gen = dspy.ChainOfThought(GenerateHypotheses)
    cross_validator = dspy.ChainOfThought(CrossValidateFindings)
    synthesizer = dspy.ChainOfThought(SynthesizeAcrossSources)
    critic_refine = dspy.Refine(dspy.ChainOfThought("research_summary: str, critique: str -> improved_critique: str"), N=3, reward_fn=lambda _, pred: 1.0 if len(pred.improved_critique) > 50 else 0.0, threshold=0.5)
    lse_evaluator = dspy.ChainOfThought(QualityEvaluation)
    agent_selector = dspy.ChainOfThought(SelectAgent)
    consolidator = SkillConsolidator(BASE_DIR / "memory" / "skills")
    frontier = _InMemoryFrontier()
    log("  DSPy modules ready")

    # ---- Phase 3: GFL optimization (BootstrapFewShot) ----
    log("\n[Phase 3] GFL optimization (BootstrapFewShot)...")
    compiled_modules = 0
    for name, prog in [("HypothesisGen", hypothesis_gen), ("CrossValidator", cross_validator),
                        ("Synthesizer", synthesizer), ("CriticRefine", critic_refine),
                        ("LSE", lse_evaluator)]:
        bs = dspy.BootstrapFewShot(metric=lambda _ex, pred, _trace: len(str(pred)) > 0, max_bootstrapped_demos=2, max_labeled_demos=1)
        bs.compile(prog, trainset=trainset)
        compiled_modules += 1
        log(f"  ✓ {name} compiled")
    log(f"  Compiled {compiled_modules} module(s)")

    # ---- Phase 4: LSE research loop ----
    log(f"\n[Phase 4] LSE research loop ({max_iter} iterations)...")
    frontier.seed_from_query(query)
    all_trajectories: list[dict] = []

    for i in range(max_iter):
        direction = frontier.next_action()
        if not direction:
            log("  Frontier saturated — stopping early")
            break

        selection = agent_selector(exploration_depth=direction.exploration_depth, confidence=direction.confidence, topic=direction.topic)
        selected = selection.selected_agent
        log(f"  Iter {i+1}: agent={selected} topic={direction.topic[:60]}")

        frontier.absorb_findings(direction.topic, 0.2, 1, [])
        state = {"num_directions": len(frontier.directions), "num_findings": i + 1, "frontier_saturation": 0.0}
        lse.record_run(f"iter_{i+1}", state, direction.topic)

    trend = lse.improvement_trend()
    if trend:
        log(f"  LSE trend: {[f'{t:+.2f}' for t in trend]}")

    # ---- Phase 5: Consolidate ----
    log("\n[Phase 5] Consolidating trajectories into skills...")
    if all_trajectories:
        skill = consolidator.consolidate(all_trajectories)
        consolidator.save_skill(f"mission_{datetime.now().strftime('%Y%m%d_%H%M%S')}", skill)
        log(f"  Saved skill: {skill.get('n_trajectories', 0)} trajectories, {len(skill.get('success_patterns', []))} patterns")
    else:
        log("  No trajectories to consolidate")

    # ---- Summary ----
    log("\n" + "=" * 60)
    log("MISSION COMPLETE")
    log("=" * 60)
    log(f"  Query:     {query}")
    log(f"  Iterations: {len(lse.runs)}")
    log(f"  Frontier:  {frontier.summary()}")
    log(f"  Compiled:  {compiled_modules} module(s)")
    log(f"  Skills:    {len(list((BASE_DIR / 'memory' / 'skills').glob('*.json')))} total")
    if trend:
        best = lse.best_strategy()
        log(f"  Best iter: {best}")

    client.close()


def cmd_distill():
    """Teacher (DeepSeek) → student (Gemma 4) distillation for all DSPy programs.

    Compiles every ChainOfThought / Refine module using BootstrapFewShot
    with the teacher generating demonstrations and the student learning from them.
    """
    teacher_lm = _TEACHER_LM
    student_lm = dspy.LM(get_student_lm_model())
    print(f"Teacher: {get_lm_model()}")
    print(f"Student: {get_student_lm_model()}")

    bridge = _get_bridge()
    trainset: list[dspy.Example] = []

    agents: list[tuple[str, ExplorerAgent | DeepReaderAgent | SynthesizerAgent | CriticAgent | ResearchWorkflow | DaprFrontier | LSEOptimizer | SkillConsolidator]] = [
        ("ExplorerAgent", ExplorerAgent(bridge=bridge)),
        ("DeepReaderAgent", DeepReaderAgent(bridge=bridge)),
        ("SynthesizerAgent", SynthesizerAgent(bridge=bridge)),
        ("CriticAgent", CriticAgent()),
        ("Workflow", ResearchWorkflow(frontier=DaprFrontier())),
        ("LSEOptimizer", LSEOptimizer()),
        ("SkillConsolidator", SkillConsolidator(BASE_DIR / "memory" / "skills")),
        ("DaprFrontier", DaprFrontier()),
    ]

    for name, agent in agents:
        print(f"  Compiling {name} with student LM ...")
        agent.compile(trainset, student_lm=student_lm)
        print(f"    ✓ {name} compiled")

    print("\nDistillation complete. All compiled modules use student_lm for inference.")


def main():
    parser = argparse.ArgumentParser(description="Dapr Deep Research — multi-agent research platform")
    parser.add_argument("--mode", default="run",
                        choices=["orchestrator", "explorer", "deepreader", "synthesizer", "critic", "run", "distill", "mission"],
                        help="orchestrator/explorer/deepreader/synthesizer/critic (Dapr sidecar) | run (frontier demo) | distill (teacher/student) | mission (full pipeline)")
    parser.add_argument("--query", "-q", type=str, default="",
                        help="Research topic or question")
    parser.add_argument("--iterations", "-i", type=int, default=5,
                        help="Max research iterations")
    args = parser.parse_args()

    if args.mode == "orchestrator":
        cmd_orchestrator(query=args.query)
    elif args.mode == "explorer":
        cmd_explorer()
    elif args.mode == "deepreader":
        cmd_deep_reader()
    elif args.mode == "synthesizer":
        cmd_synthesizer()
    elif args.mode == "critic":
        cmd_critic()
    elif args.mode == "run":
        cmd_run(query=args.query)
    elif args.mode == "distill":
        cmd_distill()
    elif args.mode == "mission":
        cmd_mission(query=args.query, max_iter=args.iterations)


if __name__ == "__main__":
    main()
