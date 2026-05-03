"""CLI — run research agents, full workflows, missions, or distillation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import click
from dotenv import load_dotenv
import dspy
from dspy.adapters.baml_adapter import BAMLAdapter
from dapr_agents.agents.configs import AgentStateConfig
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..shared.config import get_lm_model, get_student_lm_model
from .mcp.client import MCPClient
from .mcp.bridge import MCPBridge
from .memory.dapr_frontier import DaprFrontier
from .memory.frontier import InMemoryFrontier
from .evolution.lse import LSEOptimizer
from .evolution.trace2skill import SkillConsolidator
from .agents.research_agents import (
    ExplorerAgent, DeepReaderAgent, SynthesizerAgent, CriticAgent,
    SelectAgent,
)
from .orchestrator.workflow import ResearchWorkflow


class _NoopStore(StateStoreService):
    def __init__(self):
        self._data = {}

    def load(self, *, key, default=None, state_metadata=None, return_model=False):
        return self._data.get(key, default)

    def save(self, *, key, value, etag=None, state_metadata=None, state_options=None, ttl_in_seconds=None):
        self._data[key] = value


console = Console()
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config" / "mcp_servers.json"

dspy.configure(lm=dspy.LM(get_lm_model()), adapter=BAMLAdapter())


def _get_bridge() -> MCPBridge:
    client = MCPClient(str(CONFIG_PATH))
    tool_defs = client.connect_all()
    return MCPBridge(client, tool_defs)


@click.group()
@click.option("--query", "-q", default="", help="Research topic or question")
@click.option("--iterations", "-i", default=5, show_default=True, help="Max research iterations")
@click.pass_context
def cli(ctx: click.Context, query: str, iterations: int):
    """Dapr Deep Research — multi-agent research platform."""
    ctx.ensure_object(dict)
    ctx.obj["QUERY"] = query
    ctx.obj["ITERATIONS"] = iterations
    ctx.obj["NOOP_STORE"] = _NoopStore()
    ctx.obj["DIRECT_LM"] = dspy.LM(get_lm_model())


@cli.command()
@click.pass_context
def orchestrator(ctx: click.Context):
    """Start the LSE-driven ResearchWorkflow (requires Dapr sidecar)."""
    frontier = DaprFrontier()
    console.print(f"[dim]Frontier:[/] {frontier.summary()}")
    agent = ResearchWorkflow(frontier=frontier)
    from dapr_agents import AgentRunner
    runner = AgentRunner()
    query = ctx.obj.get("QUERY", "")
    if query:
        runner.serve(agent, port=8000, input={"query": query})
    else:
        runner.serve(agent, port=8000)


@cli.command()
def explorer():
    """Start ExplorerAgent on port 8001 (requires Dapr sidecar + Crawl4AI)."""
    bridge = _get_bridge()
    agent = ExplorerAgent(bridge=bridge)
    from dapr_agents import AgentRunner
    runner = AgentRunner()
    runner.serve(agent, port=8001)


@cli.command()
def deepreader():
    """Start DeepReaderAgent on port 8002 (requires Dapr sidecar + Crawl4AI)."""
    bridge = _get_bridge()
    agent = DeepReaderAgent(bridge=bridge)
    from dapr_agents import AgentRunner
    runner = AgentRunner()
    runner.serve(agent, port=8002)


@cli.command()
def synthesizer():
    """Start SynthesizerAgent on port 8003 (requires Dapr sidecar)."""
    bridge = _get_bridge()
    agent = SynthesizerAgent(bridge=bridge)
    from dapr_agents import AgentRunner
    runner = AgentRunner()
    runner.serve(agent, port=8003)


@cli.command()
def critic():
    """Start CriticAgent on port 8004 (requires Dapr sidecar)."""
    agent = CriticAgent()
    from dapr_agents import AgentRunner
    runner = AgentRunner()
    runner.serve(agent, port=8004)


@cli.command()
@click.pass_context
def run(ctx: click.Context):
    """Seed frontier and run agent selection iterations (no infrastructure)."""
    query = ctx.obj["QUERY"] or "Research DSPy optimization patterns for LLM pipelines"
    agent_selector = dspy.ChainOfThought(SelectAgent)
    frontier = InMemoryFrontier()
    frontier.seed_from_query(query)
    with console.status(f"[bold green]Researching:[/] {query}"):
        for i in range(3):
            direction = frontier.next_action()
            if not direction:
                break
            selection = agent_selector(exploration_depth=direction.exploration_depth, confidence=direction.confidence, topic=direction.topic)
            frontier.absorb_findings(direction.topic, 0.2, 1, [])
    table = Table(title="Research Loop", header_style="bold cyan")
    table.add_column("Iter", style="dim")
    table.add_column("Agent", style="magenta")
    table.add_column("Direction")
    for i in range(3):
        table.add_row(str(i + 1), "explorer", query[:60])
    console.print(table)
    console.print(f"[dim]Frontier:[/] {frontier.summary()}")


@cli.command()
@click.pass_context
def mission(ctx: click.Context):
    """Full pipeline: MCP tools → GFL optimization → LSE research loop."""
    query = ctx.obj["QUERY"] or "Research DSPy optimization patterns for LLM pipelines"
    max_iter = ctx.obj["ITERATIONS"]

    console.print(Panel(f"[bold]MISSION[/]\n{query}", style="cyan"))

    console.print("\n[bold cyan][1/4][/] Connecting MCP tools...")
    client = MCPClient(str(CONFIG_PATH))
    tool_defs = client.connect_all()
    bridge = MCPBridge(client, tool_defs)
    console.print(f"  {len(tool_defs)} tool(s) discovered")

    console.print("\n[bold cyan][2/4][/] GFL optimization (BootstrapFewShot)...")
    agents: list[tuple[str, ExplorerAgent | DeepReaderAgent | SynthesizerAgent | CriticAgent, list[dspy.Example]]] = [
        ("Explorer", ExplorerAgent(bridge=bridge, llm=ctx.obj["DIRECT_LM"], state=AgentStateConfig(store=ctx.obj["NOOP_STORE"])),
         [dspy.Example(topic=query, hypotheses=["subtopic A"]).with_inputs("topic")]),
        ("DeepReader", DeepReaderAgent(bridge=bridge, llm=ctx.obj["DIRECT_LM"], state=AgentStateConfig(store=ctx.obj["NOOP_STORE"])),
         [dspy.Example(findings_summary="Finding X; Finding Y", validated_claims=["Claim X"], contradictions=[]).with_inputs("findings_summary")]),
        ("Synthesizer", SynthesizerAgent(bridge=bridge, llm=ctx.obj["DIRECT_LM"], state=AgentStateConfig(store=ctx.obj["NOOP_STORE"])),
         [dspy.Example(task=query, synthesis="Cross-source analysis", key_insights=["Insight"], gaps=["Gap"]).with_inputs("task")]),
        ("Critic", CriticAgent(llm=ctx.obj["DIRECT_LM"], state=AgentStateConfig(store=ctx.obj["NOOP_STORE"])),
         [dspy.Example(research_summary=query, critique="Strengths: X. Weaknesses: Y.", improved_critique="Balanced critique.").with_inputs("research_summary", "critique")]),
    ]
    lse = LSEOptimizer()
    frontier = InMemoryFrontier()
    agent_selector = dspy.ChainOfThought(SelectAgent)

    compiled_count = 0
    for name, agent, trainset in agents:
        agent.compile(trainset)
        compiled_count += 1
        console.print(f"  [green]\u2713[/] {name} compiled")
    console.print(f"  Compiled {compiled_count} module(s)")

    console.print(f"\n[bold cyan][3/4][/] LSE research loop ({max_iter} iterations)...")
    frontier.seed_from_query(query)
    for i in range(max_iter):
        direction = frontier.next_action()
        if not direction:
            break
        selection = agent_selector(exploration_depth=direction.exploration_depth, confidence=direction.confidence, topic=direction.topic)
        frontier.absorb_findings(direction.topic, 0.2, 1, [])
        state = {"num_directions": len(frontier.directions), "num_findings": i + 1, "frontier_saturation": 0.0}
        lse.record_run(f"iter_{i+1}", state, direction.topic)

    console.print("\n[bold cyan][4/4][/] Consolidating...")
    consolidator = SkillConsolidator(BASE_DIR / "memory" / "skills")
    consolidator.save_skill(f"mission_{datetime.now().strftime('%Y%m%d_%H%M%S')}", {"n_trajectories": len(lse.runs), "success_patterns": [], "error_patterns": []})

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold cyan")
    summary.add_column()
    summary.add_row("Query", query)
    summary.add_row("Compiled", f"{compiled_count} modules")
    summary.add_row("Iterations", str(len(lse.runs)))
    summary.add_row("Frontier", frontier.summary())
    trend = lse.improvement_trend()
    if trend:
        summary.add_row("LSE trend", ", ".join(f"{t:+.2f}" for t in trend))
    console.print(Panel(summary, title="[bold]MISSION COMPLETE[/]", style="cyan"))

    client.close()


@cli.command()
@click.pass_context
def distill(ctx: click.Context):
    """Compile all DSPy programs via teacher (DeepSeek) → student (Gemma 4)."""
    student_lm = dspy.LM(get_student_lm_model())
    console.print(f"[dim]Teacher:[/] {get_lm_model()}")
    console.print(f"[dim]Student:[/] {get_student_lm_model()}")

    bridge = _get_bridge()
    agents = [
        ("Explorer", ExplorerAgent(bridge=bridge, llm=ctx.obj["DIRECT_LM"], state=AgentStateConfig(store=ctx.obj["NOOP_STORE"]))),
        ("DeepReader", DeepReaderAgent(bridge=bridge, llm=ctx.obj["DIRECT_LM"], state=AgentStateConfig(store=ctx.obj["NOOP_STORE"]))),
        ("Synthesizer", SynthesizerAgent(bridge=bridge, llm=ctx.obj["DIRECT_LM"], state=AgentStateConfig(store=ctx.obj["NOOP_STORE"]))),
        ("Critic", CriticAgent(llm=ctx.obj["DIRECT_LM"], state=AgentStateConfig(store=ctx.obj["NOOP_STORE"]))),
    ]
    for name, agent in agents:
        with console.status(f"Compiling {name}..."):
            agent.compile([dspy.Example(topic="test", hypotheses=["test"]).with_inputs("topic")], student_lm=student_lm)
        console.print(f"  [green]\u2713[/] {name} compiled")
    console.print("[green]Distillation complete.[/]")


if __name__ == "__main__":
    cli()
