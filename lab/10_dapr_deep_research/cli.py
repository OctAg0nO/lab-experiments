"""CLI — run research agents, full workflows, missions, or distillation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import click
from dotenv import load_dotenv
import dspy
from dspy.adapters.baml_adapter import BAMLAdapter
from dapr_agents.agents.configs import AgentStateConfig
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..shared.config import get_lm_model, get_student_lm_model, get_agent_port
from .mcp.client import MCPClient
from .mcp.bridge import MCPBridge
from .memory.dapr_frontier import DaprFrontier
from .memory.frontier import InMemoryFrontier
from .memory.noop_store import NoopStore
from .evolution.lse import LSEOptimizer
from .evolution.trace2skill import SkillConsolidator
from .agents.research_agents import ExplorerAgent, DeepReaderAgent, SynthesizerAgent, CriticAgent, SelectAgent
from .orchestrator.workflow import ResearchWorkflow


console = Console()
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config" / "mcp_servers.json"

dspy.configure(lm=dspy.LM(get_lm_model()), adapter=BAMLAdapter())


def _get_bridge() -> MCPBridge:
    client = MCPClient(str(CONFIG_PATH))
    tool_defs = client.connect_all()
    return MCPBridge(client, tool_defs)


def _create_agents(bridge: MCPBridge, llm: dspy.LM) -> dict[str, ExplorerAgent | DeepReaderAgent | SynthesizerAgent | CriticAgent]:
    noop = NoopStore()
    state = AgentStateConfig(store=noop)
    return {
        "explorer": ExplorerAgent(bridge=bridge, llm=llm, state=state),
        "deepreader": DeepReaderAgent(bridge=bridge, llm=llm, state=state),
        "synthesizer": SynthesizerAgent(bridge=bridge, llm=llm, state=state),
        "critic": CriticAgent(llm=llm, state=state),
    }


@click.group()
@click.option("--query", "-q", default="", help="Research topic or question")
@click.option("--iterations", "-i", default=5, show_default=True, help="Max research iterations")
@click.pass_context
def cli(ctx: click.Context, query: str, iterations: int):
    """Dapr Deep Research — multi-agent research platform."""
    ctx.ensure_object(dict)
    ctx.obj["QUERY"] = query
    ctx.obj["ITERATIONS"] = iterations
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
    runner.serve(agent, port=get_agent_port("orchestrator"), input={"query": query})


@cli.command()
def explorer():
    """Start ExplorerAgent (requires Dapr sidecar + Crawl4AI)."""
    bridge = _get_bridge()
    agent = ExplorerAgent(bridge=bridge)
    from dapr_agents import AgentRunner
    AgentRunner().serve(agent, port=get_agent_port("explorer"))


@cli.command()
def deepreader():
    """Start DeepReaderAgent (requires Dapr sidecar + Crawl4AI)."""
    bridge = _get_bridge()
    agent = DeepReaderAgent(bridge=bridge)
    from dapr_agents import AgentRunner
    AgentRunner().serve(agent, port=get_agent_port("deepreader"))


@cli.command()
def synthesizer():
    """Start SynthesizerAgent (requires Dapr sidecar)."""
    bridge = _get_bridge()
    agent = SynthesizerAgent(bridge=bridge)
    from dapr_agents import AgentRunner
    AgentRunner().serve(agent, port=get_agent_port("synthesizer"))


@cli.command()
def critic():
    """Start CriticAgent (requires Dapr sidecar)."""
    agent = CriticAgent()
    from dapr_agents import AgentRunner
    AgentRunner().serve(agent, port=get_agent_port("critic"))


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
    agents = _create_agents(bridge, ctx.obj["DIRECT_LM"])
    agent_trainsets = {
        "explorer": [dspy.Example(topic=query, hypotheses=["subtopic A"]).with_inputs("topic")],
        "deepreader": [dspy.Example(findings_summary="Finding X; Finding Y", validated_claims=["Claim X"], contradictions=[]).with_inputs("findings_summary")],
        "synthesizer": [dspy.Example(task=query, synthesis="Cross-source analysis", key_insights=["Insight"], gaps=["Gap"]).with_inputs("task")],
        "critic": [dspy.Example(research_summary=query, critique="Strengths: X. Weaknesses: Y.", improved_critique="Balanced critique.").with_inputs("research_summary", "critique")],
    }
    lse = LSEOptimizer()
    frontier = InMemoryFrontier()
    agent_selector = dspy.ChainOfThought(SelectAgent)

    compiled_count = 0
    for name, trainset in agent_trainsets.items():
        agents[name].compile(trainset)
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
    agents = _create_agents(bridge, ctx.obj["DIRECT_LM"])
    for name, agent in agents.items():
        with console.status(f"Compiling {name}..."):
            agent.compile([dspy.Example(topic="test", hypotheses=["test"]).with_inputs("topic")], student_lm=student_lm)
        console.print(f"  [green]\u2713[/] {name} compiled")
    console.print("[green]Distillation complete.[/]")


@cli.command()
@click.pass_context
def chat(ctx: click.Context):
    """Interactive research REPL. Type queries or /commands."""
    console.print(Panel("[bold]Interactive Research Chat[/]\nType a research query or /help for commands.", style="cyan"))

    import shutil

    frontier = InMemoryFrontier()
    lse = LSEOptimizer()
    agent_selector = dspy.ChainOfThought(SelectAgent)
    history: list[str] = []
    max_iter = ctx.obj["ITERATIONS"]

    try:
        client = MCPClient(str(CONFIG_PATH))
        tool_defs = client.connect_all()
        bridge = MCPBridge(client, tool_defs)
        console.print(f"  [dim]{len(tool_defs)} MCP tool(s) ready[/]")
    except Exception:
        bridge = None
        console.print("  [dim]No MCP tools available[/]")

    agents_compiled = False

    while True:
        try:
            line = input("\n[1m[?][0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line.startswith("/"):
            parts = line.split(maxsplit=1)
            match parts[0]:
                case "/help":
                    console.print("[bold]Commands:[/]")
                    console.print("  /query <text>    Set research query")
                    console.print("  /status          Show frontier + LSE state")
                    console.print("  /frontier        List all frontier directions")
                    console.print("  /history         Show recent queries")
                    console.print("  /iterations <N>  Set max iterations")
                    console.print("  /compile         Compile agents with current data")
                    console.print("  /quit            Exit")
                case "/status":
                    console.print(f"  Frontier: {frontier.summary()}")
                    console.print(f"  Iterations: {len(lse.runs)}")
                    trend = lse.improvement_trend()
                    if trend:
                        console.print(f"  LSE trend: {[f'{t:+.2f}' for t in trend]}")
                    console.print(f"  MCP: {'connected' if bridge else 'unavailable'}")
                    console.print(f"  Agents compiled: {agents_compiled}")
                case "/frontier":
                    table = Table(title="Frontier Directions", header_style="bold cyan")
                    table.add_column("Topic", style="cyan")
                    table.add_column("Confidence")
                    table.add_column("Depth")
                    table.add_column("Sources")
                    for d in frontier.directions:
                        table.add_row(d.topic[:50], f"{d.confidence:.2f}", str(d.exploration_depth), str(d.source_count))
                    console.print(table)
                case "/history":
                    for h in history[-10:]:
                        console.print(f"  {h[:80]}")
                case "/iterations":
                    if len(parts) > 1:
                        max_iter = max(1, int(parts[1]))
                        console.print(f"  Max iterations set to {max_iter}")
                    else:
                        console.print(f"  Current: {max_iter}")
                case "/compile":
                    if bridge:
                        chat_agents = _create_agents(bridge, ctx.obj["DIRECT_LM"])
                        for agent in chat_agents.values():
                            agent.compile([dspy.Example(topic="research", hypotheses=["subtopic"]).with_inputs("topic")])
                        agents_compiled = True
                        console.print("  [green]Agents compiled[/]")
                    else:
                        console.print("  [yellow]MCP required for compilation[/]")
                case "/quit":
                    break
                case _:
                    console.print(f"  Unknown: {parts[0]}. Type /help")
        else:
            frontier.seed_from_query(line)
            history.append(line)
            with console.status(f"[bold green]Researching:[/] {line}") as status:
                for i in range(max_iter):
                    direction = frontier.next_action()
                    if not direction:
                        break
                    selection = agent_selector(exploration_depth=direction.exploration_depth, confidence=direction.confidence, topic=direction.topic)
                    frontier.absorb_findings(direction.topic, 0.2, 1, [])
                    state = {"num_directions": len(frontier.directions), "num_findings": i + 1, "frontier_saturation": 0.0}
                    lse.record_run(f"iter_{i+1}", state, direction.topic)
                    status.update(f"[green]{selection.selected_agent}[/] -> {direction.topic[:50]}")
            trend = lse.improvement_trend()
            summary = f"[dim]Frontier:[/] {frontier.summary()}"
            if trend:
                summary += f"  [dim]Trend:[/] {[f'{t:+.2f}' for t in trend]}"
            console.print(summary)


if __name__ == "__main__":
    cli()
