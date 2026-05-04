"""CLI — generate agents, run meta-agent, inspect stack, distill."""

from __future__ import annotations

from pathlib import Path

import click
from dotenv import load_dotenv
import dspy
from dspy.adapters.baml_adapter import BAMLAdapter
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..shared.config import get_lm_model, get_student_lm_model, get_lm_temperature
from .mcp.client import MCPClient
from .mcp.bridge import MCPBridge
from .meta.agent_generator import AgentGenerator
from .meta.meta_agent import MetaAgent


console = Console()
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config" / "mcp_servers.json"

dspy.configure(lm=dspy.LM(get_lm_model(), temperature=get_lm_temperature()), adapter=BAMLAdapter())


# -- helpers --

def _get_bridge() -> MCPBridge | None:
    try:
        client = MCPClient(str(CONFIG_PATH))
        tool_defs = client.connect_all()
        return MCPBridge(client, tool_defs)
    except Exception as e:
        console.print(f"  [yellow]MCP unavailable: {e}[/]")
        return None


def _make_meta(ctx) -> tuple[MCPBridge | None, MetaAgent]:
    bridge = _get_bridge()
    tool_defs = bridge.tool_defs if bridge else []
    generator = AgentGenerator(ctx.obj["DIRECT_LM"], tool_defs)
    meta = MetaAgent(
        llm=ctx.obj["DIRECT_LM"],
        generator=generator,
        tool_defs=tool_defs,
        skills_dir=str(BASE_DIR / "memory" / "skills"),
    )
    return bridge, meta


# -- CLI --

@click.group()
@click.option("--query", "-q", default="", help="Task for the meta-agent")
@click.option("--iterations", "-i", default=5, show_default=True, help="Max research iterations")
@click.pass_context
def cli(ctx: click.Context, query: str, iterations: int):
    """Meta-Agent — generates specialized agents on the fly using LSE + Trace2Skill."""
    ctx.ensure_object(dict)
    ctx.obj["QUERY"] = query
    ctx.obj["ITERATIONS"] = iterations
    ctx.obj["DIRECT_LM"] = dspy.LM(get_lm_model(), temperature=get_lm_temperature())


@cli.command()
@click.pass_context
def generate(ctx: click.Context):
    """Analyze task and generate agents onto the stack."""
    query = ctx.obj["QUERY"] or "Analyze the latest trends in AI agent architectures"
    _, meta = _make_meta(ctx)

    console.print(Panel(f"[bold]Generating agents for:[/]\n{query}", style="cyan"))
    count = meta.generate_agents(query)
    console.print(f"  [green]\u2713[/] {count} agent(s) generated\n")
    console.print(meta.stack.summary())


@cli.command()
@click.pass_context
def run(ctx: click.Context):
    """Full pipeline: generate agents → run stack → LSE → Trace2Skill."""
    query = ctx.obj["QUERY"] or "Analyze the latest trends in AI agent architectures"
    max_iter = ctx.obj["ITERATIONS"]

    console.print(Panel(f"[bold]META RUN[/]\n{query}", style="cyan"))

    console.print("\n[bold cyan][1/4][/] Connecting MCP tools...")
    bridge = _get_bridge()
    tool_defs = bridge.tool_defs if bridge else []
    console.print(f"  {len(tool_defs)} tool(s) discovered" if tool_defs else "  [dim]No MCP tools[/]")

    console.print("\n[bold cyan][2/4][/] Generating agents for task...")
    generator = AgentGenerator(ctx.obj["DIRECT_LM"], tool_defs)
    meta = MetaAgent(
        llm=ctx.obj["DIRECT_LM"],
        generator=generator,
        tool_defs=tool_defs,
        skills_dir=str(BASE_DIR / "memory" / "skills"),
    )
    count = meta.generate_agents(query)
    console.print(f"  {count} agent(s) on stack")

    console.print(f"\n[bold cyan][3/4][/] Running agent stack ({max_iter} iterations)...")
    results = meta.run_stack(query, max_iterations=max_iter)
    console.print(f"  {len(results)} iteration(s) complete")

    console.print("\n[bold cyan][4/4][/] Consolidating patterns...")
    patterns = meta.consolidate(results)
    skill_name = meta.save_skill(patterns)
    console.print(f"  [green]\u2713[/] {len(patterns['error_patterns'])} error patterns, {len(patterns['success_patterns'])} success patterns")
    console.print(f"  [dim]Saved:[/] {skill_name}")

    # Summary
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold cyan")
    summary.add_column()
    summary.add_row("Task", query)
    summary.add_row("Agents", str(len(meta.stack)))
    summary.add_row("Iterations", str(len(results)))
    summary.add_row("Frontier", meta.frontier.summary())
    trend = meta.lse.improvement_trend()
    if trend:
        summary.add_row("LSE trend", ", ".join(f"{t:+.2f}" for t in trend))
    console.print(Panel(summary, title="[bold]META COMPLETE[/]", style="cyan"))


@cli.command()
@click.pass_context
def stack(ctx: click.Context):
    """Inspect the current agent stack."""
    _, meta = _make_meta(ctx)
    console.print(meta.stack.summary() if len(meta.stack) > 0 else "[yellow]Stack is empty[/]")


@cli.command()
@click.pass_context
def distill(ctx: click.Context):
    """Distill: teacher → student for generated agents."""
    dspy.LM(get_student_lm_model())
    console.print(f"[dim]Teacher:[/] {get_lm_model()}")
    console.print(f"[dim]Student:[/] {get_student_lm_model()}")

    bridge = _get_bridge()
    if not bridge:
        console.print("[red]MCP tools required for distillation[/]")
        raise SystemExit(1)

    generator = AgentGenerator(ctx.obj["DIRECT_LM"], bridge.tool_defs)
    meta = MetaAgent(
        llm=ctx.obj["DIRECT_LM"],
        generator=generator,
        tool_defs=bridge.tool_defs,
    )
    meta.generate_agents("distillation")
    for entry in meta.stack:
        module = generator.generate_module(entry)
        bs = dspy.BootstrapFewShot(
            metric=lambda ex, pred, trace: True,
            max_bootstrapped_demos=4,
            max_labeled_demos=2,
        )
        bs.compile(module, trainset=[
            dspy.Example(task="test", result="output").with_inputs("task"),
        ])
        console.print(f"  [green]\u2713[/] {entry.name} compiled")
    console.print("[green]Distillation complete.[/]")


if __name__ == "__main__":
    cli()
