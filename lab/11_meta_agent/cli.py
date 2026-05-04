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
from .evolution.gfl import GFLPipeline


console = Console()
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config" / "mcp_servers.json"

dspy.configure(lm=dspy.LM(get_lm_model(), temperature=get_lm_temperature()), adapter=BAMLAdapter())


def _get_bridge() -> MCPBridge | None:
    client = MCPClient(str(CONFIG_PATH))
    try:
        tool_defs = client.connect_all()
    except FileNotFoundError:
        console.print(f"  [red]Config not found: {CONFIG_PATH}[/]")
        return None
    except Exception as e:
        console.print(f"  [yellow]MCP connect failed: {e}[/]")
        return None
    return MCPBridge(client, tool_defs)


def _make_meta(ctx, bridge=None) -> MetaAgent:
    tool_defs = bridge.tool_defs if bridge else []
    generator = AgentGenerator(ctx.obj["DIRECT_LM"], tool_defs)
    return MetaAgent(
        llm=ctx.obj["DIRECT_LM"],
        generator=generator,
        tool_defs=tool_defs,
        skills_dir=str(BASE_DIR / "memory" / "skills"),
    )


# -- CLI --

@click.group()
@click.option("--query", "-q", default="", help="Task for the meta-agent")
@click.option("--iterations", "-i", default=5, show_default=True, help="Max iterations")
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
    bridge = _get_bridge()
    meta = _make_meta(ctx, bridge)

    console.print(Panel(f"[bold]Generating agents for:[/]\n{query}", style="cyan"))
    count = meta.generate_agents(query)
    console.print(f"  [green]\u2713[/] {count} agent(s) generated\n")
    console.print(meta.stack.summary())

    if bridge:
        bridge.client.close()


@cli.command()
@click.pass_context
def run(ctx: click.Context):
    """Full pipeline: generate agents -> run stack -> LSE -> Trace2Skill."""
    query = ctx.obj["QUERY"] or "Analyze the latest trends in AI agent architectures"
    max_iter = ctx.obj["ITERATIONS"]

    console.print(Panel(f"[bold]META RUN[/]\n{query}", style="cyan"))

    console.print("\n[bold cyan][1/4][/] Connecting MCP tools...")
    bridge = _get_bridge()
    tool_defs = bridge.tool_defs if bridge else []
    console.print(f"  {len(tool_defs)} tool(s) discovered" if tool_defs else "  [dim]No MCP tools[/]")

    console.print("\n[bold cyan][2/4][/] Generating agents for task...")
    meta = _make_meta(ctx, bridge)
    count = meta.generate_agents(query)
    console.print(f"  {count} agent(s) on stack")

    console.print(f"\n[bold cyan][3/4][/] Running agent stack ({max_iter} iterations)...")
    results = meta.run_stack(query, max_iterations=max_iter)
    console.print(f"  {len(results)} iteration(s) complete")

    console.print("\n[bold cyan][4/4][/] Consolidating patterns...")
    patterns = meta.consolidate(results)
    skill_name = meta.save_skill(patterns)
    console.print(f"  [green]\u2713[/] {len(patterns['error_patterns'])} error, {len(patterns['success_patterns'])} success")
    console.print(f"  [dim]Saved:[/] {skill_name}")

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

    if bridge:
        bridge.client.close()


@cli.command()
@click.pass_context
def stack(ctx: click.Context):
    """Inspect the last generated agent stack."""
    console.print("[yellow]The agent stack is created per 'generate' or 'run' command.[/]")
    console.print("  Use --query to generate, then inspect with 'stack' is not supported.")
    console.print("  Run 'generate' or 'run' instead to see the stack summary.")


@cli.command()
@click.pass_context
def optimize(ctx: click.Context):
    """Optimize generated agents using GEPA + BetterTogether chaining."""
    query = ctx.obj["QUERY"] or "Optimize agent for research tasks"
    console.print(Panel(f"[bold]Optimizing agents for:[/]\n{query}", style="green"))

    bridge = _get_bridge()
    meta = _make_meta(ctx, bridge)
    meta.generate_agents(query)

    if not meta.stack:
        console.print("[red]No agents to optimize[/]")
        return

    for entry in meta.stack:
        module = meta._generator.generate_module(entry)
        trainset = [
            dspy.Example(task=query, result="quality output").with_inputs("task"),
        ]

        def metric(ex, pred, trace=None):
            pred_str = str(pred)
            return 1.0 if len(pred_str) > 20 else 0.0

        try:
            gepa = dspy.GEPA(metric=metric, max_full_evals=3)
            optimized = gepa.compile(module, trainset=trainset)
            console.print(f"  [green]\u2713[/] {entry.name} optimized via GEPA")
        except Exception as e:
            console.print(f"  [yellow]GEPA failed for {entry.name}: {e}[/]")

    console.print("[green]Optimization complete.[/]")
    if bridge:
        bridge.client.close()


@cli.command()
@click.pass_context
def gfl(ctx: click.Context):
    """Run full Generative Feedback Loop pipeline on generated agents."""
    query = ctx.obj["QUERY"] or "Classify user intent from query text"
    console.print(Panel(f"[bold]GFL Pipeline for:[/]\n{query}", style="green"))

    trainset = [
        dspy.Example(query="Book a flight", intent="booking", confidence="high").with_inputs("query"),
        dspy.Example(query="What's the weather", intent="inquiry", confidence="high").with_inputs("query"),
        dspy.Example(query="Cancel my order", intent="cancellation", confidence="high").with_inputs("query"),
    ]
    devset = [
        dspy.Example(query="Order pizza", intent="booking", confidence="high").with_inputs("query"),
    ]

    def metric(ex, pred, trace=None):
        return 1.0 if getattr(pred, "intent", "") == ex.intent else 0.0

    pipeline = GFLPipeline(
        trainset=trainset, devset=devset, metric=metric,
        reflection_lm=ctx.obj["DIRECT_LM"],
    )

    class ClassifyIntent(dspy.Signature):
        """Classify user query intent."""
        query: str = dspy.InputField()
        intent: str = dspy.OutputField()
        confidence: str = dspy.OutputField()

    program = dspy.ChainOfThought(ClassifyIntent)
    results = pipeline.run_full(program)

    console.print("\n[bold]GFL Results:[/]")
    for name, (prog, score) in results.items():
        delta = score - results["baseline"][1]
        console.print(f"  {name:25s} {score:.0%}  ({delta:+.0%})")

    console.print("[green]GFL pipeline complete.[/]")


@cli.command()
@click.pass_context
def distill(ctx: click.Context):
    """Distill: teacher -> student for generated agents."""
    student_lm = dspy.LM(get_student_lm_model())
    console.print(f"[dim]Teacher:[/] {get_lm_model()}")
    console.print(f"[dim]Student:[/] {get_student_lm_model()}")

    bridge = _get_bridge()
    if not bridge:
        console.print("[red]MCP tools required for distillation[/]")
        raise SystemExit(1)

    meta = _make_meta(ctx, bridge)
    meta.generate_agents("distillation")
    for entry in meta.stack:
        module = meta._generator.generate_module(entry)
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
    bridge.client.close()


if __name__ == "__main__":
    cli()
