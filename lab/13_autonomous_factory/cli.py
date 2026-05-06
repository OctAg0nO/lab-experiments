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
from lab.shared.mcp import MCPClient, MCPBridge
from .meta.agent_generator import AgentGenerator
from .meta.meta_agent import MetaAgent, ResourceBudget
from .evolution.gfl import GFLPipeline

console = Console()
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config" / "mcp_servers.json"

dspy.configure(lm=dspy.LM(get_lm_model(), temperature=get_lm_temperature()), adapter=BAMLAdapter())


def _get_bridge() -> MCPBridge | None:
    config = MCPClient.load_config_with_auth(str(CONFIG_PATH))
    client = MCPClient.__new__(MCPClient)
    client.config = config
    client._loop = __import__("asyncio").new_event_loop()
    client._thread = __import__("threading").Thread(target=client._loop.run_forever, daemon=True)
    client._thread.start()
    client._servers = {}
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
        budget=ctx.obj["BUDGET"],
    )


@click.group()
@click.option("--query", "-q", default="", help="Task for the meta-agent")
@click.option("--iterations", "-i", default=5, show_default=True, help="Max iterations")
@click.option("--max-llm", default=100, show_default=True, help="Max LLM calls budget")
@click.option("--max-time", default=300, show_default=True, help="Max wall seconds")
@click.option("--max-agents", default=10, show_default=True, help="Max agents to generate")
@click.pass_context
def cli(ctx, query, iterations, max_llm, max_time, max_agents):
    ctx.ensure_object(dict)
    ctx.obj["DIRECT_LM"] = dspy.LM(get_lm_model(), temperature=get_lm_temperature())
    ctx.obj["STUDENT_LM"] = dspy.LM(get_student_lm_model(), temperature=get_lm_temperature()) if get_student_lm_model() else None
    ctx.obj["BUDGET"] = ResourceBudget(max_llm_calls=max_llm, max_wall_time=max_time, max_agents=max_agents)
    ctx.obj["QUERY"] = query
    ctx.obj["ITERATIONS"] = iterations


@cli.command()
@click.pass_context
def generate(ctx):
    """Analyze task and generate agents onto the stack."""
    bridge = _get_bridge()
    meta = _make_meta(ctx, bridge)
    agents = meta.generate(ctx.obj["QUERY"])
    table = Table(title="Generated Agent Stack")
    table.add_column("Type", style="cyan")
    table.add_column("Tools", style="green")
    table.add_column("Quality", style="yellow")
    for a in agents:
        table.add_row(a.agent_type, ", ".join(a.tool_names), f"{a.quality:.2f}")
    console.print(table)
    if bridge:
        bridge.client.close()


@cli.command()
@click.pass_context
def run(ctx):
    """Full pipeline: generate -> run stack -> LSE -> consolidate."""
    bridge = _get_bridge()
    meta = _make_meta(ctx, bridge)
    result = meta.run(ctx.obj["QUERY"], max_iterations=ctx.obj["ITERATIONS"])
    console.print(Panel(f"[bold]Result:[/] {result.get('summary', 'done')[:200]}"))
    console.print(f"  Agents: {result.get('agents_generated', 0)}  |  "
                  f"Quality: {result.get('avg_quality', 0):.2f}  |  "
                  f"Improvement: {result.get('net_improvement', 0):+.2f}")
    if bridge:
        bridge.client.close()


@cli.command()
@click.pass_context
def optimize(ctx):
    """Generate agents then run GEPA optimization."""
    bridge = _get_bridge()
    meta = _make_meta(ctx, bridge)
    meta.generate(ctx.obj["QUERY"])
    meta.optimize(optimizer="gepa")
    console.print("[green]Optimization complete[/]")
    if bridge:
        bridge.client.close()


@cli.command()
@click.pass_context
def gfl(ctx):
    """Run full GFL pipeline (BootstrapFewShot, MIPROv2, GEPA)."""
    bridge = _get_bridge()
    meta = _make_meta(ctx, bridge)
    meta.generate(ctx.obj["QUERY"])
    pipeline = GFLPipeline(meta.llm, meta.tool_defs)
    results = pipeline.run_all()
    table = Table(title="GFL Results")
    table.add_column("Optimizer", style="cyan")
    table.add_column("Score", style="green")
    for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        table.add_row(name, f"{score:.3f}")
    console.print(table)
    if bridge:
        bridge.client.close()


@cli.command()
@click.pass_context
def stack(ctx):
    """Inspect the current agent stack."""
    bridge = _get_bridge()
    meta = _make_meta(ctx, bridge)
    meta.generate(ctx.obj["QUERY"])
    agents = meta.stack.list()
    if not agents:
        console.print("[yellow]No agents on stack[/]")
    else:
        table = Table(title="Agent Stack")
        table.add_column("#", style="dim")
        table.add_column("Type", style="cyan")
        table.add_column("Tools", style="green")
        table.add_column("Score", style="yellow")
        for i, a in enumerate(agents, 1):
            table.add_row(str(i), a.agent_type, ", ".join(a.tool_names), f"{a.quality:.2f}")
        console.print(table)
    if bridge:
        bridge.client.close()


@cli.command()
@click.pass_context
def distill(ctx):
    """Teacher -> student compilation."""
    bridge = _get_bridge()
    meta = _make_meta(ctx, bridge)
    meta.generate(ctx.obj["QUERY"])
    result = meta.distill(student_lm=ctx.obj["STUDENT_LM"])
    console.print(f"[green]Distillation complete:[/] student quality={result.get('student_quality', 0):.3f}")
    if bridge:
        bridge.client.close()


@cli.command(name="list-servers")
def list_servers():
    """List all configured MCP servers and their status."""
    import json
    config = json.loads(Path(str(CONFIG_PATH)).read_text())
    servers = config.get("mcpServers", {})
    table = Table(title="MCP Servers")
    table.add_column("Server", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Transport", style="yellow")
    table.add_column("Tools")
    for name, cfg in servers.items():
        enabled = "✅" if cfg.get("enabled", True) else "❌"
        transport = cfg.get("type", "stdio")
        desc = cfg.get("description", "")[:50]
        table.add_row(name, enabled, transport, desc)
    console.print(table)


@cli.command(name="health")
def health_check():
    """Check connectivity of all configured MCP servers."""
    client = MCPClient(str(CONFIG_PATH))
    try:
        client.connect_all()
        results = client.health_check()
        table = Table(title="MCP Health Check")
        table.add_column("Server", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Tools", style="yellow")
        table.add_column("Latency", style="white")
        for name, status in results.items():
            s = status["status"]
            status_icon = "✅" if s == "ok" else "❌" if s == "error" else "⚪"
            tools = str(status.get("tools", 0))
            latency = f'{status.get("latency_ms", 0)}ms'
            table.add_row(name, f"{status_icon} {s}", tools, latency)
        console.print(table)
    finally:
        client.close()


def main():
    cli()


if __name__ == "__main__":
    main()
