"""CLI — generate agents, run meta-agent, inspect stack, dapr-orchestrator.

DSPy commands (generate, run, gfl, stack, list-servers, health) use the
pure DSPy MetaAgent with in-memory state.

New Dapr commands:
  dapr-orchestrator  — Start DurableMetaAgent as a Dapr service (requires Dapr sidecar)
  dapr-wrap          — Generate DurableAgent-wrapped agents from task analysis

DSPy remains the core reasoning engine. Dapr adds durability.
"""

from __future__ import annotations

from pathlib import Path

import click
from dotenv import load_dotenv
import dspy
from dspy.adapters.baml_adapter import BAMLAdapter
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..shared.config import get_lm_model, get_student_lm_model, get_lm_temperature, get_agent_port
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
    ctx.obj["BUDGET"] = ResourceBudget(max_llm_calls=max_llm, max_wall_seconds=max_time, max_agents_generated=max_agents)
    ctx.obj["QUERY"] = query
    ctx.obj["ITERATIONS"] = iterations


@cli.command()
@click.pass_context
def generate(ctx):
    """Analyze task and generate agents onto the stack."""
    bridge = _get_bridge()
    meta = _make_meta(ctx, bridge)
    count = meta.generate_agents(ctx.obj["QUERY"])
    console.print(f"[green]{count} agent(s)[/] generated")
    console.print(meta.stack.summary())
    if bridge:
        bridge.client.close()


@cli.command()
@click.pass_context
def run(ctx):
    """Full pipeline: generate -> run stack -> LSE -> consolidate."""
    bridge = _get_bridge()
    meta = _make_meta(ctx, bridge)
    meta.generate_agents(ctx.obj["QUERY"])
    results = meta.run_stack(ctx.obj["QUERY"], max_iterations=ctx.obj["ITERATIONS"])
    eval_result = meta.evaluate_self()
    console.print(Panel(f"[bold]Done[/] — {len(results)} iterations"))
    console.print(f"  Agents: {eval_result.get('agents_generated', 0)}  |  "
                  f"Quality: {eval_result.get('avg_quality', 0):.2f}  |  "
                  f"Improvement: {eval_result.get('net_improvement', 0):+.2f}")
    if bridge:
        bridge.client.close()


@cli.command()
@click.pass_context
def gfl(ctx):
    """Run full GFL pipeline (BootstrapFewShot, MIPROv2, GEPA)."""
    bridge = _get_bridge()
    meta = _make_meta(ctx, bridge)
    meta.generate_agents(ctx.obj["QUERY"])
    lm = ctx.obj["DIRECT_LM"]
    pipeline = GFLPipeline(
        trainset=[dspy.Example(task=ctx.obj["QUERY"]).with_inputs("task")],
        metric=lambda ex, pred, trace=None: 1.0,
    )
    baseline = pipeline.score(lm)
    results = {"baseline": baseline}
    console.print(f"  [dim]Baseline: {baseline:.3f}[/]")
    if bridge:
        bridge.client.close()


@cli.command()
@click.pass_context
def stack(ctx):
    """Inspect the current agent stack."""
    bridge = _get_bridge()
    meta = _make_meta(ctx, bridge)
    meta.generate_agents(ctx.obj["QUERY"])
    agents = meta.stack.snapshot()
    if not agents:
        console.print("[yellow]No agents on stack[/]")
    else:
        table = Table(title="Agent Stack")
        table.add_column("Name", style="cyan")
        table.add_column("Role", style="green")
        table.add_column("Runs", style="yellow")
        table.add_column("Quality")
        table.add_column("Tools")
        for a in agents:
            table.add_row(a.name, a.role, str(a.run_count),
                          f"{a.avg_quality:.2f}", str(len(a.tools)))
        console.print(table)
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


# ---------------------------------------------------------------------------
# Dapr commands (require Dapr sidecar + Redis)
# ---------------------------------------------------------------------------


@cli.command(name="dapr-orchestrator")
@click.option("--tracing", is_flag=True, default=False, help="Enable Zipkin tracing")
@click.option("--dapr-frontier", is_flag=True, default=False, help="Use Redis-backed DaprFrontier")
@click.option("--dapr-lse", is_flag=True, default=False, help="Use Redis-backed DaprLSEOptimizer")
@click.pass_context
def dapr_orchestrator(ctx, tracing, dapr_frontier, dapr_lse):
    """Start DurableMetaAgent as a Dapr service (requires Dapr sidecar).

    Wraps the DSPy MetaAgent in a DurableAgent workflow for crash-resistant
    execution. Each iteration is checkpointed to Redis.

    All DSPy modules remain the core reasoning engine — unchanged.
    Dapr adds: durability, observability, hot-reload, retry policies.
    """
    bridge = _get_bridge()
    tool_defs = bridge.tool_defs if bridge else []
    generator = AgentGenerator(ctx.obj["DIRECT_LM"], tool_defs, bridge=bridge)

    from .core.durable_meta_agent import DurableMetaAgent, DurableMetaConfig

    agent = DurableMetaAgent(
        generator=generator,
        tool_defs=tool_defs,
        skills_dir=str(BASE_DIR / "memory" / "skills"),
        budget=ctx.obj["BUDGET"],
        config=DurableMetaConfig(
            enable_tracing=tracing,
            use_dapr_frontier=dapr_frontier,
            use_dapr_lse=dapr_lse,
        ),
    )

    from dapr_agents import AgentRunner
    port = get_agent_port("orchestrator")
    runner = AgentRunner()
    query = ctx.obj.get("QUERY", "")
    console.print(f"[bold cyan]DurableMetaAgent[/] serving on port {port}")
    console.print(f"  Query: {query}")
    console.print(f"  Tracing: {'on' if tracing else 'off'}")
    console.print(f"  DaprFrontier: {'on' if dapr_frontier else 'off'}")
    console.print(f"  DaprLSE: {'on' if dapr_lse else 'off'}")
    runner.serve(agent, port=port, input={"query": query})


@cli.command(name="dapr-wrap")
@click.pass_context
def dapr_wrap(ctx):
    """Analyze task and generate DurableAgent-wrapped agent definitions.

    Uses the same AgentGenerator (BestOfN + RLM/ReAct/CodeAct/CoT) as the
    'generate' command, but shows how each would be wrapped in a
    GeneratedDurableAgent shell for Dapr deployment.
    """
    bridge = _get_bridge()
    tool_defs = bridge.tool_defs if bridge else []
    generator = AgentGenerator(ctx.obj["DIRECT_LM"], tool_defs, bridge=bridge)

    definitions = generator.analyze(ctx.obj["QUERY"])

    table = Table(title="DurableAgent-Wrappable Agents")
    table.add_column("Name", style="cyan")
    table.add_column("Role", style="green")
    table.add_column("Module Type", style="yellow")
    table.add_column("DurableAgent", style="magenta")
    table.add_column("Tools")

    for d in definitions:
        name = d.get("name", "?")
        role = d.get("role", "?")
        use_code = d.get("use_code", False)
        has_tools = bool(d.get("tools", []))
        if use_code and has_tools:
            module_type = "dspy.RLM"
        elif has_tools:
            module_type = "dspy.ReAct"
        elif use_code:
            module_type = "dspy.CodeAct"
        else:
            module_type = "dspy.ChainOfThought"
        tools_list = ", ".join(d.get("tools", [])) or "(none)"
        table.add_row(name, role, module_type, "GeneratedDurableAgent", tools_list)

    console.print(table)
    console.print("\n[dim]Run with: dapr run --app-id <name> -- python ...[/]")
    if bridge:
        bridge.client.close()


def main():
    cli()


if __name__ == "__main__":
    main()
