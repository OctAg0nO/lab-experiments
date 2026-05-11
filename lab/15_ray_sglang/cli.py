"""CLI — generate agents, run meta-agent, inspect stack, dapr-orchestrator.

DSPy commands (generate, run, gfl, stack, list-servers, health) use the
pure DSPy MetaAgent with in-memory state.

New in Lab 15:
  --sglang-endpoint  Use SGLang server as the inference backend
  --ray              Distribute agent execution via Ray tasks

Dapr commands:
  dapr-orchestrator  — Start DurableMetaAgent as a Dapr service (requires Dapr sidecar)
  dapr-wrap          — Generate DurableAgent-wrapped agents from task analysis

DSPy remains the core reasoning engine. Dapr adds durability. Ray adds parallelism.
"""

from __future__ import annotations

import atexit
import functools
import json
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
from .meta.meta_agent import MetaAgent, MetaConfig, ResourceBudget
from .evolution.gfl import GFLPipeline

console = Console()
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config" / "mcp_servers.json"

# Bridge cache — created once, shared across commands, closed on exit
_bridge_cache: MCPBridge | None = None


def _get_bridge() -> MCPBridge | None:
    global _bridge_cache
    if _bridge_cache is not None:
        return _bridge_cache

    config = MCPClient.load_config_with_auth(str(CONFIG_PATH))
    client = MCPClient.__new__(MCPClient)
    client.config = config
    # Async loop + thread for MCP client. hasattr guards protect
    # against MCPClient internal refactors (dunder attribute access).
    if not hasattr(client, '_loop'):
        client._loop = __import__("asyncio").new_event_loop()
    if not hasattr(client, '_thread'):
        client._thread = __import__("threading").Thread(
            target=client._loop.run_forever, daemon=True
        )
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
    _bridge_cache = MCPBridge(client, tool_defs)
    return _bridge_cache


def _close_bridge():
    """Close the cached bridge on process exit."""
    global _bridge_cache
    if _bridge_cache is not None:
        _bridge_cache.client.close()
        _bridge_cache = None


atexit.register(_close_bridge)


_ray_cleanup_registered = False
_generator_cache: dict[str, AgentGenerator] = {}


def _health_check_sglang(endpoint: str) -> bool:
    """Verify SGLang server is reachable before starting the agent."""
    import urllib.request
    import urllib.error
    try:
        resp = urllib.request.urlopen(f"{endpoint.rstrip('/v1')}/health", timeout=5)
        return resp.status == 200
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError):
        return False


def with_bridge(func):
    """Decorator: injects (generator, tool_defs) from cached MCP bridge."""
    @functools.wraps(func)
    def wrapper(ctx, *args, **kwargs):
        bridge = _get_bridge()
        tool_defs = bridge.tool_defs if bridge else []

        # Cache AgentGenerator by identity of tool_defs to avoid
        # recreating DSPy modules (BestOfN, ChainOfThought) per command
        cache_key = str(id(tool_defs))
        if cache_key not in _generator_cache:
            _generator_cache[cache_key] = AgentGenerator(
                ctx.obj["DIRECT_LM"], tool_defs, bridge=bridge
            )
        generator = _generator_cache[cache_key]

        return func(ctx, generator, tool_defs, *args, **kwargs)
    return wrapper


def _make_meta(ctx, generator, tool_defs) -> MetaAgent:
    return MetaAgent(config=MetaConfig(
        llm=ctx.obj["DIRECT_LM"],
        generator=generator,
        tool_defs=tool_defs,
        skills_dir=str(BASE_DIR / "memory" / "skills"),
        budget=ctx.obj["BUDGET"],
        executor=ctx.obj["EXECUTOR"],
    ))


@click.group()
@click.option("--query", "-q", default="", help="Task for the meta-agent")
@click.option("--iterations", "-i", default=5, show_default=True, help="Max iterations")
@click.option("--max-llm", default=100, show_default=True, help="Max LLM calls budget")
@click.option("--max-time", default=300, show_default=True, help="Max wall seconds")
@click.option("--max-agents", default=10, show_default=True, help="Max agents to generate")
@click.option("--sglang-endpoint", default="", help="SGLang server base URL (e.g. http://localhost:30000/v1). Overrides default LM.")
@click.option("--ray/--no-ray", default=False, help="Use Ray for parallel agent execution")
@click.pass_context
def cli(ctx, query, iterations, max_llm, max_time, max_agents, sglang_endpoint, ray):
    ctx.ensure_object(dict)

    # Configure LM: use SGLang endpoint if provided, otherwise default
    if sglang_endpoint:
        if not _health_check_sglang(sglang_endpoint):
            console.print(
                f"  [yellow]Warning: SGLang server at {sglang_endpoint} "
                f"is not responding. Check that the server is running.[/]"
            )
        lm = dspy.LM(
            model=f"openai/{get_lm_model()}",
            base_url=sglang_endpoint,
            api_key="None",
            temperature=get_lm_temperature(),
        )
    else:
        lm = dspy.LM(get_lm_model(), temperature=get_lm_temperature())

    dspy.configure(lm=lm, adapter=BAMLAdapter())
    ctx.obj["DIRECT_LM"] = lm
    ctx.obj["SGLANG_ENDPOINT"] = sglang_endpoint
    ctx.obj["STUDENT_LM"] = dspy.LM(get_student_lm_model(), temperature=get_lm_temperature()) if get_student_lm_model() else None
    ctx.obj["BUDGET"] = ResourceBudget(max_llm_calls=max_llm, max_wall_seconds=max_time, max_agents_generated=max_agents)
    ctx.obj["QUERY"] = query
    ctx.obj["ITERATIONS"] = iterations

    # Set up executor: RayModuleExecutor if --ray, otherwise InProcessExecutor
    global _ray_cleanup_registered

    if ray:
        import warnings
        try:
            from .ray.executor import RayModuleExecutor
            executor = RayModuleExecutor(num_gpus=0, num_cpus=1, timeout=300)
            ctx.obj["EXECUTOR"] = executor
            if not _ray_cleanup_registered:
                atexit.register(lambda: __import__("ray").shutdown())
                _ray_cleanup_registered = True
        except Exception as e:
            warnings.warn(f"Failed to initialize Ray: {e}. Falling back to InProcessExecutor.")
            from .ray.executor import InProcessExecutor
            ctx.obj["EXECUTOR"] = InProcessExecutor()
    else:
        from .ray.executor import InProcessExecutor
        ctx.obj["EXECUTOR"] = InProcessExecutor()


@cli.command()
@click.pass_context
@with_bridge
def generate(ctx, generator, tool_defs):
    """Analyze task and generate agents onto the stack."""
    meta = _make_meta(ctx, generator, tool_defs)
    count = meta.generate_agents(ctx.obj["QUERY"])
    console.print(f"[green]{count} agent(s)[/] generated")
    console.print(meta.stack.summary())


@cli.command()
@click.pass_context
@with_bridge
def run(ctx, generator, tool_defs):
    """Full pipeline: generate -> run stack -> LSE -> consolidate."""
    meta = _make_meta(ctx, generator, tool_defs)
    meta.generate_agents(ctx.obj["QUERY"])
    results = meta.run_stack(ctx.obj["QUERY"], max_iterations=ctx.obj["ITERATIONS"])
    eval_result = meta.evaluate_self()
    console.print(Panel(f"[bold]Done[/] — {len(results)} iterations"))
    console.print(f"  Agents: {eval_result.get('agents_generated', 0)}  |  "
                  f"Quality: {eval_result.get('avg_quality', 0):.2f}  |  "
                  f"Improvement: {eval_result.get('net_improvement', 0):+.2f}")


@cli.command()
@click.pass_context
@with_bridge
def gfl(ctx, generator, tool_defs):
    """Run full GFL pipeline (BootstrapFewShot, MIPROv2, GEPA)."""
    meta = _make_meta(ctx, generator, tool_defs)
    meta.generate_agents(ctx.obj["QUERY"])
    pipeline = GFLPipeline(
        trainset=[dspy.Example(task=ctx.obj["QUERY"]).with_inputs("task")],
        metric=lambda ex, pred, trace=None: 1.0,
    )
    baseline = pipeline.score(ctx.obj["DIRECT_LM"])
    console.print(f"  [dim]Baseline: {baseline:.3f}[/]")


@cli.command()
@click.pass_context
@with_bridge
def stack(ctx, generator, tool_defs):
    """Inspect the current agent stack."""
    meta = _make_meta(ctx, generator, tool_defs)
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


@cli.command(name="list-servers")
def list_servers():
    """List all configured MCP servers and their status."""
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
@with_bridge
def dapr_orchestrator(ctx, generator, tool_defs, tracing, dapr_frontier, dapr_lse):
    """Start DurableMetaAgent as a Dapr service (requires Dapr sidecar)."""
    sglang = ctx.obj.get("SGLANG_ENDPOINT", "")
    if sglang:
        console.print(
            "  [yellow]Warning: --sglang-endpoint is set but dapr-orchestrator "
            "uses DaprChatClient (not SGLang) for inference. "
            "Agent generation uses SGLang but the workflow loop uses Dapr's LLM provider.[/]"
        )

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
        executor=ctx.obj["EXECUTOR"],
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
@with_bridge
def dapr_wrap(ctx, generator, tool_defs):
    """Analyze task and generate DurableAgent-wrapped agent definitions."""
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


# ---------------------------------------------------------------------------
# Swarm commands (multi-agent coordination via Dapr pub/sub)
# ---------------------------------------------------------------------------


@cli.command(name="swarm-coordinator")
@click.option("--tracing", is_flag=True, default=False, help="Enable Zipkin tracing")
@click.option("--workers", default=2, show_default=True, help="Expected worker count")
@click.pass_context
@with_bridge
def swarm_coordinator(ctx, generator, tool_defs, tracing, workers):
    """Start SwarmCoordinator — dispatches tasks to worker agents."""
    from .swarm.coordinator import SwarmCoordinator

    agent = SwarmCoordinator(enable_tracing=tracing)
    from dapr_agents import AgentRunner
    runner = AgentRunner()
    query = ctx.obj.get("QUERY", "")
    port = get_agent_port("orchestrator")
    console.print(f"[bold cyan]SwarmCoordinator[/] on port {port}")
    console.print(f"  Query: {query}")
    console.print(f"  Workers expected: {workers}")
    runner.serve(agent, port=port, input={
        "query": query,
        "worker_app_ids": [f"swarm-worker-{i}" for i in range(workers)],
    })


@cli.command(name="swarm-worker")
@click.option("--domain", default="general", help="Research domain for this worker")
@click.option("--worker-id", default="swarm-worker-0", help="Unique worker ID")
@click.option("--port", default=8001, type=int, help="HTTP port")
@click.option("--tracing", is_flag=True, default=False, help="Enable Zipkin tracing")
@click.pass_context
@with_bridge
def swarm_worker(ctx, generator, tool_defs, domain, worker_id, port, tracing):
    """Start SwarmMetaAgent — subscribes to swarm tasks."""
    from .swarm.worker import SwarmMetaAgent
    from .core.durable_meta_agent import DurableMetaConfig

    agent = SwarmMetaAgent(
        generator=generator,
        tool_defs=tool_defs,
        skills_dir=str(BASE_DIR / "memory" / "skills"),
        budget=ctx.obj["BUDGET"],
        config=DurableMetaConfig(enable_tracing=tracing),
        agent_id=worker_id,
        domain=domain,
    )
    console.print(f"[bold cyan]SwarmMetaAgent[/] {worker_id} on port {port} (domain: {domain})")
    agent.run_worker(port=port)


@cli.command(name="swarm")
@click.option("--workers", default=2, show_default=True, help="Number of worker agents")
@click.option("--tracing", is_flag=True, default=False, help="Enable Zipkin tracing")
@click.pass_context
@with_bridge
def swarm(ctx, generator, tool_defs, workers, tracing):
    """Run a full swarm in-process (coordinator + workers)."""
    from .swarm.coordinator import SwarmCoordinator
    from .swarm.worker import SwarmMetaAgent
    from .core.durable_meta_agent import DurableMetaConfig

    coordinator = SwarmCoordinator(enable_tracing=tracing)

    console.print(f"[bold cyan]Swarm[/] with {workers} workers")
    console.print(f"  Query: {ctx.obj.get('QUERY', '')}")
    console.print(f"  Tracing: {'on' if tracing else 'off'}")
    console.print("  [dim]Use swarm-coordinator and swarm-worker separately in production[/]")
    console.print(f"\nFrontier: {coordinator.frontier.summary()}")


# ---------------------------------------------------------------------------
# LiveKit commands (voice + A2UI integration)
# ---------------------------------------------------------------------------


@cli.command(name="livekit-worker")
@click.option("--stt", default="deepgram/nova-3", help="STT model")
@click.option("--tts", default="cartesia/sonic-3", help="TTS model (for livekit backend)")
@click.option("--tts-voice", default="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc", help="TTS voice ID (for livekit backend)")
@click.option("--tts-backend", default="livekit", help="TTS backend: livekit (cloud) or qwen3 (local)")
@click.option("--qwen3-mode", default="custom_voice", help="Qwen3-TTS mode: custom_voice, voice_design, voice_clone")
@click.option("--qwen3-speaker", default="Vivian", help="Qwen3-TTS speaker: Vivian, Ryan, Serena, Eric, Aiden, etc.")
@with_bridge
@click.pass_context
def livekit_worker(ctx, generator, tool_defs, stt, tts, tts_voice, tts_backend, qwen3_mode, qwen3_speaker):
    """Start a LiveKit agent worker with OctAg0nO brain.

    Requires LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET in .env.
    Start the LiveKit server first: docker run -p 7880:7880 livekit/livekit-server
    """
    sglang = ctx.obj.get("SGLANG_ENDPOINT", "")
    if not sglang:
        console.print("  [yellow]Warning: --sglang-endpoint not set. "
                      "Voice agent will use default LLM.[/]")

    from .livekit.worker import run_worker
    from .meta.meta_agent import MetaAgent, MetaConfig

    meta = MetaAgent(config=MetaConfig(
        llm=ctx.obj["DIRECT_LM"],
        generator=generator,
        tool_defs=tool_defs,
        skills_dir=str(BASE_DIR / "memory" / "skills"),
        budget=ctx.obj["BUDGET"],
        executor=ctx.obj["EXECUTOR"],
    ))

    console.print("[bold cyan]LiveKit Worker[/] starting...")
    console.print(f"  STT: {stt}")
    console.print(f"  TTS: {tts if tts_backend == 'livekit' else f'Qwen3 ({qwen3_mode}/{qwen3_speaker})'}")
    console.print(f"  SGLang: {sglang or 'default LLM'}")

    run_worker(
        meta,
        stt_model=stt,
        tts_model=tts,
        tts_voice=tts_voice,
        tts_backend=tts_backend,
        qwen3_mode=qwen3_mode,
        qwen3_speaker=qwen3_speaker,
    )


def main():
    cli()


if __name__ == "__main__":
    main()
