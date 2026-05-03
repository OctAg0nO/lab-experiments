"""
CLI for the Super Deep Research platform.

Usage:
    python -m lab.09_super_deep_research.cli --query "your research topic"
    python -m lab.09_super_deep_research.cli --chat
    python -m lab.09_super_deep_research.cli --status
    python -m lab.09_super_deep_research.cli --list-servers
    python -m lab.09_super_deep_research.cli --disable-server openrouter
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
import dspy

from .mcp.client import MCPClient
from .frontier import ResearchFrontier
from .memory.store import MemoryStore
from .orchestrator import ResearchOrchestrator

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config" / "mcp_servers.json"


def _get_memory() -> MemoryStore:
    return MemoryStore(BASE_DIR / "memory")


def _get_frontier() -> ResearchFrontier:
    return ResearchFrontier(persist_path=str(BASE_DIR / "memory" / "frontier.json"))


def _load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def _save_config(config: dict):
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_list_servers():
    config = _load_config()
    servers = config.get("mcpServers", {})
    if not servers:
        print("No MCP servers configured.")
        return
    print(f"{'SERVER':20s} {'ENABLED':10s} {'TRANSPORT':10s} {'DESCRIPTION'}")
    print("-" * 70)
    for name, cfg in servers.items():
        enabled = "✅" if cfg.get("enabled", True) else "❌"
        transport = cfg.get("type", "stdio")
        desc = cfg.get("description", "")[:40]
        print(f"{name:20s} {enabled:10s} {transport:10s} {desc}")


def cmd_enable_server(name: str):
    config = _load_config()
    servers = config.get("mcpServers", {})
    if name not in servers:
        print(f"Server '{name}' not found in config.")
        return
    servers[name]["enabled"] = True
    _save_config(config)
    print(f"✅ {name} enabled.")


def cmd_disable_server(name: str):
    config = _load_config()
    servers = config.get("mcpServers", {})
    if name not in servers:
        print(f"Server '{name}' not found in config.")
        return
    servers[name]["enabled"] = False
    _save_config(config)
    print(f"❌ {name} disabled.")


def cmd_list_skills():
    memory = _get_memory()
    skills = memory.load_skills()
    if not skills:
        print("No skills accumulated yet.")
        return
    print(f"{'SKILL':30s} {'SAVED'}")
    print("-" * 50)
    for s in skills:
        name = s.get("name", Path(s.get("_file", "unknown")).stem) if "_file" in s else "skill"
        saved = s.get("saved", "?")[:19]
        print(f"{name:30s} {saved}")


def cmd_list_frontier():
    frontier = _get_frontier()
    print(f"Frontier: {frontier.summary()}")
    print()
    if not frontier.directions:
        print("No research directions yet. Run --query to seed.")
        return
    print(f"{'TOPIC':50s} {'CONF':6s} {'DEPTH':6s} {'UCB':6s}")
    print("-" * 70)
    total_exp = frontier.total_explorations
    for d in sorted(frontier.directions, key=lambda x: x.ucb_score(total_exp), reverse=True):
        ucb = d.ucb_score(total_exp)
        ucb_str = "∞" if ucb == float("inf") else f"{ucb:.2f}"
        print(f"{d.topic[:48]:50s} {d.confidence:.2f}  {d.exploration_depth:6d} {ucb_str:6s}")


def cmd_status():
    memory = _get_memory()
    frontier = _get_frontier()
    config = _load_config()
    servers = config.get("mcpServers", {})
    n_enabled = sum(1 for s in servers.values() if s.get("enabled", True))
    n_disabled = sum(1 for s in servers.values() if not s.get("enabled", True))
    print("=" * 50)
    print("SUPER DEEP RESEARCH — STATUS")
    print("=" * 50)
    print(f"  MCP servers:    {len(servers)} ({n_enabled} enabled, {n_disabled} disabled)")
    print(f"  {memory.summary()}")
    print(f"  {frontier.summary()}")


def cmd_reset_memory():
    import shutil
    memory_dir = BASE_DIR / "memory"
    if memory_dir.exists():
        shutil.rmtree(memory_dir)
        memory_dir.mkdir(parents=True)
        print("🧹 Memory reset complete.")
    else:
        print("No memory to reset.")


def cmd_query(query: str, max_iterations: int):
    lm = dspy.LM("deepseek/deepseek-v4-flash")
    dspy.configure(lm=lm)

    memory = _get_memory()
    frontier = _get_frontier()
    print(f"  {memory.summary()}")
    print(f"  {frontier.summary()}")
    print()

    client = MCPClient(str(CONFIG_PATH))
    try:
        tool_defs = client.connect_all()
        if not tool_defs:
            raise RuntimeError("No MCP tools discovered.")
        print(f"\nActive servers: {len(set(td['server'] for td in tool_defs))}")
        orchestrator = ResearchOrchestrator(
            mcp_client=client, tool_defs=tool_defs, lm=lm,
            memory=memory, frontier=frontier, max_iterations=max_iterations,
        )
        print(f"\n{'='*60}")
        print(f"QUERY: {query[:80]}...")
        print(f"{'='*60}\n")
        report = orchestrator.run(query)
        print(f"\n{'='*60}")
        print("RESEARCH COMPLETE")
        print(f"{'='*60}")
        print(f"  Iterations:      {report['iterations']}")
        print(f"  {report['frontier']}")
        print(f"  {report['memory']}")
        improvement = report.get("improvement_trend", [])
        if improvement:
            print(f"  LSE trend: {[f'{x:+.2f}' for x in improvement]}")
    finally:
        client.close()


def cmd_chat(max_iterations: int):
    """Interactive research chat REPL."""
    lm = dspy.LM("deepseek/deepseek-v4-flash")
    dspy.configure(lm=lm)

    memory = _get_memory()
    frontier = _get_frontier()
    prev_skills = memory.load_skills()
    print(f"  {memory.summary()}")
    if prev_skills:
        print(f"  Loaded {len(prev_skills)} skill(s) from previous runs")
    print(f"  {frontier.summary()}")

    client = MCPClient(str(CONFIG_PATH))
    try:
        tool_defs = client.connect_all()
        if not tool_defs:
            raise RuntimeError("No MCP tools discovered.")
        print(f"\nActive servers: {len(set(td['server'] for td in tool_defs))}")
        print(f"\n{'='*50}")
        print("Chat mode — enter research queries. Type /help for commands.")
        print(f"{'='*50}")
        while True:
            try:
                line = input("\n🔬 ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line.startswith("/"):
                _handle_slash_command(line, memory, frontier, client, tool_defs, lm, max_iterations)
            else:
                orchestrator = ResearchOrchestrator(
                    mcp_client=client, tool_defs=tool_defs, lm=lm,
                    memory=memory, frontier=frontier, max_iterations=max_iterations,
                )
                print()
                report = orchestrator.run(line)
                print(f"\n  Iterations: {report['iterations']}")
                print(f"  {report['frontier']}")
    finally:
        client.close()


def _handle_slash_command(cmd: str, memory, frontier, client, tool_defs, lm, max_iterations):
    parts = cmd.split(maxsplit=1)
    match parts[0]:
        case "/help":
            print("  /query <text>    — run a research query")
            print("  /status          — show current frontier + memory")
            print("  /skills          — list accumulated skills")
            print("  /frontier        — show research frontier")
            print("  /reset           — reset memory")
            print(f"  /iterations <N>  — set max iterations (current: {max_iterations})")
            print("  /quit            — exit chat")
        case "/status":
            print(f"  {memory.summary()}")
            print(f"  {frontier.summary()}")
        case "/skills":
            skills = memory.load_skills()
            if skills:
                print(f"  {len(skills)} skill(s)")
                for s in skills[:5]:
                    print(f"    - {str(s)[:80]}")
            else:
                print("  No skills yet.")
        case "/frontier":
            for d in frontier.directions:
                print(f"  {d.topic[:50]:50s} conf={d.confidence:.2f} depth={d.exploration_depth}")
        case "/reset":
            import shutil
            memory_dir = Path(memory.base)
            if memory_dir.exists():
                shutil.rmtree(memory_dir)
                memory_dir.mkdir(parents=True)
                memory.__init__(memory.base)
                frontier.__init__(persist_path=str(Path(memory.base) / "frontier.json"))
                print("  Memory reset.")
        case "/quit":
            sys.exit(0)
        case _:
            print(f"  Unknown command: {parts[0]}. Type /help")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Super Deep Research — self-evolving agentic research platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m lab.09_super_deep_research.cli --query \"DSPy optimization benchmarks\"\n"
            "  python -m lab.09_super_deep_research.cli --chat\n"
            "  python -m lab.09_super_deep_research.cli --status\n"
            "  python -m lab.09_super_deep_research.cli --list-servers\n"
            "  python -m lab.09_super_deep_research.cli --disable-server openrouter\n"
        ),
    )

    parser.add_argument("--query", "-q", type=str, help="Single research query to run")
    parser.add_argument("--chat", "-c", action="store_true", help="Interactive chat mode")
    parser.add_argument("--iterations", "-i", type=int, default=6, help="Max research iterations")

    parser.add_argument("--status", "-s", action="store_true", help="Show platform status")
    parser.add_argument("--list-servers", action="store_true", help="List MCP servers")
    parser.add_argument("--enable-server", type=str, metavar="NAME", help="Enable an MCP server")
    parser.add_argument("--disable-server", type=str, metavar="NAME", help="Disable an MCP server")
    parser.add_argument("--list-skills", action="store_true", help="List accumulated skills")
    parser.add_argument("--list-frontier", action="store_true", help="List research frontier")
    parser.add_argument("--reset-memory", action="store_true", help="Reset all persisted memory")

    args = parser.parse_args()

    if args.list_servers:
        cmd_list_servers()
    elif args.enable_server:
        cmd_enable_server(args.enable_server)
    elif args.disable_server:
        cmd_disable_server(args.disable_server)
    elif args.list_skills:
        cmd_list_skills()
    elif args.list_frontier:
        cmd_list_frontier()
    elif args.status:
        cmd_status()
    elif args.reset_memory:
        cmd_reset_memory()
    elif args.query:
        cmd_query(args.query, args.iterations)
    elif args.chat:
        cmd_chat(args.iterations)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
