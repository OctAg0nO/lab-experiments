"""
Super Deep Research — self-evolving agentic research platform.

Pipeline:
  1. MCP client connects to Crawl4AI, OpenRouter, fetch, filesystem
  2. ResearchFrontier seeds from user query (autonomous discovery)
  3. Orchestrator dispatches specialized agents (Explorer, DeepReader,
     Synthesizer, Critic) with LSE meta-optimization
  4. Findings persisted to Knowledge Graph + Skill Library
  5. Trajectories consolidated via Trace2Skill across runs
"""

from pathlib import Path
from dotenv import load_dotenv
import dspy

from lab.shared.mcp import MCPClient
from .frontier import ResearchFrontier
from .memory.store import MemoryStore
from .orchestrator import ResearchOrchestrator

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

lm = dspy.LM("deepseek/deepseek-v4-flash")
dspy.configure(lm=lm)


def main():
    base_dir = Path(__file__).parent
    config_path = base_dir / "config" / "mcp_servers.json"

    if not config_path.exists():
        raise FileNotFoundError(f"MCP config not found: {config_path}")

    # ---- Memory ----
    memory = MemoryStore(base_dir / "memory")
    print(f"  {memory.summary()}")

    # ---- Frontier ----
    frontier = ResearchFrontier(persist_path=str(base_dir / "memory" / "frontier.json"))
    prev_skills = memory.load_skills()
    if prev_skills:
        print(f"  Loaded {len(prev_skills)} skill(s) from previous runs")

    # ---- MCP ----
    client = MCPClient(str(config_path))
    try:
        tool_defs = client.connect_all()
        if not tool_defs:
            raise RuntimeError("No MCP tools discovered.")

        print(f"\nDiscovered {len(tool_defs)} MCP tool(s):")
        for td in tool_defs:
            print(f"  [{td['server']}] {td['name']}")

        # ---- MCP tool capabilities summary ----
        servers = set(td["server"] for td in tool_defs)
        print(f"\nActive servers: {', '.join(sorted(servers))}")

        # ---- Research ----
        orchestrator = ResearchOrchestrator(
            mcp_client=client,
            tool_defs=tool_defs,
            lm=lm,
            memory=memory,
            frontier=frontier,
            max_iterations=6,
        )

        query = (
            "Research how DSPy's Generative Feedback Loops optimize LLM "
            "pipelines — compare BootstrapFewShot, MIPROv2, and GEPA. "
            "Find recent benchmarks and real-world applications."
        )

        print(f"\n{'='*60}")
        print(f"RESEARCH: {query[:80]}...")
        print(f"{'='*60}")

        report = orchestrator.run(query)

        # ---- Results ----
        print(f"\n{'='*60}")
        print("RESEARCH COMPLETE")
        print(f"{'='*60}")
        print(f"  Iterations:      {report['iterations']}")
        print(f"  {report['frontier']}")
        print(f"  {report['memory']}")
        improvement = report.get("improvement_trend", [])
        if improvement:
            print(f"  LSE improvement trend: {[f'{x:+.2f}' for x in improvement]}")
        best = report.get("best_strategy")
        if best:
            print(f"  Best strategy:   {best}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
