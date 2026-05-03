"""
CLI — start individual Dapr agents or the multi-app research run.

Usage:
    dapr run --app-id orchestrator --app-protocol grpc --app-port 8000 --resources-path ./resources -- \
        python -m lab.10_dapr_deep_research --mode orchestrator
    dapr run --app-id explorer-agent --app-protocol grpc --app-port 8001 --resources-path ./resources -- \
        python -m lab.10_dapr_deep_research --mode explorer
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
import dspy
from dspy.adapters.baml_adapter import BAMLAdapter

from .mcp.client import MCPClient
from .mcp.bridge import MCPBridge
from .memory.dapr_frontier import DaprFrontier
from .agents.research_agents import ExplorerAgent, DeepReaderAgent, SynthesizerAgent, CriticAgent
from .orchestrator.workflow import ResearchWorkflow

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config" / "mcp_servers.json"

dspy.configure(lm=dspy.LM("deepseek/deepseek-v4-flash"), adapter=BAMLAdapter())


def _get_bridge() -> MCPBridge:
    client = MCPClient(str(CONFIG_PATH))
    tool_defs = client.connect_all()
    return MCPBridge(client, tool_defs)


def cmd_orchestrator():
    frontier = DaprFrontier()
    print(f"Frontier: {frontier.summary()}")
    agent = ResearchWorkflow(frontier=frontier)
    from dapr_agents import AgentRunner
    runner = AgentRunner()
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


def cmd_run():
    """Single-process programmatic run (no Dapr sidecar needed for dev)."""
    client = MCPClient(str(CONFIG_PATH))
    client.connect_all()
    frontier = DaprFrontier()
    print(f"Frontier: {frontier.summary()}")
    print("Running research loop programmatically...")
    query = "Research DSPy optimization patterns for LLM pipelines"
    frontier.seed_from_query(query)
    for i in range(3):
        direction = frontier.next_action()
        if not direction:
            break
        print(f"  Iteration {i+1}: {direction.topic[:60]}")
        frontier.absorb_findings(direction.topic, 0.2, 1, [])
    print(f"Done. {frontier.summary()}")
    client.close()


def main():
    parser = argparse.ArgumentParser(description="Dapr Deep Research — multi-agent research platform")
    parser.add_argument("--mode", choices=["orchestrator", "explorer", "deepreader", "synthesizer", "critic", "run"], default="run")
    args = parser.parse_args()

    modes = {
        "orchestrator": cmd_orchestrator,
        "explorer": cmd_explorer,
        "deepreader": cmd_deep_reader,
        "synthesizer": cmd_synthesizer,
        "critic": cmd_critic,
        "run": cmd_run,
    }
    modes[args.mode]()


if __name__ == "__main__":
    main()
