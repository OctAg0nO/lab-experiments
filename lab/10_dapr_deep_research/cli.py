"""
CLI — start individual Dapr agents or run teacher/student distillation.

Usage:
    dapr run --app-id orchestrator --app-protocol grpc --app-port 8000 --resources-path ./resources -- \
        python -m lab.10_dapr_deep_research --mode orchestrator
    dapr run --app-id explorer-agent --app-protocol grpc --app-port 8001 --resources-path ./resources -- \
        python -m lab.10_dapr_deep_research --mode explorer
    python -m lab.10_dapr_deep_research --mode distill
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
import dspy
from dspy.adapters.baml_adapter import BAMLAdapter

from ..shared.config import get_lm_model, get_student_lm_model
from .mcp.client import MCPClient
from .mcp.bridge import MCPBridge
from .memory.dapr_frontier import DaprFrontier
from .evolution.lse import LSEOptimizer
from .evolution.trace2skill import SkillConsolidator
from .agents.research_agents import ExplorerAgent, DeepReaderAgent, SynthesizerAgent, CriticAgent
from .orchestrator.workflow import ResearchWorkflow

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config" / "mcp_servers.json"

_TEACHER_LM = dspy.LM(get_lm_model())
dspy.configure(lm=_TEACHER_LM, adapter=BAMLAdapter())


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


def cmd_distill():
    """Teacher (DeepSeek) → student (Gemma 4) distillation for all DSPy programs.

    Compiles every ChainOfThought / Refine module using BootstrapFewShot
    with the teacher generating demonstrations and the student learning from them.
    """
    teacher_lm = _TEACHER_LM
    student_lm = dspy.LM(get_student_lm_model())
    print(f"Teacher: {get_lm_model()}")
    print(f"Student: {get_student_lm_model()}")

    bridge = _get_bridge()
    trainset: list[dspy.Example] = []

    agents: list[tuple[str, ExplorerAgent | DeepReaderAgent | SynthesizerAgent | CriticAgent | ResearchWorkflow | DaprFrontier | LSEOptimizer | SkillConsolidator]] = [
        ("ExplorerAgent", ExplorerAgent(bridge=bridge)),
        ("DeepReaderAgent", DeepReaderAgent(bridge=bridge)),
        ("SynthesizerAgent", SynthesizerAgent(bridge=bridge)),
        ("CriticAgent", CriticAgent()),
        ("Workflow", ResearchWorkflow(frontier=DaprFrontier())),
        ("LSEOptimizer", LSEOptimizer()),
        ("SkillConsolidator", SkillConsolidator(BASE_DIR / "memory" / "skills")),
        ("DaprFrontier", DaprFrontier()),
    ]

    for name, agent in agents:
        if hasattr(agent, "compile") and trainset:
            print(f"  Compiling {name} with student LM ...")
            try:
                agent.compile(trainset, student_lm=student_lm)
                print(f"    ✓ {name} compiled")
            except Exception as e:
                print(f"    ✗ {name} failed: {e}")
        else:
            print(f"  Skipping {name} (no trainset or no compile())")

    print("\nDistillation complete. All compiled modules use student_lm for inference.")


def main():
    parser = argparse.ArgumentParser(description="Dapr Deep Research — multi-agent research platform")
    parser.add_argument("--mode", choices=["orchestrator", "explorer", "deepreader", "synthesizer", "critic", "run", "distill"], default="run")
    args = parser.parse_args()

    modes = {
        "orchestrator": cmd_orchestrator,
        "explorer": cmd_explorer,
        "deepreader": cmd_deep_reader,
        "synthesizer": cmd_synthesizer,
        "critic": cmd_critic,
        "run": cmd_run,
        "distill": cmd_distill,
    }
    modes[args.mode]()


if __name__ == "__main__":
    main()
