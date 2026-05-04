"""AgentGenerator — dynamically creates DSPy agents using ReAct, BestOfN, and tool integration."""

from __future__ import annotations

import json
import logging

import dspy

from .agent_stack import AgentEntry

logger = logging.getLogger(__name__)


class AnalyzeTask(dspy.Signature):
    """Analyze a user task and determine what sub-agents are needed."""
    task: str = dspy.InputField()
    num_agents: int = dspy.OutputField(desc="How many distinct sub-agents needed")
    agent_definitions: str = dspy.OutputField(
        desc="JSON list: [{\"name\", \"role\", \"goal\", \"tools\", \"use_code\"}]"
    )


class EvaluateQuality(dspy.Signature):
    """Evaluate the quality of an agent's prediction for a research task."""
    task: str = dspy.InputField()
    agent_role: str = dspy.InputField()
    prediction: str = dspy.InputField(desc="The agent's output")
    quality_score: float = dspy.OutputField(desc="Quality from 0.0 to 1.0")
    reasoning: str = dspy.OutputField(desc="Why this score was given")


def _build_tools(tool_names: list[str], bridge) -> list:
    """Build DSPy tool callables from MCP bridge for a list of tool names."""
    if not bridge:
        return []
    all_fns = bridge.get_dspy_tools()
    if not tool_names:
        return all_fns
    return [fn for fn in all_fns if fn.__name__ in tool_names]


class AgentGenerator:
    """Generates DSPy agents using BestOfN, ReAct, and tool integration.

    DSPy features used:
    - dspy.ChainOfThought for task analysis
    - dspy.BestOfN to sample multiple candidates and pick the best
    - dspy.ReAct for tool-using agents (vs plain ChainOfThought)
    - dspy.PythonInterpreter for code-capable agents
    """

    def __init__(self, llm: dspy.LM, tool_defs: list[dict] | None = None, bridge=None):
        self._llm = llm
        self._tool_defs = tool_defs or []
        self._bridge = bridge
        self._module_cache: dict[str, dspy.Module] = {}
        self._evaluator = dspy.ChainOfThought(EvaluateQuality)

        # Use BestOfN to sample 3 candidate analyses, pick best by number of agents
        self._analyzer = dspy.BestOfN(
            dspy.ChainOfThought(AnalyzeTask),
            N=3,
            reward_fn=lambda ex, pred: (
                getattr(pred, "num_agents", 0)
                if hasattr(pred, "agent_definitions") and pred.agent_definitions
                else 0
            ),
            threshold=0.5,
        )

    # -- task analysis with BestOfN --

    def analyze(self, task: str) -> list[dict]:
        best = self._analyzer(task=task)
        result = best if hasattr(best, "agent_definitions") else None
        if not result or not result.agent_definitions:
            logger.warning("AnalyzeTask returned no agent_definitions, using defaults")
            return self._default_agents(task)
        try:
            agents = json.loads(result.agent_definitions)
            if not isinstance(agents, list):
                logger.warning("agent_definitions not a list, using defaults")
                return self._default_agents(task)
            return agents
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Failed to parse agent_definitions: %s", e)
            return self._default_agents(task)

    # -- agent entry generation --

    def generate(self, definition: dict) -> AgentEntry:
        return AgentEntry(
            name=definition.get("name", f"agent_{id(definition)}"),
            role=definition.get("role", "assistant"),
            goal=definition.get("goal", "help with the task"),
            signature="task -> result",
            tools=definition.get("tools", []),
            use_code=definition.get("use_code", False),
            prompt_template=definition.get("prompt", ""),
        )

    def _default_agents(self, task: str) -> list[dict]:
        return [
            {
                "name": "searcher", "role": "Web Researcher",
                "goal": f"Search and gather information about: {task}",
                "tools": ["search", "fetch", "md"], "use_code": False,
            },
            {
                "name": "analyzer", "role": "Content Analyst",
                "goal": f"Analyze and extract insights from research about: {task}",
                "tools": ["chat"], "use_code": False,
            },
            {
                "name": "synthesizer", "role": "Research Synthesizer",
                "goal": f"Synthesize findings into coherent report about: {task}",
                "tools": [], "use_code": True,
            },
        ]

    # -- module generation: RLM + ReAct + CodeAct + CoT --

    def generate_module(self, entry: AgentEntry) -> dspy.Module:
        if entry.name in self._module_cache:
            return self._module_cache[entry.name]

        prompt = entry.prompt_template or f"You are {entry.role}. {entry.goal}"
        tools = _build_tools(entry.tools, self._bridge)

        if entry.use_code and tools:
            # RLM: REPL-based agent with code execution, MCP tools, and sub-LLM queries
            module = dspy.RLM(
                "task: str -> result: str",
                tools=tools,
                max_iterations=10,
                max_llm_calls=16,
            )
        elif tools:
            # ReAct: agentic loop with tools (thought + action + observation)
            module = dspy.ReAct(
                "task: str -> result: str",
                tools=tools,
                max_iters=10,
            )
        elif entry.use_code:
            # CodeAct: tool-use via code actions (code-capable, no external tools)
            module = dspy.CodeAct("task: str -> result: str")
        else:
            # Plain ChainOfThought for simple reasoning agents
            sig_cls = type(
                entry.name,
                (dspy.Signature,),
                {"__doc__": prompt, "task": dspy.InputField(), "result": dspy.OutputField()},
            )
            module = dspy.ChainOfThought(sig_cls)

        self._module_cache[entry.name] = module
        return module

    # -- quality evaluation --

    def evaluate(self, task: str, agent_role: str, prediction: str) -> float:
        result = self._evaluator(
            task=task,
            agent_role=agent_role,
            prediction=str(prediction)[:1000],
        )
        if hasattr(result, "quality_score"):
            return max(0.0, min(1.0, float(result.quality_score)))
        return 0.5

    # -- module lifecycle --

    def clear_cache(self):
        self._module_cache.clear()
