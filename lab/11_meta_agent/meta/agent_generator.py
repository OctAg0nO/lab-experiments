"""AgentGenerator — dynamically creates DSPy agents for a given task."""

from __future__ import annotations

import dspy

from .agent_stack import AgentEntry


class AnalyzeTask(dspy.Signature):
    """Analyze a user task and determine what sub-agents are needed."""
    task: str = dspy.InputField()
    num_agents: int = dspy.OutputField(desc="How many distinct sub-agents needed")
    agent_definitions: str = dspy.OutputField(
        desc="JSON list: [{\"name\", \"role\", \"goal\", \"signature\", \"tools\"}]"
    )


class GenerateSignature(dspy.Signature):
    """Generate a DSPy signature string for a given agent role and goal."""
    role: str = dspy.InputField()
    goal: str = dspy.InputField(desc="What this agent should accomplish")
    available_tools: str = dspy.InputField(desc="Available MCP tools JSON")
    dspy_signature: str = dspy.OutputField(desc="DSPy signature like 'input_field -> output_field'")
    input_field: str = dspy.OutputField(desc="Input field name and type")
    output_field: str = dspy.OutputField(desc="Output field name and type")


class AgentGenerator:
    """Uses DSPy CoT to analyze tasks and generate agent definitions on the fly."""

    def __init__(self, llm: dspy.LM, tool_defs: list[dict] | None = None):
        self._analyzer = dspy.ChainOfThought(AnalyzeTask)
        self._sig_gen = dspy.ChainOfThought(GenerateSignature)
        self._llm = llm
        self._tool_defs = tool_defs or []

    def analyze(self, task: str) -> list[dict]:
        """Determine what agents are needed for a task."""
        result = self._analyzer(task=task)
        if not hasattr(result, "agent_definitions") or not result.agent_definitions:
            return self._default_agents(task)
        import json
        try:
            agents = json.loads(result.agent_definitions)
            return agents if isinstance(agents, list) else self._default_agents(task)
        except (json.JSONDecodeError, TypeError):
            return self._default_agents(task)

    def generate(self, definition: dict) -> AgentEntry:
        """Generate a full AgentEntry from a definition dict."""
        tools_json = str([t.get("name", t.get("server", "?")) for t in self._tool_defs])
        sig_result = self._sig_gen(
            role=definition.get("role", "assistant"),
            goal=definition.get("goal", "help with the task"),
            available_tools=tools_json,
        )
        signature = (
            getattr(sig_result, "dspy_signature", "task -> result")
            or "task -> result"
        )
        return AgentEntry(
            name=definition.get("name", f"agent_{id(definition)}"),
            role=definition.get("role", "assistant"),
            goal=definition.get("goal", "help with the task"),
            signature=signature,
            tools=definition.get("tools", []),
            prompt_template=definition.get("prompt", ""),
        )

    def _default_agents(self, task: str) -> list[dict]:
        """Fallback agents when LLM analysis fails."""
        return [
            {
                "name": "searcher",
                "role": "Web Researcher",
                "goal": f"Search and gather information about: {task}",
                "signature": "query: str -> findings: str",
                "tools": ["search", "fetch", "md"],
            },
            {
                "name": "analyzer",
                "role": "Content Analyst",
                "goal": f"Analyze and extract insights from research about: {task}",
                "signature": "content: str -> insights: str, gaps: str",
                "tools": ["chat"],
            },
            {
                "name": "synthesizer",
                "role": "Research Synthesizer",
                "goal": f"Synthesize findings into coherent report about: {task}",
                "signature": "findings: str -> report: str, key_points: list[str]",
                "tools": [],
            },
        ]

    def generate_module(self, entry: AgentEntry) -> dspy.Module:
        """Create an executable DSPy module from an AgentEntry."""
        sig_field = entry.signature.replace("->", ",").replace(":", ",").split(",")
        if len(sig_field) >= 4:
            input_desc = sig_field[0].strip()
            output_desc = sig_field[2].strip()
            sig_str = f"{input_desc} -> {output_desc}"
        else:
            sig_str = entry.signature

        prompt = entry.prompt_template or f"You are {entry.role}. {entry.goal}"
        sig = dspy.Signature(sig_str)
        sig.__doc__ = prompt
        return dspy.ChainOfThought(sig)
