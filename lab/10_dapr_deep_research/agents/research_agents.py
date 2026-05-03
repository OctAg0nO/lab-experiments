"""
DurableAgent subclasses — each wraps a DSPy RLM for workflow-backed execution.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
import dspy
from dapr_agents import DurableAgent
from dapr_agents.llm import DaprChatClient
from dapr_agents.agents.configs import (
    AgentStateConfig, AgentExecutionConfig,
    ToolExecutionMode,
)
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.workflow import workflow_entry

from ..mcp.bridge import MCPBridge


# ---------------------------------------------------------------------------
# Pydantic output models (same as lab/09)
# ---------------------------------------------------------------------------

class FoundDirection(BaseModel):
    topic: str = Field(description="Research topic discovered")
    relevance: str = Field(description="Why this matters")
    seed_query: str = Field(description="Search query to explore further")

class ExplorationResult(BaseModel):
    directions: list[FoundDirection] = Field(description="Discovered research directions")

class ExtractedFinding(BaseModel):
    claim: str = Field(description="Main claim or finding")
    evidence: str = Field(description="Supporting evidence")
    source: str = Field(description="Source URL")
    confidence: str = Field(description="high/medium/low")

class DeepReadResult(BaseModel):
    findings: list[ExtractedFinding] = Field(description="Extracted findings")
    summary: str = Field(description="Content summary")

class SynthesisReport(BaseModel):
    synthesis: str = Field(description="Cross-source synthesis")
    key_insights: list[str] = Field(description="Key insights")
    gaps: list[str] = Field(description="Knowledge gaps")

class Critique(BaseModel):
    strengths: list[str] = Field(description="Strengths")
    weaknesses: list[str] = Field(description="Weaknesses")
    follow_ups: list[str] = Field(description="Next directions")


def _rlm_factory(signature: str, tools: list, max_iter: int, max_calls: int) -> dspy.RLM:
    return dspy.RLM(signature, tools=tools, max_iterations=max_iter, max_llm_calls=max_calls, verbose=False)


class ExplorerAgent(DurableAgent):
    """Discovers research directions using DSPy RLM + search tools."""

    def __init__(self, bridge: MCPBridge, **kwargs):
        dspy_tools = bridge.get_dspy_tools()
        search_tools = [t for t in dspy_tools if t.__name__ in ("search", "chat", "model_list")] or dspy_tools
        self._rlm = _rlm_factory("task: str -> result: ExplorationResult", search_tools, 8, 12)
        super().__init__(
            name="ExplorerAgent",
            role="Research Explorer",
            goal="Discover novel research directions and topics",
            instructions=["Identify unexplored angles", "Return diverse directions", "Be specific"],
            llm=DaprChatClient(component_name="llm-provider"),
            tools=bridge.get_agent_tools(),
            state=AgentStateConfig(store=StateStoreService(store_name="research-state")),
            execution=AgentExecutionConfig(max_iterations=10, tool_execution_mode=ToolExecutionMode.PARALLEL),
            **kwargs,
        )

    @workflow_entry
    def explore(self, ctx, input: dict) -> dict:
        result = self._rlm(task=input["topic"])
        directions = result.result.directions if hasattr(result, "result") and result.result else []
        ctx.set_state("explorer_result", {
            "topic": input["topic"],
            "directions": [{"topic": d.topic, "relevance": d.relevance, "seed_query": d.seed_query} for d in directions],
        })
        return ctx.get_state("explorer_result")


class DeepReaderAgent(DurableAgent):
    """Deep content analysis using DSPy RLM + fetch tools."""

    def __init__(self, bridge: MCPBridge, **kwargs):
        dspy_tools = bridge.get_dspy_tools()
        fetch_tools = [t for t in dspy_tools if t.__name__ in ("fetch", "md", "crawl")] or dspy_tools
        self._rlm = _rlm_factory("topic: str, url: str -> result: DeepReadResult", fetch_tools, 10, 16)
        super().__init__(
            name="DeepReaderAgent",
            role="Content Analyst",
            goal="Extract structured findings from web content",
            instructions=["Read thoroughly", "Extract specific claims with evidence", "Rate confidence"],
            llm=DaprChatClient(component_name="llm-provider"),
            tools=bridge.get_agent_tools(),
            state=AgentStateConfig(store=StateStoreService(store_name="research-state")),
            execution=AgentExecutionConfig(max_iterations=10, tool_execution_mode=ToolExecutionMode.PARALLEL),
            **kwargs,
        )

    @workflow_entry
    def deep_read(self, ctx, input: dict) -> dict:
        url = input.get("url") or input["topic"]
        result = self._rlm(topic=input["topic"], url=url)
        findings = result.result.findings if hasattr(result, "result") and result.result and hasattr(result.result, "findings") else []
        ctx.set_state("deepread_result", {
            "topic": input["topic"],
            "findings": [{"claim": f.claim, "evidence": f.evidence, "source": f.source, "confidence": f.confidence} for f in findings],
            "summary": result.result.summary if hasattr(result, "result") and hasattr(result.result, "summary") else "",
        })
        return ctx.get_state("deepread_result")


class SynthesizerAgent(DurableAgent):
    """Cross-source synthesis using DSPy RLM."""

    def __init__(self, bridge: MCPBridge, **kwargs):
        dspy_tools = bridge.get_dspy_tools()
        self._rlm = _rlm_factory("task: str, findings: str -> result: SynthesisReport", dspy_tools, 8, 12)
        super().__init__(
            name="SynthesizerAgent",
            role="Research Synthesizer",
            goal="Synthesize findings across sources",
            instructions=["Identify patterns", "Highlight contradictions", "Suggest gaps"],
            llm=DaprChatClient(component_name="llm-provider"),
            tools=bridge.get_agent_tools(),
            state=AgentStateConfig(store=StateStoreService(store_name="research-state")),
            execution=AgentExecutionConfig(max_iterations=8, tool_execution_mode=ToolExecutionMode.PARALLEL),
            **kwargs,
        )

    @workflow_entry
    def synthesize(self, ctx, input: dict) -> dict:
        import json
        result = self._rlm(task=input["topic"], findings=json.dumps(input.get("findings", [])))
        r = result.result if hasattr(result, "result") and result.result else None
        ctx.set_state("synthesis_result", {
            "topic": input["topic"],
            "synthesis": r.synthesis if r and hasattr(r, "synthesis") else "",
            "insights": r.key_insights if r and hasattr(r, "key_insights") else [],
            "gaps": r.gaps if r and hasattr(r, "gaps") else [],
        })
        return ctx.get_state("synthesis_result")


class CriticAgent(DurableAgent):
    """Evaluates research quality and identifies gaps."""

    def __init__(self, **kwargs):
        self._rlm = _rlm_factory("research_summary: str -> result: Critique", [], 6, 8)
        super().__init__(
            name="CriticAgent",
            role="Research Critic",
            goal="Evaluate research quality and find gaps",
            instructions=["Be critical but constructive", "Identify missing angles", "Prioritize follow-ups"],
            llm=DaprChatClient(component_name="llm-provider"),
            state=AgentStateConfig(store=StateStoreService(store_name="research-state")),
            execution=AgentExecutionConfig(max_iterations=6),
            **kwargs,
        )

    @workflow_entry
    def critique(self, ctx, input: dict) -> dict:
        result = self._rlm(research_summary=input.get("summary", ""))
        r = result.result if hasattr(result, "result") and result.result else None
        ctx.set_state("critique_result", {
            "strengths": r.strengths if r and hasattr(r, "strengths") else [],
            "weaknesses": r.weaknesses if r and hasattr(r, "weaknesses") else [],
            "follow_ups": r.follow_ups if r and hasattr(r, "follow_ups") else [],
        })
        return ctx.get_state("critique_result")
