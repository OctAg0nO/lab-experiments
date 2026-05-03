"""
DurableAgent subclasses — each uses DSPy modules (RLM, ChainOfThought,
Parallel, BestOfN, Refine, ProgramOfThought) for workflow-backed execution.
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
# Pydantic output models
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

# ---------------------------------------------------------------------------
# DSPy signatures
# ---------------------------------------------------------------------------

class GenerateHypotheses(dspy.Signature):
    """Generate diverse research hypotheses from a topic."""
    topic: str = dspy.InputField()
    hypotheses: list[str] = dspy.OutputField(desc="Diverse hypotheses to explore")

class CrossValidateFindings(dspy.Signature):
    """Cross-validate findings from multiple sources for consistency."""
    findings_summary: str = dspy.InputField()
    validated_claims: list[str] = dspy.OutputField(desc="Claims supported by multiple sources")
    contradictions: list[str] = dspy.OutputField(desc="Conflicting information found")

class SynthesizeAcrossSources(dspy.Signature):
    """Synthesize findings from multiple sources into a coherent report."""
    task: str = dspy.InputField()
    synthesis: str = dspy.OutputField(desc="Cross-source synthesis")
    key_insights: list[str] = dspy.OutputField(desc="Key insights")
    gaps: list[str] = dspy.OutputField(desc="Knowledge gaps")

class SelectAgent(dspy.Signature):
    """Select the best agent for a research task based on frontier state."""
    exploration_depth: int = dspy.InputField(desc="How many times explored (0 = new)")
    confidence: float = dspy.InputField(desc="Current confidence 0-1")
    topic: str = dspy.InputField()
    selected_agent: str = dspy.OutputField(desc="explorer, deepreader, or synthesizer")

class ComputeConfidenceDelta(dspy.Signature):
    """Determine confidence increase from research findings."""
    topic: str = dspy.InputField()
    agent_type: str = dspy.InputField(desc="explorer, deepreader, or synthesizer")
    num_findings: int = dspy.InputField(desc="Number of findings collected")
    findings_summary: str = dspy.InputField(desc="Key findings summary")
    exploration_depth: int = dspy.InputField(desc="Times explored")
    confidence_delta: float = dspy.OutputField(desc="Confidence increase 0.0–0.5")
    reasoning: str = dspy.OutputField(desc="Why this delta")

class AssessSaturation(dspy.Signature):
    """Assess whether continued exploration of a direction is still valuable."""
    topic: str = dspy.InputField()
    confidence: float = dspy.InputField()
    exploration_depth: int = dspy.InputField()
    source_count: int = dspy.InputField()
    is_saturated: bool = dspy.OutputField(desc="Whether saturated")
    reasoning: str = dspy.OutputField(desc="Why")

class CritiqueReasoning(dspy.Signature):
    """Critique research findings and identify gaps."""
    research_summary: str = dspy.InputField()
    critique: str = dspy.OutputField(desc="Critical analysis")

# ---------------------------------------------------------------------------
# RLM factory
# ---------------------------------------------------------------------------

def _rlm_factory(signature: str, max_iter: int = 20, max_calls: int = 50, tools: list | None = None):
    return dspy.RLM(signature, tools=tools, max_iterations=max_iter, max_llm_calls=max_calls, verbose=False)

# ---------------------------------------------------------------------------
# ExplorerAgent — multi-stage DSPy pipeline
# Uses: RLM (discovery) + ChainOfThought (hypothesis gen) + Parallel (batch)
# ---------------------------------------------------------------------------

class ExplorerAgent(DurableAgent):
    def __init__(self, bridge: MCPBridge, llm: dspy.LM | None = None, state: AgentStateConfig | None = None, **kwargs):
        dspy_tools = bridge.get_dspy_tools()
        search_tools = [t for t in dspy_tools if t.__name__ in ("search", "chat", "model_list")] or dspy_tools
        self._rlm = _rlm_factory("task: str -> result: ExplorationResult", max_iter=8, max_calls=12, tools=search_tools)
        self._hypothesis_gen = dspy.ChainOfThought(GenerateHypotheses)
        self._hypothesis_best = dspy.BestOfN(dspy.ChainOfThought(GenerateHypotheses), N=3, reward_fn=lambda _, pred: len(pred.hypotheses) if hasattr(pred, "hypotheses") else 0, threshold=0.5)
        super().__init__(
            name="ExplorerAgent", role="Research Explorer",
            goal="Discover novel research directions and topics",
            instructions=["Identify unexplored angles", "Return diverse directions", "Be specific"],
            llm=llm or DaprChatClient(component_name="llm-provider"),
            tools=bridge.get_agent_tools(),
            state=state or AgentStateConfig(store=StateStoreService(store_name="research-state")),
            execution=AgentExecutionConfig(max_iterations=10, tool_execution_mode=ToolExecutionMode.PARALLEL),
            **kwargs,
        )

    def compile(self, trainset: list[dspy.Example], student_lm: dspy.LM | None = None):
        teacher = self._hypothesis_gen
        student = dspy.ChainOfThought(GenerateHypotheses) if student_lm else teacher
        if student_lm:
            student.set_lm(student_lm)
        bs = dspy.BootstrapFewShot(metric=lambda _ex, pred, _trace: len(pred.hypotheses) > 0, max_bootstrapped_demos=4, max_labeled_demos=2)
        compiled = bs.compile(student, teacher=teacher, trainset=trainset)
        if student_lm:
            compiled.set_lm(student_lm)
        self._hypothesis_gen = compiled

    @workflow_entry
    def explore(self, ctx, input: dict) -> dict:
        rlm_result = self._rlm(task=input["topic"])
        directions = rlm_result.result.directions if hasattr(rlm_result, "result") and rlm_result.result else []
        hyp = self._hypothesis_gen(topic=input["topic"])
        best = self._hypothesis_best(topic=input["topic"])
        all_topics = [d.topic for d in directions]
        if hyp.hypotheses:
            all_topics.extend(hyp.hypotheses[:3])
        if hasattr(best, "hypotheses") and best.hypotheses:
            all_topics.extend(best.hypotheses[:2])
        ctx.set_state("explorer_result", {"topic": input["topic"], "directions": [{"topic": t} for t in set(all_topics)]})
        return ctx.get_state("explorer_result")


# ---------------------------------------------------------------------------
# DeepReaderAgent — DSPy CoT + RLM for structured content extraction
# ---------------------------------------------------------------------------

class DeepReaderAgent(DurableAgent):
    def __init__(self, bridge: MCPBridge, llm: dspy.LM | None = None, **kwargs):
        dspy_tools = bridge.get_dspy_tools()
        fetch_tools = [t for t in dspy_tools if t.__name__ in ("fetch", "md", "crawl")] or dspy_tools
        self._rlm = _rlm_factory("topic: str, url: str -> result: DeepReadResult", max_iter=10, max_calls=16, tools=fetch_tools)
        self._cross_validator = dspy.ChainOfThought(CrossValidateFindings)
        super().__init__(
            name="DeepReaderAgent", role="Content Analyst",
            goal="Extract structured findings from web content",
            instructions=["Read thoroughly", "Extract specific claims with evidence", "Rate confidence"],
            llm=llm or DaprChatClient(component_name="llm-provider"),
            tools=bridge.get_agent_tools(),
            state=AgentStateConfig(store=StateStoreService(store_name="research-state")),
            execution=AgentExecutionConfig(max_iterations=10, tool_execution_mode=ToolExecutionMode.PARALLEL),
            **kwargs,
        )

    def compile(self, trainset: list[dspy.Example], student_lm: dspy.LM | None = None):
        teacher = self._cross_validator
        student = dspy.ChainOfThought(CrossValidateFindings) if student_lm else teacher
        if student_lm:
            student.set_lm(student_lm)
        bs = dspy.BootstrapFewShot(metric=lambda _ex, pred, _trace: hasattr(pred, "validated_claims") and len(pred.validated_claims) > 0, max_bootstrapped_demos=4, max_labeled_demos=2)
        compiled = bs.compile(student, teacher=teacher, trainset=trainset)
        if student_lm:
            compiled.set_lm(student_lm)
        self._cross_validator = compiled

    @workflow_entry
    def deep_read(self, ctx, input: dict) -> dict:
        url = input.get("url") or input["topic"]
        result = self._rlm(topic=input["topic"], url=url)
        findings = result.result.findings if hasattr(result, "result") and result.result and hasattr(result.result, "findings") else []
        findings_text = "; ".join(f"{f.claim} ({f.source})" for f in findings[:5])
        validation = self._cross_validator(findings_summary=findings_text) if findings_text else None
        ctx.set_state("deepread_result", {
            "topic": input["topic"],
            "findings": [{"claim": f.claim, "evidence": f.evidence, "source": f.source, "confidence": f.confidence} for f in findings],
            "summary": result.result.summary if hasattr(result, "result") and hasattr(result.result, "summary") else "",
            "validated_claims": validation.validated_claims if validation and hasattr(validation, "validated_claims") else [],
            "contradictions": validation.contradictions if validation and hasattr(validation, "contradictions") else [],
        })
        return ctx.get_state("deepread_result")


# ---------------------------------------------------------------------------
# SynthesizerAgent — DSPy RLM + ChainOfThought with unified signature
# ---------------------------------------------------------------------------

class SynthesizerAgent(DurableAgent):
    def __init__(self, bridge: MCPBridge, llm: dspy.LM | None = None, **kwargs):
        dspy_tools = bridge.get_dspy_tools()
        self._rlm = _rlm_factory("task: str -> result: SynthesisReport", max_iter=8, max_calls=12, tools=dspy_tools)
        self._synthesizer = dspy.ChainOfThought(SynthesizeAcrossSources)
        super().__init__(
            name="SynthesizerAgent", role="Research Synthesizer",
            goal="Synthesize findings across sources",
            instructions=["Identify patterns", "Highlight contradictions", "Suggest gaps"],
            llm=llm or DaprChatClient(component_name="llm-provider"),
            tools=bridge.get_agent_tools(),
            state=AgentStateConfig(store=StateStoreService(store_name="research-state")),
            execution=AgentExecutionConfig(max_iterations=8, tool_execution_mode=ToolExecutionMode.PARALLEL),
            **kwargs,
        )

    def compile(self, trainset: list[dspy.Example], student_lm: dspy.LM | None = None):
        teacher = self._synthesizer
        student = dspy.ChainOfThought(SynthesizeAcrossSources) if student_lm else teacher
        if student_lm:
            student.set_lm(student_lm)
        bs = dspy.BootstrapFewShot(metric=lambda _ex, pred, _trace: hasattr(pred, "synthesis") and len(pred.synthesis) > 50, max_bootstrapped_demos=4, max_labeled_demos=2)
        compiled = bs.compile(student, teacher=teacher, trainset=trainset)
        if student_lm:
            compiled.set_lm(student_lm)
        self._synthesizer = compiled

    @workflow_entry
    def synthesize(self, ctx, input: dict) -> dict:
        result = self._rlm(task=f"Synthesize: {input['topic']}")
        r = result.result if hasattr(result, "result") and result.result else None
        cot = self._synthesizer(task=input["topic"])
        ctx.set_state("synthesis_result", {
            "topic": input["topic"],
            "synthesis": r.synthesis if r and hasattr(r, "synthesis") else cot.synthesis if cot and hasattr(cot, "synthesis") else "",
            "insights": r.key_insights if r and hasattr(r, "key_insights") else (cot.key_insights if cot and hasattr(cot, "key_insights") else []),
            "gaps": r.gaps if r and hasattr(r, "gaps") else (cot.gaps if cot and hasattr(cot, "gaps") else []),
        })
        return ctx.get_state("synthesis_result")


# ---------------------------------------------------------------------------
# CriticAgent — DSPy Refine with feedback loop
# Uses: RLM (initial critique) + Refine (iterative improvement)
# ---------------------------------------------------------------------------

class CriticAgent(DurableAgent):
    def __init__(self, llm: dspy.LM | None = None, **kwargs):
        self._rlm = _rlm_factory("research_summary: str -> result: Critique", max_iter=6, max_calls=8)
        self._refine = dspy.Refine(dspy.ChainOfThought("research_summary: str, critique: str -> improved_critique: str"), N=3, reward_fn=lambda _, pred: 1.0 if len(pred.improved_critique) > 50 else 0.0, threshold=0.5)
        self._comparison = dspy.MultiChainComparison(CritiqueReasoning, n=3)
        self._rlm_second = _rlm_factory("research_summary: str, refinement_guidance: str -> result: Critique", max_iter=4, max_calls=6)
        super().__init__(
            name="CriticAgent", role="Research Critic",
            goal="Evaluate research quality and find gaps",
            instructions=["Be critical but constructive", "Identify missing angles", "Prioritize follow-ups"],
            llm=llm or DaprChatClient(component_name="llm-provider"),
            state=AgentStateConfig(store=StateStoreService(store_name="research-state")),
            execution=AgentExecutionConfig(max_iterations=6),
            **kwargs,
        )

    def compile(self, trainset: list[dspy.Example], student_lm: dspy.LM | None = None):
        teacher = self._refine
        if student_lm:
            inner = dspy.ChainOfThought("research_summary: str, critique: str -> improved_critique: str")
            inner.set_lm(student_lm)
            student = dspy.Refine(inner, N=3, reward_fn=lambda _, pred: 1.0 if len(pred.improved_critique) > 50 else 0.0, threshold=0.5)
        else:
            student = teacher
        bs = dspy.BootstrapFewShot(metric=lambda _ex, pred, _trace: hasattr(pred, "improved_critique") and len(pred.improved_critique) > 100, max_bootstrapped_demos=4, max_labeled_demos=2)
        compiled = bs.compile(student, teacher=teacher, trainset=trainset)
        if student_lm:
            compiled.set_lm(student_lm)
        self._refine = compiled

    @workflow_entry
    def critique(self, ctx, input: dict) -> dict:
        summary = input.get("summary", "")
        first_pass = self._rlm(research_summary=summary)
        r = first_pass.result if hasattr(first_pass, "result") and first_pass.result else None
        comparison = self._comparison(research_summary=summary) if r else None
        comparison_critique = comparison.critique if comparison and hasattr(comparison, "critique") else ""
        refine_input = comparison_critique or str(r.follow_ups if r else [])
        refined = self._refine(research_summary=summary, critique=refine_input) if (r or comparison) and refine_input else None
        guidance = refined.improved_critique if refined and hasattr(refined, "improved_critique") else comparison_critique
        second_pass = self._rlm_second(research_summary=summary, refinement_guidance=guidance) if guidance else first_pass
        r2 = second_pass.result if hasattr(second_pass, "result") and second_pass.result else r
        ctx.set_state("critique_result", {
            "strengths": r2.strengths if r2 and hasattr(r2, "strengths") else [],
            "weaknesses": r2.weaknesses if r2 and hasattr(r2, "weaknesses") else [],
            "follow_ups": r2.follow_ups if r2 and hasattr(r2, "follow_ups") else [],
            "refined": guidance,
        })
        return ctx.get_state("critique_result")
