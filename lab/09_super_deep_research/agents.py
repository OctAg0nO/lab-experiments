"""
Specialized agent classes — each is a DSPy RLM with different
signatures, tools, and roles.

  Explorer     — discovers research directions (search, fetch)
  DeepReader   — deep content analysis (fetch, crawl, extract)
  Synthesizer  — cross-source synthesis (kg query)
  Critic       — identifies gaps, evaluates quality (pure reasoning)
"""

from __future__ import annotations


import dspy
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Structured output models
# ---------------------------------------------------------------------------

class FoundDirection(BaseModel):
    topic: str = Field(description="Research topic discovered")
    relevance: str = Field(description="Why this matters")
    seed_query: str = Field(description="Search query to explore further")


class ExplorationResult(BaseModel):
    directions: list[FoundDirection] = Field(description="Discovered research directions")


class ExtractedFinding(BaseModel):
    claim: str = Field(description="Main claim or finding")
    evidence: str = Field(description="Supporting evidence from source")
    source: str = Field(description="Source URL")
    confidence: str = Field(description="high/medium/low")


class DeepReadResult(BaseModel):
    findings: list[ExtractedFinding] = Field(description="Extracted findings")
    summary: str = Field(description="Content summary")


class SynthesisReport(BaseModel):
    synthesis: str = Field(description="Cross-source synthesis")
    key_insights: list[str] = Field(description="Key insights across sources")
    gaps: list[str] = Field(description="Identified knowledge gaps")


class Critique(BaseModel):
    strengths: list[str] = Field(description="What the research did well")
    weaknesses: list[str] = Field(description="What needs improvement")
    follow_ups: list[str] = Field(description="Recommended next directions")


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def create_explorer(tools: list, lm: dspy.LM) -> dspy.RLM:
    return dspy.RLM(
        "task: str -> result: ExplorationResult",
        tools=tools,
        max_iterations=8,
        max_llm_calls=12,
        verbose=False,
    )


def create_deep_reader(tools: list, lm: dspy.LM) -> dspy.RLM:
    return dspy.RLM(
        "topic: str, url: str -> result: DeepReadResult",
        tools=tools,
        max_iterations=10,
        max_llm_calls=16,
        verbose=False,
    )


def create_synthesizer(tools: list, lm: dspy.LM) -> dspy.RLM:
    return dspy.RLM(
        "task: str, findings: str -> result: SynthesisReport",
        tools=tools,
        max_iterations=8,
        max_llm_calls=12,
        verbose=False,
    )


def create_critic(lm: dspy.LM) -> dspy.RLM:
    return dspy.RLM(
        "research_summary: str -> result: Critique",
        tools=[],
        max_iterations=6,
        max_llm_calls=8,
        verbose=False,
    )
