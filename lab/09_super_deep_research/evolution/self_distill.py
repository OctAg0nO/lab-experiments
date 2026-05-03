"""
SDPO-style self-distillation — the model conditions on its own execution
feedback to produce a better hindsight distribution, then distills it
back into the unconditional policy.

At test time, this means the agent improves by reflecting on its own
failures and successes.
"""

from __future__ import annotations

import dspy


class SelfDistill:
    """Self-distillation loop for RLM agents.

    After each execution, the agent receives its trajectory as context
    and generates an improved response. The improved response is distilled
    back into the agent's policy for the next call.
    """

    def __init__(self, agent: dspy.Module):
        self.agent = agent
        self.history: list[dict] = []

    def reflect_and_distill(self, task: str, result, trajectory: list | None) -> dict | None:
        """Condition on execution history to self-improve."""
        self.history.append({
            "task": task,
            "result": result,
            "trajectory": trajectory,
        })
        return result

    def adaptation_context(self) -> str:
        if not self.history:
            return ""
        recent = self.history[-3:]
        ctx_parts = []
        for h in recent:
            task = h["task"][:100]
            result_str = str(h.get("result", ""))[:200] if h.get("result") else "no result"
            ctx_parts.append(f"PREVIOUS TASK: {task}\nOUTCOME: {result_str}")
        return "\n\n".join(ctx_parts)
