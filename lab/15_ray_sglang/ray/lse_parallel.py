"""Parallel LSE — distributed self-optimization via Ray tasks.

Capped at min(available_GPUs, 10) branches to prevent overfitting
and uncontrolled compute cost. Uses RayModuleExecutor when available
instead of creating its own Ray remote functions.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import dspy

logger = logging.getLogger(__name__)

MAX_LSE_BRANCHES = 10


def _evaluate_branch(
    agent_module: dspy.Module,
    test_cases: list[dspy.Example],
    metric: Callable,
) -> float:
    """Evaluate one LSE branch. Designed to run as a Ray remote task."""
    scores: list[float] = []
    for example in test_cases:
        try:
            prediction = agent_module(**example.inputs())
            score = metric(example, prediction)
            scores.append(float(score))
        except Exception as e:
            logger.debug("LSE branch evaluation failed: %s", e)
            scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0


def parallel_lse_evaluate(
    branches: list[dspy.Module],
    test_cases: list[dspy.Example],
    metric: Callable,
    max_branches: int = MAX_LSE_BRANCHES,
    executor: Any | None = None,
) -> list[float]:
    """Evaluate LSE branches, optionally in parallel via Ray executor.

    Reuses a RayModuleExecutor if provided, avoiding duplicate Ray init.
    Falls back to sequential evaluation on error or no executor.

    Args:
        branches: DSPy modules (candidate prompts/agents to evaluate)
        test_cases: DSPy examples to evaluate on
        metric: (example, prediction) -> float
        max_branches: Hard cap to prevent runaway cost
        executor: RayModuleExecutor instance (if None, runs sequential)

    Returns:
        Average quality scores per branch, same order as input
    """
    capped_branches = branches[:max_branches]
    if len(branches) > max_branches:
        logger.warning(
            "LSE branches capped to %d (requested %d).",
            max_branches, len(branches),
        )

    if not executor or len(capped_branches) <= 1:
        return [_evaluate_branch(b, test_cases, metric) for b in capped_branches]

    from .executor import RayModuleExecutor

    if not isinstance(executor, RayModuleExecutor):
        return [_evaluate_branch(b, test_cases, metric) for b in capped_branches]

    try:
        # Flatten: run all branches × test_cases as parallel Ray tasks.
        # Keeps maximum parallelism (all branches, all test cases at once).
        batch_kwargs = [
            {key: getattr(ex, key) for key in ex.inputs().keys()}
            for _ in capped_branches
            for ex in test_cases
        ]
        batch_modules = [b for b in capped_branches for _ in test_cases]

        predictions = executor.execute_batch(batch_modules, batch_kwargs)

        # Slice predictions back into per-branch scores using clean offsets
        scores = []
        step = len(test_cases)
        for i, branch in enumerate(capped_branches):
            start = i * step
            branch_preds = predictions[start:start + step]
            if len(branch_preds) != len(test_cases):
                raise ValueError(
                    f"Expected {len(test_cases)} predictions for branch {i}, got {len(branch_preds)}"
                )
            branch_scores = [
                metric(ex, pred) for ex, pred in zip(test_cases, branch_preds)
            ]
            scores.append(sum(branch_scores) / len(branch_scores) if branch_scores else 0.0)
        return scores
    except Exception as e:
        logger.error("LSE parallel evaluation failed: %s. Falling back to sequential.", e)
        return [_evaluate_branch(b, test_cases, metric) for b in capped_branches]
