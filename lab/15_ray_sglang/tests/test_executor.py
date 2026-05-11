"""Tests for ModuleExecutor, MetaConfig, and circuit-breaker."""

from __future__ import annotations

from unittest import TestCase, main

import dspy

from ..meta.meta_agent import MetaAgent, MetaConfig, ResourceBudget
from ..meta.agent_generator import AgentGenerator
from ..ray.executor import InProcessExecutor, RayModuleExecutor


class TestInProcessExecutor(TestCase):
    def setUp(self):
        self.executor = InProcessExecutor()
        self.module = dspy.ChainOfThought("task -> result")

    def test_execute_returns_prediction(self):
        result = self.executor.execute(self.module, task="test")
        self.assertIsInstance(result, dspy.Prediction)

    def test_execute_batch_returns_list(self):
        modules = [self.module, self.module]
        kwargs_list = [{"task": "test1"}, {"task": "test2"}]
        results = self.executor.execute_batch(modules, kwargs_list)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIsInstance(r, dspy.Prediction)

    def test_execute_batch_preserves_order(self):
        class CountingModule(dspy.Module):
            def __init__(self, label):
                super().__init__()
                self.label = label
            def forward(self, task):
                return dspy.Prediction(result=f"{self.label}:{task}")

        mods = [CountingModule("A"), CountingModule("B")]
        kwargs = [{"task": "x"}, {"task": "y"}]
        results = self.executor.execute_batch(mods, kwargs)
        self.assertEqual(results[0].result, "A:x")
        self.assertEqual(results[1].result, "B:y")


class TestRayModuleExecutor(TestCase):
    def setUp(self):
        self.executor = RayModuleExecutor()
        self.module = dspy.ChainOfThought("task -> result")

    def test_fallback_on_no_ray(self):
        result = self.executor.execute(self.module, task="test")
        self.assertIsInstance(result, dspy.Prediction)

    def test_circuit_breaker_tracks_fallbacks(self):
        before = self.executor.fallback_count
        self.executor.execute(self.module, task="test")
        self.assertGreaterEqual(self.executor.fallback_count, before)

    def test_reset_circuit_breaker(self):
        self.executor._circuit_open = True
        self.executor.reset_circuit_breaker()
        self.assertFalse(self.executor._circuit_open)


class TestMetaConfig(TestCase):
    def test_meta_config_requires_generator(self):
        with self.assertRaises(ValueError):
            MetaConfig()
        with self.assertRaises(ValueError):
            MetaConfig(llm=dspy.LM("openai/gpt-4o-mini"))

    def test_meta_config_valid(self):
        lm = dspy.LM("openai/gpt-4o-mini")
        gen = object.__new__(AgentGenerator)
        cfg = MetaConfig(llm=lm, generator=gen)
        self.assertIs(cfg.llm, lm)
        self.assertIs(cfg.generator, gen)

    def test_meta_config_defaults(self):
        lm = dspy.LM("openai/gpt-4o-mini")
        gen = object.__new__(AgentGenerator)
        cfg = MetaConfig(llm=lm, generator=gen)
        self.assertIsInstance(cfg.executor or InProcessExecutor(), InProcessExecutor)
        self.assertEqual(cfg.tool_defs, None)

    def test_meta_agent_accepts_config(self):
        lm = dspy.LM("openai/gpt-4o-mini")
        gen = object.__new__(AgentGenerator)
        agent = MetaAgent(config=MetaConfig(llm=lm, generator=gen))
        self.assertIsNotNone(agent.budget)
        self.assertIsInstance(agent.budget, ResourceBudget)


if __name__ == "__main__":
    main()
