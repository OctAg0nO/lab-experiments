"""GFL Optimizer — Generative Feedback Loops for meta-agent optimization.

Integrates all optimizers from lab 07:
- BootstrapFewShot: trace -> demo pipeline
- MIPROv2: instruction + demo joint Bayesian optimization
- GEPA: reflective prompt evolution via Pareto frontier
- Sequential chaining: GEPA -> BootstrapFewShot
- Evaluation harness
"""

from __future__ import annotations

from typing import Literal

import dspy


class GFLPipeline:
    """Generative Feedback Loop pipeline for optimizing generated agents.

    Usage:
        pipeline = GFLPipeline(metric=my_metric, trainset=..., devset=...)
        results = pipeline.run_full(program)
    """

    def __init__(
        self,
        trainset: list[dspy.Example],
        devset: list[dspy.Example] | None = None,
        metric: callable | None = None,
        reflection_lm: dspy.LM | None = None,
    ):
        self.trainset = trainset
        self.devset = devset or trainset[:5]
        self._metric = metric or (lambda ex, pred, trace=None: 1.0)
        self._reflection_lm = reflection_lm
        self._evaluator = dspy.Evaluate(
            devset=self.devset,
            metric=self._metric,
            num_threads=4,
            display_progress=False,
        )

    def score(self, program) -> float:
        return self._evaluator(program).score / 100.0

    # -- individual optimizers --

    def bootstrap_fewshot(
        self, program: dspy.Module,
        max_bootstrapped: int = 6,
        max_labeled: int = 4,
        teacher: dspy.Module | None = None,
    ) -> dspy.Module:
        bs = dspy.BootstrapFewShot(
            metric=self._metric,
            max_bootstrapped_demos=max_bootstrapped,
            max_labeled_demos=max_labeled,
        )
        return bs.compile(program, teacher=teacher or program, trainset=self.trainset)

    def mipro(
        self, program: dspy.Module, auto: Literal["light", "medium", "heavy"] | None = "light",
    ) -> dspy.Module:
        mipro = dspy.MIPROv2(
            metric=self._metric,
            auto=auto,
            num_threads=4,
        )
        return mipro.compile(program, trainset=self.trainset)

    def gepa(
        self, program: dspy.Module,
        max_evals: int = 2,
        trainset: list[dspy.Example] | None = None,
    ) -> dspy.Module:
        gepa = dspy.GEPA(
            metric=self._metric,
            max_full_evals=max_evals,
            reflection_lm=self._reflection_lm,
            num_threads=4,
        )
        return gepa.compile(program, trainset=trainset or self.trainset[:10])

    def sequential(self, program: dspy.Module) -> dspy.Module:
        gepa = dspy.GEPA(
            metric=self._metric,
            max_full_evals=1,
            reflection_lm=self._reflection_lm,
            num_threads=4,
        )
        gepa_opt = gepa.compile(program, trainset=self.trainset[:10])
        bs = dspy.BootstrapFewShot(
            metric=self._metric,
            max_bootstrapped_demos=4,
            max_labeled_demos=2,
        )
        return bs.compile(gepa_opt, trainset=self.trainset)

    def distill(
        self, program: dspy.Module,
        student_lm: dspy.LM,
        teacher: dspy.Module | None = None,
    ) -> dspy.Module:
        teacher = teacher or program
        sig_cls = type("StudentSig", (dspy.Signature,), {
            "__doc__": getattr(getattr(teacher, "signature", None), "__doc__", "task -> result"),
            "task": dspy.InputField(),
            "result": dspy.OutputField(),
        })
        student = dspy.ChainOfThought(sig_cls)
        if hasattr(student, "set_lm"):
            student.set_lm(student_lm)
        bs = dspy.BootstrapFewShot(
            metric=self._metric,
            max_bootstrapped_demos=6,
            max_labeled_demos=4,
        )
        return bs.compile(student, teacher=teacher, trainset=self.trainset)

    # -- full pipeline --

    def run_full(self, program: dspy.Module) -> dict[str, tuple[dspy.Module, float]]:
        results = {}

        baseline_score = self.score(program)
        results["baseline"] = (program, baseline_score)

        bs_prog = self.bootstrap_fewshot(program)
        results["bootstrap_fewshot"] = (bs_prog, self.score(bs_prog))

        mipro_prog = self.mipro(program)
        results["mipro"] = (mipro_prog, self.score(mipro_prog))

        gepa_prog = self.gepa(program)
        results["gepa"] = (gepa_prog, self.score(gepa_prog))

        seq_prog = self.sequential(program)
        results["sequential"] = (seq_prog, self.score(seq_prog))

        return results
