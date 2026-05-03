"""
DSPy optimizer showcase.

Compares:
- BootstrapFewShot (fast, small datasets)
- MIPROv2 (Bayesian prompt + few-shot search)
- GEPA (genetic prompt evolution)
- BetterTogether (chain optimizers)
"""

from pathlib import Path
from dotenv import load_dotenv
import dspy

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

lm = dspy.LM("deepseek/deepseek-v4-flash")
dspy.configure(lm=lm)

# ---------------------------------------------------------------------------
# Toy dataset: number classification
# ---------------------------------------------------------------------------

class NumClassify(dspy.Signature):
    """Classify a number."""
    number: str = dspy.InputField()
    parity: str = dspy.OutputField()
    prime: str = dspy.OutputField()

trainset = [
    dspy.Example(number="2", parity="even", prime="prime").with_inputs("number"),
    dspy.Example(number="3", parity="odd", prime="prime").with_inputs("number"),
    dspy.Example(number="4", parity="even", prime="composite").with_inputs("number"),
    dspy.Example(number="5", parity="odd", prime="prime").with_inputs("number"),
    dspy.Example(number="6", parity="even", prime="composite").with_inputs("number"),
    dspy.Example(number="7", parity="odd", prime="prime").with_inputs("number"),
    dspy.Example(number="9", parity="odd", prime="composite").with_inputs("number"),
    dspy.Example(number="10", parity="even", prime="composite").with_inputs("number"),
    dspy.Example(number="11", parity="odd", prime="prime").with_inputs("number"),
    dspy.Example(number="12", parity="even", prime="composite").with_inputs("number"),
]

devset = [
    dspy.Example(number="8", parity="even", prime="composite").with_inputs("number"),
    dspy.Example(number="13", parity="odd", prime="prime").with_inputs("number"),
    dspy.Example(number="14", parity="even", prime="composite").with_inputs("number"),
    dspy.Example(number="15", parity="odd", prime="composite").with_inputs("number"),
]


def exact_match(example, prediction, trace=None):
    return example.parity == prediction.parity and example.prime == prediction.prime


program = dspy.ChainOfThought(NumClassify)
evaluator = dspy.Evaluate(devset=devset, metric=exact_match, num_threads=4)

print("=== Baseline (unoptimized) ===")
baseline = evaluator(program)
print(f"Accuracy: {baseline:.2%}\n")

# ---------------------------------------------------------------------------
# 1. BootstrapFewShot — fastest, good for small data
# ---------------------------------------------------------------------------
print("=== BootstrapFewShot ===")
bs = dspy.BootstrapFewShot(metric=exact_match, max_bootstrapped_demos=4, max_labeled_demos=4)
bs_program = bs.compile(dspy.ChainOfThought(NumClassify), trainset=trainset)
bs_score = evaluator(bs_program)
print(f"Accuracy: {bs_score:.2%}\n")

# ---------------------------------------------------------------------------
# 2. GEPA — genetic prompt evolution
# ---------------------------------------------------------------------------
print("=== GEPA ===")
gepa = dspy.GEPA(metric=exact_match)
gepa_program = gepa.compile(dspy.ChainOfThought(NumClassify), trainset=trainset)
gepa_score = evaluator(gepa_program)
print(f"Accuracy: {gepa_score:.2%}\n")

# ---------------------------------------------------------------------------
# 3. MIPROv2 — Bayesian prompt + few-shot search (requires optuna)
# ---------------------------------------------------------------------------
print("=== MIPROv2 ===")
mipro = dspy.MIPROv2(metric=exact_match, auto="light", num_threads=4)
mipro_program = mipro.compile(dspy.ChainOfThought(NumClassify), trainset=trainset)
mipro_score = evaluator(mipro_program)
print(f"Accuracy: {mipro_score:.2%}\n")

# ---------------------------------------------------------------------------
# 4. BetterTogether — chain GEPA → BootstrapFinetune → GEPA
# ---------------------------------------------------------------------------
print("=== BetterTogether (GEPA → BootstrapFinetune → GEPA) ===")
bt = dspy.BetterTogether(
    metric=exact_match,
    p=dspy.GEPA(metric=exact_match, max_iters=3),
    w=dspy.BootstrapFinetune(metric=exact_match),
    strategy="p -> w -> p"
)
bt_program = bt.compile(dspy.ChainOfThought(NumClassify), trainset=trainset)
bt_score = evaluator(bt_program)
print(f"Accuracy: {bt_score:.2%}\n")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("=" * 50)
print("OPTIMIZER COMPARISON")
print("=" * 50)
print(f"Baseline:           {baseline:.2%}")
print(f"BootstrapFewShot:   {bs_score:.2%}")
print(f"GEPA:               {gepa_score:.2%}")
print(f"MIPROv2:            {mipro_score:.2%}")
print(f"BetterTogether:     {bt_score:.2%}")
