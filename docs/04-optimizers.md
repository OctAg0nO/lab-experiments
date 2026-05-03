# 04-optimizers — DSPy Optimizer Comparison

> **File:** `lab/04-optimizers/main.py`
> **Concepts:** `dspy.BootstrapFewShot`, `dspy.GEPA`, `dspy.MIPROv2`, `dspy.BetterTogether`, optimizer strategies and trade-offs.

## Purpose

Compare four DSPy optimizers on a small toy classification task. Each optimizer takes the same program and training data but uses a different strategy to improve the prompt or the few-shot demonstrations.

## Setup

```python
from pathlib import Path
import dspy

lm = dspy.LM("deepseek/deepseek-v4-flash")
dspy.configure(lm=lm)
```

---

## Task: Number Classification

### Signature

```python
class NumClassify(dspy.Signature):
    """Classify a number."""
    number: str = dspy.InputField()
    parity: str = dspy.OutputField()
    prime: str = dspy.OutputField()
```

Two output fields: `parity` (even/odd) and `prime` (prime/composite). The program predicts both from a single input.

### Dataset

10 training examples and 4 evaluation examples:

```python
trainset = [
    dspy.Example(number="2", parity="even", prime="prime").with_inputs("number"),
    dspy.Example(number="3", parity="odd", prime="prime").with_inputs("number"),
    # ... 8 more
]
devset = [
    dspy.Example(number="8", parity="even", prime="composite").with_inputs("number"),
    dspy.Example(number="13", parity="odd", prime="prime").with_inputs("number"),
    dspy.Example(number="14", parity="odd", prime="composite").with_inputs("number"),
    dspy.Example(number="15", parity="odd", prime="composite").with_inputs("number"),
]
```

### Metric

```python
def exact_match(example, prediction, trace=None):
    return example.parity == prediction.parity and example.prime == prediction.prime
```

Both fields must match exactly for a correct prediction.

### Base Program and Evaluator

```python
program = dspy.ChainOfThought(NumClassify)
evaluator = dspy.Evaluate(devset=devset, metric=exact_match, num_threads=4)
```

---

## Optimizer Catalog

### 1. dspy.BootstrapFewShot

**Fastest optimizer.** Good for small datasets and rapid iteration.

```python
from dspy import BootstrapFewShot

optimizer = BootstrapFewShot(
    metric=exact_match,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
)
optimized = optimizer.compile(dspy.ChainOfThought(NumClassify), trainset=trainset)
```

| Parameter | Description |
|-----------|-------------|
| `metric` | Judging function to select high-quality traces. |
| `max_bootstrapped_demos` | Max auto-generated demonstrations per prompt. |
| `max_labeled_demos` | Max hand-labeled demonstrations per prompt. |

**How it works:** Runs the program on training examples, checks the metric, keeps correct traces as few-shot examples, and injects them into the prompt.

**Best for:** When you have a small labeled dataset and want a quick accuracy boost with no extra LLM calls during optimization.

---

### 2. dspy.GEPA (Genetic Prompt Evolution)

**Genetic algorithm** that evolves prompt instructions through mutation and selection. Presented at ICLR 2026.

```python
from dspy import GEPA

optimizer = GEPA(
    metric=exact_match,
    max_full_evals=10,     # Number of generations (default)
)
optimized = optimizer.compile(dspy.ChainOfThought(NumClassify), trainset=trainset)
```

| Parameter | Description |
|-----------|-------------|
| `metric` | Fitness function for evaluating prompt candidates. |
| `max_full_evals` | Maximum evaluations over the full training set per generation. |
| `num_candidates` | Number of prompt candidates per generation (default varies). |

**How it works:**
1. Start with the original prompt as the seed.
2. Generate mutated prompt variants.
3. Evaluate each variant on the training set using the metric.
4. Select the best performers, mutate them, repeat.
5. Return the best prompt found.

**Best for:** When you want to discover non-obvious prompt phrasings that improve performance. More expensive than BootstrapFewShot but can find better prompts.

---

### 3. dspy.MIPROv2 (Bayesian Instruction Search)

**Bayesian optimization** over both instructions and few-shot examples. Uses uncertainty-guided search to find the best prompt configuration.

```python
from dspy import MIPROv2

optimizer = MIPROv2(
    metric=exact_match,
    auto="light",        # Search budget: "light", "medium", "heavy"
    num_threads=4,       # Parallel evaluation threads
)
optimized = optimizer.compile(dspy.ChainOfThought(NumClassify), trainset=trainset)
```

| Parameter | Description |
|-----------|-------------|
| `metric` | Evaluation function. |
| `auto` | Budget level. `"light"` for quick experiments, `"medium"` and `"heavy"` for thorough search. |
| `num_threads` | Threads for parallel evaluation. |
| `num_candidates` | Number of candidate instructions to try (overrides `auto` if set directly). |

**Dependency:** Requires `optuna` for Bayesian optimization (`pip install optuna`).

**How it works:**
1. Generate many candidate instructions and demo sets.
2. Use Bayesian optimization (via Optuna) to select promising combinations.
3. Evaluate top candidates on the training set.
4. Return the best combination.

**Best for:** When you have a moderate to large training set and want the best possible prompt through systematic search. The most thorough optimizer.

---

### 4. dspy.BetterTogether (Chain Optimizers)

**Compose multiple optimizers** in sequence. Each optimizer refines the output of the previous one.

```python
from dspy import BetterTogether, GEPA, BootstrapFinetune

optimizer = BetterTogether(
    metric=exact_match,
    p=GEPA(metric=exact_match, max_full_evals=3),
    w=BootstrapFinetune(metric=exact_match),
)
optimized = optimizer.compile(
    dspy.ChainOfThought(NumClassify),
    trainset=trainset,
    strategy="p -> w -> p",  # Optimizer chain: GEPA → BootstrapFinetune → GEPA
)
```

#### Constructor Parameters

| Parameter | Description |
|-----------|-------------|
| `metric` | Shared metric across all sub-optimizers. |
| `**optimizers` | Named optimizers (`p`, `w`, etc.) that form the building blocks of the strategy. |

#### Compile Parameters

| Parameter | Description |
|-----------|-------------|
| `strategy` | Chain specification. Arrow-separated optimizer names: `"p -> w -> p"`. |
| `trainset` | Training examples. |

**How it works:**
1. Apply `GEPA` (p) to the program.
2. Take the result, apply `BootstrapFinetune` (w).
3. Take that result, apply `GEPA` (p) again.

The strategy string `"p -> w -> p"` references the keyword argument names in the constructor.

**Best for:** When no single optimizer is sufficient. Combining search strategies (GEPA) with fine-tuning (BootstrapFinetune) can yield better results than either alone.

---

## Run Commands

```bash
# Run all optimizers and print comparison table
uv run python lab/04-optimizers/main.py

# Expected output (approximate):
# === Baseline (unoptimized) ===
# Accuracy: 25.00%
#
# === BootstrapFewShot ===
# Accuracy: 75.00%
#
# === GEPA ===
# Accuracy: 100.00%
#
# === MIPROv2 ===
# Accuracy: 100.00%
#
# === BetterTogether ===
# Accuracy: 100.00%
#
# ==================================================
# OPTIMIZER COMPARISON
# ==================================================
# Baseline:           25.00%
# BootstrapFewShot:   75.00%
# GEPA:               100.00%
# MIPROv2:            100.00%
# BetterTogether:     100.00%
```

Note: Actual results depend on the LM and randomness. The toy dataset is small enough that GEPA, MIPROv2, and BetterTogether often reach 100%.

---

## Optimizer Comparison Table

| Optimizer | Strategy | Speed | Typical Improvement | Best For |
|-----------|----------|-------|-------------------|----------|
| `BootstrapFewShot` | Trace & inject demonstrations | Fastest | Moderate | Small datasets, quick iteration |
| `GEPA` | Genetic prompt evolution | Medium | High | Discovering better prompt phrasings |
| `MIPROv2` | Bayesian instruction + demo search | Slowest | Highest | Maximum accuracy, systematic search |
| `BetterTogether` | Chain multiple optimizers | Depends on strategy | Potentially highest | When one optimizer isn't enough |

### Trade-off Guidance

| If you... | Use |
|------------|-----|
| Have a tiny dataset (<20 examples) | `BootstrapFewShot` |
| Want quick results with minimal cost | `BootstrapFewShot` |
| Need the best possible prompt | `MIPROv2` (with `auto="heavy"`) |
| Want to experiment with prompt evolution | `GEPA` |
| Have resources to chain optimizers | `BetterTogether` |

---

## Key Takeaways

- **BootstrapFewShot** is the simplest and fastest. It adds demonstrations to the prompt without changing the instruction.
- **GEPA** mutates the instruction itself through a genetic algorithm. It can find prompts that work better than the original.
- **MIPROv2** jointly searches instruction text and demonstration selection using Bayesian optimization. It is the most thorough but slowest.
- **BetterTogether** chains optimizers in sequence via a strategy string like `"p -> w -> p"`. The arrow-based syntax maps to named optimizer instances.
- **Compile** is the universal API: `optimizer.compile(program, trainset=trainset)`. Every optimizer returns an improved program with the same interface.
- **Metrics** serve double duty: they guide both evaluation and optimization. A good metric is essential for good results.
