# 07 — Generative Feedback Loops (GFL)

> Source: `lab/07-generative-feedback-loops/main.py`

DSPy's zero-gradient optimization engine. The GFL mechanism: run forward, collect traces, evaluate against a metric, update instructions and demonstrations, repeat. No weight updates, no backprop. The LLM's generative capability drives optimization; the metric provides selection pressure.

---

## The GFL Loop

```
┌─────────────────────────────────────────────────────────┐
│                 GENERATIVE FEEDBACK LOOP                 │
│                                                         │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────┐  │
│   │  TRACE       │    │ FEEDBACK     │    │ PROGRAM  │  │
│   │  COLLECT     │───→│ GENERATE     │───→│  UPDATE  │  │
│   └──────────────┘    └──────────────┘    └──────────┘  │
│         ↑                                  │            │
│         └──────────────── LOOP ────────────┘            │
│                                                         │
│   • No gradients              • Program-level updates   │
│   • No weight updates         • Prompt + demos only     │
│   • LLM generates signal      • Metric selects          │
└─────────────────────────────────────────────────────────┘
```

Four steps, repeated:
1. **Trace Collection** — run the program forward, capture all inputs, outputs, and intermediates
2. **Feedback Generation** — evaluate each trace against the metric (scalar or textual)
3. **Program Update** — revise instructions and/or demonstrations based on feedback
4. **Repeat** — loop until convergence or budget exhaustion

Five GFL optimizers are demonstrated: BootstrapFewShot, MIPROv2, GEPA, Sequential (GEPA then BootstrapFewShot), and Teacher/Student distillation.

---

## Task: Multi-label Intent Classification

### `ClassifyIntent(dspy.Signature)`

Classify a user query into an intent category with a confidence level.

```python
class ClassifyIntent(dspy.Signature):
    """Classify user query intent and confidence level."""
    query: str = dspy.InputField()
    intent: str = dspy.OutputField()
    confidence: str = dspy.OutputField()
```

| Field | Direction | Type | Description |
|-------|-----------|------|-------------|
| `query` | Input | `str` | The user's natural language query |
| `intent` | Output | `str` | One of: `booking`, `inquiry`, `cancellation`, `account`, `command`, `entertainment`, `tracking`, `transaction`, `support` |
| `confidence` | Output | `str` | One of: `low`, `medium`, `high` |

### Training data

20 labeled examples spanning 9 intent classes. Each example is a `dspy.Example` with `query`, `intent`, and `confidence` fields, with `query` set as the input field via `.with_inputs("query")`.

```python
def build_examples(data):
    return [
        dspy.Example(query=q, intent=i, confidence=c).with_inputs("query")
        for q, i, c in data
    ]
```

### Dev set

5 held-out examples for evaluation, same format as the training set.

---

## Metrics

### `intent_metric(example, prediction, trace=None)`

Standard metric for BootstrapFewShot and MIPROv2 (3-arg signature).

```python
def intent_metric(example, prediction, trace=None):
    intent_ok = example.intent == prediction.intent
    confidence_ok = prediction.confidence in ("low", "medium", "high")
    return intent_ok and confidence_ok
```

| Param | Type | Description |
|-------|------|-------------|
| `example` | `dspy.Example` | The ground-truth example |
| `prediction` | `dspy.Prediction` | The model's prediction |
| `trace` | `list` or `None` | Execution trace (not used here) |

Returns `bool` — exact match on intent AND valid confidence value.

### `gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None)`

GEPA metric with the required 5-argument signature. Returns a float score.

```python
def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    intent_ok = gold.intent == pred.intent
    confidence_ok = pred.confidence in ("low", "medium", "high")
    return float(intent_ok and confidence_ok)
```

| Param | Type | Description |
|-------|------|-------------|
| `gold` | `dspy.Example` | The ground-truth example |
| `pred` | `dspy.Prediction` | The model's prediction |
| `trace` | ignored | Required by GEPA signature |
| `pred_name` | ignored | Required by GEPA signature |
| `pred_trace` | ignored | Required by GEPA signature |

Returns `float` — `1.0` for correct, `0.0` for incorrect.

**Important:** GEPA requires exactly 5 positional arguments. Using a 3-arg metric with GEPA will fail.

---

## Evaluation

### `dspy.Evaluate`

```python
evaluator = dspy.Evaluate(
    devset=devset,
    metric=intent_metric,
    num_threads=4,
    display_progress=True,
)
```

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `devset` | `list[dspy.Example]` | required | Evaluation examples |
| `metric` | `callable` | required | Metric function |
| `num_threads` | `int` | `4` | Parallel threads for eval |
| `display_progress` | `bool` | `True` | Show progress bar |

### `eval_score(program) -> float`

Convenience wrapper. `Evaluate.__call__` returns an `EvaluationResult` whose `.score` is a percentage (0-100). This function divides by 100 to get a 0.0-1.0 fraction.

```python
def eval_score(program) -> float:
    return evaluator(program).score / 100.0
```

---

## Optimizer 1: `dspy.BootstrapFewShot`

**Mechanism:** Run teacher on training examples, collect execution traces, keep only traces where the metric succeeds, attach them as demonstrations on the student.

```python
bs = dspy.BootstrapFewShot(
    metric=intent_metric,
    max_bootstrapped_demos=6,
    max_labeled_demos=4,
)
bs_program = bs.compile(
    dspy.ChainOfThought(ClassifyIntent),
    trainset=trainset,
)
```

### Constructor

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `metric` | `callable` | required | 3-arg metric (example, prediction, trace) |
| `max_bootstrapped_demos` | `int` | `4` | Max teacher-generated demonstrations |
| `max_labeled_demos` | `int` | `0` | Max raw labeled examples as demos |

### `.compile(student, teacher=None, trainset=...)`

| Param | Type | Description |
|-------|------|-------------|
| `student` | `dspy.Module` | The program to optimize |
| `teacher` | `dspy.Module` or `None` | Teacher for generating demos. Defaults to student. |
| `trainset` | `list[dspy.Example]` | Training examples |

Returns the compiled program with `.demos` populated.

### When to use

- First optimizer to try on any task
- Simple tasks where a few demonstrations suffice
- Starting point before more expensive optimizers

---

## Optimizer 2: `dspy.MIPROv2`

**Mechanism:** (1) Bootstrap candidate demonstrations. (2) Propose instruction variants via GroundedProposer. (3) Bayesian search over the instruction x demonstration space.

```python
mipro = dspy.MIPROv2(
    metric=intent_metric,
    auto="light",
    num_threads=4,
)
mipro_program = mipro.compile(
    dspy.ChainOfThought(ClassifyIntent),
    trainset=trainset,
)
```

### Constructor

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `metric` | `callable` | required | 3-arg metric |
| `auto` | `str` | `"light"` | Optimization budget: `"light"` or `"full"` |
| `num_threads` | `int` | `4` | Parallel threads |

### `.compile(program, trainset=...)`

Returns an optimized program with revised instructions. Access optimized instructions via:

```python
optimized_sig = mipro_program.predictors()[0].signature
print(optimized_sig.instructions)
```

### When to use

- Production pipelines where you want both better instructions AND demonstrations
- Tasks where the default instructions are getting you 70-80% and you need to push higher
- When you have enough budget for Bayesian search over instructions

---

## Optimizer 3: `dspy.GEPA`

**Mechanism:** Reads full execution traces, diagnoses failure modes, proposes targeted instruction mutations, selects via Pareto frontier. Uses a separate reflection LM for the mutation engine. ICLR 2026 Oral.

```python
gepa = dspy.GEPA(
    metric=gepa_metric,
    max_full_evals=2,
    reflection_lm=dspy.LM("deepseek/deepseek-v4-flash"),
    num_threads=4,
)
gepa_program = gepa.compile(
    dspy.ChainOfThought(ClassifyIntent),
    trainset=trainset[:10],
)
```

### Constructor

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `metric` | `callable` | required | **5-arg metric** (gold, pred, trace, pred_name, pred_trace) |
| `max_full_evals` | `int` | `5` | Number of full evaluation cycles |
| `reflection_lm` | `dspy.LM` or `None` | `None` | LM used for analyzing traces and mutating instructions |
| `num_threads` | `int` | `4` | Parallel threads |

### `.compile(program, trainset=...)`

Returns an evolved program with mutated instructions.

### GEPA metric signature

GEPA metrics MUST accept 5 positional arguments:

```python
def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    ...
    return float(...)  # 0.0 to 1.0
```

Passing a standard 3-arg metric (example, prediction, trace) to GEPA will raise an error.

### When to use

- Hard tasks where BootstrapFewShot and MIPROv2 plateau
- Cutting-edge research where you want reflective prompt evolution
- Tasks with clear failure modes that the reflection LM can diagnose

---

## Optimizer 4: Sequential GEPA → BootstrapFewShot

**Mechanism:** Run GEPA to optimize instructions first, then BootstrapFewShot to attach demonstrations to the already-optimized prompt. Two-stage pipeline.

```python
# Stage 1: GEPA optimizes instructions
seq_program = dspy.ChainOfThought(ClassifyIntent)
seq_gepa = dspy.GEPA(
    metric=gepa_metric,
    max_full_evals=1,
    reflection_lm=dspy.LM("deepseek/deepseek-v4-flash"),
    num_threads=4,
)
seq_program = seq_gepa.compile(seq_program, trainset=trainset[:10])

# Stage 2: BootstrapFewShot attaches demonstrations
seq_bs = dspy.BootstrapFewShot(
    metric=intent_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=2,
)
seq_program = seq_bs.compile(seq_program, trainset=trainset)
```

### Why this order

GEPA changes instructions. BootstrapFewShot's demonstrations are instruction-specific. If you add demos first then mutate instructions, the demos may no longer be good matches. Instructions first, demos second.

### When to use

- When neither GEPA nor BootstrapFewShot alone is enough
- When you want the full pipeline: good instructions + good demonstrations
- When you have budget for two optimizer passes

---

## Optimizer 5: Teacher/Student Distillation

**Mechanism:** A strong teacher LM (DeepSeek) generates bootstrapped demonstrations. A weaker student LM (Gemma 4 via Ollama) uses those demonstrations as few-shot examples. The student never sees the teacher's weights, only the demonstrations.

```python
student_lm = dspy.LM("ollama_chat/gemma4")

# Student alone (no teacher, no demos)
student_alone = dspy.ChainOfThought(ClassifyIntent)
student_alone.set_lm(student_lm)
alone_score = eval_score(student_alone)

# Teacher generates demos, student uses them
teacher = dspy.ChainOfThought(ClassifyIntent)       # default LM (DeepSeek)
student = dspy.ChainOfThought(ClassifyIntent)
student.set_lm(student_lm)                           # switch to Gemma 4

ts_bs = dspy.BootstrapFewShot(
    metric=intent_metric,
    max_bootstrapped_demos=6,
    max_labeled_demos=4,
)
distilled = ts_bs.compile(student, teacher=teacher, trainset=trainset)

# IMPORTANT: set the student LM again after compile
distilled.set_lm(student_lm)
distilled_score = eval_score(distilled)
```

### Key steps

1. Create the teacher with the default (strong) LM
2. Create the student, call `student.set_lm(student_lm)` to assign the weak model
3. Call `BootstrapFewShot.compile(student, teacher=teacher, trainset=...)` — the teacher generates demos, they are attached to the student
4. Call `compiled.set_lm(student_lm)` again **after** compile — compile may reset the LM assignment
5. Evaluate the compiled student (it runs on the weak LM with the teacher's demos)

### `module.set_lm(lm)`

```python
module.set_lm(lm: dspy.LM) -> None
```

Assigns a specific LM to a module and all its submodules (predictors). This isolates model choice per-module, enabling teacher/student patterns where different modules use different models.

### When to use

- Model compression: distill a large model into a smaller one
- Cost reduction: run a cheap model with expensive-model demonstrations
- Privacy: generate demos with a cloud API, run inference with a local model
- Latency: trade some accuracy for faster inference

---

## Summary

| Optimizer | What it optimizes | Metric signature | Key parameter |
|-----------|------------------|-----------------|---------------|
| BootstrapFewShot | Demonstrations | 3-arg | `max_bootstrapped_demos` |
| MIPROv2 | Instructions + Demos | 3-arg | `auto="light"` or `"full"` |
| GEPA | Instructions (via reflection) | **5-arg** | `reflection_lm`, `max_full_evals` |
| Sequential | Instructions then Demos | both | Two-stage pipeline |
| Teacher/Student | Demo transfer | 3-arg | `teacher=` kwarg in `compile()` |

### Typical workflow

1. Start with BootstrapFewShot (cheapest, often gives the biggest jump)
2. Try MIPROv2 if you need better instructions
3. Try GEPA if the task is hard and you have a strong reflection LM
4. Chain them: GEPA → BootstrapFewShot for the full pipeline
5. Distill to a cheaper model if the optimized program is too expensive
