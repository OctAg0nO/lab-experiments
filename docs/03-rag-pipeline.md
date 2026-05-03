# 03-rag-pipeline — Retrieval-Augmented Generation with Optimization

> **File:** `lab/03-rag-pipeline/main.py`
> **Concepts:** `dspy.ColBERTv2` retriever, `dspy.Module` for RAG pipelines, `dspy.BootstrapFewShot` optimizer, `dspy.Evaluate`, custom metrics, dataset loading.

## Purpose

Build an end-to-end RAG (Retrieval-Augmented Generation) pipeline, evaluate its baseline accuracy, then optimize it with BootstrapFewShot to improve performance through demonstrations.

## Setup

```python
import dspy
from dspy.datasets import HotPotQA

lm = dspy.LM("deepseek/deepseek-v4-flash")
dspy.configure(lm=lm)
```

---

## Dataset Loading

### dspy.datasets.HotPotQA

A multi-hop QA dataset with questions, answers, and supporting facts.

```python
dataset = HotPotQA(
    train_seed=2024,    # Random seed for reproducibility
    train_size=200,     # Number of training examples
    eval_size=50,       # Number of evaluation examples
)
```

Returns a dict-like object with `.train` and `.dev` attributes:

```python
trainset = [dspy.Example(**x).with_inputs("question") for x in dataset.train]
devset   = [dspy.Example(**x).with_inputs("question") for x in dataset.dev]
```

### dspy.Example

A data container with attribute access and metadata:

```python
ex = dspy.Example(question="...", answer="...").with_inputs("question")
# .with_inputs("question") marks which fields are model inputs (vs. labels)
```

| Method | Description |
|--------|-------------|
| `dspy.Example(**fields)` | Create an example with arbitrary key-value fields. |
| `.with_inputs(*field_names)` | Designate fields as model inputs. Other fields are treated as labels. |
| `.copy(**overrides)` | Create a copy with modified fields. |

**Important:** Call `.with_inputs()` to tell DSPy which fields to pass to the model. Fields not marked as inputs are assumed to be labels used for evaluation.

---

## RAG Module

A custom DSPy module that chains retrieval and generation.

```python
class RAG(dspy.Module):
    def __init__(self, k=3):
        self.k = k
        self.retrieve = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question, k=self.k)
        return self.generate(context=context, question=question)
```

### dspy.ColBERTv2

A sparse neural retriever (ColBERT v2) that scores document-query relevance via late interaction.

```python
retriever = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
contexts = retriever(query="What is the capital of France?", k=3)
# contexts -> list[str] of top-k document passages
```

| API | Description |
|-----|-------------|
| `dspy.ColBERTv2(url=...,)` | Constructor. Points to a running ColBERTv2 API endpoint. |
| `.query(query, k=3)` | Retrieve top-k passages. Returns `list[str]`. |
| `.forward(query, k=3)` | Same as `.query()`. Used when composing in modules. |

**Note:** The URL points to a public ColBERTv2 endpoint serving Wikipedia 2017 abstracts. In production, you would host your own ColBERTv2 index.

### Context-Generation Chain

The generator uses a multi-field signature that includes both the retrieved context and the original question:

```python
self.generate = dspy.ChainOfThought("context, question -> answer")
```

This signature:
- Takes `context: str` (the retrieved passages, concatenated) and `question: str`
- Produces `answer: str`

The ChainOfThought module reasons over the context before answering, which helps with multi-hop questions.

---

## Custom Metric

A metric function evaluates predictions against ground truth labels. It receives an example (with label fields) and a prediction, and returns a score (bool or float).

```python
def gold_answer_metric(example, prediction, trace=None):
    return example.answer == prediction.answer
```

### Metric Function Signature

```python
def metric_name(
    example: dspy.Example,     # Gold example with label fields
    prediction: dspy.Prediction, # Model output
    trace: list | None         # Internal trace (set by optimizers, can be None)
) -> bool | float:
```

| Parameter | Description |
|-----------|-------------|
| `example` | The gold example. Access labels via `example.answer`, `example.parity`, etc. |
| `prediction` | The model's output. Access predictions via `prediction.answer`, etc. |
| `trace` | Set by optimizers during bootstrapping. Ignore for simple metrics. |

**Return value:** `True`/`False` for accuracy, or a float for partial credit.

---

## Evaluation

### dspy.Evaluate

Runs a program over a development set and computes aggregate metrics.

```python
evaluator = dspy.Evaluate(
    devset=devset,           # List of dspy.Example
    metric=gold_answer_metric, # Callable: (example, prediction, trace) -> score
    num_threads=8,           # Parallel threads for evaluation
    display_progress=True,   # Show progress bar
    display_table=False,     # Print per-example results table
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `devset` | required | Iterable of `dspy.Example` to evaluate on. |
| `metric` | required | Scoring function. |
| `num_threads` | 1 | Number of parallel threads. |
| `display_progress` | `True` | Show tqdm progress bar. |
| `display_table` | `False` | Print a pandas-style results table. |
| `return_all_scores` | `True` | Also return per-example scores. |

### Evaluation Result

```python
result = evaluator(rag)
# result.score -> int (sum of metric scores, usually 0-100 for accuracy %)
# result.scores -> list of per-example metric values (if return_all_scores=True)
accuracy = result.score / 100.0  # Normalize to 0-1
```

The `.score` is an integer (sum of all metric results). For boolean metrics, it equals the number of correct predictions. Divide by `len(devset)` (or 100) to get a 0-1 accuracy.

---

## Optimization

### dspy.BootstrapFewShot

An optimizer that traces the program on training examples to build few-shot demonstrations.

```python
optimizer = dspy.BootstrapFewShot(
    metric=gold_answer_metric,  # Metric to optimize for
    max_bootstrapped_demos=4,   # Max auto-generated demonstrations per prompt
    max_labeled_demos=4,        # Max hand-labeled demonstrations per prompt
)
optimized_rag = optimizer.compile(rag, trainset=trainset)
```

| Parameter | Description |
|-----------|-------------|
| `metric` | The metric used to judge which bootstrapped traces are high-quality. |
| `max_bootstrapped_demos` | Max number of examples where DSPy runs the program to generate a demonstration. |
| `max_labeled_demos` | Max number of examples used directly as demonstrations (no program trace needed). |
| `max_rounds` | Max bootstrapping rounds (default 1 for simple programs). |

### How BootstrapFewShot Works

1. For each training example, run the program to get a prediction.
2. Evaluate the prediction against the metric.
3. If the prediction is correct (metric returns True/positive), save the input, output, and reasoning trace as a demonstration.
4. Select up to `max_bootstrapped_demos + max_labeled_demos` demonstrations per prompt.
5. Inject demonstrations into the prompt for future calls.

### Compile

```python
optimized_program = optimizer.compile(
    program,            # dspy.Module or predictor to optimize
    trainset=trainset,  # Training examples
    num_threads=4       # Threads for bootstrapping
)
```

Returns the same module with optimized internal prompts. The module keeps its original interface:

```python
optimized_rag(question="...")  # Same call signature as before
```

---

## Full Pipeline

```python
# 1. Load data
dataset = HotPotQA(train_seed=2024, train_size=200, eval_size=50)
trainset = [dspy.Example(**x).with_inputs("question") for x in dataset.train]
devset   = [dspy.Example(**x).with_inputs("question") for x in dataset.dev]

# 2. Define metric
def gold_answer_metric(example, prediction, trace=None):
    return example.answer == prediction.answer

# 3. Baseline evaluation
rag = RAG()
evaluator = dspy.Evaluate(devset=devset, metric=gold_answer_metric, num_threads=8)
baseline = evaluator(rag).score / 100.0

# 4. Optimize
optimizer = dspy.BootstrapFewShot(metric=gold_answer_metric, max_bootstrapped_demos=4, max_labeled_demos=4)
optimized_rag = optimizer.compile(rag, trainset=trainset)

# 5. Optimized evaluation
optimized_score = evaluator(optimized_rag).score / 100.0
```

---

## Run Commands

```bash
# Run the full pipeline
uv run python lab/03-rag-pipeline/main.py

# Expected output (approximate):
# Baseline accuracy: 34.00%
# Optimized accuracy: 48.00%
# Improvement: +14.00%
```

Note: The ColBERTv2 endpoint must be reachable. If it is down, the script will fail at the retrieval step.

---

## Key Takeaways

- **RAG modules** compose a retriever (`dspy.ColBERTv2`) and a generator (`ChainOfThought`). The retriever is not a DSPy predictor and is not optimized.
- **BootstrapFewShot** improves accuracy by providing worked examples in the prompt. It traces the program, keeps correct traces, and injects them as demonstrations.
- **Metrics** are plain functions that compare predictions to labels. They drive both evaluation and optimization.
- **dspy.Evaluate** runs the program in parallel (set via `num_threads`) and aggregates scores.
- **Normalize scores**: `result.score / 100.0` converts the integer sum to 0-1 when metric returns bool.
- **with_inputs()** is required on examples to tell DSPy which fields are model inputs vs. labels.
