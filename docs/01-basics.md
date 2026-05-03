# 01-basics — DSPy Signatures, Predictors, and Module Composition

> **File:** `lab/01-basics/main.py`
> **Concepts:** Short-form signatures, class-based signatures, `dspy.Predict`, `dspy.ChainOfThought`, custom `dspy.Module` pipelines.

## Purpose

Introduce the core DSPy abstractions: how to define input/output schemas (signatures), choose the right predictor module, and compose predictors into multi-stage pipelines.

## Setup

```python
from typing import Literal
import dspy

lm = dspy.LM("deepseek/deepseek-v4-flash")
dspy.configure(lm=lm)
```

The LM must be configured once at the top of the script. Every predictor call routes through this global LM.

---

## Signatures

Signatures are DSPy's way of declaring what an LM call should consume and produce. Two forms:

### Short-Form Signature

A compact string syntax: `"input_field1, input_field2 -> output_field1: type, output_field2: type"`

```python
math = dspy.ChainOfThought("question -> answer: float")
result = math(question="What is 15 * 37?")
# result.answer -> float
```

The `: float` suffix constrains the output type. DSPy uses this for structured parsing.

### Class-Based Signature

A declarative class with typed fields. Supports richer metadata through docstrings and field annotations.

```python
class Classify(dspy.Signature):
    """Classify the sentiment and topic of a message."""
    sentence: str = dspy.InputField()
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()
    topic: str = dspy.OutputField()

classifier = dspy.ChainOfThought(Classify)
result = classifier(sentence="DSPy makes LLM programming elegant and reliable")
# result.sentiment -> one of "positive", "negative", "neutral"
# result.topic -> str
```

| Element | Description |
|---------|-------------|
| `dspy.Signature` | Base class for all signatures |
| `dspy.InputField()` | Marks a field as model input |
| `dspy.OutputField()` | Marks a field as model output |
| `Literal[...]` | Constrains output to a fixed set of values (DSPy parses into exact match) |
| Docstring | Becomes the task instruction in the prompt |

### Field Types Supported

| Type | Example | Behavior |
|------|---------|----------|
| `str` | `name: str` | Free-text string output |
| `int` | `age: int` | Parsed integer |
| `float` | `score: float` | Parsed float |
| `Literal["a", "b"]` | `sentiment: Literal["pos", "neg"]` | Constrained classification |
| `list[str]` | `sections: list[str]` | Parsed list of strings |
| `bool` | `is_valid: bool` | Parsed boolean |

---

## Predictors

DSPy provides different predictor modules that wrap a signature with different inference strategies.

### dspy.Predict

Direct prediction with no reasoning trace. Faster and cheaper than ChainOfThought.

```python
class Extract(dspy.Signature):
    """Extract structured info from text."""
    text: str = dspy.InputField()
    name: str = dspy.OutputField()
    age: int = dspy.OutputField()

extractor = dspy.Predict(Extract)
result = extractor(text="Alice is 32 years old and works at Acme Corp.")
# result.name -> "Alice", result.age -> 32
```

**When to use:** Simple extraction, classification, or formatting tasks where step-by-step reasoning adds no value.

**API:**
| Method / Attribute | Description |
|--------------------|-------------|
| `dspy.Predict(signature)` | Constructor. Accepts a signature class or short-form string. |
| `predictor(**inputs) -> dspy.Prediction` | Forward pass. Returns a Prediction with typed fields. |
| `predictor.activate_optimizer(optimizer)` | Attach an optimizer for compilation (see lab 04). |

### dspy.ChainOfThought

Generates a step-by-step reasoning trace before producing the final output. Improves accuracy on tasks that benefit from intermediate reasoning.

```python
math = dspy.ChainOfThought("question -> answer: float")
result = math(question="What is 15 * 37?")
# result.answer -> 555.0
```

```python
classifier = dspy.ChainOfThought(Classify)
result = classifier(sentence="...")
# Internally: "We need to classify the sentiment..."
```

**When to use:** Multi-step reasoning, math, analysis, classification requiring justification.

**API:**
| Method / Attribute | Description |
|--------------------|-------------|
| `dspy.ChainOfThought(signature)` | Constructor. Same signature formats as Predict. |
| `cot(**inputs) -> dspy.Prediction` | Forward pass. Returns prediction with same fields as output signature. |
| `cot.activate_optimizer(optimizer)` | Attach an optimizer. |

**Key difference from Predict:**

| Aspect | Predict | ChainOfThought |
|--------|---------|----------------|
| Reasoning trace | None | Produces "rationale" before answer |
| Speed | Faster | Slower (more tokens) |
| Cost | Cheaper | More expensive |
| Accuracy on complex tasks | Lower | Higher |

---

## Custom Modules

A `dspy.Module` composes multiple predictors into a reusable pipeline with learnable parameters.

### Pattern

```python
class MyModule(dspy.Module):
    def __init__(self):
        self.sub_predictor = dspy.ChainOfThought(...)
        self.other_step = dspy.Predict(...)

    def forward(self, **inputs):
        intermediate = self.sub_predictor(...)
        result = self.other_step(...)
        return dspy.Prediction(...)
```

- `__init__`: Define sub-modules as instance attributes. DSPy auto-discovers them for optimization.
- `forward`: Accept keyword arguments matching input fields, return `dspy.Prediction`.

### Example: ArticleWriter

A two-stage pipeline that first outlines an article, then drafts each section.

```python
class Outline(dspy.Signature):
    topic: str = dspy.InputField()
    title: str = dspy.OutputField()
    sections: list[str] = dspy.OutputField()

class DraftSection(dspy.Signature):
    topic: str = dspy.InputField()
    heading: str = dspy.InputField()
    content: str = dspy.OutputField()

class ArticleWriter(dspy.Module):
    def __init__(self):
        self.outline = dspy.ChainOfThought(Outline)
        self.draft = dspy.ChainOfThought(DraftSection)

    def forward(self, topic):
        o = self.outline(topic=topic)
        sections = [self.draft(topic=topic, heading=h) for h in o.sections]
        return dspy.Prediction(title=o.title, sections=sections)

writer = ArticleWriter()
article = writer(topic="The future of AI agents")
# article.title -> str
# article.sections -> list[Prediction] each with .content
```

**Key design rules:**
1. Sub-modules must be set as `self.*` attributes in `__init__` so DSPy can find them during compilation.
2. `forward` must accept the top-level input fields and return a `dspy.Prediction`.
3. Sub-modules can call each other arbitrarily, including loops over dynamic outputs (like `o.sections`).
4. Optimizers (lab 04) recursively compile all sub-modules when optimizing the parent.

### dspy.Prediction

A dict-like object with attribute access. Returned by every predictor forward pass.

```python
pred = dspy.Prediction(title="...", sections=[...])
pred.title     # attribute access
pred["title"]  # dict access
```

---

## Run Commands

```bash
# Run the full script
uv run python lab/01-basics/main.py

# Expected output (approximately):
# Math answer: 555.0
# Sentiment: positive | Topic: DSPy
# Extracted: Alice, age 32
# Title: The Future of AI Agents
# Sections: 5 drafted
```

## Key Takeaways

- **Signatures** define the contract between your code and the LM. Use short-form for quick experiments, class-based for richer schemas.
- **Predict** is direct and fast. **ChainOfThought** adds reasoning for harder tasks.
- **Modules** compose predictors into pipelines. DSPy discovers sub-modules via `self.*` attributes.
- **Output fields** can be typed with `str`, `int`, `float`, `Literal`, `list[str]`, and `bool`.
- **Multi-stage pipelines** are straightforward: chain predictors in `forward`, passing outputs as inputs to the next step.
