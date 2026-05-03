# 06 — Advanced DSPy Patterns

> Source: `lab/06-advanced/main.py`

Covers advanced execution patterns: multi-chain reasoning comparison, parallel batch execution, ensemble prediction, adapter switching for structured output, synchronous streaming, and async execution.

---

## `dspy.MultiChainComparison`

Compare multiple reasoning chains and synthesize a final answer. Runs the same signature N times, then compares the chains and picks the strongest result.

```python
class CompareAnswers(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

mc = dspy.MultiChainComparison(CompareAnswers, n=3)
result = mc(question="Which number is larger: 0.8 or 0.11?")
print(result.answer)
```

### Constructor

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `signature` | `type[dspy.Signature]` | required | The signature to compare across chains |
| `n` | `int` | `3` | Number of reasoning chains to generate and compare |

### Call

`mc(**kwargs) -> dspy.Prediction`

Runs the signature `n` times, then applies cross-chain comparison to produce the final output.

### When to use

- Questions where different reasoning paths can lead to different answers
- Tasks that benefit from majority-vote or meta-reasoning across chains
- Reducing variance in single-pass predictions

---

## `Module.batch()` — Parallel Execution

Replaces the legacy `dspy.Parallel`. Runs the same module across multiple examples concurrently.

```python
class Summarize(dspy.Signature):
    text: str = dspy.InputField()
    summary: str = dspy.OutputField()

prog = dspy.ChainOfThought(Summarize)
texts = [
    "Python is a high-level general-purpose programming language.",
    "DSPy is a framework for programming language models.",
    "Dapr provides distributed application runtime capabilities.",
]
examples = [dspy.Example(text=t).with_inputs("text") for t in texts]
results = prog.batch(examples, num_threads=3)
for r in results:
    print(r.summary)
```

### Signature

```python
module.batch(
    examples: list[dspy.Example],
    num_threads: int = 4,
) -> list[dspy.Prediction]
```

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `examples` | `list[dspy.Example]` | required | List of examples, each with `.with_inputs()` called |
| `num_threads` | `int` | `4` | Number of concurrent threads |

### Returns

`list[dspy.Prediction]` — one prediction per example, in the same order.

### Requirements

- Each example must have `.with_inputs(...)` set
- The module must be callable (Predict, ChainOfThought, ReAct, etc.)
- Thread count should match your rate limits and LM backend capacity

---

## `dspy.Ensemble`

Combine predictions from multiple modules. The ensemble runs all modules and merges outputs.

```python
class Classify(dspy.Signature):
    text: str = dspy.InputField()
    label: str = dspy.OutputField()

ensemble = dspy.Ensemble()
compiled = ensemble.compile([
    dspy.ChainOfThought(Classify),
    dspy.Predict(Classify),
])
result = compiled(text="This movie was surprisingly good!")
print(result.label)
```

### Constructor

`dspy.Ensemble()`

No constructor parameters.

### `.compile(modules) -> dspy.Module`

| Param | Type | Description |
|-------|------|-------------|
| `modules` | `list[dspy.Module]` | Modules whose predictions will be combined |

Returns a compiled module that, when called, runs all input modules and merges their outputs.

### How merging works

For each output field, the ensemble collects values from each constituent module. For categorical fields (like labels), the ensemble uses voting. The exact merging strategy depends on output field type.

### When to use

- Reducing prediction variance
- Combining different module types (Predict + ChainOfThought)
- Production pipelines where robustness matters more than latency

---

## Adapter Switching

DSPy supports pluggable adapters that control how signatures are serialized and deserialized. Set the active adapter via `dspy.configure()`.

### `dspy.JSONAdapter()`

Forces structured JSON output. Useful for extracting typed fields (integers, lists, nested objects).

```python
dspy.configure(adapter=dspy.JSONAdapter())

class Person(dspy.Signature):
    description: str = dspy.InputField()
    name: str = dspy.OutputField()
    age: int = dspy.OutputField()

extract = dspy.Predict(Person)
result = extract(description="A 28-year-old software engineer named Maya living in Berlin")
print(f"name={result.name}, age={result.age}")  # name=Maya, age=28
```

Key behavior:
- Output fields with non-string types (int, float, bool, list) are parsed from JSON
- The LM is instructed to return a JSON object matching the output schema
- Works with Predict, ChainOfThought, and other modules

### `dspy.ChatAdapter()`

The default chat-style adapter. Resets the adapter back to standard behavior after using JSONAdapter or other structured adapters.

```python
dspy.configure(adapter=dspy.ChatAdapter())
```

### Other built-in adapters

| Adapter | Description |
|---------|-------------|
| `dspy.ChatAdapter` | Default chat completion format |
| `dspy.JSONAdapter` | Structured JSON output |
| `dspy.XMLAdapter` | XML-tagged output format |
| `dspy.BAMLAdapter` | Boundary-prompted structured output via BAML |

---

## `dspy.streamify()`

Convert a synchronous DSPy module into a generator that yields tokens as they arrive.

```python
dspy.configure(adapter=dspy.ChatAdapter())  # ChatAdapter required for streaming

stream_gen = dspy.streamify(
    dspy.ChainOfThought("topic -> haiku"),
    async_streaming=False,
)
for chunk in stream_gen(topic="programming"):
    print(chunk, end="", flush=True)
```

### Signature

```python
dspy.streamify(
    module: dspy.Module,
    async_streaming: bool = False,
) -> Callable[..., Generator[str, None, None]]
```

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `module` | `dspy.Module` | required | The module to convert to a streaming generator |
| `async_streaming` | `bool` | `False` | If True, returns an async generator instead |

### Returns

A callable that, when invoked with the module's inputs, yields strings (token chunks).

### Requirements

- The active adapter must support streaming. Use `ChatAdapter` (not JSONAdapter).
- The underlying LM must support streaming (most do).

---

## `dspy.asyncify()` and `module.acall()`

### `dspy.asyncify(module)`

Convert a synchronous module into an async-compatible callable.

```python
async_module = dspy.asyncify(dspy.ChainOfThought("question -> answer"))
result = await async_module(question="What is the async equivalent in DSPy?")
```

### `module.acall(**kwargs)`

The built-in async call method available on all DSPy modules. No conversion needed.

```python
prog = dspy.ChainOfThought("question -> answer")
result = await prog.acall(question="What is the async equivalent in DSPy?")
print(result.answer)
```

### Signature

```python
await module.acall(**kwargs) -> dspy.Prediction
```

| Param | Type | Description |
|-------|------|-------------|
| `**kwargs` | `Any` | Input fields matching the module's signature |

### Returns

`dspy.Prediction` — same as the synchronous call.

### When to use `acall` vs `asyncify`

- Use `acall` directly — it is available on every module and requires no wrapping
- Use `asyncify` only if you need to pass the module to an API that expects a callable (not a method)

### Example — full async pipeline

```python
import asyncio

async def main():
    prog = dspy.ChainOfThought("question -> answer")
    result = await prog.acall(question="What is the async equivalent in DSPy?")
    print(f"Async answer: {result.answer}")

asyncio.run(main())
```

---

## Quick Reference

| API | Purpose |
|-----|---------|
| `dspy.MultiChainComparison(Sig, n=3)` | Compare N reasoning chains and pick best |
| `module.batch(examples, num_threads=4)` | Run module on multiple examples in parallel |
| `dspy.Ensemble().compile([mod1, mod2])` | Combine predictions from multiple modules |
| `dspy.JSONAdapter()` | Structured JSON output from any module |
| `dspy.ChatAdapter()` | Default adapter, resets from JSON/XML |
| `dspy.streamify(module)` | Convert module to token-streaming generator |
| `dspy.asyncify(module)` | Convert module to async callable |
| `module.acall(**kwargs)` | Async call on any module |
