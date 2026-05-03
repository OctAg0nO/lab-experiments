# 02-react-tools — ReAct Agent Loop with Tools

> **File:** `lab/02-react-tools/main.py`
> **Concepts:** `dspy.ReAct`, custom function tools, `dspy.Tool`, `dspy.PythonInterpreter`, reasoning trace inspection.

## Purpose

Demonstrate the ReAct agent pattern (thought, action, observation) using DSPy's built-in ReAct module. The agent calls tools in a loop, interprets observations, and decides when to produce a final answer.

## Setup

```python
import dspy

lm = dspy.LM("deepseek/deepseek-v4-flash")
dspy.configure(lm=lm)
```

---

## Tools

Tools are functions the agent can invoke. DSPy converts function signatures and docstrings into LLM-readable tool descriptions.

### Function Tool Pattern

A plain Python function with type annotations and a docstring automatically becomes a tool:

```python
def search(query: str) -> list[str]:
    """Search the knowledge base for information."""
    db = {
        "population of paris": "2.1 million",
        "capital of france": "Paris",
        "height of eiffel tower": "330 meters",
        "currency of japan": "Japanese Yen",
    }
    query_lower = query.lower()
    for key, value in db.items():
        if query_lower in key:
            return [f"{key}: {value}"]
    return [f"No results found for '{query}'"]
```

How DSPy reads the tool:
| Source | Becomes |
|--------|---------|
| Function name (`search`) | Tool name shown to LLM |
| Docstring (`Search the knowledge base...`) | Tool description |
| Parameter name + type (`query: str`) | Tool parameter schema |
| Return type (`-> list[str]`) | Tool output type |

```python
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    allowed = {"+", "-", "*", "/", "(", ")", " ", ".", "0", "1", "2",
               "3", "4", "5", "6", "7", "8", "9"}
    if not all(c in allowed for c in expression):
        return "Error: invalid characters in expression"
    return str(eval(expression))
```

**Safety note:** The `calculator` example uses `eval()` with a whitelist of allowed characters. This prevents arbitrary code injection but is illustrative only. In production, use `ast.literal_eval` or a proper expression parser.

---

## dspy.ReAct

The ReAct module wraps a signature with an agentic loop: the LLM generates thoughts, selects tools, observes results, and repeats until it produces a final answer.

### Basic Usage

```python
agent = dspy.ReAct("question -> answer", tools=[search, calculator])
result = agent(question="What is the population of Paris multiplied by 2?")
# result.answer -> str (e.g., "4.2 million")
```

The agent internally:
1. Reads the question
2. Decides to call `search(query="population of paris")`
3. Receives `"population of paris: 2.1 million"`
4. Decides to call `calculator(expression="2.1 * 2")`
5. Receives `"4.2"`
6. Produces final answer: `"4.2 million"`

### Constructor

```python
dspy.ReAct(
    signature,           # Short-form string or class-based Signature
    tools=[],            # List of function tools or dspy.Tool instances
    max_iters=10,        # Max thought-action-observation steps (default 10)
    verbose=False,       # Print trace during execution
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `signature` | `str` or `dspy.Signature` | Defines input/output schema. Short-form: `"question -> answer"`. |
| `tools` | `list` | Functions or `dspy.Tool` instances the agent can call. |
| `max_iters` | `int` | Maximum iterations before forcing a final answer (default 10). |
| `verbose` | `bool` | Print each thought/action/observation step to stderr. |

### Return Value

`agent(**inputs)` returns a `dspy.Prediction` matching the output fields of the signature. For `"question -> answer"`:

```python
result.answer  # str — the final answer
```

---

## dspy.Tool

For explicit control over tool name and description, wrap your function:

```python
tool = dspy.Tool(
    func=search,
    name="knowledge_search",
    desc="Search a structured knowledge base for factual information.",
)
```

When you pass a plain function, DSPy auto-generates name and desc from the function name and docstring. Use `dspy.Tool` when you need to override these for clarity or disambiguation.

---

## dspy.PythonInterpreter

A built-in tool that executes arbitrary Python code in a sandboxed REPL. The LLM writes code, DSPy runs it, and returns stdout.

```python
code_agent = dspy.ReAct(
    "question -> answer",
    tools=[dspy.PythonInterpreter]
)
code_result = code_agent(question="Compute the sum of squares from 1 to 10")
# Internally the LLM writes: sum(x**2 for x in range(1, 11))
# result.answer -> str (e.g., "385")
```

The PythonInterpreter is useful when:
- The task requires computation the LLM can't do reliably in its head.
- You want the LLM to generate and execute algorithms.
- The logic depends on iteration, recursion, or data structures.

**Security:** The interpreter runs in a restricted namespace. It is not a full sandbox and should not be used with untrusted prompts.

---

## Reasoning Trace Inspection

After a ReAct run, inspect the full thought-action-observation trace:

```python
dspy.inspect_history(n=1)
```

This prints the last `n` LLM interactions, showing:
- The system prompt with tool descriptions
- Each thought step
- Each tool call and its result
- The final answer generation

Use this for debugging: see what the LLM was thinking, which tools it chose, and where the chain went wrong.

---

## dspy.ReAct Signature Variants

| Signature | Input Field | Output Field | Use Case |
|-----------|-------------|--------------|----------|
| `"question -> answer"` | `question: str` | `answer: str` | Standard Q&A |
| `"problem -> solution: str, steps: list[str]"` | `problem: str` | `solution, steps` | Multi-field output |
| `ClassBasedReAct(dspy.Signature)` | Custom fields | Custom fields | Complex schemas |

All predictor signatures work. The ReAct loop fills input fields from the caller and generates output fields as the final answer.

---

## Run Commands

```bash
# Run the full script
uv run python lab/02-react-tools/main.py

# Expected output:
# Q: What is the population of Paris multiplied by 2?
# A: 4.2 million
# (then inspect_history shows the reasoning trace)
# ---
# Q: Compute the sum of squares from 1 to 10
# A: 385
```

---

## Key Takeaways

- **Function tools** become LLM-callable tools via their signature and docstring. Type hints and docstrings are critical for good LLM performance.
- **dspy.ReAct** manages the thought-action-observation loop automatically. It routes tool results back as observations.
- **dspy.PythonInterpreter** lets the LLM write and execute code as a tool. Powerful but be mindful of security boundaries.
- **dspy.inspect_history(n=1)** is your primary debugging tool. Always check it when the agent produces wrong answers.
- Tools are composable: pass multiple tools and the LLM decides which to call based on the task description.
