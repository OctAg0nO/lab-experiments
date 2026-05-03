# 05-rlm — Recursive Language Model (RLM)

> **File:** `lab/05-rlm/main.py`
> **Concepts:** `dspy.RLM`, trajectory inspection, REPL-based code generation, structured extraction from unstructured text.

## Purpose

Demonstrate the Recursive Language Model (RLM), an experimental DSPy module where the LLM writes Python code, executes it in a sandboxed REPL, observes results, and iteratively refines its approach until it produces a final answer.

## Setup

```python
import dspy

lm = dspy.LM("deepseek/deepseek-v4-flash")
dspy.configure(lm=lm)
```

---

## dspy.RLM

The RLM module implements a recursive computation pattern: the LLM generates Python code, the code runs in a persistent REPL, and the LLM sees the output and decides what to do next.

### Constructor

```python
rlm = dspy.RLM(
    signature,              # Input/output schema (short-form or class-based)
    max_iterations=15,      # Max write-execute-observe cycles
    max_llm_calls=20,       # Max total LLM calls across all cycles
    tools=None,             # Optional list of additional tools
    verbose=False,          # Print each step's code and output
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `signature` | required | Defines input and output fields. |
| `max_iterations` | `15` | Max number of write-execute-observe cycles. The LLM gets one LLM call per cycle by default. |
| `max_llm_calls` | `20` | Hard cap on total LLM calls. Preents runaway costs. |
| `tools` | `None` | Additional tools beyond the built-in REPL. |
| `verbose` | `False` | Print each code block, execution result, and reasoning step. |

### Signature Format

RLM uses the same signature format as all DSPy predictors, but it is especially effective with structured output types:

```python
# Short-form with typed outputs
"clinical_note -> diagnosis, medications: list[str], procedures: list[str]"

# Class-based
class MedicalExtraction(dspy.Signature):
    clinical_note: str = dspy.InputField()
    diagnosis: str = dspy.OutputField()
    medications: list[str] = dspy.OutputField()
    procedures: list[str] = dspy.OutputField()
```

Key difference from standard predictors: RLM output fields should be structured types (`list[str]`, `dict`, etc.) since the LLM will write code to construct them.

### Forward Pass

```python
result = rlm(clinical_note=text_block)
# result.diagnosis -> str
# result.medications -> list[str]
# result.procedures -> list[str]
```

The returned `dspy.Prediction` has one attribute per output field in the signature.

---

## How RLM Works

The RLM loop follows a recursive pattern:

```
1. LLM receives the input + previous state
2. LLM writes Python code to process data, call sub-LLMs, etc.
3. Code executes in a persistent REPL (state carries across iterations)
4. LLM sees the output (stdout, errors, intermediate variables)
5. If not done: go to step 1
6. If done: extract final answer from REPL state
```

### REPL State Persistence

The REPL namespace persists across iterations. Variables defined in one step are available in subsequent steps. This allows the LLM to build answers incrementally:

```
Iteration 1: text = load_data()
Iteration 2: lines = text.split('\n')  # 'text' from iteration 1 is available
Iteration 3: diagnosis = extract_diagnosis(lines)  # 'lines' from iteration 2
```

### Sub-LLM Calls

The RLM can call sub-LLMs within its code for semantic analysis tasks (classification, summarization, extraction). This is a key differentiator from standard code-gen: the RLMs code can delegate semantic subtasks back to the LLM.

---

## Trajectory Inspection

After execution, inspect what code the LLM wrote at each step:

```python
if hasattr(result, "trajectory"):
    print(f"RLM trajectory: {len(result.trajectory)} steps")
    for i, step in enumerate(result.trajectory):
        print(f"  Step {i+1}: {step.get('reasoning', '')[:120]}...")
```

The trajectory is a list of step dicts. Each step typically contains:

| Key | Description |
|-----|-------------|
| `reasoning` | The LLMs thought process before writing code. |
| `code` | The Python code the LLM wrote and executed. |
| `output` | The stdout/stderr from code execution. |
| `error` | Any execution error (if the code crashed). |

**Trajectory structure** (approximately):

```python
[
    {"reasoning": "I need to parse the clinical note...", "code": "lines = ...", "output": "..."},
    {"reasoning": "Now I need to search for diagnosis...", "code": "diagnosis = ...", "output": "..."},
    # ...
]
```

Use the trajectory for:
- Debugging why the RLM produced a wrong answer
- Understanding the RLMs problem-solving strategy
- Extracting reusable code patterns

---

## Example: Clinical Note Extraction

The lab uses a realistic clinical note (67-year-old male with anterior STEMI) and extracts structured fields.

```python
text_block = """
Patient: John Doe (MRN: 88472)
Date of Admission: 2025-11-12
...
ECG: ST elevation in leads V2-V4 consistent with anterior STEMI.
...
"""

rlm = dspy.RLM(
    "clinical_note -> diagnosis, medications: list[str], procedures: list[str]",
    max_iterations=15,
    max_llm_calls=20,
    verbose=False,
)

result = rlm(clinical_note=text_block)
print(f"Diagnosis:    {result.diagnosis}")
print(f"Medications:  {result.medications}")
print(f"Procedures:   {result.procedures}")
```

**What the LLM typically does internally:**
1. Split the text into sections (History, Vitals, Labs, etc.).
2. Search for keywords like "diagnosis", "Assessment", "Discharge Plan".
3. Extract medications by pattern matching (`Aspirin 81 mg`, etc.).
4. Extract procedures (cath lab, stent placement).
5. Write the final structured answer into the output fields.

---

## Run Commands

```bash
# Run the RLM extraction
uv run python lab/05-rlm/main.py

# Expected output (approximately):
# === RLM: Clinical Note Extraction ===
# Processing 943 character clinical note...
#
# Diagnosis:    Anterior ST-elevation myocardial infarction (STEMI)
# Medications:  ['Aspirin 81 mg daily', 'Clopidogrel 75 mg daily', 'Atorvastatin 80 mg daily', 'Metoprolol 25 mg BID']
# Procedures:   ['Cardiac catheterization', 'Drug-eluting stent placement to proximal LAD']
#
# RLM trajectory: 4 steps
#   Step 1: I need to parse the clinical note and identify sections...
#   Step 2: Now I will search for the diagnosis in the Assessment section...
#   Step 3: I found the diagnosis. Let me extract medications from Discharge Plan...
#   Step 4: I have all fields. Let me structure the final answer...
```

---

## RLM vs. Other DSPy Modules

| Module | LLM Writes Code | REPL Persistence | Iterative Refinement | Best For |
|--------|----------------|------------------|---------------------|----------|
| `Predict` | No | N/A | No | Simple extraction and classification |
| `ChainOfThought` | No | N/A | No (single step) | Multi-step reasoning in text |
| `ReAct` | No* | No | Yes (tool loop) | Tool use and multi-step search |
| `ProgramOfThought` | Yes | No (fresh each call) | No | Code generation from descriptions |
| `RLM` | Yes | Yes (across iterations) | Yes | Complex extraction, data analysis, tasks requiring iteration |

\* ReAct can use PythonInterpreter tool, but the REPL state is not preserved across tool calls and the focus is on tool use, not code generation.

---

## Key Takeaways

- **RLM writes Python code** to process inputs, call sub-LLMs for semantic analysis, and build answers. It is not just an LLM answering directly.
- **REPL state persists** across iterations. Variables defined in one step are available in the next. This enables incremental solution building.
- **Sub-LLM calls** from within the RLMs code enable hybrid symbolic-neural computation: code for structure, LM calls for semantics.
- **Trajectory inspection** reveals the RLMs problem-solving strategy. Use `result.trajectory` to debug and understand.
- **Structured output types** (`list[str]`, `dict`) are natural for RLM since the LLM constructs them programmatically.
- **Compute budget** is controlled by `max_iterations` (write-execute cycles) and `max_llm_calls` (total LLM calls including sub-LLM invocations).

### When to Use RLM

- **Complex extraction** from semi-structured text (clinical notes, legal documents, logs).
- **Data analysis** requiring iteration (e.g., compute statistics, then visualize, then summarize).
- **Multi-step reasoning** where intermediate state matters (the LLM needs to remember what it computed earlier).
- **Tasks that benefit from code** for precision (regex, arithmetic, string manipulation) combined with LLM calls for semantics.

### When NOT to Use RLM

- Simple classification or extraction (use `Predict` or `ChainOfThought`).
- Tasks with no benefit from code execution.
- Latency-sensitive applications (RLM makes multiple LLM calls).
- Cost-sensitive applications (each iteration is an LLM call plus code execution).
