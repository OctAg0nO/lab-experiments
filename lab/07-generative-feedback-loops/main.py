"""
Generative Feedback Loops (GFL) — DSPy's zero-gradient optimization engine.

The GFL mechanism:
  1. Trace Collection     — run the program forward, capture inputs/outputs/intermediates
  2. Feedback Generation  — evaluate each trace against a metric (scalar or textual)
  3. Program Update       — update instructions/demonstrations based on feedback
  4. Repeat               — loop until convergence or budget exhaustion

No gradients. No weight updates. The LLM's generative capability drives
optimization, and the metric provides selection pressure.

Reference: https://octagono.org/blog/dspy-generative-feedback-loops/
"""

from pathlib import Path
from dotenv import load_dotenv
import dspy

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

lm = dspy.LM("deepseek/deepseek-v4-flash")
dspy.configure(lm=lm)

# ============================================================================
# Task: Multi-label intent classification
# ============================================================================

TRAIN_DATA = [
    ("Book a flight to Tokyo for next week", "booking", "high"),
    ("What's the weather in Paris?", "inquiry", "high"),
    ("I need to cancel my hotel reservation", "cancellation", "high"),
    ("Show me my recent transactions", "account", "high"),
    ("Turn on the living room lights", "command", "high"),
    ("Tell me a joke", "entertainment", "medium"),
    ("Is my order shipped yet?", "tracking", "high"),
    ("Play some jazz music", "command", "medium"),
    ("Transfer $500 to my savings", "transaction", "high"),
    ("Who is the CEO of Acme Corp?", "inquiry", "medium"),
    ("Set an alarm for 7 AM", "command", "high"),
    ("I want to return a product", "support", "high"),
    ("What movies are playing nearby?", "inquiry", "medium"),
    ("Schedule a meeting for Friday", "booking", "high"),
    ("Lock the front door", "command", "high"),
    ("What's my account balance?", "account", "high"),
    ("Find me a good Italian restaurant", "inquiry", "medium"),
    ("Remind me to call John at 3 PM", "command", "high"),
    ("I'm having trouble logging in", "support", "high"),
    ("Translate hello to Spanish", "command", "medium"),
]

DEV_DATA = [
    ("Order a pizza for delivery", "booking", "high"),
    ("How far is the Moon from Earth?", "inquiry", "medium"),
    ("Delete my account", "cancellation", "high"),
    ("Dim the bedroom lights to 50%", "command", "high"),
    ("Where is my package?", "tracking", "high"),
]


def build_examples(data):
    return [
        dspy.Example(query=q, intent=i, confidence=c).with_inputs("query")
        for q, i, c in data
    ]


trainset = build_examples(TRAIN_DATA)
devset = build_examples(DEV_DATA)


# ============================================================================
# Step 1: Define the program
# ============================================================================

class ClassifyIntent(dspy.Signature):
    """Classify user query intent and confidence level."""
    query: str = dspy.InputField()
    intent: str = dspy.OutputField()
    confidence: str = dspy.OutputField()


# ============================================================================
# Step 2: Define metrics
# ============================================================================

def intent_metric(example, prediction, trace=None):
    """Exact match on intent. Confidence must be one of low/medium/high."""
    intent_ok = example.intent == prediction.intent
    confidence_ok = prediction.confidence in ("low", "medium", "high")
    return intent_ok and confidence_ok


def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """GEPA metric: accepts 5 required args, returns float score."""
    intent_ok = gold.intent == pred.intent
    confidence_ok = pred.confidence in ("low", "medium", "high")
    return float(intent_ok and confidence_ok)


# Shared Evaluate instance. __call__ returns EvaluationResult whose .score
# is a percentage (0-100). We divide by 100 to get a 0.0-1.0 fraction.
evaluator = dspy.Evaluate(
    devset=devset,
    metric=intent_metric,
    num_threads=4,
    display_progress=True,
)


def eval_score(program) -> float:
    """Evaluate program, return fraction 0.0–1.0."""
    return evaluator(program).score / 100.0


# ============================================================================
# Step 3: Baseline
# ============================================================================

print("=" * 65)
print("GENERATIVE FEEDBACK LOOPS — DSPy Optimization Demo")
print("=" * 65)

program = dspy.ChainOfThought(ClassifyIntent)
baseline_score = eval_score(program)
print(f"\nBaseline accuracy: {baseline_score:.0%}\n")

pred = program(query="Cancel my dinner reservation")
print(f"  Baseline prediction: intent={pred.intent}, confidence={pred.confidence}")
print()

# ============================================================================
# Step 4: BootstrapFewShot
# ============================================================================
# Mechanism: run teacher on training examples, collect traces, keep only
# traces where metric succeeds, attach as student demonstrations.

print("\u2014" * 65)
print("1. BootstrapFewShot (Trace \u2192 Demo pipeline)")
print("\u2014" * 65)

bs = dspy.BootstrapFewShot(
    metric=intent_metric,
    max_bootstrapped_demos=6,
    max_labeled_demos=4,
)
bs_program = bs.compile(dspy.ChainOfThought(ClassifyIntent), trainset=trainset)
bs_score = eval_score(bs_program)
print(f"   BootstrapFewShot accuracy: {bs_score:.0%}  (\u0394 {bs_score - baseline_score:+.0%})")
print(f"   Demos attached: {len(bs_program.demos)}")
print()

# ============================================================================
# Step 5: MIPROv2
# ============================================================================
# (1) bootstrap candidate demos (2) propose instruction variants via
# GroundedProposer (3) Bayesian search over instruction x demo space.

print("\u2014" * 65)
print("2. MIPROv2 (Instruction + Demo joint optimization)")
print("\u2014" * 65)

mipro = dspy.MIPROv2(
    metric=intent_metric,
    auto="light",
    num_threads=4,
)
mipro_program = mipro.compile(
    dspy.ChainOfThought(ClassifyIntent),
    trainset=trainset,
)
mipro_score = eval_score(mipro_program)
print(f"   MIPROv2 accuracy: {mipro_score:.0%}  (\u0394 {mipro_score - baseline_score:+.0%})")

optimized_sig = mipro_program.predictors()[0].signature
print(f"   Optimized instructions: {optimized_sig.instructions}")
print()

# ============================================================================
# Step 6: GEPA
# ============================================================================
# Reads full execution traces, diagnoses failures, proposes targeted
# instruction mutations, selects via Pareto frontier. Uses a separate
# reflection LM for the mutation engine.

print("\u2014" * 65)
print("3. GEPA (Reflective Prompt Evolution)")
print("\u2014" * 65)

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
gepa_score = eval_score(gepa_program)
print(f"   GEPA accuracy: {gepa_score:.0%}  (\u0394 {gepa_score - baseline_score:+.0%})")
print()

# ============================================================================
# Step 7: Sequential — GEPA then BootstrapFewShot
# ============================================================================

print("\u2014" * 65)
print("4. Sequential: GEPA \u2192 BootstrapFewShot")
print("\u2014" * 65)

seq_program = dspy.ChainOfThought(ClassifyIntent)
seq_gepa = dspy.GEPA(
    metric=gepa_metric,
    max_full_evals=1,
    reflection_lm=dspy.LM("deepseek/deepseek-v4-flash"),
    num_threads=4,
)
seq_program = seq_gepa.compile(seq_program, trainset=trainset[:10])
seq_bs = dspy.BootstrapFewShot(
    metric=intent_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=2,
)
seq_program = seq_bs.compile(seq_program, trainset=trainset)
seq_score = eval_score(seq_program)
print(f"   Sequential accuracy: {seq_score:.0%}  (\u0394 {seq_score - baseline_score:+.0%})")
print()

# ============================================================================
# Step 8: Teacher / Student distillation
# ============================================================================
# A strong teacher LM (DeepSeek) generates bootstrapped demonstrations,
# which are then attached to a weaker student LM (Gemma 4 via Ollama).
# The student never sees the teacher's weights — only the demonstrations.

print("\u2014" * 65)
print("5. Teacher \u2192 Student distillation (DeepSeek \u2192 Gemma 4)")
print("\u2014" * 65)

student_lm = dspy.LM("ollama_chat/gemma4")

# Student using the weak model alone (no teacher)
student_alone = dspy.ChainOfThought(ClassifyIntent)
student_alone.lm = student_lm
alone_score = eval_score(student_alone)
print(f"   Student alone (Gemma 4):       {alone_score:.0%}")

# BootstrapFewShot with teacher (DeepSeek generates demos, Gemma 4 uses them)
teacher = dspy.ChainOfThought(ClassifyIntent)
student = dspy.ChainOfThought(ClassifyIntent)
student.lm = student_lm

ts_bs = dspy.BootstrapFewShot(
    metric=intent_metric,
    max_bootstrapped_demos=6,
    max_labeled_demos=4,
)
distilled = ts_bs.compile(student, teacher=teacher, trainset=trainset)
distilled_score = eval_score(distilled)
print(f"   Distilled (teacher demos):     {distilled_score:.0%}  (\u0394 {distilled_score - alone_score:+.0%})")

# The teacher's own performance for reference
teacher_score = eval_score(teacher)
print(f"   Teacher (DeepSeek) reference:  {teacher_score:.0%}")
print()

# ============================================================================
# Summary
# ============================================================================

print("=" * 65)
print("GFL OPTIMIZER COMPARISON")
print("=" * 65)
print(f"{'Baseline (no optimization)':40s} {baseline_score:.0%}")
print(f"{'BootstrapFewShot (trace \u2192 demos)':40s} {bs_score:.0%}")
print(f"{'MIPROv2 (instructions + demos)':40s} {mipro_score:.0%}")
print(f"{'GEPA (reflective evolution)':40s} {gepa_score:.0%}")
print(f"{'GEPA \u2192 BootstrapFewShot (sequential)':40s} {seq_score:.0%}")
print(f"{'Teacher \u2192 Student distillation':40s} {distilled_score:.0%}")
print()

# ============================================================================
# Conceptual breakdown
# ============================================================================

print("\u2014" * 65)
print("HOW THE GENERATIVE FEEDBACK LOOP WORKS")
print("\u2014" * 65)
print("""
     \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510
     \u2502                 GENERATIVE FEEDBACK LOOP             \u2502
     \u2502                                                     \u2502
     \u2502   \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510    \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510    \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510     \u2502
     \u2502   \u2502  TRACE   \u2502    \u2502 FEEDBACK \u2502    \u2502 PROGRAM  \u2502     \u2502
     \u2502   \u2502 COLLECT  \u2502\u2500\u2500\u2500\u2192\u2502 GENERATE \u2502\u2500\u2500\u2500\u2192\u2502  UPDATE  \u2502     \u2502
     \u2502   \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518    \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518    \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518     \u2502
     \u2502         \u2191                               \u2502          \u2502
     \u2502         \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 LOOP \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518          \u2502
     \u2502                                                     \u2502
     \u2502   \u2022 No gradients              \u2022 Program-level       \u2502
     \u2502   \u2022 No weight updates         \u2022 Prompt + demos only \u2502
     \u2502   \u2022 LLM generates signal      \u2022 Metric selects      \u2502
     \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518

  BootstrapFewShot:  trace \u2192 keep passing demos \u2192 attach to student
  MIPROv2:          trace \u2192 propose instructions \u2192 Bayesian search
  GEPA:             trace \u2192 diagnose failure \u2192 mutate \u2192 Pareto select
  Sequential:       GEPA (prompts) \u2192 BootstrapFewShot (demos)
""")
