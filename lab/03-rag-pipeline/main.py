"""
RAG pipeline with optimization.

Covers:
- Modular RAG design with dspy.Module
- ColBERTv2 sparse retrieval
- BootstrapFewShot optimizer for prompt tuning
- Custom metric for evaluation
- Train/dev split
"""

from pathlib import Path
from dotenv import load_dotenv
import dspy
from dspy.datasets import HotPotQA

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

lm = dspy.LM("deepseek/deepseek-v4-flash")
dspy.configure(lm=lm)

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
dataset = HotPotQA(train_seed=2024, train_size=200, eval_size=50)
trainset = [dspy.Example(**x).with_inputs("question") for x in dataset.train]
devset = [dspy.Example(**x).with_inputs("question") for x in dataset.dev]

# ---------------------------------------------------------------------------
# Define RAG module
# ---------------------------------------------------------------------------

class RAG(dspy.Module):
    def __init__(self, k=3):
        self.k = k
        self.retrieve = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question, k=self.k)
        return self.generate(context=context, question=question)

# ---------------------------------------------------------------------------
# Evaluate unoptimized baseline
# ---------------------------------------------------------------------------

def gold_answer_metric(example, prediction, trace=None):
    return example.answer == prediction.answer

rag = RAG()
evaluator = dspy.Evaluate(devset=devset, metric=gold_answer_metric, num_threads=8)
baseline = evaluator(rag).score / 100.0
print(f"Baseline accuracy: {baseline:.2%}")

# ---------------------------------------------------------------------------
# Optimize with BootstrapFewShot
# ---------------------------------------------------------------------------
optimizer = dspy.BootstrapFewShot(metric=gold_answer_metric, max_bootstrapped_demos=4, max_labeled_demos=4)
optimized_rag = optimizer.compile(rag, trainset=trainset)

optimized_score = evaluator(optimized_rag).score / 100.0
print(f"Optimized accuracy: {optimized_score:.2%}")
print(f"Improvement: {optimized_score - baseline:+.2%}")
