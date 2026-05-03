"""
Recursive Language Model (RLM) — experimental DSPy module.

The LLM writes Python code to examine data, call sub-LLMs for semantic
analysis, and iteratively build answers. The REPL persists state between
iterations.

Reference: Zhang, Kraska, Khattab (2025)
https://arxiv.org/abs/2501.00000 (Recursive Language Models)
"""

from pathlib import Path
from dotenv import load_dotenv
import dspy

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

lm = dspy.LM("deepseek/deepseek-v4-flash")
dspy.configure(lm=lm)

# ---------------------------------------------------------------------------
# RLM for structured extraction from text blocks
# The LLM will write code to search through text, call sub-LLMs, and
# iteratively build the final answer.
# ---------------------------------------------------------------------------

text_block = """
Patient: John Doe (MRN: 88472)
Date of Admission: 2025-11-12
Attending: Dr. Sarah Chen

History: 67-year-old male presents with acute onset chest pain radiating
to left arm, shortness of breath, and diaphoresis. Symptoms began
approximately 2 hours prior to arrival.

Vitals on Admission:
- BP: 145/92 mmHg
- HR: 108 bpm
- Temp: 37.2°C
- O2 Sat: 94% on room air

Labs:
- Troponin I: 3.42 ng/mL (elevated)
- CK-MB: 24 U/L (elevated)
- LDL: 168 mg/dL
- HbA1c: 7.1%

ECG: ST elevation in leads V2-V4 consistent with anterior STEMI.

Assessment: Anterior ST-elevation myocardial infarction (STEMI).
Patient taken emergently to cath lab. 100% occlusion of proximal LAD
identified. Drug-eluting stent placed with TIMI 3 flow post-procedure.

Discharge Plan:
- Start Aspirin 81 mg daily, Clopidogrel 75 mg daily
- Atorvastatin 80 mg daily
- Metoprolol 25 mg BID
- Cardiac rehab referral
- Follow-up with cardiology in 2 weeks
"""

print("=== RLM: Clinical Note Extraction ===")
print(f"Processing {len(text_block)} character clinical note...\n")

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

# ---------------------------------------------------------------------------
# Inspect the RLM trajectory (what code the LLM wrote)
# ---------------------------------------------------------------------------
if hasattr(result, "trajectory"):
    print(f"\nRLM trajectory: {len(result.trajectory)} steps")
    for i, step in enumerate(result.trajectory):
        print(f"  Step {i+1}: {step.get('reasoning', '')[:120]}...")
