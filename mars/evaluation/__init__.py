"""
MARS Evaluation Module — automated quality assessment for generated surveys.

Provides three core capabilities:

1. **Benchmark evaluation** (:mod:`mars.evaluation.surge_eval`)
   — Compute Citation Recall / Precision / F1 against ground-truth references.

2. **LLM-as-Judge scoring** (:mod:`mars.evaluation.llm_judge`)
   — 6-dimension quality rubric scored by an independent LLM.

3. **Citation hallucination detection** (:mod:`mars.evaluation.hallucination_checker`)
   — Verify that every cited paper exists and supports its associated claim.

Usage::

    from mars.evaluation import SurGEEvaluator, LLMJudge, HallucinationChecker

    surge = SurGEEvaluator(data_dir="data/surge")
    judge = LLMJudge()
    checker = HallucinationChecker()

    # After a MARS run:
    metrics = surge.evaluate(topic, generated_refs=...)
    scores = judge.evaluate(generated_survey=..., topic=...)
    hallu_report = checker.check(generated_survey=..., paper_db=...)
"""

from mars.evaluation.metrics import compute_citation_metrics
from mars.evaluation.surge_eval import SurGEEvaluator, SurGEResult
from mars.evaluation.llm_judge import LLMJudge, JudgeResult
from mars.evaluation.hallucination_checker import HallucinationChecker, HallucinationReport

__all__ = [
    "compute_citation_metrics",
    "SurGEEvaluator",
    "SurGEResult",
    "LLMJudge",
    "JudgeResult",
    "HallucinationChecker",
    "HallucinationReport",
]
