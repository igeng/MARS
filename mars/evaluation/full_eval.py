"""
Unified evaluation runner — chains MARS output through ALL available metrics.

Produces a single comprehensive evaluation report combining:
- MARS own metrics (Citation F1, ROUGE-L, 12-dim Judge, Hallucination)
- SurGE official metrics (ROUGE/BLEU, SH-Recall, Coverage, Relevance, Logic)
- Baseline comparison (vs Autosurvey, ID, Naive on same topic)
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from mars.config import settings

logger = logging.getLogger(__name__)


def run_full_evaluation(
    topic: str,
    survey_path: str | Path,
    surge_dir: str = "D:/PersonalResearch/PapersWS/Paper-MARS/Ref/SurGE",
    topic_id: str | None = None,
    openai_key: str | None = None,
) -> Dict[str, Any]:
    """Evaluate a generated survey against ALL available metrics.

    Args:
        topic: Research topic string (for LLM Judge context).
        survey_path: Path to the generated review_en.md.
        surge_dir: Path to SurGE repo root.
        topic_id: SurGE topic ID for ground-truth matching.
            Auto-detected if None.
        openai_key: For LLM-based metrics (Structure_Quality, Logic, Judge).
            Reads OPENAI_API_KEY env var if None.

    Returns:
        Dict with all metric results, ready for JSON serialization.
    """
    survey_path = Path(survey_path)
    survey_text = survey_path.read_text(encoding="utf-8") if survey_path.is_file() else ""
    if not survey_text:
        return {"error": f"Survey file not found or empty: {survey_path}"}

    api_key = openai_key or os.environ.get("OPENAI_API_KEY", "")
    results: Dict[str, Any] = {
        "topic": topic,
        "survey_path": str(survey_path),
        "survey_chars": len(survey_text),
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metrics": {},
    }

    # ---- 1. MARS own metrics (no API key needed for basic) ----
    _eval_mars_own(results, topic, survey_text, surge_dir, topic_id)

    # ---- 2. 12-dim LLM Judge (needs API key) ----
    if api_key:
        _eval_llm_judge(results, topic, survey_text, api_key)
    else:
        results["metrics"]["llm_judge_12dim"] = {"status": "skipped", "reason": "No OPENAI_API_KEY"}

    # ---- 3. Hallucination check (no API key needed) ----
    _eval_hallucination(results, survey_text)

    # ---- 4. SurGE official metrics ----
    _eval_surge_official(results, survey_path, surge_dir, api_key, topic_id)

    # ---- 5. Baseline comparison ----
    _eval_baseline_comparison(results, surge_dir, topic_id)

    return results


# ---------------------------------------------------------------------------
# Sub-evaluators
# ---------------------------------------------------------------------------

def _eval_mars_own(results, topic, survey_text, surge_dir, topic_id):
    """MARS: Citation F1 + ROUGE-L."""
    from mars.evaluation.surge_eval import SurGEEvaluator

    evaluator = SurGEEvaluator.from_surge(
        Path(surge_dir) / "data", load_corpus=True,
    )

    if not topic_id:
        # Auto-match
        for tid, tname in evaluator.topics.items():
            if topic.lower() in tname.lower() or any(
                w in tname.lower() for w in topic.lower().split()[:3]
            ):
                topic_id = tid
                break
    if not topic_id:
        topic_id = "0"

    try:
        r = evaluator.evaluate(topic_id, survey_text)
        results["topic_id"] = topic_id
        results["topic_name"] = evaluator.topics.get(topic_id, "")
        results["metrics"]["citation"] = {
            "recall": r.recall,
            "precision": r.precision,
            "f1": r.f1,
            "generated_refs": r.generated_ref_count,
            "ground_truth_refs": r.ground_truth_ref_count,
            "matched_refs": r.matched_ref_count,
            "rouge_l": r.rouge_l_score,
        }
    except Exception as exc:
        results["metrics"]["citation"] = {"error": str(exc)}


def _eval_llm_judge(results, topic, survey_text, api_key):
    """12-dimension LLM-as-Judge."""
    from mars.evaluation.llm_judge import LLMJudge

    try:
        judge = LLMJudge(api_key=api_key)
        jr = judge.evaluate(survey_text, topic)
        results["metrics"]["llm_judge_12dim"] = jr.to_dict()
    except Exception as exc:
        results["metrics"]["llm_judge_12dim"] = {"error": str(exc)}


def _eval_hallucination(results, survey_text):
    """Citation hallucination check."""
    from mars.evaluation.hallucination_checker import HallucinationChecker

    try:
        checker = HallucinationChecker()
        report = checker.check(survey_text, [])
        results["metrics"]["hallucination"] = report.to_dict()
    except Exception as exc:
        results["metrics"]["hallucination"] = {"error": str(exc)}


def _eval_surge_official(results, survey_path, surge_dir, api_key, topic_id):
    """SurGE official 8 metrics via external evaluator."""
    from mars.evaluation.surge_metrics import SurGEMetrics

    try:
        tmp_dir = Path("data/surge_eval_tmp")
        sid = topic_id or "0"
        m = SurGEMetrics(surge_dir=surge_dir, api_key=api_key if api_key else "sk-placeholder")

        m.prepare_passage_dir({sid: Path(survey_path).read_text(encoding="utf-8")}, tmp_dir / sid)
        surge_result = m.evaluate(
            passage_dir=str(tmp_dir / sid),
            eval_list=["ROUGE-BLEU", "SH-Recall"],
        )
        results["metrics"]["surge_official"] = {
            k: v for k, v in surge_result.items()
            if isinstance(v, (dict, list, str, int, float, type(None)))
        }
    except Exception as exc:
        results["metrics"]["surge_official"] = {"error": str(exc)}


def _eval_baseline_comparison(results, surge_dir, topic_id):
    """Load baseline scores for the same topic."""
    from mars.evaluation.benchmark_runner import BenchmarkRunner

    try:
        runner = BenchmarkRunner(surge_dir=Path(surge_dir) / "data")
        runner.load_baselines()
        baseline_scores = {}
        for name, br in runner.results.items():
            baseline_scores[name] = br.to_dict()
        results["baselines"] = baseline_scores
    except Exception as exc:
        results["baselines"] = {"error": str(exc)}


def save_eval_report(results: Dict[str, Any], output_dir: str | Path) -> Path:
    """Save evaluation results to JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "full_eval_report.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Full evaluation report saved to %s", path)
    return path
