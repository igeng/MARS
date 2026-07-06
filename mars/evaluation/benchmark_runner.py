"""
Batch benchmark runner — MARS vs SurGE baselines on 41 test topics.

Evaluates all methods using MARS's own metrics (Citation F1 + ROUGE-L) for
a fair, code-independent comparison.  SurGE's metrics (SH-Recall,
Structure_Quality, Coverage) are computed externally via SurGE's own evaluator
and are NOT integrated into MARS.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from mars.config import settings
from mars.evaluation.metrics import compute_citation_metrics, rouge_l
from mars.evaluation.surge_eval import SurGEEvaluator, SurGEResult

logger = logging.getLogger(__name__)

_SURGE_TEST_IDS = [str(i) for i in range(41)]


@dataclass
class BenchmarkResult:
    """Aggregated benchmark scores for one method."""
    method: str
    avg_recall: float = 0.0
    avg_precision: float = 0.0
    avg_f1: float = 0.0
    avg_rouge_l: float = 0.0
    per_topic: List[Dict[str, Any]] = field(default_factory=list)
    num_topics: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "num_topics": self.num_topics,
            "avg_recall": round(self.avg_recall, 4),
            "avg_precision": round(self.avg_precision, 4),
            "avg_f1": round(self.avg_f1, 4),
            "avg_rouge_l": round(self.avg_rouge_l, 4),
        }


class BenchmarkRunner:
    """Evaluate MARS + SurGE baselines using MARS's own independent metrics.

    All metrics are computed by MARS code — no SurGE src/ code is imported.
    """

    def __init__(
        self,
        surge_dir: str = "D:/PersonalResearch/PapersWS/Paper-MARS/Ref/SurGE/data",
        output_dir: str = "data/benchmark",
    ):
        self._surge_dir = Path(surge_dir)
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._evaluator: Optional[SurGEEvaluator] = None
        self._results: Dict[str, BenchmarkResult] = {}
        self._gt_titles: Dict[str, Set[str]] = {}  # topic_id → normalized titles

    @property
    def evaluator(self) -> SurGEEvaluator:
        if self._evaluator is None:
            self._evaluator = SurGEEvaluator.from_surge(
                self._surge_dir, load_corpus=True,
            )
        return self._evaluator

    @property
    def test_topics(self) -> List[str]:
        return _SURGE_TEST_IDS

    # ------------------------------------------------------------------
    # Ground-truth title cache
    # ------------------------------------------------------------------

    def _get_gt_titles(self, topic_id: str) -> Set[str]:
        if topic_id not in self._gt_titles:
            from mars.evaluation.metrics import normalize_title
            refs = self.evaluator.get_ground_truth_refs(topic_id)
            self._gt_titles[topic_id] = {
                normalize_title(r.get("title", "")) for r in refs if r.get("title")
            }
        return self._gt_titles[topic_id]

    # ------------------------------------------------------------------
    # Load baselines
    # ------------------------------------------------------------------

    def load_baselines(self) -> None:
        for name, folder in [
            ("Autosurvey", "Autosurvey"),
            ("ID", "ID"),
            ("Naive", "Naive"),
        ]:
            self._load_one_baseline(name, folder)

    def _load_one_baseline(self, name: str, folder: str) -> None:
        baseline_dir = self._surge_dir.parent / "baselines" / folder / "output"
        if not baseline_dir.is_dir():
            logger.warning("%s: directory not found at %s", name, baseline_dir)
            return

        per_topic: List[Dict[str, Any]] = []
        c_recalls, c_precisions, c_f1s, rouges = [], [], [], []

        for topic_id in self.test_topics:
            topic_dir = baseline_dir / topic_id
            if not topic_dir.is_dir():
                continue
            # Baselines may use *.md or extensionless files
            md_files = list(topic_dir.glob("*.md"))
            if not md_files:
                md_files = [f for f in topic_dir.iterdir()
                            if f.is_file() and f.suffix in ("", ".md", ".txt")]
            if not md_files:
                continue

            survey_text = md_files[0].read_text(encoding="utf-8", errors="replace")
            # Use SurGE's own parse_refs + title2docid for accurate matching
            try:
                r = self.evaluator.evaluate(
                    topic_id, survey_text,
                    surge_passage_path=str(md_files[0]),
                )
                c_recalls.append(r.recall)
                c_precisions.append(r.precision)
                c_f1s.append(r.f1)
                rouges.append(r.rouge_l_score or 0.0)
                per_topic.append(r.to_dict())
            except KeyError:
                pass

        if per_topic:
            n = len(per_topic)
            self._results[name] = BenchmarkResult(
                method=name,
                avg_recall=sum(c_recalls) / n,
                avg_precision=sum(c_precisions) / n,
                avg_f1=sum(c_f1s) / n,
                avg_rouge_l=sum(rouges) / n,
                per_topic=per_topic,
                num_topics=n,
            )
            logger.info("%s: %d topics, F1=%.4f, ROUGE-L=%.4f", name, n, self._results[name].avg_f1, self._results[name].avg_rouge_l)

    # ------------------------------------------------------------------
    # Run MARS (slow — LLM-bound)
    # ------------------------------------------------------------------

    def run_mars(
        self,
        corpus_mode: str = "surge",
        topics: List[str] | None = None,
    ) -> None:
        from mars.crews.full_research_crew import run_full_research

        tids = topics or self.test_topics
        per_topic: List[Dict[str, Any]] = []
        c_recalls, c_precisions, c_f1s, rouges = [], [], [], []

        for i, topic_id in enumerate(tids):
            topic_name = self.evaluator.topics.get(topic_id, f"Topic {topic_id}")
            logger.info("[%d/%d] MARS on: %s", i + 1, len(tids), topic_name)
            try:
                run_full_research(topic_name, corpus_mode=corpus_mode)
                review_path = settings.OUTPUT_DIR / "review_en.md"
                if review_path.is_file():
                    survey = review_path.read_text(encoding="utf-8")
                    r = self.evaluator.evaluate(topic_id, survey)
                    c_recalls.append(r.recall)
                    c_precisions.append(r.precision)
                    c_f1s.append(r.f1)
                    rouges.append(r.rouge_l_score or 0.0)
                    per_topic.append(r.to_dict())
                    logger.info("  F1=%.4f", r.f1)
            except Exception as exc:
                logger.error("  Failed: %s", exc)

        if per_topic:
            n = len(per_topic)
            self._results["MARS"] = BenchmarkResult(
                method="MARS",
                avg_recall=sum(c_recalls) / n,
                avg_precision=sum(c_precisions) / n,
                avg_f1=sum(c_f1s) / n,
                avg_rouge_l=sum(rouges) / n,
                per_topic=per_topic,
                num_topics=n,
            )

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------

    def compare(self) -> str:
        if not self._results:
            self.load_baselines()
        h = (
            "| Method | Topics | Recall | Precision | F1 | ROUGE-L |\n"
            "|--------|--------|--------|-----------|----|---------|\n"
        )
        rows = []
        for name in ["Naive", "ID", "Autosurvey", "MARS"]:
            br = self._results.get(name)
            if br:
                rows.append(
                    f"| {name} | {br.num_topics} | {br.avg_recall:.4f} "
                    f"| {br.avg_precision:.4f} | {br.avg_f1:.4f} "
                    f"| {br.avg_rouge_l:.4f} |"
                )
        return h + "\n".join(rows)

    def compare_csv(self) -> str:
        """Return CSV for external plotting."""
        lines = ["method,topics,recall,precision,f1,rouge_l"]
        for name in ["Naive", "ID", "Autosurvey", "MARS"]:
            br = self._results.get(name)
            if br:
                lines.append(
                    f"{name},{br.num_topics},{br.avg_recall:.4f},"
                    f"{br.avg_precision:.4f},{br.avg_f1:.4f},{br.avg_rouge_l:.4f}"
                )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_results(self) -> Path:
        summary = {m: br.to_dict() for m, br in self._results.items()}
        (self._output_dir / "benchmark_results.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        (self._output_dir / "benchmark_comparison.md").write_text(
            f"# MARS vs SurGE Baselines\n\n{self.compare()}\n", encoding="utf-8",
        )
        (self._output_dir / "benchmark_comparison.csv").write_text(
            self.compare_csv(), encoding="utf-8",
        )
        logger.info("Saved to %s", self._output_dir)
        return self._output_dir

    @property
    def results(self) -> Dict[str, BenchmarkResult]:
        return dict(self._results)
