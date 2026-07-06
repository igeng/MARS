"""
SurGE evaluation metrics wrapper — calls the official SurGE evaluator.

This module wraps SurGE's evaluation code (``SurGE/src/``) as an external
library.  SurGE's *evaluation metrics* (SH-Recall, Structure_Quality, Coverage,
Relevance, Logic, ROUGE, BLEU) are directly invoked via this wrapper.

SurGE's *generation methods* (baselines/Autosurvey, baselines/ID, etc.) are
NOT integrated — they remain comparison targets.

Usage::

    from mars.evaluation.surge_metrics import SurGEMetrics

    metrics = SurGEMetrics(surge_dir="D:/.../SurGE")
    result = metrics.evaluate(passage_dir="output/my_run/")
    print(result["ROUGE-BLEU"])
    print(result["SH-Recall"])
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SurGEMetrics:
    """Evaluate MARS output using SurGE's official evaluation pipeline.

    Parameters:
        surge_dir: Path to the SurGE repository root (contains ``src/``
            and ``data/``).
        device: GPU device ID (default "0"; use "-1" for CPU).
        api_key: OpenAI API key for LLM-based metrics (Structure_Quality).
            When None, reads from ``OPENAI_API_KEY`` env var.
    """

    # SurGE metric names as used in --eval_list
    ALL_METRICS = [
        "ROUGE-BLEU",
        "SH-Recall",
        "Structure_Quality",
        "Coverage",
        "Relevance-Paper",
        "Relevance-Section",
        "Relevance-Sentence",
        "Logic",
    ]

    def __init__(
        self,
        surge_dir: str = "D:/PersonalResearch/PapersWS/Paper-MARS/Ref/SurGE",
        device: str = "0",
        api_key: Optional[str] = None,
    ):
        self._surge_dir = Path(surge_dir)
        self._data_dir = self._surge_dir / "data"
        self._device = device
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

        # Register SurGE src/ on sys.path
        src_path = str(self._surge_dir / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        self._evaluator = None

    @property
    def evaluator(self):
        """Lazy-load the SurGE evaluator."""
        if self._evaluator is None:
            from evaluator import SurGEvaluator as _SurGEvaluator

            self._evaluator = _SurGEvaluator(
                device=self._device,
                survey_path=str(self._data_dir / "surveys.json"),
                corpus_path=str(self._data_dir / "corpus.json"),
                api_key=self._api_key,
            )
            logger.info("SurGE evaluator initialised (device=%s).", self._device)
        return self._evaluator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        passage_dir: str | Path,
        eval_list: List[str] | None = None,
        save_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        """Run SurGE evaluation on a directory of generated surveys.

        Args:
            passage_dir: Directory containing one subdirectory per survey_id,
                each with the generated survey file.
            eval_list: Metrics to compute. Default: all 8 metrics.
            save_path: Path to save the evaluation result JSON.

        Returns:
            Dict with per-metric results.
        """
        metrics = eval_list or ["ALL"]
        result = self.evaluator.eval_all(
            passage_dir=str(passage_dir),
            eval_list=metrics,
            save_path=str(save_path) if save_path else None,
        )
        return result

    def evaluate_single(
        self,
        survey_path: str | Path,
        eval_list: List[str] | None = None,
    ) -> Dict[str, Any]:
        """Evaluate a single generated survey file.

        Args:
            survey_path: Path to the generated survey file.
            eval_list: Metrics to compute. Default: all 8 metrics.

        Returns:
            Dict with per-metric results.
        """
        metrics = eval_list or ["ALL"]
        result = self.evaluator.single_eval(
            passage_path=str(survey_path),
            eval_list=metrics,
        )
        return result

    # ------------------------------------------------------------------
    # Convenience: prepare MARS output for SurGE evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def prepare_passage_dir(
        survey_map: Dict[str, str],       # survey_id → markdown text
        output_dir: str | Path,
    ) -> Path:
        """Write generated surveys into SurGE's expected directory format.

        SurGE expects: ``<passage_dir>/<survey_id>/<survey_file>.md``

        Args:
            survey_map: ``{survey_id: survey_markdown_text}`` mapping.
            output_dir: Where to create the passage directory.

        Returns:
            Path to the prepared passage directory.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for survey_id, text in survey_map.items():
            sid_dir = out / str(survey_id)
            sid_dir.mkdir(parents=True, exist_ok=True)
            (sid_dir / "generated_survey.md").write_text(
                text, encoding="utf-8",
            )
        logger.info(
            "Prepared %d surveys for SurGE evaluation in %s",
            len(survey_map), out,
        )
        return out

    @staticmethod
    def list_metrics() -> List[str]:
        """Return all available SurGE metric names."""
        return list(SurGEMetrics.ALL_METRICS)
