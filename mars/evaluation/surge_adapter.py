"""
SurGE adapter — converts the official SurGE dataset into MARS's evaluation format.

Reads ``surveys.json`` (205 ground-truth surveys) and optionally ``corpus.json``
(1.09M papers) from the SurGE dataset, producing ``topics`` + ``ground_truth_refs``
that :class:`SurGEEvaluator` can consume directly.

Usage::

    from mars.evaluation.surge_adapter import SurGEAdapter

    adapter = SurGEAdapter(surge_dir="D:/.../SurGE/data")
    adapter.export_to("data/surge")  # writes topics.json + ground_truth_refs.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class SurGEAdapter:
    """Bridge between the official SurGE dataset and MARS's evaluator format.

    Parameters:
        surge_dir: Path to the SurGE ``data/`` directory containing
            ``surveys.json`` and (optionally) ``corpus.json``.
        load_corpus: When True and ``corpus.json`` exists, build a
            doc_id → title lookup for paper-level citation evaluation.
    """

    def __init__(self, surge_dir: str | Path, load_corpus: bool = True):
        self._surge_dir = Path(surge_dir)
        self._surveys: List[Dict[str, Any]] = []
        self._corpus_lookup: Dict[int, Dict[str, Any]] = {}
        self._has_corpus = False

        # Load surveys
        surveys_path = self._surge_dir / "surveys.json"
        if surveys_path.is_file():
            raw = json.loads(surveys_path.read_text(encoding="utf-8"))
            self._surveys = raw if isinstance(raw, list) else list(raw.values())
            logger.info("Loaded %d surveys from %s", len(self._surveys), surveys_path)
        else:
            logger.warning("surveys.json not found at %s", surveys_path)

        # Optionally load corpus
        if load_corpus:
            corpus_path = self._surge_dir / "corpus.json"
            if corpus_path.is_file():
                logger.info("Loading corpus from %s (this may take a moment)...", corpus_path)
                # Corpus is a dict: {doc_id: {...}} or list of dicts
                raw_corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
                if isinstance(raw_corpus, dict):
                    # Convert string keys (JSON) to int
                    self._corpus_lookup = {
                        int(k): v for k, v in raw_corpus.items()
                    }
                elif isinstance(raw_corpus, list):
                    for paper in raw_corpus:
                        doc_id = paper.get("doc_id")
                        if doc_id is not None:
                            self._corpus_lookup[int(doc_id)] = paper
                self._has_corpus = True
                logger.info("Corpus loaded: %d papers", len(self._corpus_lookup))
            else:
                logger.info(
                    "corpus.json not found at %s — paper-level citation metrics "
                    "will be unavailable. Download corpus.json from SurGE Google "
                    "Drive for full evaluation.",
                    corpus_path,
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def survey_count(self) -> int:
        return len(self._surveys)

    @property
    def has_corpus(self) -> bool:
        return self._has_corpus

    @property
    def topic_count(self) -> int:
        return len(self.get_topics())

    def get_topics(self) -> Dict[str, str]:
        """Return ``{survey_id: survey_title}`` mapping for all 205 surveys."""
        return {
            str(s["survey_id"]): s["survey_title"]
            for s in self._surveys
        }

    def get_ground_truth_refs(self, survey_id: str) -> List[Dict[str, str]]:
        """Return ground-truth references for a survey.

        When corpus is available, returns full ``{title, doi, year}`` dicts.
        When corpus is unavailable, returns ``{doc_id}``-only entries (paper-level
        citation metrics disabled; topic-level evaluation still works).
        """
        survey = self._find_survey(survey_id)
        if survey is None:
            return []

        doc_ids = survey.get("all_cites", [])
        refs: List[Dict[str, str]] = []

        for doc_id in doc_ids:
            doc_id = int(doc_id)
            if self._has_corpus and doc_id in self._corpus_lookup:
                paper = self._corpus_lookup[doc_id]
                refs.append({
                    "title": paper.get("Title", ""),
                    "doi": "",  # SurGE corpus doesn't include DOIs
                    "year": str(paper.get("Year", "")),
                    "doc_id": str(doc_id),
                })
            elif self._has_corpus:
                refs.append({
                    "title": f"doc_{doc_id}",  # fallback: ID not in corpus
                    "doi": "",
                    "year": "",
                    "doc_id": str(doc_id),
                })
            else:
                # No corpus: store doc_id only for count-based metrics
                refs.append({
                    "title": f"doc_{doc_id}",
                    "doi": "",
                    "year": "",
                    "doc_id": str(doc_id),
                })

        return refs

    def get_survey_metadata(self, survey_id: str) -> Optional[Dict[str, Any]]:
        """Return full metadata for a survey (category, authors, abstract, structure)."""
        survey = self._find_survey(survey_id)
        if survey is None:
            return None
        return {
            "survey_id": survey["survey_id"],
            "survey_title": survey["survey_title"],
            "authors": survey.get("authors", []),
            "year": survey.get("year", ""),
            "category": survey.get("category", ""),
            "abstract": survey.get("abstract", ""),
            "structure": survey.get("structure", []),
            "cite_count": len(survey.get("all_cites", [])),
        }

    def get_structure(self, survey_id: str) -> List[Dict[str, Any]]:
        """Return the ground-truth hierarchical structure for a survey."""
        survey = self._find_survey(survey_id)
        if survey is None:
            return []
        return survey.get("structure", [])

    def search_topic(self, keyword: str) -> List[Dict[str, Any]]:
        """Search for SurGE surveys matching a keyword (in title or category).

        Returns matching surveys with survey_id, title, category, and cite_count.
        """
        keyword_lower = keyword.lower()
        matches = []
        for s in self._surveys:
            title = s.get("survey_title", "")
            category = s.get("category", "")
            if keyword_lower in title.lower() or keyword_lower in category.lower():
                matches.append({
                    "survey_id": s["survey_id"],
                    "survey_title": title,
                    "category": category,
                    "cite_count": len(s.get("all_cites", [])),
                    "year": s.get("year", ""),
                })
        return matches

    def export_to(self, output_dir: str | Path) -> Path:
        """Export SurGE data into MARS's evaluator format.

        Writes ``topics.json`` and ``ground_truth_refs.json`` into *output_dir*,
        ready for use by :class:`SurGEEvaluator`.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        topics = self.get_topics()
        gt_refs: Dict[str, List[Dict[str, str]]] = {}
        for survey_id in topics:
            gt_refs[survey_id] = self.get_ground_truth_refs(survey_id)

        (out / "topics.json").write_text(
            json.dumps(topics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (out / "ground_truth_refs.json").write_text(
            json.dumps(gt_refs, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(
            "Exported %d topics + %d ground-truth ref sets to %s",
            len(topics), len(gt_refs), out,
        )
        return out

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find_survey(self, survey_id: str) -> Optional[Dict[str, Any]]:
        sid = int(survey_id)
        for s in self._surveys:
            if s["survey_id"] == sid:
                return s
        return None
