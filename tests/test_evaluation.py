"""
Tests for the mars.evaluation module (SurGE, LLM Judge, Hallucination Checker).
"""

from __future__ import annotations

import pytest


# ============================================================================
# Metrics tests
# ============================================================================

class TestCitationMetrics:
    def test_perfect_match(self) -> None:
        from mars.evaluation.metrics import compute_citation_metrics

        gen = ["Paper A", "Paper B", "Paper C"]
        gt = ["Paper A", "Paper B", "Paper C"]
        result = compute_citation_metrics(gen, gt)
        assert result["recall"] == 1.0
        assert result["precision"] == 1.0
        assert result["f1"] == 1.0
        assert result["matched_count"] == 3

    def test_partial_match(self) -> None:
        from mars.evaluation.metrics import compute_citation_metrics

        gen = ["Paper A", "Paper B", "Paper X"]
        gt = ["Paper A", "Paper C", "Paper D"]
        result = compute_citation_metrics(gen, gt)
        assert result["recall"] == pytest.approx(1 / 3, abs=0.01)
        assert result["precision"] == pytest.approx(1 / 3, abs=0.01)

    def test_empty_generated(self) -> None:
        from mars.evaluation.metrics import compute_citation_metrics

        result = compute_citation_metrics([], ["Paper A"])
        assert result["recall"] == 0.0
        assert result["precision"] == 0.0
        assert result["f1"] == 0.0

    def test_empty_ground_truth(self) -> None:
        from mars.evaluation.metrics import compute_citation_metrics

        result = compute_citation_metrics(["Paper A"], [])
        assert result["recall"] == 0.0
        assert result["precision"] == 0.0

    def test_fuzzy_match_case_insensitive(self) -> None:
        from mars.evaluation.metrics import compute_citation_metrics

        gen = ["Federated Learning Privacy"]
        gt = ["federated learning privacy"]
        result = compute_citation_metrics(gen, gt, fuzzy=True)
        assert result["recall"] == 1.0

    def test_fuzzy_match_ignores_punctuation(self) -> None:
        from mars.evaluation.metrics import compute_citation_metrics

        gen = ["Deep Learning: A Survey"]
        gt = ["deep learning a survey"]
        result = compute_citation_metrics(gen, gt, fuzzy=True)
        assert result["recall"] == 1.0

    def test_rouge_l_identical(self) -> None:
        from mars.evaluation.metrics import rouge_l

        score = rouge_l("hello world", "hello world")
        assert score == 1.0

    def test_rouge_l_different(self) -> None:
        from mars.evaluation.metrics import rouge_l

        score = rouge_l("hello world", "goodbye mars")
        assert score < 0.5


# ============================================================================
# SurGE evaluator tests
# ============================================================================

class TestSurGEEvaluator:
    def test_construction_loads_builtin_data(self) -> None:
        from mars.evaluation.surge_eval import SurGEEvaluator

        evaluator = SurGEEvaluator(data_dir="data/surge_test_nonexistent")
        assert len(evaluator.topic_ids) == 12
        assert "federated_learning_privacy" in evaluator.topic_ids

    def test_get_ground_truth_refs(self) -> None:
        from mars.evaluation.surge_eval import SurGEEvaluator

        evaluator = SurGEEvaluator()
        refs = evaluator.get_ground_truth_refs("federated_learning_privacy")
        assert len(refs) >= 5
        assert "title" in refs[0]
        assert "doi" in refs[0]

    def test_evaluate_returns_result(self) -> None:
        from mars.evaluation.surge_eval import SurGEEvaluator

        evaluator = SurGEEvaluator()

        # Build a fake survey with some matching refs
        gt_refs = evaluator.get_ground_truth_refs("federated_learning_privacy")
        fake_ref_section = "\n".join(
            f"[{i+1}] {r['title']}. Authors. Venue, {r['year']}."
            for i, r in enumerate(gt_refs[:4])
        )
        fake_survey = f"## Survey\n\nSome content.\n\n## References\n{fake_ref_section}"

        result = evaluator.evaluate("federated_learning_privacy", fake_survey)
        assert result.topic_id == "federated_learning_privacy"
        assert result.ground_truth_ref_count == len(gt_refs)
        assert result.matched_ref_count >= 3  # fuzzy matching

    def test_evaluate_unknown_topic_raises(self) -> None:
        from mars.evaluation.surge_eval import SurGEEvaluator

        evaluator = SurGEEvaluator()
        with pytest.raises(KeyError, match="Unknown topic"):
            evaluator.evaluate("nonexistent_topic", "content")

    def test_aggregate_results(self) -> None:
        from mars.evaluation.surge_eval import SurGEEvaluator, SurGEResult

        evaluator = SurGEEvaluator()
        results = [
            SurGEResult("t1", "Topic 1", 0.8, 0.7, 0.75, 10, 12, 8),
            SurGEResult("t2", "Topic 2", 0.6, 0.9, 0.72, 15, 10, 6),
        ]
        agg = evaluator.aggregate_results(results)
        assert agg["num_topics"] == 2
        assert agg["avg_recall"] == pytest.approx(0.7, abs=0.01)
        assert agg["avg_precision"] == pytest.approx(0.8, abs=0.01)


# ============================================================================
# LLM Judge tests (no API calls)
# ============================================================================

class TestLLMJudge:
    def test_parse_valid_json_response(self) -> None:
        from mars.evaluation.llm_judge import LLMJudge

        raw = '''```json
{
  "citation_coverage": {"score": 8, "reasoning": "Good coverage"},
  "citation_accuracy": {"score": 7, "reasoning": "Minor issues"},
  "content_comprehensiveness": {"score": 9, "reasoning": "Comprehensive"},
  "structural_coherence": {"score": 8, "reasoning": "Well organized"},
  "critical_analysis_depth": {"score": 6, "reasoning": "Could be deeper"},
  "writing_quality": {"score": 8, "reasoning": "Good academic prose"},
  "overall_comment": "A solid survey with room for deeper analysis."
}
```'''
        parsed = LLMJudge._parse_response(raw)
        assert parsed["citation_coverage"] == 8
        assert parsed["critical_analysis_depth"] == 6
        assert "solid survey" in parsed["overall_comment"]

    def test_parse_flat_json_response(self) -> None:
        from mars.evaluation.llm_judge import LLMJudge

        raw = '{"citation_coverage": 7, "citation_accuracy": 8, "content_comprehensiveness": 6, "structural_coherence": 7, "critical_analysis_depth": 5, "writing_quality": 9, "overall_comment": "ok"}'
        parsed = LLMJudge._parse_response(raw)
        assert parsed["citation_coverage"] == 7
        assert parsed["writing_quality"] == 9

    def test_parse_garbled_response_returns_fallback(self) -> None:
        from mars.evaluation.llm_judge import LLMJudge

        parsed = LLMJudge._parse_response("not json at all")
        assert parsed["citation_coverage"] == 0
        assert "could not parse" in parsed["overall_comment"].lower()

    def test_judge_result_passed_threshold(self) -> None:
        from mars.evaluation.llm_judge import JudgeResult

        r = JudgeResult(
            citation_coverage=8, citation_accuracy=7,
            content_comprehensiveness=8, structural_coherence=7,
            critical_analysis_depth=7, writing_quality=8,
            overall_score=7.5,
        )
        assert r.passed(threshold=7.0) is True
        assert r.passed(threshold=8.0) is False


# ============================================================================
# Hallucination checker tests
# ============================================================================

class TestHallucinationChecker:
    def test_empty_survey(self) -> None:
        from mars.evaluation.hallucination_checker import HallucinationChecker

        checker = HallucinationChecker()
        report = checker.check("No references here.", [])
        assert report.total_citations == 0

    def test_verified_from_local_db(self) -> None:
        from mars.evaluation.hallucination_checker import HallucinationChecker

        paper_db = [
            {"title": "Federated Learning: Challenges and Methods", "doi": "10.1234/fl"},
        ]
        survey = (
            "## Survey\nSome text about federated learning [1].\n\n"
            "## References\n"
            "[1] Federated Learning: Challenges and Methods. Authors. Venue, 2020.\n"
        )
        checker = HallucinationChecker()
        report = checker.check(survey, paper_db)
        assert report.total_citations > 0

    def test_unchecked_when_no_local_match(self) -> None:
        from mars.evaluation.hallucination_checker import HallucinationChecker

        paper_db = [{"title": "Some Other Paper", "doi": ""}]
        survey = (
            "## Survey\nText [1].\n\n"
            "## References\n"
            "[1] Unknown Paper Title. Authors. Venue, 2020.\n"
        )
        checker = HallucinationChecker(enable_api_check=False)
        report = checker.check(survey, paper_db)
        if report.total_citations > 0:
            assert report.verdicts[0].status in ("unchecked", "fabricated")

    def test_report_to_dict(self) -> None:
        from mars.evaluation.hallucination_checker import HallucinationChecker

        checker = HallucinationChecker()
        report = checker.check("text [1].\n\n## References\n[1] T. A. V, 2020.\n", [])
        d = report.to_dict()
        assert "total_citations" in d
        assert "fabrication_rate" in d
        assert "verdicts" in d
