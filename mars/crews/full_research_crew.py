"""
Full Research Crew - 完整研究流程（含迭代精炼）

Flow:
  User topic
    → Researcher Agent (domain analysis)
    → Searcher Agent (bulk retrieval, 50 papers, saves paper_search.json)
    → Analyzer Agent (deep analysis, top 20 papers)
    → Connector Agent (relationship analysis + saves connection_analysis.json
                        + connection_analysis_report.md)
    → Evaluator Agent (quality evaluation, saves analysis_report.md)
    → Summarizer Agent (English review → saves review_en.md)
    —— Iterative Refinement Loop (EXP-1.1) ——
    → LLM-as-Judge evaluates review
    → if score < threshold: revision task → re-evaluate (max N rounds)
    ——
    → Summarizer Agent (Chinese translation → saves review_zh.md)
    → Summarizer Agent (final synthesis → saves full_research_report.md)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from crewai import Crew, Process

from mars.agents.analyzer import create_analyzer_agent
from mars.agents.connector import create_connector_agent
from mars.agents.evaluator import create_evaluator_agent
from mars.agents.researcher import create_researcher_agent
from mars.agents.searcher import create_searcher_agent
from mars.agents.summarizer import create_summarizer_agent
from mars.config import settings
from mars.tasks.task_definitions import (
    create_connection_analysis_task,
    create_deep_analysis_task,
    create_domain_analysis_task,
    create_english_review_task,
    create_full_research_synthesis_task,
    create_paper_search_task,
    create_quality_evaluation_task,
    create_refinement_task,
    create_review_generation_task,
)

logger = logging.getLogger(__name__)


def _build_analysis_crew(topic: str) -> Crew:
    """Build the analysis-phase crew (domain → search → deep → connect → evaluate)."""
    max_papers = settings.MAX_PAPERS_PER_SEARCH
    analysis_limit = settings.MAX_PAPERS_FOR_ANALYSIS

    researcher = create_researcher_agent()
    searcher = create_searcher_agent()
    analyzer = create_analyzer_agent()
    connector = create_connector_agent()
    evaluator = create_evaluator_agent()

    domain_analysis_task = create_domain_analysis_task(researcher, topic)
    bulk_search_task = create_paper_search_task(
        searcher, topic, max_papers, context=[domain_analysis_task]
    )
    deep_analysis_task = create_deep_analysis_task(
        analyzer, topic, limit=analysis_limit, context=[bulk_search_task]
    )
    connection_analysis_task = create_connection_analysis_task(
        connector, topic, context=[bulk_search_task]
    )
    quality_evaluation_task = create_quality_evaluation_task(
        evaluator, limit=analysis_limit, context=[deep_analysis_task]
    )

    return Crew(
        agents=[researcher, searcher, analyzer, connector, evaluator],
        tasks=[
            domain_analysis_task,
            bulk_search_task,
            deep_analysis_task,
            connection_analysis_task,
            quality_evaluation_task,
        ],
        process=Process.sequential,
        verbose=True,
        memory=settings.ENABLE_MEMORY,
    )


# ---------------------------------------------------------------------------
# Refinement loop (EXP-1.1)
# ---------------------------------------------------------------------------

def _read_generated_file(filename: str) -> Optional[str]:
    """Read a file saved by the file_writer tool from the output directory."""
    path = settings.OUTPUT_DIR / filename
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return None


def _run_refinement_loop(
    topic: str,
    initial_draft: str,
) -> str:
    """Iteratively refine a survey draft using LLM-as-Judge feedback.

    Args:
        topic: Research topic string.
        initial_draft: The first-pass survey text.

    Returns:
        The (possibly refined) final survey text.
    """
    max_rounds = settings.REFINEMENT_MAX_ROUNDS
    threshold = settings.REFINEMENT_THRESHOLD

    if max_rounds <= 1:
        logger.info("Refinement disabled (REFINEMENT_MAX_ROUNDS=%d).", max_rounds)
        return initial_draft

    current_draft = initial_draft
    refinement_log: list[dict] = []

    for round_num in range(1, max_rounds + 1):
        logger.info("Refinement round %d/%d starting...", round_num, max_rounds)

        # 1. Evaluate current draft with LLM-as-Judge
        try:
            from mars.evaluation.llm_judge import LLMJudge

            judge = LLMJudge()
            judge_result = judge.evaluate(current_draft, topic)
            score = judge_result.overall_score
            logger.info(
                "Round %d — Judge score: %.1f/10 (threshold: %.1f)",
                round_num, score, threshold,
            )
        except Exception as exc:
            logger.warning("LLM Judge unavailable in round %d: %s — stopping refinement.", round_num, exc)
            # Without a judge we can't evaluate → keep current draft
            refinement_log.append({"round": round_num, "score": None, "error": str(exc)})
            break

        refinement_log.append({
            "round": round_num,
            "score": score,
            "dimensions": judge_result.to_dict(),
        })

        # 2. Check if quality is sufficient
        if score >= threshold:
            logger.info(
                "Quality threshold met (%.1f ≥ %.1f) — refinement complete after %d round(s).",
                score, threshold, round_num,
            )
            break

        # 3. Build structured feedback for the revision prompt
        feedback = _format_judge_feedback(judge_result)

        # 4. Run a refinement crew (summarizer only, one revision task)
        if round_num < max_rounds:
            logger.info("Launching revision crew for round %d...", round_num)
            try:
                revised = _run_revision_step(topic, current_draft, feedback, round_num)
                if revised:
                    current_draft = revised
                else:
                    logger.warning("Revision step returned empty — keeping previous draft.")
            except Exception as exc:
                logger.error("Revision step failed in round %d: %s", round_num, exc)
                # Continue with current draft rather than losing progress
        else:
            logger.info("Max rounds reached — keeping best draft.")

    # Save refinement log
    _save_refinement_log(refinement_log)

    return current_draft


def _format_judge_feedback(judge_result) -> str:
    """Convert a JudgeResult into human-readable feedback for the revision prompt."""
    d = judge_result.to_dict()
    lines = [
        f"**Overall Score**: {d['overall_score']}/10",
        f"**Overall Comment**: {d.get('overall_comment', 'N/A')}",
        "",
        "**Dimension Scores**:",
    ]
    dims = [
        ("citation_coverage", "Citation Coverage"),
        ("citation_accuracy", "Citation Accuracy"),
        ("content_comprehensiveness", "Content Comprehensiveness"),
        ("structural_coherence", "Structural Coherence"),
        ("critical_analysis_depth", "Critical Analysis Depth"),
        ("writing_quality", "Writing Quality"),
    ]
    for key, label in dims:
        score = d.get(key, "N/A")
        lines.append(f"- {label}: **{score}/10**")

    # Add specific improvement suggestions based on lowest scores
    sorted_dims = sorted(
        [(k, v) for k, v in dims if d.get(k, 10) < 7],
        key=lambda x: d.get(x[0], 10),
    )
    if sorted_dims:
        lines.append("")
        lines.append("**Priority improvements needed**:")
        for key, label in sorted_dims:
            lines.append(f"- **{label}** (score: {d.get(key)}/10) — needs significant improvement")

    return "\n".join(lines)


def _run_revision_step(
    topic: str,
    current_draft: str,
    feedback: str,
    round_num: int,
) -> Optional[str]:
    """Run a single revision crew step and return the revised text."""
    summarizer = create_summarizer_agent()
    output_filename = f"review_en_round{round_num}.md"

    revision_task = create_refinement_task(
        agent=summarizer,
        topic=topic,
        previous_survey=current_draft,
        evaluation_feedback=feedback,
        output_filename=output_filename,
    )

    crew = Crew(
        agents=[summarizer],
        tasks=[revision_task],
        process=Process.sequential,
        verbose=True,
    )

    crew.kickoff()

    # Read the revised file
    revised = _read_generated_file(output_filename)
    return revised or _read_generated_file("review_en.md")


def _save_refinement_log(log: list[dict]) -> None:
    """Persist the refinement log as JSON."""
    path = settings.OUTPUT_DIR / "refinement_log.json"
    try:
        path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Refinement log saved to %s", path)
    except OSError as exc:
        logger.warning("Failed to save refinement log: %s", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_full_research_crew(topic: str) -> Crew:
    """Build the complete 8-task full research crew (for backward compatibility).

    For iterative refinement, use :func:`run_full_research`.
    """
    max_papers = settings.MAX_PAPERS_PER_SEARCH
    analysis_limit = settings.MAX_PAPERS_FOR_ANALYSIS

    researcher = create_researcher_agent()
    searcher = create_searcher_agent()
    analyzer = create_analyzer_agent()
    connector = create_connector_agent()
    evaluator = create_evaluator_agent()
    summarizer = create_summarizer_agent()

    domain_analysis_task = create_domain_analysis_task(researcher, topic)
    bulk_search_task = create_paper_search_task(
        searcher, topic, max_papers, context=[domain_analysis_task]
    )
    deep_analysis_task = create_deep_analysis_task(
        analyzer, topic, limit=analysis_limit, context=[bulk_search_task]
    )
    connection_analysis_task = create_connection_analysis_task(
        connector, topic, context=[bulk_search_task]
    )
    quality_evaluation_task = create_quality_evaluation_task(
        evaluator, limit=analysis_limit, context=[deep_analysis_task]
    )
    english_review_task = create_english_review_task(
        summarizer, topic,
        context=[
            domain_analysis_task, bulk_search_task, deep_analysis_task,
            connection_analysis_task, quality_evaluation_task,
        ],
    )
    chinese_review_task = create_review_generation_task(
        summarizer, topic,
        context=[
            domain_analysis_task, bulk_search_task, deep_analysis_task,
            connection_analysis_task, quality_evaluation_task,
            english_review_task,
        ],
    )
    synthesis_task = create_full_research_synthesis_task(
        summarizer, topic,
        context=[
            domain_analysis_task, bulk_search_task, deep_analysis_task,
            connection_analysis_task, quality_evaluation_task,
            english_review_task, chinese_review_task,
        ],
    )

    return Crew(
        agents=[researcher, searcher, analyzer, connector, evaluator, summarizer],
        tasks=[
            domain_analysis_task, bulk_search_task, deep_analysis_task,
            connection_analysis_task, quality_evaluation_task,
            english_review_task, chinese_review_task, synthesis_task,
        ],
        process=Process.sequential,
        verbose=True,
        memory=settings.ENABLE_MEMORY,
    )


def run_full_research(topic: str) -> str:
    """Run the full research workflow with iterative refinement.

    1. Analysis phase (domain → search → deep → connect → evaluate)
    2. First-pass English review
    3. Refinement loop (LLM-as-Judge → revision → re-evaluate)
    4. Chinese translation + final synthesis
    """
    logger.info("=== MARS Full Research: %s ===", topic)
    logger.info(
        "Refinement: max_rounds=%d, threshold=%.1f",
        settings.REFINEMENT_MAX_ROUNDS,
        settings.REFINEMENT_THRESHOLD,
    )

    # ---- Phase 1: Analysis ----
    logger.info("Phase 1: Running analysis crew...")
    analysis_crew = _build_analysis_crew(topic)
    analysis_crew.kickoff()

    # ---- Phase 2: First-pass English review ----
    logger.info("Phase 2: Generating initial English review...")
    summarizer = create_summarizer_agent()
    english_task = create_english_review_task(summarizer, topic)

    first_pass_crew = Crew(
        agents=[summarizer],
        tasks=[english_task],
        process=Process.sequential,
        verbose=True,
    )
    first_pass_crew.kickoff()

    initial_draft = _read_generated_file("review_en.md") or ""
    logger.info("Initial draft: %d chars", len(initial_draft))

    # ---- Phase 3: Iterative refinement (EXP-1.1) ----
    logger.info("Phase 3: Iterative refinement loop...")
    refined_draft = _run_refinement_loop(topic, initial_draft)

    # If the refinement produced a better draft, write it as the final version
    if refined_draft and refined_draft != initial_draft:
        out_path = settings.OUTPUT_DIR / "review_en.md"
        out_path.write_text(refined_draft, encoding="utf-8")
        logger.info("Refined draft saved (%d chars → %d chars).",
                     len(initial_draft), len(refined_draft))

    # ---- Phase 4: Chinese translation + synthesis ----
    logger.info("Phase 4: Chinese translation + synthesis report...")
    chinese_task = create_review_generation_task(summarizer, topic)
    synthesis_task = create_full_research_synthesis_task(summarizer, topic)

    final_crew = Crew(
        agents=[summarizer],
        tasks=[chinese_task, synthesis_task],
        process=Process.sequential,
        verbose=True,
    )
    result = final_crew.kickoff()

    logger.info("=== MARS Full Research complete ===")
    return str(result)
