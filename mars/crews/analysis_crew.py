"""
Analysis Crew - 深度分析流程

Flow:
  Paper list → Analyzer Agent (full-text retrieval + deep analysis)
             → Evaluator Agent (quality assessment)
             → Return detailed analysis report
"""

from __future__ import annotations

from crewai import Crew, Process

from mars.agents.analyzer import create_analyzer_agent
from mars.agents.evaluator import create_evaluator_agent
from mars.config import settings
from mars.tasks.task_definitions import (
    create_deep_analysis_task,
    create_quality_evaluation_task,
)


def create_analysis_crew(papers_info: str, max_papers: int | None = None) -> Crew:
    """
    Build and return the Analysis Crew for a set of papers.

    Args:
        papers_info: Structured string describing papers to analyze
                     (titles, URLs, abstracts, etc.).
        max_papers:  Maximum number of papers to deeply analyze.
    """
    limit = max_papers or settings.MAX_PAPERS_FOR_ANALYSIS

    analyzer = create_analyzer_agent()
    evaluator = create_evaluator_agent()

    deep_analysis_task = create_deep_analysis_task(
        analyzer, topic="", papers_info=papers_info, limit=limit,
    )
    quality_evaluation_task = create_quality_evaluation_task(
        evaluator, limit=limit, context=[deep_analysis_task],
    )

    return Crew(
        agents=[analyzer, evaluator],
        tasks=[deep_analysis_task, quality_evaluation_task],
        process=Process.sequential,
        verbose=True,
    )


def run_analysis(papers_info: str, max_papers: int | None = None) -> str:
    """Run the analysis workflow and return the result string."""
    crew = create_analysis_crew(papers_info, max_papers)
    result = crew.kickoff()
    return str(result)
