"""
Full Research Crew - 完整研究流程

Flow:
  User topic
    → Researcher Agent (domain analysis)
    → Searcher Agent (bulk retrieval, 50 papers)
    → [Parallel]
        ├─ Analyzer Agent (deep analysis, top 20 papers)
        ├─ Connector Agent (relationship analysis, all 50 papers)
        └─ Evaluator Agent (quality evaluation, top 20 papers)
    → Summarizer Agent (comprehensive review)
    → Return complete research report
"""

from __future__ import annotations

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
    create_paper_search_task,
    create_quality_evaluation_task,
    create_review_generation_task,
)


def create_full_research_crew(topic: str) -> Crew:
    """
    Build and return the full research crew for an end-to-end research workflow.

    Args:
        topic: The research topic or question from the user.
    """
    max_papers = settings.MAX_PAPERS_PER_SEARCH
    analysis_limit = settings.MAX_PAPERS_FOR_ANALYSIS

    researcher = create_researcher_agent()
    searcher = create_searcher_agent()
    analyzer = create_analyzer_agent()
    connector = create_connector_agent()
    evaluator = create_evaluator_agent()
    summarizer = create_summarizer_agent()

    # ---- Sequential phase 1: Domain analysis ----
    domain_analysis_task = create_domain_analysis_task(researcher, topic)

    # ---- Sequential phase 1: Bulk paper retrieval ----
    bulk_search_task = create_paper_search_task(
        searcher, topic, max_papers, context=[domain_analysis_task]
    )

    # ---- Parallel phase: Analysis + Connection + Evaluation ----
    deep_analysis_task = create_deep_analysis_task(
        analyzer, topic, limit=analysis_limit, context=[bulk_search_task]
    )

    connection_analysis_task = create_connection_analysis_task(
        connector, topic, context=[bulk_search_task]
    )

    quality_evaluation_task = create_quality_evaluation_task(
        evaluator, limit=analysis_limit, context=[deep_analysis_task]
    )

    # ---- Sequential phase 2: Comprehensive review ----
    comprehensive_review_task = create_review_generation_task(
        summarizer,
        topic,
        context=[
            domain_analysis_task,
            bulk_search_task,
            deep_analysis_task,
            connection_analysis_task,
            quality_evaluation_task,
        ],
    )

    return Crew(
        agents=[researcher, searcher, analyzer, connector, evaluator, summarizer],
        tasks=[
            domain_analysis_task,
            bulk_search_task,
            deep_analysis_task,
            connection_analysis_task,
            quality_evaluation_task,
            comprehensive_review_task,
        ],
        process=Process.sequential,
        verbose=True,
        memory=True,
    )


def run_full_research(topic: str) -> str:
    """Run the full research workflow and return the result string."""
    crew = create_full_research_crew(topic)
    result = crew.kickoff()
    return str(result)
