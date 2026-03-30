"""
Connection Crew - 关联分析流程

Flow:
  Multiple papers → Connector Agent (citation analysis + similarity)
                 → Summarizer Agent (literature review generation)
                 → Return relationship graph + review report
"""

from __future__ import annotations

from crewai import Crew, Process

from mars.agents.connector import create_connector_agent
from mars.agents.summarizer import create_summarizer_agent
from mars.config import settings
from mars.tasks.task_definitions import (
    create_connection_analysis_task,
    create_english_review_task,
    create_review_generation_task,
)


def create_connection_crew(papers_info: str, topic: str) -> Crew:
    """
    Build and return the Connection Crew.

    Args:
        papers_info: Structured info about the papers to analyze
                     (titles, IDs, abstracts, venues, years).
        topic:       The original research topic/theme.
    """
    connector = create_connector_agent()
    summarizer = create_summarizer_agent()

    connection_analysis_task = create_connection_analysis_task(
        connector, topic, papers_info=papers_info,
    )
    chinese_review_task = create_review_generation_task(
        summarizer, topic, context=[connection_analysis_task],
    )
    english_review_task = create_english_review_task(
        summarizer, topic, context=[connection_analysis_task, chinese_review_task],
    )

    return Crew(
        agents=[connector, summarizer],
        tasks=[connection_analysis_task, chinese_review_task, english_review_task],
        process=Process.sequential,
        verbose=True,
        memory=settings.ENABLE_MEMORY,
    )


def run_connection(papers_info: str, topic: str) -> str:
    """Run the connection workflow and return the result string."""
    crew = create_connection_crew(papers_info, topic)
    result = crew.kickoff()
    return str(result)
