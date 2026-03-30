"""
Search Crew - 基础检索流程

Flow:
  User topic → Researcher Agent (domain analysis + venue recommendation)
             → Searcher Agent (paper retrieval + initial ranking)
             → Return paper list
"""

from __future__ import annotations

from crewai import Crew, Process

from mars.agents.researcher import create_researcher_agent
from mars.agents.searcher import create_searcher_agent
from mars.config import settings
from mars.tasks.task_definitions import (
    create_domain_analysis_task,
    create_paper_search_task,
)


def create_search_crew(topic: str, max_results: int | None = None) -> Crew:
    """
    Build and return the Search Crew for a given research topic.

    Args:
        topic:       The research topic or question from the user.
        max_results: Maximum number of papers to retrieve (default from settings).
    """
    max_papers = max_results or settings.MAX_PAPERS_PER_SEARCH

    researcher = create_researcher_agent()
    searcher = create_searcher_agent()

    domain_analysis_task = create_domain_analysis_task(researcher, topic)
    paper_search_task = create_paper_search_task(
        searcher, topic, max_papers, context=[domain_analysis_task]
    )

    return Crew(
        agents=[researcher, searcher],
        tasks=[domain_analysis_task, paper_search_task],
        process=Process.sequential,
        verbose=True,
        memory=True,
    )


def run_search(topic: str, max_results: int | None = None) -> str:
    """Run the search workflow and return the result string."""
    crew = create_search_crew(topic, max_results)
    result = crew.kickoff()
    return str(result)
