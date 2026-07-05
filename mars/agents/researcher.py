"""
Researcher Agent - 领域分析师

Understands the user's research topic, identifies the research domain,
matches relevant CCF-ranked venues, and recommends the most relevant
journals/conferences for paper retrieval.
"""

from __future__ import annotations

from crewai import Agent

from mars.tools.ccf_database import CCFDatabaseQueryTool
from mars.tools.keyword_expander import KeywordExpanderTool
from mars.services.llm_gateway import get_llm_by_task
from mars.config.settings import settings


def create_researcher_agent() -> Agent:
    """Create and return the Researcher Agent."""
    return Agent(
        role="领域分析师 (Research Domain Analyst)",
        goal=(
            "深入理解用户的研究主题，精准识别研究领域和子领域，"
            "结合CCF目录推荐最相关的期刊和会议，为后续论文检索奠定基础。"
        ),
        backstory=(
            "你是一位资深的计算机科学研究顾问，拥有20年的学术出版经验。"
            "你精通CCF目录，熟悉各个期刊和会议的研究方向和影响力。"
            "你的专长是帮助研究者精准定位最适合的发表平台和检索来源，"
            "能够快速从一个模糊的研究方向中提炼出精确的检索策略。"
        ),
        llm=get_llm_by_task("researcher"),
        tools=[
            CCFDatabaseQueryTool(),
            KeywordExpanderTool(),
        ],
        verbose=True,
        allow_delegation=False,
        max_iter=settings.AGENT_MAX_ITER,
    )
