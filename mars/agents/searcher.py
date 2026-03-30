"""
Searcher Agent - 论文检索师

Builds search queries from the research domain analysis,
queries multiple academic databases (DBLP, Semantic Scholar, arXiv),
filters and ranks results, and returns a structured paper list.
"""

from __future__ import annotations

from crewai import Agent

from mars.tools.arxiv_api import ArXivSearchTool
from mars.tools.dblp_search import DBLPSearchTool
from mars.tools.file_manager import FileWriterTool
from mars.tools.keyword_expander import KeywordExpanderTool
from mars.tools.semantic_scholar import SemanticScholarSearchTool
from mars.services.llm_gateway import get_llm_by_task


def create_searcher_agent() -> Agent:
    """Create and return the Searcher Agent."""
    return Agent(
        role="论文检索师 (Academic Paper Searcher)",
        goal=(
            "根据研究领域分析结果，构建高效的检索查询，"
            "从DBLP、Semantic Scholar、arXiv等多个学术数据库中"
            "检索高质量的相关论文，并进行初步过滤和排序，"
            "返回结构化的论文列表供后续分析使用。"
        ),
        backstory=(
            "你是一位专业的学术文献检索专家，精通各种学术数据库的检索技巧。"
            "你擅长从海量文献中快速定位高质量论文，"
            "能够根据用户需求优化检索策略，组合使用多种检索工具，"
            "并对结果进行智能排序和去重处理。"
        ),
        llm=get_llm_by_task("searcher"),
        tools=[
            DBLPSearchTool(),
            SemanticScholarSearchTool(),
            ArXivSearchTool(),
            KeywordExpanderTool(),
            FileWriterTool(),
        ],
        verbose=True,
        allow_delegation=False,
        max_iter=8,
    )
