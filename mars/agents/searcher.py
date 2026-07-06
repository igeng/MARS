"""
Searcher Agent - 论文检索师

Supports two retrieval modes:

- **online** (default): DBLP + Semantic Scholar + arXiv APIs
- **surge**: SurGE 1.09M arXiv corpus via local TF-IDF search

Mode is controlled by ``settings.CORPUS_MODE`` or the *corpus_mode* parameter.
"""

from __future__ import annotations

import logging
from typing import List

from crewai import Agent
from crewai.tools import BaseTool

from mars.tools.arxiv_api import ArXivSearchTool
from mars.tools.dblp_search import DBLPSearchTool
from mars.tools.file_manager import FileWriterTool
from mars.tools.keyword_expander import KeywordExpanderTool
from mars.tools.semantic_scholar import SemanticScholarSearchTool
from mars.services.llm_gateway import get_llm_by_task
from mars.config.settings import settings

logger = logging.getLogger(__name__)


def _build_online_tools() -> List[BaseTool]:
    """Tools for online (live API) retrieval mode."""
    return [
        DBLPSearchTool(),
        SemanticScholarSearchTool(),
        ArXivSearchTool(),
        KeywordExpanderTool(),
        FileWriterTool(),
    ]


def _build_surge_tools() -> List[BaseTool]:
    """Tools for SurGE corpus retrieval mode."""
    from mars.tools.surge_corpus_tool import SurgeCorpusSearchTool
    logger.info("Searcher using SurGE corpus (offline) mode.")
    return [
        SurgeCorpusSearchTool(),
        SemanticScholarSearchTool(),  # kept for paper verification
        KeywordExpanderTool(),
        FileWriterTool(),
    ]


def create_searcher_agent(corpus_mode: str | None = None) -> Agent:
    """Create and return the Searcher Agent.

    Args:
        corpus_mode: "online" (DBLP/S2/arXiv APIs) or "surge" (SurGE corpus).
            Defaults to ``settings.CORPUS_MODE``.
    """
    mode = (corpus_mode or settings.CORPUS_MODE).lower()
    tools = _build_surge_tools() if mode == "surge" else _build_online_tools()

    backstory_suffix = (
        "你使用 SurGE 语料库进行离线检索，检索结果带有 doc_id 用于精确评估。"
        if mode == "surge"
        else "你精通各种学术数据库的检索技巧，能快速定位高质量论文。"
    )

    return Agent(
        role="论文检索师 (Academic Paper Searcher)",
        goal=(
            "根据研究领域分析结果，构建高效的检索查询，"
            "从多个学术数据库中检索高质量的相关论文，"
            "并进行初步过滤和排序，"
            "返回结构化的论文列表供后续分析使用。"
        ),
        backstory=(
            "你是一位专业的学术文献检索专家。"
            "你擅长从海量文献中快速定位高质量论文，"
            "能够根据用户需求优化检索策略，组合使用多种检索工具，"
            "并对结果进行智能排序和去重处理。"
            + backstory_suffix
        ),
        llm=get_llm_by_task("searcher"),
        tools=tools,
        verbose=True,
        allow_delegation=False,
        max_iter=settings.AGENT_MAX_ITER,
    )
