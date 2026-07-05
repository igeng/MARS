"""
Summarizer Agent - 综述生成师

Integrates analysis results from multiple papers,
generates a comprehensive literature review report
covering research status, main methods, and open problems.
"""

from __future__ import annotations

from crewai import Agent

from mars.tools.file_manager import FileWriterTool
from mars.services.llm_gateway import get_llm_by_task
from mars.config.settings import settings


def create_summarizer_agent() -> Agent:
    """Create and return the Summarizer Agent."""
    return Agent(
        role="综述生成师 (Literature Review Synthesizer)",
        goal=(
            "整合多篇论文的分析结果，生成高质量的领域综述报告，"
            "全面覆盖研究现状、主要方法、核心发现和开放问题，"
            "为研究者提供清晰的领域全局视图和未来研究方向建议。"
        ),
        backstory=(
            "你是一位学术写作专家，擅长将零散的研究成果整合成连贯的综述文章。"
            "你能准确把握领域的研究现状，提炼关键发现，"
            "并指出未来的研究方向。你的综述报告逻辑清晰、"
            "结构完整，能够帮助读者快速掌握一个研究领域的全貌。"
        ),
        llm=get_llm_by_task("summarizer"),
        tools=[
            FileWriterTool(),
        ],
        verbose=True,
        allow_delegation=False,
        max_iter=settings.AGENT_MAX_ITER,
    )
