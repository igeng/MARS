"""
Analyzer Agent - 深度分析师

Downloads paper PDFs (when available), extracts key information,
analyzes research methods, innovations, experimental design,
and generates structured paper analysis reports.
"""

from __future__ import annotations

from crewai import Agent

from mars.tools.arxiv_api import ArXivSearchTool
from mars.tools.file_manager import FileWriterTool
from mars.tools.pdf_parser import PDFParserTool
from mars.tools.semantic_scholar import SemanticScholarSearchTool
from mars.services.llm_gateway import get_llm_by_task
from mars.config.settings import settings


def create_analyzer_agent() -> Agent:
    """Create and return the Analyzer Agent."""
    return Agent(
        role="深度分析师 (Deep Paper Analyst)",
        goal=(
            "对检索到的论文进行深度解析，提取论文的核心贡献、研究方法、"
            "实验设计和主要结论，生成结构化的论文分析报告，"
            "帮助研究者快速理解每篇论文的学术价值。"
        ),
        backstory=(
            "你是一位细心的学术文献分析师，擅长快速理解复杂的学术论文。"
            "你能准确提取论文的核心贡献、方法论和实验结果，"
            "并用清晰的结构化语言呈现出来。"
            "你特别擅长处理长文本，能够从数十页的论文中提炼出最关键的信息。"
        ),
        llm=get_llm_by_task("analyzer"),
        tools=[
            ArXivSearchTool(),
            SemanticScholarSearchTool(),
            PDFParserTool(),
            FileWriterTool(),
        ],
        verbose=True,
        allow_delegation=False,
        max_iter=settings.AGENT_MAX_ITER,
    )
