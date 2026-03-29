"""
Evaluator Agent - 质量评估师

Evaluates academic quality (novelty, impact) of papers,
assesses relevance of search results,
and evaluates completeness of the literature review report.
"""

from __future__ import annotations

from crewai import Agent

from mars.tools.file_manager import FileWriterTool
from mars.tools.semantic_scholar import SemanticScholarSearchTool
from mars.utils.llm_factory import get_deepseek_llm


def create_evaluator_agent() -> Agent:
    """Create and return the Evaluator Agent."""
    return Agent(
        role="质量评估师 (Academic Quality Evaluator)",
        goal=(
            "对论文的学术质量进行客观评估，包括创新性、技术深度、"
            "实验有效性和写作质量，提供多维度的质量评分和专业审稿意见，"
            "帮助研究者筛选高价值文献和识别潜在研究缺口。"
        ),
        backstory=(
            "你是一位严格的学术期刊审稿人，拥有丰富的论文评审经验。"
            "你能客观评估论文的学术价值，给出专业的评审意见和改进建议。"
            "你的评估标准严格但公正，能够识别论文的真正创新点，"
            "也能指出其局限性和改进空间。"
        ),
        llm=get_deepseek_llm(),
        tools=[
            SemanticScholarSearchTool(),
            FileWriterTool(),
        ],
        verbose=True,
        allow_delegation=False,
        max_iter=8,
    )
