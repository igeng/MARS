"""
Connector Agent - 关联分析师

Analyzes citation relationships between multiple papers,
computes topic similarity, builds paper relationship networks,
and identifies research hotspots and development trends.
"""

from __future__ import annotations

from crewai import Agent

from mars.tools.citation_network import CitationNetworkTool
from mars.tools.file_manager import FileWriterTool
from mars.tools.semantic_scholar import SemanticScholarSearchTool
from mars.services.llm_gateway import get_llm_by_task


def create_connector_agent() -> Agent:
    """Create and return the Connector Agent."""
    return Agent(
        role="关联分析师 (Research Connection Analyst)",
        goal=(
            "分析多篇论文之间的引用关系和主题相似度，"
            "构建论文关系网络，识别研究热点和发展趋势，"
            "帮助研究者把握领域发展脉络和研究前沿。"
        ),
        backstory=(
            "你是一位学术情报分析专家，擅长从海量文献中发现隐藏的关联关系。"
            "你能构建论文的引用网络，识别研究主题的演进路径，"
            "通过分析引用模式和主题聚类，揭示领域内的研究热点、"
            "新兴方向和潜在合作机会。"
        ),
        llm=get_llm_by_task("connector"),
        tools=[
            CitationNetworkTool(),
            SemanticScholarSearchTool(),
            FileWriterTool(),
        ],
        verbose=True,
        allow_delegation=False,
        max_iter=8,
    )
