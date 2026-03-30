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

from crewai import Crew, Process, Task

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
    comprehensive_review_task = Task(
        description=(
            f"整合所有分析结果，生成一份高质量的完整研究综述报告。\n\n"
            f"综述主题：{topic}\n\n"
            f"综合参考以下分析结果：\n"
            f"- 领域分析（期刊/会议推荐）\n"
            f"- 论文检索结果（{max_papers}篇）\n"
            f"- 深度分析报告（前{analysis_limit}篇）\n"
            f"- 关联分析报告（引用网络、主题聚类、趋势）\n"
            f"- 质量评估报告（评分和审稿意见）\n\n"
            f"综述报告结构：\n"
            f"# {{topic}} 研究综述\n\n"
            f"## 1. 研究背景与意义\n"
            f"## 2. 研究现状综述\n"
            f"### 2.1 主要研究方向\n"
            f"### 2.2 代表性工作\n"
            f"## 3. 核心技术方法\n"
            f"## 4. 方法对比分析\n"
            f"## 5. 开放挑战\n"
            f"## 6. 未来研究方向\n"
            f"## 7. 参考文献\n\n"
            f"使用file_writer工具保存到 comprehensive_review.md\n"
            f"综述长度：不少于2000字，引用至少15篇检索到的论文"
        ),
        expected_output=(
            "一份完整的研究综述报告（Markdown格式，≥2000字），"
            "覆盖研究背景、现状、方法、挑战和未来方向，含参考文献"
        ),
        agent=summarizer,
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
    )


def run_full_research(topic: str) -> str:
    """Run the full research workflow and return the result string."""
    crew = create_full_research_crew(topic)
    result = crew.kickoff()
    return str(result)
