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
    domain_analysis_task = Task(
        description=(
            f"对研究主题进行全面的领域分析：\n\n"
            f"研究主题：{topic}\n\n"
            f"1. 识别主研究领域、2-4个子领域\n"
            f"2. 提取5-10个核心关键词（中英文）\n"
            f"3. 使用keyword_expander扩展关键词\n"
            f"4. 使用ccf_database_query查询相关CCF期刊/会议\n"
            f"5. 推荐Top-8最相关的发表平台"
        ),
        expected_output=(
            "结构化的领域分析报告：研究域、子领域、关键词列表、"
            "推荐期刊/会议（含CCF等级）"
        ),
        agent=researcher,
    )

    # ---- Sequential phase 1: Bulk paper retrieval ----
    bulk_search_task = Task(
        description=(
            f"根据领域分析结果，批量检索论文。\n\n"
            f"研究主题：{topic}\n"
            f"检索目标：最多{max_papers}篇高质量相关论文\n\n"
            f"1. 综合使用DBLP、Semantic Scholar、arXiv进行多轮检索\n"
            f"2. 优先检索CCF A/B类论文\n"
            f"3. 按相关性和引用量排序\n"
            f"4. 返回结构化论文列表（含Semantic Scholar ID便于后续分析）"
        ),
        expected_output=(
            f"最多{max_papers}篇论文的结构化列表（含标题、作者、期刊、年份、"
            f"引用量、URL、SS ID）"
        ),
        agent=searcher,
        context=[domain_analysis_task],
    )

    # ---- Parallel phase: Analysis + Connection + Evaluation ----
    deep_analysis_task = Task(
        description=(
            f"对检索结果中排名靠前的论文进行深度解析（最多{analysis_limit}篇）。\n\n"
            f"研究主题：{topic}\n\n"
            f"对每篇论文提取：\n"
            f"- 核心贡献（3-5点）\n"
            f"- 研究方法\n"
            f"- 实验设置和主要结果\n"
            f"- 结论和局限性\n\n"
            f"将分析结果保存到 analysis_results.json"
        ),
        expected_output=(
            f"前{analysis_limit}篇论文的深度分析报告（JSON格式）"
        ),
        agent=analyzer,
        context=[bulk_search_task],
    )

    connection_analysis_task = Task(
        description=(
            f"对全部检索到的论文进行关联分析。\n\n"
            f"研究主题：{topic}\n\n"
            f"1. 构建引用网络（使用Semantic Scholar ID）\n"
            f"2. 进行主题聚类分析\n"
            f"3. 识别研究趋势（新兴方向、成熟方向、研究空白）\n"
            f"4. 识别领域核心论文\n\n"
            f"将分析结果保存到 connection_analysis.json"
        ),
        expected_output=(
            "论文关联分析报告：引用网络摘要、主题聚类、研究趋势"
        ),
        agent=connector,
        context=[bulk_search_task],
    )

    quality_evaluation_task = Task(
        description=(
            f"对深度分析的论文进行学术质量多维评估（最多{analysis_limit}篇）。\n\n"
            f"评估维度：创新性、技术深度、实验有效性、写作质量\n"
            f"每篇给出0-10分的各维度评分和综合评分\n"
            f"提供优点、不足和改进建议\n\n"
            f"将评估结果保存到 quality_evaluation.json"
        ),
        expected_output=(
            f"前{analysis_limit}篇论文的质量评估报告（含多维度评分和审稿意见）"
        ),
        agent=evaluator,
        context=[deep_analysis_task],
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
