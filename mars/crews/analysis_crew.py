"""
Analysis Crew - 深度分析流程

Flow:
  Paper list → Analyzer Agent (full-text retrieval + deep analysis)
             → Evaluator Agent (quality assessment)
             → Return detailed analysis report
"""

from __future__ import annotations

from crewai import Crew, Process, Task

from mars.agents.analyzer import create_analyzer_agent
from mars.agents.evaluator import create_evaluator_agent
from mars.config import settings


def create_analysis_crew(papers_info: str, max_papers: int | None = None) -> Crew:
    """
    Build and return the Analysis Crew for a set of papers.

    Args:
        papers_info: Structured string describing papers to analyze
                     (titles, URLs, abstracts, etc.).
        max_papers:  Maximum number of papers to deeply analyze.
    """
    limit = max_papers or settings.MAX_PAPERS_FOR_ANALYSIS

    analyzer = create_analyzer_agent()
    evaluator = create_evaluator_agent()

    # Task 1: Deep paper analysis
    deep_analysis_task = Task(
        description=(
            f"对以下论文列表进行深度解析（最多{limit}篇）：\n\n"
            f"{papers_info}\n\n"
            f"请完成以下工作：\n"
            f"1. 对每篇论文使用arxiv_search或semantic_scholar_search获取详细信息\n"
            f"2. 如果有PDF链接，使用pdf_parser工具提取全文\n"
            f"3. 提取每篇论文的：\n"
            f"   - 摘要和关键词\n"
            f"   - 研究方法（理论分析/实验/仿真等）\n"
            f"   - 主要贡献（3-5点）\n"
            f"   - 实验设置（数据集、评估指标、主要结果）\n"
            f"   - 结论和未来工作方向\n"
            f"4. 使用file_writer工具将分析结果保存到analysis_results.json\n\n"
            f"每篇论文的分析要深入、专业，突出其核心创新点。"
        ),
        expected_output=(
            "每篇论文的详细分析报告，包含：\n"
            "- title: 论文标题\n"
            "- abstract: 摘要\n"
            "- keywords: 关键词\n"
            "- research_method: 研究方法\n"
            "- main_contributions: 主要贡献列表\n"
            "- experiments: 实验设置和结果\n"
            "- conclusions: 结论\n"
            "- future_work: 未来工作"
        ),
        agent=analyzer,
    )

    # Task 2: Quality evaluation
    quality_evaluation_task = Task(
        description=(
            f"对深度分析的论文进行学术质量评估：\n\n"
            f"评估维度：\n"
            f"1. 创新性（novelty）：0-10分\n"
            f"   - 研究问题是否新颖？方法是否有突破？\n"
            f"2. 技术深度（technical_depth）：0-10分\n"
            f"   - 理论分析是否严谨？技术细节是否充分？\n"
            f"3. 实验有效性（experimental_validity）：0-10分\n"
            f"   - 实验设计是否合理？结果是否可信？\n"
            f"4. 写作质量（writing_quality）：0-10分\n"
            f"   - 论文结构是否清晰？表达是否准确？\n"
            f"5. 综合评分（overall）：以上四项加权平均\n\n"
            f"对每篇论文还要给出：\n"
            f"- 优点（strengths）：2-3条\n"
            f"- 不足（weaknesses）：1-2条\n"
            f"- 改进建议（suggestions）：1-2条\n\n"
            f"使用file_writer工具将评估结果保存到quality_evaluation.json。"
        ),
        expected_output=(
            "每篇论文的质量评估报告，包含：\n"
            "- paper_title: 论文标题\n"
            "- quality_scores: 各维度评分（novelty, technical_depth, "
            "experimental_validity, writing_quality, overall）\n"
            "- strengths: 优点列表\n"
            "- weaknesses: 不足列表\n"
            "- suggestions: 改进建议"
        ),
        agent=evaluator,
        context=[deep_analysis_task],
    )

    return Crew(
        agents=[analyzer, evaluator],
        tasks=[deep_analysis_task, quality_evaluation_task],
        process=Process.sequential,
        verbose=True,
    )


def run_analysis(papers_info: str, max_papers: int | None = None) -> str:
    """Run the analysis workflow and return the result string."""
    crew = create_analysis_crew(papers_info, max_papers)
    result = crew.kickoff()
    return str(result)
