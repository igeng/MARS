"""
Connection Crew - 关联分析流程

Flow:
  Multiple papers → Connector Agent (citation analysis + similarity)
                 → Summarizer Agent (literature review generation)
                 → Return relationship graph + review report
"""

from __future__ import annotations

from crewai import Crew, Process, Task

from mars.agents.connector import create_connector_agent
from mars.agents.summarizer import create_summarizer_agent


def create_connection_crew(papers_info: str, topic: str) -> Crew:
    """
    Build and return the Connection Crew.

    Args:
        papers_info: Structured info about the papers to analyze
                     (titles, IDs, abstracts, venues, years).
        topic:       The original research topic/theme.
    """
    connector = create_connector_agent()
    summarizer = create_summarizer_agent()

    # Task 1: Citation and relationship analysis
    connection_analysis_task = Task(
        description=(
            f"对以下论文集合进行关联分析：\n\n"
            f"{papers_info}\n\n"
            f"请完成以下工作：\n"
            f"1. 如果论文有Semantic Scholar ID，使用citation_network_builder"
            f"   工具构建引用网络\n"
            f"2. 分析论文之间的主题相似度（基于关键词和研究方法）\n"
            f"3. 识别论文集合中的主题聚类（2-4个聚类）\n"
            f"4. 分析研究趋势：\n"
            f"   - 新兴研究方向（近2年大量出现的主题）\n"
            f"   - 成熟稳定的研究方向\n"
            f"   - 潜在的研究空白\n"
            f"5. 识别领域内的核心论文（高引用、奠基性工作）\n"
            f"6. 使用file_writer工具保存网络分析结果到connection_analysis.json\n\n"
            f"重点关注：{topic}"
        ),
        expected_output=(
            "论文关联分析报告，包含：\n"
            "- citation_network_summary: 引用网络摘要（节点数、边数、核心论文）\n"
            "- topic_clusters: 主题聚类列表，每个聚类包含主题名和相关论文\n"
            "- research_trends: 研究趋势分析（新兴、成熟、空白方向）\n"
            "- key_papers: 领域核心论文列表"
        ),
        agent=connector,
    )

    # Task 2: Literature review synthesis
    review_generation_task = Task(
        description=(
            f"基于论文分析和关联分析结果，生成一份专业的领域综述报告。\n\n"
            f"综述主题：{topic}\n\n"
            f"综述报告结构：\n"
            f"1. **引言**：研究背景和综述意义（约300字）\n"
            f"2. **研究现状**：主要研究方向和代表性工作（约500字）\n"
            f"3. **主要方法**：按技术类别梳理主要研究方法（约600字）\n"
            f"4. **对比分析**：各方法的优缺点对比（表格形式）\n"
            f"5. **开放问题**：当前未解决的关键挑战（约300字）\n"
            f"6. **未来方向**：有潜力的研究方向和建议（约300字）\n"
            f"7. **参考文献**：按顺序列出引用的论文\n\n"
            f"要求：\n"
            f"- 综述语言：中文为主，专业术语保留英文\n"
            f"- 引用具体论文支撑观点\n"
            f"- 内容客观准确，覆盖全面\n"
            f"- 使用file_writer工具将综述保存到literature_review.md"
        ),
        expected_output=(
            "一份完整的领域综述报告（Markdown格式），包含：\n"
            "- 标题和摘要\n"
            "- 引言、研究现状、主要方法、对比分析\n"
            "- 开放问题、未来方向\n"
            "- 参考文献列表"
        ),
        agent=summarizer,
        context=[connection_analysis_task],
    )

    return Crew(
        agents=[connector, summarizer],
        tasks=[connection_analysis_task, review_generation_task],
        process=Process.sequential,
        verbose=True,
    )


def run_connection(papers_info: str, topic: str) -> str:
    """Run the connection workflow and return the result string."""
    crew = create_connection_crew(papers_info, topic)
    result = crew.kickoff()
    return str(result)
