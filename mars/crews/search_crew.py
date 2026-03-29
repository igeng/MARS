"""
Search Crew - 基础检索流程

Flow:
  User topic → Researcher Agent (domain analysis + venue recommendation)
             → Searcher Agent (paper retrieval + initial ranking)
             → Return paper list
"""

from __future__ import annotations

from crewai import Crew, Process, Task

from mars.agents.researcher import create_researcher_agent
from mars.agents.searcher import create_searcher_agent
from mars.config import settings


def create_search_crew(topic: str, max_results: int | None = None) -> Crew:
    """
    Build and return the Search Crew for a given research topic.

    Args:
        topic:       The research topic or question from the user.
        max_results: Maximum number of papers to retrieve (default from settings).
    """
    max_papers = max_results or settings.MAX_PAPERS_PER_SEARCH

    researcher = create_researcher_agent()
    searcher = create_searcher_agent()

    # Task 1: Domain analysis and venue recommendation
    domain_analysis_task = Task(
        description=(
            f"分析以下研究主题，识别研究领域和子领域，"
            f"并推荐最相关的CCF推荐期刊/会议列表：\n\n"
            f"研究主题：{topic}\n\n"
            f"请完成以下工作：\n"
            f"1. 识别主研究领域和2-4个子领域\n"
            f"2. 提取3-8个核心关键词（包含英文）\n"
            f"3. 使用keyword_expander工具扩展关键词\n"
            f"4. 使用ccf_database_query工具查询相关期刊/会议\n"
            f"5. 推荐Top-5最相关的期刊/会议，注明CCF等级\n\n"
            f"输出格式要求：结构化的JSON格式，包含research_domain、"
            f"sub_domains、keywords、recommended_venues字段。"
        ),
        expected_output=(
            "一个结构化的JSON报告，包含：\n"
            "- research_domain: 主研究领域名称\n"
            "- sub_domains: 子领域列表\n"
            "- keywords: 核心关键词列表（中英文）\n"
            "- recommended_venues: 推荐期刊/会议列表，每项包含name、"
            "ccf_rank、relevance_reason"
        ),
        agent=researcher,
    )

    # Task 2: Paper retrieval
    paper_search_task = Task(
        description=(
            f"根据领域分析结果，在多个学术数据库中检索相关论文。\n\n"
            f"原始研究主题：{topic}\n"
            f"目标检索数量：{max_papers}篇\n\n"
            f"请完成以下工作：\n"
            f"1. 参考领域分析任务的关键词和推荐期刊\n"
            f"2. 使用dblp_search工具在DBLP中检索（重点检索A/B类会议期刊）\n"
            f"3. 使用semantic_scholar_search工具补充检索（按引用量过滤）\n"
            f"4. 使用arxiv_search工具检索最新预印本\n"
            f"5. 合并去重，按相关性和质量排序\n"
            f"6. 返回最多{max_papers}篇论文的结构化列表\n\n"
            f"输出要求：每篇论文包含title、authors、venue、year、"
            f"citation_count（如有）、url、relevance_reason。"
        ),
        expected_output=(
            f"最多{max_papers}篇论文的结构化列表，每篇包含：\n"
            "- title: 论文标题\n"
            "- authors: 作者列表\n"
            "- venue: 发表期刊/会议\n"
            "- year: 发表年份\n"
            "- citation_count: 引用次数（如可获取）\n"
            "- url: 论文链接\n"
            "- relevance_reason: 与研究主题的相关性说明"
        ),
        agent=searcher,
        context=[domain_analysis_task],
    )

    return Crew(
        agents=[researcher, searcher],
        tasks=[domain_analysis_task, paper_search_task],
        process=Process.sequential,
        verbose=True,
    )


def run_search(topic: str, max_results: int | None = None) -> str:
    """Run the search workflow and return the result string."""
    crew = create_search_crew(topic, max_results)
    result = crew.kickoff()
    return str(result)
