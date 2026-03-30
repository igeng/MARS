"""
Reusable task factory functions for all 6 MARS agents.

Each ``create_*_task`` function returns a ``crewai.Task`` that can be
plugged into any Crew.  Keeping task definitions separate from crew
assembly makes it easy to compose new workflows.
"""

from __future__ import annotations

from crewai import Agent, Task


# ---------------------------------------------------------------------------
# 1. Researcher Agent tasks
# ---------------------------------------------------------------------------

def create_domain_analysis_task(
    agent: Agent,
    topic: str,
    *,
    context: list[Task] | None = None,
) -> Task:
    """Task: analyse a research topic, identify CCF venues."""
    return Task(
        description=(
            f"分析以下研究主题，识别研究领域和子领域，"
            f"并推荐最相关的CCF推荐期刊/会议列表：\n\n"
            f"研究主题：{topic}\n\n"
            f"请完成以下工作：\n"
            f"1. 识别主研究领域和2-4个子领域\n"
            f"2. 提取3-8个核心关键词（包含英文）\n"
            f"3. 使用keyword_expander工具扩展关键词\n"
            f"4. 使用ccf_database_query工具查询相关期刊/会议\n"
            f"5. 推荐Top-5最相关的期刊/会议，注明CCF等级\n"
            f"6. 使用file_writer工具将结果保存到domain_analysis.json\n\n"
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
        agent=agent,
        context=context or [],
    )


# ---------------------------------------------------------------------------
# 2. Searcher Agent tasks
# ---------------------------------------------------------------------------

def create_paper_search_task(
    agent: Agent,
    topic: str,
    max_papers: int = 50,
    *,
    context: list[Task] | None = None,
) -> Task:
    """Task: search academic databases for papers on a topic."""
    return Task(
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
            f"6. 返回最多{max_papers}篇论文的结构化列表\n"
            f"7. 使用file_writer工具将检索结果保存到paper_search.json\n\n"
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
        agent=agent,
        context=context or [],
    )


# ---------------------------------------------------------------------------
# 3. Analyzer Agent tasks
# ---------------------------------------------------------------------------

def create_deep_analysis_task(
    agent: Agent,
    topic: str,
    papers_info: str = "",
    limit: int = 20,
    *,
    context: list[Task] | None = None,
) -> Task:
    """Task: deeply analyse papers — contributions, methods, experiments."""
    papers_block = f"\n\n{papers_info}" if papers_info else ""
    return Task(
        description=(
            f"对检索结果中排名靠前的论文进行深度解析（最多{limit}篇）。"
            f"{papers_block}\n\n"
            f"研究主题：{topic}\n\n"
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
            f"前{limit}篇论文的详细分析报告，包含：\n"
            "- title: 论文标题\n"
            "- abstract: 摘要\n"
            "- keywords: 关键词\n"
            "- research_method: 研究方法\n"
            "- main_contributions: 主要贡献列表\n"
            "- experiments: 实验设置和结果\n"
            "- conclusions: 结论\n"
            "- future_work: 未来工作"
        ),
        agent=agent,
        context=context or [],
    )


# ---------------------------------------------------------------------------
# 4. Connector Agent tasks
# ---------------------------------------------------------------------------

def create_connection_analysis_task(
    agent: Agent,
    topic: str,
    papers_info: str = "",
    *,
    context: list[Task] | None = None,
) -> Task:
    """Task: build citation network, topic clusters, research trends."""
    papers_block = f"\n\n{papers_info}" if papers_info else ""
    return Task(
        description=(
            f"对全部检索到的论文进行关联分析。{papers_block}\n\n"
            f"研究主题：{topic}\n\n"
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
        agent=agent,
        context=context or [],
    )


# ---------------------------------------------------------------------------
# 5. Summarizer Agent tasks
# ---------------------------------------------------------------------------

def create_review_generation_task(
    agent: Agent,
    topic: str,
    *,
    context: list[Task] | None = None,
) -> Task:
    """Task: synthesise a rigorous Chinese academic literature review."""
    return Task(
        description=(
            f"基于论文分析和关联分析结果，生成一份严谨深入的中文学术文献综述报告。\n\n"
            f"综述主题：{topic}\n\n"
            f"【综述结构要求】\n"
            f"1. **标题**：简明扼要，反映研究主题\n"
            f"2. **摘要**（约200字）：概括综述的主要内容、方法和结论\n"
            f"3. **引言**（约500字）：\n"
            f"   - 研究背景与动机\n"
            f"   - 该领域的重要性与应用价值\n"
            f"   - 综述的范围与组织结构说明\n"
            f"4. **研究现状**（约800字）：\n"
            f"   - 领域发展历程与重要里程碑\n"
            f"   - 主要研究流派与代表性工作（引用具体论文）\n"
            f"   - 近年来的研究热点与进展\n"
            f"5. **主要方法与技术**（约1000字）：\n"
            f"   - 按技术类别系统梳理各种研究方法\n"
            f"   - 每类方法的原理、代表论文及适用场景\n"
            f"   - 方法之间的演进关系\n"
            f"6. **对比分析**：\n"
            f"   - 各方法/模型在主要基准上的性能对比表格\n"
            f"   - 各方法的优势与局限性分析\n"
            f"7. **开放问题与挑战**（约400字）：\n"
            f"   - 当前尚未解决的核心难题\n"
            f"   - 理论与实践层面的主要挑战\n"
            f"8. **未来研究方向**（约400字）：\n"
            f"   - 有潜力的新兴研究方向\n"
            f"   - 跨领域融合机会\n"
            f"   - 具体可行的研究建议\n"
            f"9. **结论**（约200字）：总结全文核心观点\n"
            f"10. **参考文献**：按引用顺序列出所有引用论文，"
            f"格式：[序号] 作者. 论文标题. 期刊/会议, 年份. DOI/URL\n\n"
            f"【写作要求】\n"
            f"- 语言：中文为主，专业术语首次出现时附英文原文\n"
            f"- 客观严谨，用具体数据和实验结果支撑观点\n"
            f"- 每个观点必须引用具体论文（格式：[序号]）\n"
            f"- 内容深度：适合领域研究人员阅读\n"
            f"- 总字数不少于3000字\n"
            f"- 使用file_writer工具将综述保存到 review_zh.md"
        ),
        expected_output=(
            "一份完整的中文学术文献综述（Markdown格式），包含：\n"
            "- 标题、摘要、引言\n"
            "- 研究现状、主要方法与技术、对比分析表格\n"
            "- 开放问题、未来研究方向、结论\n"
            "- 完整参考文献列表（格式规范，含DOI/URL）\n"
            "- 总字数不少于3000字"
        ),
        agent=agent,
        context=context or [],
    )


def create_english_review_task(
    agent: Agent,
    topic: str,
    *,
    context: list[Task] | None = None,
) -> Task:
    """Task: synthesise a rigorous English academic literature review."""
    return Task(
        description=(
            f"Based on the paper analysis and relationship analysis results, "
            f"generate a rigorous and comprehensive English academic literature review.\n\n"
            f"Review Topic: {topic}\n\n"
            f"[Required Structure]\n"
            f"1. **Title**: Concise and reflective of the research topic\n"
            f"2. **Abstract** (~200 words): Summarize the scope, methods, and key findings\n"
            f"3. **Introduction** (~500 words):\n"
            f"   - Research background and motivation\n"
            f"   - Importance and application value of the field\n"
            f"   - Scope of the review and organization\n"
            f"4. **Research Overview** (~800 words):\n"
            f"   - Historical development and key milestones\n"
            f"   - Major research schools and representative works (cite specific papers)\n"
            f"   - Recent research trends and advances\n"
            f"5. **Methods and Techniques** (~1000 words):\n"
            f"   - Systematic taxonomy of research methodologies\n"
            f"   - Principles, representative papers, and use cases for each category\n"
            f"   - Evolution and relationships between approaches\n"
            f"6. **Comparative Analysis**:\n"
            f"   - Performance comparison table on major benchmarks\n"
            f"   - Strengths and limitations of each approach\n"
            f"7. **Open Problems and Challenges** (~400 words):\n"
            f"   - Core unsolved problems in the field\n"
            f"   - Theoretical and practical challenges\n"
            f"8. **Future Research Directions** (~400 words):\n"
            f"   - Promising emerging directions\n"
            f"   - Cross-domain fusion opportunities\n"
            f"   - Concrete actionable research suggestions\n"
            f"9. **Conclusion** (~200 words): Summarize key insights\n"
            f"10. **References**: List all cited papers in citation order, "
            f"format: [N] Authors. Title. Venue, Year. DOI/URL\n\n"
            f"[Writing Requirements]\n"
            f"- Language: English throughout\n"
            f"- Objective and rigorous; support claims with specific data and results\n"
            f"- Every claim must cite a specific paper (format: [N])\n"
            f"- Depth: suitable for domain researchers\n"
            f"- Minimum 3000 words\n"
            f"- Use file_writer tool to save the review to review_en.md"
        ),
        expected_output=(
            "A complete English academic literature review (Markdown format) including:\n"
            "- Title, abstract, introduction\n"
            "- Research overview, methods/techniques, comparative analysis table\n"
            "- Open problems, future directions, conclusion\n"
            "- Complete reference list (properly formatted with DOI/URL)\n"
            "- Minimum 3000 words"
        ),
        agent=agent,
        context=context or [],
    )


# ---------------------------------------------------------------------------
# 6. Evaluator Agent tasks
# ---------------------------------------------------------------------------

def create_quality_evaluation_task(
    agent: Agent,
    limit: int = 20,
    *,
    context: list[Task] | None = None,
) -> Task:
    """Task: multi-dimensional quality evaluation of papers."""
    return Task(
        description=(
            f"对深度分析的论文进行学术质量评估（最多{limit}篇）：\n\n"
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
        agent=agent,
        context=context or [],
    )
