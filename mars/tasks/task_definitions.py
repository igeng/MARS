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
            f"对以下论文进行全面深入的学术解析（最多{limit}篇）。"
            f"{papers_block}\n\n"
            f"研究主题：{topic}\n\n"
            f"【任务要求】\n"
            f"对每篇论文，请依次完成：\n"
            f"1. 使用arxiv_search或semantic_scholar_search检索该论文，获取：\n"
            f"   - 作者、年份、发表会议/期刊、摘要\n"
            f"   - 论文下载链接（arxiv URL、DOI或PDF链接）\n"
            f"2. 如有PDF链接，使用pdf_parser工具提取全文内容\n"
            f"3. 对每篇论文进行深度分析，提取以下所有要素：\n"
            f"   - 背景（Background）：该研究的领域背景和历史发展\n"
            f"   - 挑战（Challenges）：该研究要解决的核心难题\n"
            f"   - 研究问题（Research Questions）：明确的研究问题和目标\n"
            f"   - 动机（Motivation）：为何要开展该研究\n"
            f"   - 贡献（Contributions）：论文的主要创新点（3-5条）\n"
            f"   - 方法解析（Methodology）：核心技术方案和架构的详细说明\n"
            f"   - 实验设置（Experimental Setup）：数据集、评估指标、实验环境\n"
            f"   - 对比方法（Baselines）：与哪些方法进行了比较\n"
            f"   - 实验维度（Experimental Aspects）：从哪些方面进行了评估\n"
            f"   - 实验结果（Results）：主要实验数据和发现\n"
            f"   - 结论（Conclusion）：论文的主要结论和未来工作方向\n\n"
            f"输出每篇论文的结构化Markdown分析文本，包含上述所有要素，"
            f"并确保每篇都附有可用的下载链接。"
        ),
        expected_output=(
            f"前{limit}篇论文的完整深度分析，以Markdown格式输出，每篇包含：\n"
            "- 标题、作者、年份、发表会议/期刊、下载链接\n"
            "- 背景（Background）\n"
            "- 挑战（Challenges）\n"
            "- 研究问题（Research Questions）\n"
            "- 动机（Motivation）\n"
            "- 主要贡献（Contributions，3-5条）\n"
            "- 方法解析（Methodology）\n"
            "- 实验设置（Experimental Setup）\n"
            "- 对比方法（Baselines）\n"
            "- 实验维度（Experimental Aspects）\n"
            "- 实验结果（Results）\n"
            "- 结论（Conclusion）"
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
    """Task: build citation network, topic clusters, research trends; save JSON + Markdown."""
    papers_block = f"\n\n{papers_info}" if papers_info else ""
    return Task(
        description=(
            f"对全部检索到的论文进行全面深入的关联分析。{papers_block}\n\n"
            f"研究主题：{topic}\n\n"
            f"【输出说明】本任务将生成两份文件：\n"
            f"1. connection_analysis.json — 机器可读的结构化关联数据\n"
            f"2. connection_analysis_report.md — 供人阅读的深度叙述分析报告（2000+字）\n\n"
            f"【第一步：构建结构化关联数据（保存为JSON）】\n"
            f"1. 如果论文有Semantic Scholar ID，使用citation_network_builder"
            f"   工具构建引用网络\n"
            f"2. 分析论文之间的主题相似度（基于关键词、研究方法、实验基准）\n"
            f"3. 识别论文集合中的主题聚类（3-5个聚类），每个聚类列出所有相关论文\n"
            f"4. 分析研究趋势：\n"
            f"   - 新兴研究方向（近2年大量出现的主题，列举具体论文）\n"
            f"   - 成熟稳定的研究方向（经典论文和奠基工作）\n"
            f"   - 潜在的研究空白（缺失的连接和未被充分研究的问题）\n"
            f"5. 识别领域内的核心论文（高引用、奠基性工作，至少列出10篇）\n"
            f"6. 使用file_writer工具将结构化数据保存到 connection_analysis.json，"
            f"   字段包括：citation_network_summary、topic_clusters、"
            f"   research_trends、key_papers、similarity_groups、"
            f"   methodology_lineage（方法传承链）、benchmark_groups（共用实验基准）\n\n"
            f"【第二步：生成深度关联分析报告（保存为Markdown）】\n"
            f"完成JSON保存后，基于以上分析结果撰写一份深度叙述报告，"
            f"使用file_writer工具保存到 connection_analysis_report.md，"
            f"报告结构如下（不少于2000字）：\n"
            f"1. **论文集合全景概述**：论文总数、年份分布（列出各年数量）、"
            f"   主要来源（会议/期刊及论文数），主要研究机构和高频作者群体\n"
            f"2. **引用网络深度解析**：核心高引用节点论文的地位与贡献分析"
            f"   （每篇2-3段），奠基性论文→发展性论文→最新工作的影响链条\n"
            f"3. **主题聚类详细分析**：每个聚类的核心论文（列举具体标题），"
            f"   聚类内论文之间的相互借鉴关系，聚类间的交叉点和联结\n"
            f"4. **研究演进路径**：按时间线展示领域关键里程碑，"
            f"   重要研究范式转变，每个阶段的代表性工作（列举论文标题和年份）\n"
            f"5. **论文间深层关联分析**：方法论传承（哪些论文方法影响了哪些后续工作），"
            f"   问题域关联（研究同一核心问题的论文群），"
            f"   实验基准关联（使用相同数据集/评测基准的论文组）\n"
            f"6. **研究热点与空白分析**：当前最活跃的研究主题（近2年高频关键词），"
            f"   从网络结构中识别出的研究空白，最有潜力的跨领域融合机会\n"
            f"7. **推荐阅读路径**：入门必读（3-5篇奠基论文），"
            f"   技术进阶（5-8篇方法深度论文），前沿追踪（近2年重要工作）\n\n"
            f"重点关注：{topic}"
        ),
        expected_output=(
            "两份输出文件：\n"
            "1. connection_analysis.json：结构化关联数据，包含：\n"
            "   - citation_network_summary: 引用网络摘要（节点数、边数）\n"
            "   - topic_clusters: 3-5个主题聚类，每个聚类含全部相关论文列表\n"
            "   - research_trends: 研究趋势（新兴/成熟/空白方向，各含具体论文）\n"
            "   - key_papers: 至少10篇核心论文列表\n"
            "   - similarity_groups: 高相似度论文组\n"
            "   - methodology_lineage: 方法传承关系\n"
            "   - benchmark_groups: 共用实验基准的论文组\n"
            "2. connection_analysis_report.md：深度叙述报告（不少于2000字），包含：\n"
            "   - 论文集合全景概述（数量/年份/来源统计）\n"
            "   - 引用网络深度解析（核心论文分析）\n"
            "   - 各聚类详细内部关系分析\n"
            "   - 研究演进路径（时间线）\n"
            "   - 论文间深层关联（方法传承/问题域/实验基准）\n"
            "   - 研究热点与空白\n"
            "   - 推荐阅读路径"
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
    """Task: translate the English literature review into Chinese."""
    return Task(
        description=(
            f"将上一步生成的英文文献综述（review_en.md）翻译成高质量的中文版本。\n\n"
            f"综述主题：{topic}\n\n"
            f"【翻译要求】\n"
            f"1. 保持与英文版完全一致的结构和内容，不要省略或增删任何部分\n"
            f"2. 专业术语首次出现时附英文原文，格式：中文术语（English Term）\n"
            f"3. 语言流畅自然，符合中文学术写作规范\n"
            f"4. 保留所有参考文献引用标记（[N]格式）\n"
            f"5. 参考文献列表中的作者名和标题保持英文，仅翻译说明性文字\n"
            f"6. 使用file_writer工具将中文综述保存到 review_zh.md"
        ),
        expected_output=(
            "完整的中文学术文献综述（Markdown格式），内容与英文版一致，包含：\n"
            "- 标题、摘要、引言\n"
            "- 研究现状、主要方法与技术、对比分析表格\n"
            "- 开放问题、未来研究方向、结论\n"
            "- 完整参考文献列表（格式规范，含DOI/URL）"
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
    """Task: multi-dimensional quality evaluation of papers, producing comprehensive analysis_report.md."""
    return Task(
        description=(
            f"基于上一步的深度解析结果，对每篇论文（最多{limit}篇）进行学术质量评估，"
            f"并生成完整的综合分析报告（Markdown格式）。\n\n"
            f"【评估维度】\n"
            f"1. 创新性（novelty）：0-10分 — 研究问题和方法是否新颖突破\n"
            f"2. 技术深度（technical_depth）：0-10分 — 理论是否严谨，技术细节是否充分\n"
            f"3. 实验有效性（experimental_validity）：0-10分 — 实验设计是否合理，结果是否可信\n"
            f"4. 写作质量（writing_quality）：0-10分 — 论文结构是否清晰，表达是否准确\n"
            f"5. 综合评分（overall）：以上四项加权平均\n\n"
            f"【每篇论文还需提供】\n"
            f"- 优势（Strengths）：2-3条主要优点\n"
            f"- 不足（Weaknesses）：1-2条主要局限\n"
            f"- 改进建议（Suggestions）：1-2条具体改进方向\n\n"
            f"【最终输出格式要求】\n"
            f"直接输出完整的Markdown报告，结构如下：\n"
            f"1. 报告标题（含分析日期和论文总数）\n"
            f"2. 概述段落\n"
            f"3. 评分汇总表格（各论文在novelty、technical_depth、"
            f"experimental_validity、writing_quality、overall五个维度的评分对比）\n"
            f"4. 逐篇论文完整分析（每篇包含：元信息+下载链接、背景、挑战、"
            f"研究问题、动机、贡献、方法解析、实验设置、对比方法、实验维度、"
            f"实验结果、结论、质量评分、优势、不足、改进建议）\n"
            f"5. 研究缺口识别与未来方向总结\n\n"
            f"注意：\n"
            f"1. 使用file_writer工具将完整报告保存到 analysis_report.md\n"
            f"2. 同时将完整Markdown内容作为任务结果直接输出（供后续任务使用）"
        ),
        expected_output=(
            "一份完整的Markdown格式综合分析报告，包含：\n"
            "- 报告标题和概述（含分析日期、论文总数）\n"
            "- 评分汇总表格（novelty, technical_depth, experimental_validity, "
            "writing_quality, overall各维度对比）\n"
            "- 每篇论文的完整分析（元信息含下载链接、深度分析各要素、"
            "质量评分+优势+不足+改进建议）\n"
            "- 研究缺口识别与未来方向总结"
        ),
        agent=agent,
        context=context or [],
    )


# ---------------------------------------------------------------------------
# 7. Synthesizer tasks (full research workflow only)
# ---------------------------------------------------------------------------

def create_full_research_synthesis_task(
    agent: Agent,
    topic: str,
    *,
    context: list[Task] | None = None,
) -> Task:
    """Task: generate a comprehensive final synthesis report for the full research workflow.

    This is distinct from review_zh.md (Chinese translation) — it provides an
    executive-level synthesis combining statistics, quality rankings, connection
    insights, and actionable future directions in a single authoritative document.
    """
    return Task(
        description=(
            f"基于所有前序任务的完整结果，生成一份全面深入的综合研究报告。\n\n"
            f"研究主题：{topic}\n\n"
            f"【注意】本报告与 review_zh.md（纯文献综述翻译）定位不同："
            f"本报告是执行层面的综合分析，侧重统计数据、质量评级、关联洞见和"
            f"可操作建议，而非学术综述正文。\n\n"
            f"【报告结构要求（总字数不少于3000字）】\n\n"
            f"# 一、报告封面与执行摘要\n"
            f"- 报告标题、研究主题、生成日期\n"
            f"- 执行摘要（300-500字）：本次研究的核心发现、领域现状一句话判断、"
            f"  最关键的5个研究洞见（每条需引用具体论文）\n\n"
            f"# 二、检索与分析全景统计\n"
            f"- 检索论文总数，覆盖年份范围\n"
            f"- 发表年份分布表格（每年论文数量）\n"
            f"- 来源分布表格（按会议/期刊名称及论文数量排列，至少列出Top-10）\n"
            f"- 高质量论文数量（CCF A类、B类、C类、其他分别列出）\n"
            f"- 深度分析论文数量；关联分析覆盖论文数量\n\n"
            f"# 三、领域研究全景\n"
            f"- 基于领域分析结果：主研究域及3-5个子领域名称\n"
            f"- 每个子领域的代表性论文（列举具体标题）及最新进展\n"
            f"- 核心关键词云分析（列出频率最高的10-15个关键词及其出现论文数）\n"
            f"- 主要研究机构和高产作者分析\n\n"
            f"# 四、Top-10 精选论文深度点评\n"
            f"按综合质量评分从高到低列出前10篇论文，每篇提供：\n"
            f"- 标题、作者、年份、发表会议/期刊\n"
            f"- 四维评分：novelty / technical_depth / experimental_validity / "
            f"  writing_quality（各项分数，精确到0.1）\n"
            f"- 综合评分（overall）\n"
            f"- 核心贡献（2-3句话，具体）\n"
            f"- 在领域中的地位和影响（1-2句话）\n\n"
            f"# 五、论文关联网络核心洞见\n"
            f"- 核心影响力论文（引用量最高的5篇）及其影响链条分析\n"
            f"- 主要研究聚类（至少3个）的代表论文与聚类特征\n"
            f"- 关键方法传承链：识别出至少3条 \"A论文方法 -> B论文发展 -> C论文改进\" 链条\n"
            f"- 研究范式演变：按时间顺序列出3-5个重要范式转变节点（附论文和年份）\n\n"
            f"# 六、质量评估综合分析\n"
            f"- 四维度评分汇总表（所有深度分析论文，含平均分、最高分、最低分）\n"
            f"- 领域整体学术质量水平判断（每个维度的平均水平及分析）\n"
            f"- 高质量论文特征总结（优秀论文的共同特点）\n"
            f"- 常见问题与不足（多篇论文共同存在的局限）\n\n"
            f"# 七、研究空白与未来方向\n"
            f"- 当前研究主要局限（从多篇论文综合分析，不少于3条，每条举例说明）\n"
            f"- 关键未解问题清单（3-5个具体、可操作的科学问题）\n"
            f"- 最有前景的研究方向（3-5个方向，每个方向给出具体研究建议和参考文献）\n"
            f"- 跨领域融合机会（与哪些其他领域结合有潜力，为什么）\n\n"
            f"# 八、本次研究输出文件导读指南\n"
            f"列表说明本次研究生成的所有文件：\n"
            f"| 文件名 | 内容说明 | 适合读者 | 建议用途 |\n"
            f"|--------|----------|----------|----------|\n"
            f"（依次说明：domain_analysis.json、paper_search.json、"
            f"connection_analysis.json、connection_analysis_report.md、"
            f"analysis_report.md、review_en.md、review_zh.md、"
            f"full_research_report.md）\n\n"
            f"推荐阅读顺序（从概览到深入）\n\n"
            f"【写作要求】\n"
            f"- 总字数不少于3000字\n"
            f"- 语言：中文为主，专业术语首次出现附英文原文\n"
            f"- 每个结论必须有具体论文标题或数据支撑，不能空泛\n"
            f"- 所有表格数据要准确，不使用\u300c约\u300d等模糊表述\n"
            f"- 使用file_writer工具将报告保存到 full_research_report.md"
        ),
        expected_output=(
            "一份全面深入的综合研究报告（Markdown格式，不少于3000字），包含：\n"
            "- 执行摘要（5个具体研究洞见）\n"
            "- 检索统计（年份分布表、来源分布表、CCF等级统计）\n"
            "- 领域全景（子领域、关键词、机构分析）\n"
            "- Top-10精选论文深度点评（含四维评分）\n"
            "- 关联网络核心洞见（影响链条、方法传承、范式演变）\n"
            "- 质量评估汇总表（所有论文，含均值/最高/最低）\n"
            "- 研究空白与未来方向（可操作的具体建议）\n"
            "- 文件导读指南（表格形式）"
        ),
        agent=agent,
        context=context or [],
    )
