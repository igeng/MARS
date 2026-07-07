# MARS 项目技术文档

> **目标读者**：本文档面向 Claude Code、Codex 等 AI 编程智能体，旨在帮助新的智能体快速、准确地理解 MARS 项目的完整技术架构，以便高效地进行后续优化与迭代工作。

---

## 目录

1. [项目概述](#1-项目概述)
2. [技术栈](#2-技术栈)
3. [目录结构](#3-目录结构)
4. [核心架构](#4-核心架构)
5. [配置系统](#5-配置系统)
6. [LLM 网关](#6-llm-网关)
7. [智能体（Agents）](#7-智能体agents)
8. [工具（Tools）](#8-工具tools)
9. [任务定义（Tasks）](#9-任务定义tasks)
10. [工作流（Crews）](#10-工作流crews)
11. [CLI 接口](#11-cli-接口)
12. [REST API](#12-rest-api)
13. [数据库](#13-数据库)
14. [工具函数](#14-工具函数)
15. [测试](#15-测试)
16. [依赖管理](#16-依赖管理)
17. [关键设计决策与注意事项](#17-关键设计决策与注意事项)
18. [常见扩展点](#18-常见扩展点)

---

## 1. 项目概述

MARS（Multi-Agent Research System）是一个**多智能体学术文献智能检索与分析系统**。系统接收用户输入的研究主题，通过多个专业化 AI 智能体协作，自动完成以下全流程：

```
用户输入研究主题
  → 领域分析（识别子领域、关键词、推荐 CCF 期刊会议）
  → 批量论文检索（DBLP + Semantic Scholar + arXiv，最多 50 篇）
  → 深度论文解析（每篇从背景到结论的全方位分析，最多 20 篇）
  → 关联分析（引用网络、主题聚类、研究趋势）
  → 质量评估（四维评分 + 专业审稿意见）
  → 双语综述生成（英文 3000+ 字 + 中文翻译版）
  → 综合研究报告（统计 + 排名 + 洞见 + 导读）
```

系统最终产出 **10 份文件**（完整流程），支持通过 CLI、REST API 或 Python import 三种方式调用。

---

## 2. 技术栈

| 层次 | 技术 | 说明 |
|------|------|------|
| 多智能体框架 | `crewai >= 0.80.0` | 智能体编排、任务调度 |
| LLM 路由 | `litellm`（crewai 内置） | 统一路由各 LLM 提供商 |
| LLM SDK | `langchain-openai` | 工具直调 LLM（非 Agent 模式） |
| LLM 提供商 | Qwen（DashScope）、DeepSeek、Kimi（Moonshot）、GLM（Zhipu） | 见 [LLM 网关](#6-llm-网关) |
| Web 框架 | `fastapi + uvicorn` | REST API |
| CLI | `typer + rich` | 命令行界面 |
| 配置管理 | `pydantic-settings` | 从环境变量 / .env 读取配置 |
| 数据库 | `sqlalchemy + sqlite` | CCF 场馆表 + 论文缓存表 |
| PDF 解析 | `PyPDF2 + pymupdf(fitz)` | 提取论文全文 |
| 网络分析 | `networkx` | 引用网络图分析 |
| 数值计算 | `numpy + pandas + scikit-learn` | 数据处理、聚类 |
| 向量数据库 | `chromadb` | CrewAI memory（默认关闭） |
| 学术数据库 | DBLP API、Semantic Scholar API、arXiv API | 无需 key / 可选 key |
| 评估基准 | [SurGE](https://github.com/oneal2000/SurGE) (SIGIR 2026) | 205 CS topics + 1.09M papers |
| 日志 | stdlib `logging` | 集中配置，见 `mars/utils/logging_config.py` |

---

## 3. 目录结构

```
MARS/
├── .env.example             # 环境变量模板
├── main.py                  # 入口（调用 mars.cli:app）
├── pyproject.toml           # 项目元数据 + 依赖声明
├── requirements.txt         # pip install -r 等价依赖（与 pyproject.toml 同步）
├── MARS.md                  # 本文档（AI 编程智能体技术参考）
├── README.md                # 用户文档
├── docs/
│   ├── user-guide.md        # 详细用户指南（含 Win10 + Conda + PyCharm 部署）
│   └── technical-guide.md   # 技术指南
├── mars/
│   ├── __init__.py
│   ├── cli.py               # CLI 入口（typer app，含 _run_workflow 通用执行器）
│   ├── py.typed             # PEP 561 类型标记
│   ├── config/
│   │   ├── __init__.py      # 导出 settings 单例
│   │   └── settings.py      # MarsSettings（pydantic-settings）
│   ├── services/
│   │   └── llm_gateway.py   # LLM 工厂 + 路由 + 限流回退（RateLimitAwareLLM）
│   ├── data/
│   │   └── ccf_2025.json    # CCF 期刊/会议列表（JSON，支持版本切换）
│   ├── agents/
│   │   ├── researcher.py    # 领域分析师
│   │   ├── searcher.py      # 论文检索师
│   │   ├── analyzer.py      # 深度分析师
│   │   ├── connector.py     # 关联分析师
│   │   ├── evaluator.py     # 质量评估师
│   │   └── summarizer.py    # 综述生成师
│   ├── tasks/
│   │   ├── task_definitions.py  # 所有任务工厂函数（8 类任务）
│   │   └── prompts/             # Prompt 模板文件（.txt，str.format 占位符）
│   │       ├── domain_analysis_task.txt
│   │       ├── paper_search_task.txt
│   │       ├── deep_analysis_task.txt
│   │       ├── connection_analysis_task.txt
│   │       ├── english_review_task.txt
│   │       ├── review_generation_task.txt
│   │       ├── quality_evaluation_task.txt
│   │       └── full_research_synthesis_task.txt
│   ├── crews/
│   │   ├── search_crew.py       # 基础检索流程（3 智能体）
│   │   ├── analysis_crew.py     # 深度分析流程（2 智能体）
│   │   ├── connection_crew.py   # 关联分析流程（2 智能体）
│   │   └── full_research_crew.py # 完整研究流程（6 智能体，8 任务顺序执行）
│   ├── tools/
│   │   ├── arxiv_api.py         # arXiv 检索工具
│   │   ├── dblp_search.py       # DBLP 检索工具
│   │   ├── semantic_scholar.py  # Semantic Scholar 检索工具
│   │   ├── citation_network.py  # 引用网络构建工具
│   │   ├── keyword_expander.py  # 关键词扩展工具（调用 LLM）
│   │   ├── ccf_database.py      # CCF 场馆（从 JSON 加载，内置回退数据）
│   │   ├── pdf_parser.py        # PDF 解析工具
│   │   └── file_manager.py      # 文件读写工具
│   ├── evaluation/               # SurGE benchmark 评估模块
│   │   ├── surge_eval.py          # Citation Recall/Precision/F1
│   │   ├── surge_adapter.py       # SurGE 数据格式适配器
│   │   ├── surge_metrics.py       # SurGE 官方 8 指标包装
│   │   ├── benchmark_runner.py    # 批量对比 (MARS vs 3 baselines)
│   │   ├── llm_judge.py           # 12 维 LLM-as-Judge 评分
│   │   ├── hallucination_checker.py
│   │   ├── full_eval.py           # 一键全维度评估
│   │   └── metrics.py             # 核心指标计算
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py              # FastAPI 应用（异步任务模式，6 个端点）
│   ├── database/
│   │   ├── __init__.py
│   │   └── models.py            # SQLAlchemy 模型（CCFVenue、Paper）
│   └── utils/
│       ├── logging_config.py    # 集中日志配置（支持 run_id 注入）
│       ├── llm_factory.py       # ⚠️ 已废弃兼容层（v0.3.0 移除）
│       └── retry.py             # HTTP 重试装饰器
└── tests/
    ├── conftest.py
    ├── test_basic.py
    ├── test_new_components.py
    ├── test_usage_features.py
    └── test_gateway_and_crews.py  # RateLimitAwareLLM + Crew 编排测试
```

---

## 4. 核心架构

### 整体数据流

```
CLI / API
    ↓
Crew.kickoff(topic)          ← crewai.Crew（Process.sequential）
    ↓
Task 1: domain_analysis      ← Researcher Agent
    ↓
Task 2: paper_search         ← Searcher Agent
    ↓
Task 3: deep_analysis        ← Analyzer Agent
    ↓
Task 4: connection_analysis  ← Connector Agent
    ↓
Task 5: quality_evaluation   ← Evaluator Agent
    ↓
Task 6: english_review       ← Summarizer Agent
    ↓
Task 7: chinese_review       ← Summarizer Agent（复用同一实例）
    ↓
Task 8: synthesis_report     ← Summarizer Agent
    ↓
输出文件保存至 OUTPUT_DIR/<prefix>_<timestamp>/
```

### 关键原则

- **任务解耦**：`task_definitions.py` 中的工厂函数独立于具体 Crew，每个工厂返回一个 `crewai.Task` 实例，可被任意 Crew 复用。
- **LLM 透明回退**：通过 `llm_gateway._resolve_provider()` 实现，当首选 provider 无 key 时自动回退至 `DEFAULT_LLM_PROVIDER` 或第一个可用 provider。
- **文件持久化**：Agent 通过 `FileWriterTool` 将结果写入 `settings.OUTPUT_DIR`，CLI 每次运行创建带时间戳的子目录。
- **内存关闭**：`ENABLE_MEMORY=False`（默认），避免 ChromaDB 依赖 OpenAI 嵌入 API。

---

## 5. 配置系统

**文件**：`mars/config/settings.py`

使用 `pydantic-settings` 的 `BaseSettings`，从 `.env` 文件（项目根目录）和环境变量自动加载。

```python
from mars.config import settings   # 全局单例

settings.DASHSCOPE_API_KEY        # str，默认 ""
settings.DEEPSEEK_API_KEY         # str，默认 ""
settings.MOONSHOT_API_KEY         # str，默认 ""
settings.ZHIPU_API_KEY            # str，默认 ""

settings.QWEN_MODEL               # str，默认 "qwen3.5-flash"
settings.DEEPSEEK_MODEL           # str，默认 "deepseek-chat"
settings.KIMI_MODEL               # str，默认 "kimi-k2.5"
settings.GLM_MODEL                # str，默认 "glm-4.7-flash"

settings.DEFAULT_LLM_PROVIDER     # str，默认 "qwen"
settings.OUTPUT_DIR               # Path，默认 "./output"
settings.AGENT_MAX_ITER           # int，默认 10（所有 Agent 的 max_iter）
settings.MAX_PAPERS_PER_SEARCH    # int，默认 50
settings.MAX_PAPERS_FOR_ANALYSIS  # int，默认 20
settings.ARXIV_SEARCH_TIMEOUT     # int，默认 30（秒）
settings.ENABLE_MEMORY            # bool，默认 False
settings.DATABASE_URL             # str，默认 "sqlite:///./mars.db"
settings.API_HOST                 # str，默认 "0.0.0.0"
settings.API_PORT                 # int，默认 8000
settings.LOG_LEVEL                # str，默认 "INFO"

# GLM 限流重试参数
settings.GLM_RATE_LIMIT_MAX_RETRIES   # int，默认 3
settings.GLM_RATE_LIMIT_RETRY_DELAY   # float，默认 5.0（秒，指数退避基数）
```

**注意**：`settings` 是模块级单例，在 CLI 的 `_startup()` 中会动态修改 `settings.OUTPUT_DIR` 指向当前运行的时间戳目录。

---

## 6. LLM 网关

**文件**：`mars/services/llm_gateway.py`

这是整个系统中最复杂的模块，管理所有 LLM 实例的创建、路由和回退。

### 两类 LLM 对象

| 类型 | 返回类型 | 使用场景 | 函数 |
|------|---------|---------|------|
| ChatOpenAI | `langchain_openai.ChatOpenAI` | 工具内部直调（`KeywordExpanderTool`） | `get_llm()` |
| crewai.LLM | `crewai.LLM` | CrewAI Agent 赋值 | `get_llm_by_task()` |

### 四个 LLM 提供商配置

| Provider | 模型变量 | API Key 变量 | Base URL | LiteLLM 前缀 |
|----------|---------|-------------|----------|-------------|
| qwen | `QWEN_MODEL` | `DASHSCOPE_API_KEY` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `openai/` |
| deepseek | `DEEPSEEK_MODEL` | `DEEPSEEK_API_KEY` | `https://api.deepseek.com/v1` | `deepseek/` |
| kimi | `KIMI_MODEL` | `MOONSHOT_API_KEY` | `https://api.moonshot.cn/v1` | `openai/` |
| glm | `GLM_MODEL` | `ZHIPU_API_KEY` | `https://open.bigmodel.cn/api/paas/v4` | `openai/` |

**重要**：除 deepseek 外，所有 OpenAI 兼容端点均使用 `openai/<model>` + `base_url` 形式，**不能**使用 `dashscope/`、`moonshot/` 等特定前缀，否则 LiteLLM 会报 "LLM Provider NOT provided" 错误。

### Agent 角色 → LLM 映射

```python
_AGENT_LLM_MAP = {
    "researcher": ("qwen", None),
    "searcher":   ("qwen", None),
    "analyzer":   ("qwen", None),
    "connector":  ("qwen", None),
    "summarizer": ("qwen", None),
    "evaluator":  ("kimi", None),   # Kimi 专属，temperature 强制为 1.0
}
```

### Provider 回退逻辑

`_resolve_provider(provider)` 流程：
1. 若 `provider` 对应 key 已配置 → 直接使用
2. 若未配置 → 检查 `DEFAULT_LLM_PROVIDER` 是否可用 → 若可用则用之
3. 否则使用第一个有 key 的 provider
4. 若无任何 key → 返回原始 provider（构建成功，调用时报 auth 错误）

### Qwen3 系列特殊处理

Qwen3.x 模型默认开启 thinking 模式，会导致 `content` 为空。解决方案：在构建 `crewai.LLM` 时传入：

```python
extra_body={"enable_thinking": False}
```

此逻辑在 `_get_crewai_llm()` 中自动检测（判断 model 名前缀是否为 `qwen3`）。

### RateLimitAwareLLM

`crewai.LLM` 子类，专为 GLM 等免费模型设计，提供限流保护：

1. **指数退避重试**（最多 `GLM_RATE_LIMIT_MAX_RETRIES` 次）
2. **自动回退链**：耗尽重试后按序切换到 Qwen → Kimi
3. **签名解耦**：`call(*args, **kwargs)` 签名与 CrewAI 内部接口解耦，版本升级不会导致静默错误

`mars/utils/llm_factory.py` 为向后兼容层（⚠️ v0.3.0 将移除），新代码应直接从
`mars.services.llm_gateway` 导入。

---

## 7. 智能体（Agents）

所有 Agent 定义在 `mars/agents/` 下，每个文件导出一个 `create_*_agent()` 工厂函数，返回 `crewai.Agent` 实例。

| 文件 | Agent 角色 | LLM | max_iter | 工具 |
|------|-----------|-----|---------|------|
| `researcher.py` | 领域分析师 | qwen | 10 | CCFDatabaseQueryTool, KeywordExpanderTool |
| `searcher.py` | 论文检索师 | qwen | 10 | DBLPSearchTool, SemanticScholarSearchTool, ArXivSearchTool, KeywordExpanderTool, FileWriterTool |
| `analyzer.py` | 深度分析师 | qwen | 10 | ArXivSearchTool, SemanticScholarSearchTool, PDFParserTool, FileWriterTool |
| `connector.py` | 关联分析师 | qwen | 10 | CitationNetworkTool, SemanticScholarSearchTool, FileWriterTool |
| `evaluator.py` | 质量评估师 | kimi | 10 | SemanticScholarSearchTool, FileWriterTool |
| `summarizer.py` | 综述生成师 | qwen | 10 | FileWriterTool |

所有 Agent 的 `max_iter` 值由 `settings.AGENT_MAX_ITER` 统一控制（默认 10），
可通过 `.env` 环境变量覆盖。所有 Agent 均设置 `allow_delegation=False` 和 `verbose=True`。

---

## 8. 工具（Tools）

所有工具继承 `crewai.tools.BaseTool`，定义 `name`、`description`、`args_schema`（可选）和 `_run()` 方法。

### ArXivSearchTool（`arxiv_api.py`）

- **工具名**：`arxiv_search`
- **输入**：`query_json`（JSON 字符串，含 `query`、可选 `max_results`）
- **支持直传字段**：通过 `_coerce_direct_args` model_validator 支持直接传 `query` 字段
- **输出**：文本格式的论文列表（含 arxiv_id、标题、作者、发表日期、摘要前 400 字、PDF 链接）
- **超时**：`settings.ARXIV_SEARCH_TIMEOUT`（默认 30 秒）
- **重试**：`@retry_on_network_error(max_retries=3)`

### DBLPSearchTool（`dblp_search.py`）

- **工具名**：`dblp_search`
- **输入**：`query_json`（含 `query`、`max_results`、`year_from`、`year_to`）
- **API**：`https://dblp.org/search/publ/api`（无需 API key）
- **支持**：年份范围过滤、默认返回 20 条

### SemanticScholarSearchTool（`semantic_scholar.py`）

- **工具名**：`semantic_scholar_search`
- **输入**：`query_json`（含 `query`、`max_results`、`year_from`、`year_to`、`min_citations`）
- **API**：`https://api.semanticscholar.org/graph/v1/paper/search`
- **可选 key**：`settings.SEMANTIC_SCHOLAR_API_KEY`（有 key 可提升速率限制）
- **返回字段**：paperId、title、authors、year、venue、citationCount、abstract（前 300 字）、openAccessPdf.url

### CitationNetworkTool（`citation_network.py`）

- **工具名**：`citation_network_builder`
- **输入**：`{"paper_ids": [...], "max_refs_per_paper": 30}`（JSON 字符串）
- **功能**：调用 Semantic Scholar 批量获取引用关系，用 NetworkX 构建有向图，输出 Top-5 被引节点
- **注意**：仅支持 Semantic Scholar paper ID，需先通过 SemanticScholarSearchTool 获取 `paperId`

### KeywordExpanderTool（`keyword_expander.py`）

- **工具名**：`keyword_expander`
- **输入**：`keyword`（字符串）、可选 `provider`
- **功能**：调用 LLM（`get_llm()`，返回 ChatOpenAI）扩展关键词，输出同义词、相关概念、缩写、中文翻译

### CCFDatabaseQueryTool（`ccf_database.py`）

- **工具名**：`ccf_database_query`
- **输入**：查询字符串（研究领域名或关键词）
- **数据**：静态硬编码，约 20+ 个 A/B 类期刊会议（CVPR、ICML、NeurIPS、ACL 等）
- **匹配逻辑**：对 name、full_name 和 domains 字段进行字符串包含匹配

### PDFParserTool（`pdf_parser.py`）

- **工具名**：`pdf_parser`
- **输入**：PDF URL 或本地路径
- **功能**：优先用 `pymupdf(fitz)` 提取文本，回退到 `PyPDF2`

### FileWriterTool（`file_manager.py`）

- **工具名**：`file_writer`
- **参数**：`filename`（str）、`content`（str）、`mode`（`'w'` 或 `'a'`，默认 `'w'`）
- **写入位置**：`settings.OUTPUT_DIR`（运行时已被 CLI 重定向到时间戳子目录）
- **安全**：使用 `Path(filename).name` 防止路径穿越攻击
- **返回**：写入文件的绝对路径字符串

### FileReaderTool（`file_manager.py`）

- **工具名**：`file_reader`
- **参数**：`filename`（str）
- **读取位置**：`settings.OUTPUT_DIR`

---

## 9. 任务定义（Tasks）

**文件**：`mars/tasks/task_definitions.py`

提供 7 个任务工厂函数，每个返回 `crewai.Task`：

| 函数 | 对应 Agent | 主要输出文件 |
|------|-----------|------------|
| `create_domain_analysis_task(agent, topic)` | Researcher | `domain_analysis.json` |
| `create_paper_search_task(agent, topic, max_papers)` | Searcher | `paper_search.json` |
| `create_deep_analysis_task(agent, topic, papers_info, limit)` | Analyzer | （Markdown 文本，传递给后续任务） |
| `create_connection_analysis_task(agent, topic, papers_info)` | Connector | `connection_analysis.json`、`connection_analysis_report.md` |
| `create_quality_evaluation_task(agent, limit)` | Evaluator | `analysis_report.md` |
| `create_english_review_task(agent, topic)` | Summarizer | `review_en.md` |
| `create_review_generation_task(agent, topic)` | Summarizer | `review_zh.md`（中文翻译） |
| `create_full_research_synthesis_task(agent, topic)` | Summarizer | `full_research_report.md` |

所有工厂函数均接受可选的 `context: list[Task]` 参数，用于建立任务间依赖关系（CrewAI 会将上游任务输出注入到下游任务上下文）。

---

## 10. 工作流（Crews）

### search_crew（`mars/crews/search_crew.py`）

**流程**：领域分析 → 论文检索 → 英文综述 → 中文综述

```python
from mars.crews.search_crew import run_search
result = run_search(topic, max_results=50)
```

- 产出 5 份文件：`prompt.txt`、`run.log`、`paper_search.json`、`review_en.md`、`review_zh.md`
- 3 个 Agent：researcher、searcher、summarizer

### analysis_crew（`mars/crews/analysis_crew.py`）

**流程**：深度解析 → 质量评估

```python
from mars.crews.analysis_crew import run_analysis
result = run_analysis(papers_info, topic="", max_papers=20)
```

- 产出 3 份文件：`prompt.txt`、`run.log`、`analysis_report.md`
- 2 个 Agent：analyzer、evaluator

### connection_crew（`mars/crews/connection_crew.py`）

**流程**：引用网络分析 → 关联报告

```python
from mars.crews.connection_crew import run_connection
result = run_connection(papers_info, topic="federated learning")
```

- 产出文件：`connection_analysis.json`、`connection_analysis_report.md`
- 2 个 Agent：connector、summarizer

### full_research_crew（`mars/crews/full_research_crew.py`）

**完整流程**（8 个顺序任务）：

```
domain_analysis → bulk_search → deep_analysis → connection_analysis →
quality_evaluation → english_review → chinese_review → synthesis_report
```

```python
from mars.crews.full_research_crew import run_full_research
result = run_full_research(topic)
```

- 产出 10 份文件（见 CLI `full` 命令说明）
- 6 个 Agent：researcher、searcher、analyzer、connector、evaluator、summarizer
- **注意**：所有 Crew 均使用 `Process.sequential`，`memory=settings.ENABLE_MEMORY`

---

## 11. CLI 接口

**文件**：`mars/cli.py`

使用 `typer` 框架，入口为 `app`，在 `pyproject.toml` 中注册为 `mars` 命令。

### 可用命令

```bash
mars search  "<topic>"              # 基础检索流程
mars analyze "<papers>"             # 深度分析流程
mars connect "<papers>" --topic "<topic>"  # 关联分析流程
mars full    "<topic>"              # 完整研究流程
mars api                            # 启动 FastAPI 服务器
mars init-db                        # 初始化数据库
mars check                          # 检查系统配置状态
```

### 关键实现细节

**每次运行的目录隔离**：`_create_run_dir(prefix)` 在 `settings.OUTPUT_DIR` 下创建 `{prefix}_{timestamp}` 子目录，并通过 `_startup(run_dir)` 将 `settings.OUTPUT_DIR` 重定向到该子目录。

**日志 Tee**：`_run_context(run_dir)` 上下文管理器将 `sys.stdout` 和 `sys.stderr` 替换为 `_TeeWriter`（同时写到终端和 `run.log`），并重建 `Console` 和 logging。

**Windows 兼容**：启动时检测 `sys.platform == "win32"` 并将 stdout/stderr 重配置为 UTF-8（避免 GBK 编码 UnicodeEncodeError）。

---

## 12. REST API

**文件**：`mars/api/main.py`

使用 `FastAPI`，通过 `create_app()` 工厂函数创建应用实例，模块级 `app = create_app()` 供 `uvicorn mars.api.main:app` 使用。

### 端点（异步任务模式，v0.2.0）

所有工作流端点使用"提交-轮询"模式：`POST` 返回 HTTP 202 + `task_id`，
通过 `GET /task/{task_id}` 轮询结果。

| 方法 | 路径 | 请求体类型 | 说明 |
|------|------|----------|------|
| GET | `/health` | — | 健康检查 |
| GET | `/task/{task_id}` | — | 查询任务状态与结果 |
| POST | `/search` | `SearchRequest` | 基础检索（异步） |
| POST | `/analyze` | `AnalyzeRequest` | 深度分析（异步） |
| POST | `/connect` | `ConnectRequest` | 关联分析（异步） |
| POST | `/full-research` | `FullResearchRequest` | 完整研究（异步） |

### 请求/响应模型

```python
# 请求示例（不变）
SearchRequest(topic="federated learning", max_results=50)

# POST 响应（新）
TaskAcceptedResponse(task_id="a1b2c3d4e5f6", status="pending")

# GET /task/{task_id} 响应
TaskInfo(
    task_id="a1b2c3d4e5f6",
    status="success",        # pending → running → success/failed
    result="<result text>",  # 仅在 status=success 时有值
    error=""                 # 仅在 status=failed 时有值
)
```

### Swagger UI

启动后访问 `http://localhost:8000/docs`。

---

## 12b. SurGE Benchmark 评估

MARS 使用 **[SurGE](https://github.com/oneal2000/SurGE)** 基准进行评估（SIGIR 2026）。
SurGE 提供 205 个 CS 领域的人工撰写综述作为 ground truth，以及 1.09M 篇 arXiv 论文作为检索语料库。

**评估方式**：

```bash
# 检索模式切换到 SurGE 语料库
mars full --mode surge --eval "federated learning"

# 加载 SurGE 基准对比（Autosurvey / ID / Naive）
mars benchmark --action compare

# 单项全维度评估
mars evaluate --survey review_en.md "topic"
```

**评估模块**：`mars/evaluation/`
- `surge_eval.py` — Citation Recall/Precision/F1（doc_id 精确匹配）
- `surge_metrics.py` — 调用 SurGE 官方 8 指标
- `benchmark_runner.py` — 批量对比 41 个 topic
- `llm_judge.py` — 12 维 LLM-as-Judge 质量评分
- `hallucination_checker.py` — 引用幻觉检测

**SurGE 数据**：`surveys.json` (32MB, 205 topics) + `corpus.json` (1.6GB, 1.09M papers)
从 [SurGE Google Drive](https://drive.google.com/drive/folders/1ZZPeZvjexFcCmgFqxftKeCPn1vYeBR0Q) 下载。

---

## 13. 数据库

**文件**：`mars/database/models.py`

SQLAlchemy ORM，默认 SQLite（`sqlite:///./mars.db`）。

### 表结构

**`ccf_venues`**：CCF 推荐场馆表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | Integer PK | 自增主键 |
| name | String(50) unique | 缩写（如 CVPR） |
| full_name | String(256) | 全称 |
| ccf_rank | String(1) index | A / B / C |
| venue_type | String(20) | conference / journal |
| domains | Text | 逗号分隔的领域列表 |
| dblp_url | String(512) | DBLP 链接 |
| created_at / updated_at | DateTime | 时间戳 |

**`papers`**：论文缓存表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | Integer PK | 自增主键 |
| title | String(512) index | 论文标题 |
| authors | Text | JSON 格式作者列表 |
| venue | String(256) | 发表场馆 |
| year | Integer index | 发表年份 |
| citation_count | Integer | 引用次数 |
| doi / url / pdf_url | String | 链接字段 |
| abstract | Text | 摘要 |
| source | String(50) | dblp / semantic_scholar / arxiv |
| semantic_scholar_id | String(64) index | S2 paper ID |
| relevance_score | Float | 相关性评分 |
| created_at | DateTime | 缓存时间 |

### 辅助函数

```python
from mars.database.models import init_db, get_engine, get_session

init_db()        # CREATE TABLE IF NOT EXISTS（幂等）
get_engine()     # 返回缓存的 SQLAlchemy engine
get_session()    # 返回新 Session
```

---

## 14. 工具函数

### `mars/utils/logging_config.py`

```python
from mars.utils.logging_config import setup_logging
setup_logging(log_stream=None)   # 可选传入文件流
```

- 使用 stdlib `logging`，格式：`%(asctime)s - %(name)s - %(levelname)s - %(message)s`
- 日志级别来自 `settings.LOG_LEVEL`
- CLI 的 `_run_context` 在每次运行时传入 `log_fh`（run.log 文件句柄）

### `mars/utils/llm_factory.py`

```python
from mars.utils.llm_factory import get_llm
llm = get_llm(provider="qwen")   # 返回 ChatOpenAI，用于工具内部直调
```

封装了 `llm_gateway.get_llm()`，供工具层调用（避免循环导入）。

### `mars/utils/retry.py`

```python
from mars.utils.retry import retry_on_network_error

@retry_on_network_error(max_retries=3, base_delay=1.0, backoff_factor=2.0)
def _call_api():
    ...
```

装饰器，捕获 `requests.RequestException`，指数退避重试。各工具的 HTTP 调用均使用此装饰器。

---

## 15. 测试

测试文件位于 `tests/`，使用 `pytest`，共 5 个文件，约 135 个测试用例。

```bash
python -m pytest tests/ -v
```

**关键特性**：
- 所有测试**不需要真实 API key**（工具被 mock）
- `conftest.py` 提供公共 fixtures
- 覆盖：设置验证、工具调用、智能体创建、Crew 组装、CLI 命令、API 端点、
  RateLimitAwareLLM 重试/回退逻辑、CCF JSON 加载、Prompt 模板加载

---

## 16. 依赖管理

依赖在 `pyproject.toml`（`[project.dependencies]`）和 `requirements.txt` 中**同步维护**（两者内容等价）。

**安装方式**：
```bash
pip install -e .                    # 开发安装（从 pyproject.toml）
pip install -r requirements.txt    # 直接安装
```

**开发依赖**（`pyproject.toml` `[project.optional-dependencies].dev`）：
```
pytest, pytest-asyncio, pytest-cov, black, ruff, mypy
```

**代码风格**：`black`（行长 100）+ `ruff`（E/F/W/I 规则，忽略 E501）

---

## 17. 关键设计决策与注意事项

### 17.1 LLM 模型字符串格式（高优先级）

为 `crewai.LLM` 构建 Agent LLM 时，**必须**使用以下格式：

```python
# 正确：所有 OpenAI 兼容端点使用 openai/ 前缀 + base_url
LLM(model="openai/qwen3.5-flash", base_url="https://dashscope...", ...)
LLM(model="openai/kimi-k2.5",    base_url="https://api.moonshot...", ...)
LLM(model="openai/glm-4.7-flash", base_url="https://open.bigmodel...", ...)

# 正确：DeepSeek 使用原生前缀（LiteLLM 原生支持）
LLM(model="deepseek/deepseek-chat", base_url="https://api.deepseek.com/v1", ...)

# 错误：会触发 "LLM Provider NOT provided" 错误
LLM(model="dashscope/qwen3.5-flash", ...)
LLM(model="moonshot/kimi-k2.5", ...)
```

### 17.2 Kimi（Moonshot）模型温度

`kimi-k2.5` 只接受 `temperature=1`，已在 `_CREWAI_PROVIDER_CFG["kimi"]` 中通过 `temperature_override=1.0` 强制。**不要**为 kimi provider 传入其他温度值。

### 17.3 Qwen3 系列 thinking 模式

Qwen3.x 模型默认开启思维链（thinking），此时 API 返回的 `content` 字段为空，CrewAI/LiteLLM 会报 "Invalid response from LLM call - None or empty."。解决：

```python
LLM(..., extra_body={"enable_thinking": False})
```

`_get_crewai_llm()` 中已自动处理（检测 model 名前缀 `qwen3`）。

### 17.4 CrewAI memory 与 ChromaDB

`Crew(memory=True)` 默认使用 OpenAI 嵌入 API，无 OpenAI key 会报错。**默认保持 `ENABLE_MEMORY=False`**，仅在有兼容嵌入 API 时开启。

### 17.5 输出目录动态重定向

CLI 每次运行通过 `settings.OUTPUT_DIR = run_dir` 将输出目录指向时间戳子目录。`FileWriterTool` 调用 `settings.OUTPUT_DIR` 时读取的是运行时值，而非初始值。**注意**：`settings` 是 Pydantic 模型实例，`OUTPUT_DIR` 字段为 `Path` 类型，可直接赋值。

### 17.6 工具的 args_schema

凡定义了 `args_schema` 的工具，均注解为具体类型而非 `type[BaseModel]`：

```python
args_schema: type[ArXivSearchToolSchema] = ArXivSearchToolSchema
```

工具的输入字段通常接受 JSON 字符串（`query_json`），并通过 `_coerce_direct_args` model_validator 同时支持 Agent 直传结构化字段（`query`、`max_results` 等）。

### 17.7 FileWriterTool 安全防护

路径穿越防护：`safe_name = Path(filename).name` 只取文件名部分，忽略路径前缀。写入的文件**始终**在 `settings.OUTPUT_DIR` 下，不允许写入其他目录。

### 17.8 CCF 数据库为静态数据

`ccf_database.py` 中的 `CCF_DATABASE` 列表是硬编码的静态数据，约覆盖 20+ 个 A/B 类场馆。若需扩展，直接在该列表追加条目，格式参考现有条目。

---

## 18. 常见扩展点

### 添加新 LLM 提供商

1. 在 `settings.py` 中添加 `{NEW_PROVIDER}_API_KEY` 和 `{NEW_PROVIDER}_MODEL` 字段
2. 在 `llm_gateway.py` 的 `_CREWAI_PROVIDER_CFG` 中添加配置项
3. 在 `_PROVIDER_MAP` 中添加 `get_{new_provider}_llm()` 工厂函数
4. 在 `_PROVIDER_KEY_ATTR` 中注册
5. 可选：在 `_AGENT_LLM_MAP` 中为某些角色指定新 provider

### 添加新 Agent 角色

1. 在 `mars/agents/` 下创建新文件，实现 `create_{role}_agent()` 函数
2. 在 `llm_gateway._AGENT_LLM_MAP` 中注册角色名 → provider 映射
3. 在 `mars/tasks/task_definitions.py` 中创建对应任务工厂函数
4. 在所需 Crew 中引入并组装

### 添加新学术数据库工具

1. 在 `mars/tools/` 下创建新文件，继承 `crewai.tools.BaseTool`
2. 定义 `args_schema`（推荐），实现 `_run()` 方法
3. 对所有 HTTP 调用使用 `@retry_on_network_error` 装饰器
4. 在需要该工具的 Agent 中引入并添加到 `tools` 列表

### 修改工作流任务顺序

所有任务均独立声明，Crew 通过 `context` 参数建立依赖。修改顺序时：
1. 调整 Crew 中 `tasks=[]` 列表的顺序
2. 相应更新任务的 `context=[]` 参数（保证数据依赖正确）

### 扩展 CCF 场馆数据库

直接在 `mars/tools/ccf_database.py` 的 `CCF_DATABASE` 列表追加字典，格式：

```python
{
    "name": "缩写",
    "full_name": "全称",
    "ccf_rank": "A",      # A / B / C
    "type": "conference", # conference / journal
    "domains": ["domain1", "domain2"],
    "dblp_url": "https://dblp.org/db/conf/xxx/",
}
```
