# MARS 技术开发文档

**Multi-Agent Research System — 多智能体学术文献智能检索与分析系统**

> 版本：0.1.0 | 许可：MIT | Python ≥ 3.10

---

## 目录

1. [系统概述](#1-系统概述)
2. [系统架构](#2-系统架构)
3. [核心模块设计](#3-核心模块设计)
4. [智能体（Agent）详细设计](#4-智能体agent详细设计)
5. [工具层（Tools）设计](#5-工具层tools设计)
6. [任务（Task）定义](#6-任务task定义)
7. [工作流（Crew）编排](#7-工作流crew编排)
8. [LLM 网关服务](#8-llm-网关服务)
9. [数据库设计](#9-数据库设计)
10. [配置管理](#10-配置管理)
11. [REST API 接口](#11-rest-api-接口)
12. [CLI 命令行接口](#12-cli-命令行接口)
13. [数据流与执行流程](#13-数据流与执行流程)
14. [测试体系](#14-测试体系)
15. [技术栈与依赖](#15-技术栈与依赖)
16. [开发规范](#16-开发规范)
17. [扩展指南](#17-扩展指南)
18. [已知限制与未来规划](#18-已知限制与未来规划)

---

## 1. 系统概述

### 1.1 项目背景

MARS（Multi-Agent Research System）是一个基于 CrewAI 多智能体框架的学术文献智能检索与分析系统。系统利用 6 个专门的 AI 智能体协同工作，完成从研究领域分析、论文检索、深度解析到综述生成的完整学术研究工作流。

### 1.2 核心能力

| 能力 | 说明 |
|------|------|
| 领域分析 | 自动识别研究领域与子领域，推荐 CCF 排名期刊/会议 |
| 论文检索 | 同时查询 DBLP、Semantic Scholar、arXiv 三大学术数据库 |
| 深度解析 | 提取论文核心贡献、研究方法、实验设计和结论 |
| 关联分析 | 构建引用网络，识别主题聚类和研究趋势 |
| 综述生成 | 自动生成 ≥ 3000 字英文综述并翻译为中文（双语输出） |
| 质量评估 | 多维度学术质量评估（创新性、技术深度、实验有效性、写作质量） |

### 1.3 启动方式

MARS 提供三种启动方式：

- **命令行（CLI）模式**：通过 `mars` 命令或 `python main.py` 运行，适合本地研究使用
- **Web API 模式**：通过 `mars api` 启动 FastAPI 服务器，可通过浏览器或 HTTP 客户端访问
- **Python API 模式**：在 Python 代码中直接 `import` 调用各 `run_*` 函数，适合二次开发

> **注意**：MARS 当前不包含前端网页界面。Web API 模式提供 REST API 端点，可使用 Postman、curl 或自行开发前端进行交互。FastAPI 自带交互式文档页面（Swagger UI），可在浏览器中通过 `http://localhost:8000/docs` 访问。

---

## 2. 系统架构

### 2.1 分层架构图

```
┌──────────────────────────────────────────────────────────────────┐
│                       用户接口层 (Interface Layer)                │
│                                                                   │
│   ┌──────────────────────┐    ┌──────────────────────────────┐   │
│   │   CLI (Typer + Rich) │    │  REST API (FastAPI + Uvicorn)│   │
│   │   mars/cli.py        │    │  mars/api/main.py            │   │
│   │                      │    │  Swagger UI: /docs           │   │
│   │   命令: search,      │    │  端点: /search, /analyze,    │   │
│   │   analyze, connect,  │    │  /connect, /full-research,   │   │
│   │   full, api,         │    │  /health                     │   │
│   │   init-db, check     │    │                              │   │
│   └──────────┬───────────┘    └───────────────┬──────────────┘   │
│              │                                │                   │
└──────────────┼────────────────────────────────┼───────────────────┘
               │                                │
               ▼                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                     编排层 (Orchestration Layer)                  │
│                                                                   │
│   ┌────────────────┐ ┌────────────────┐ ┌────────────────────┐   │
│   │  search_crew   │ │ analysis_crew  │ │  connection_crew   │   │
│   │  基础检索流程  │ │ 深度分析流程   │ │  关联分析流程      │   │
│   └────────────────┘ └────────────────┘ └────────────────────┘   │
│                                                                   │
│   ┌──────────────────────────────────────────────────────────┐   │
│   │               full_research_crew                          │   │
│   │               完整研究流程（6 个 Agent 协同）             │   │
│   └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│   CrewAI Process: Sequential (顺序执行，上下文依赖)              │
│   任务工厂: mars/tasks/task_definitions.py                       │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                     智能体层 (Agent Layer)                        │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Researcher   │  │   Searcher   │  │   Analyzer   │           │
│  │  领域分析师   │  │  论文检索师  │  │  深度分析师  │           │
│  │  LLM: Qwen   │  │  LLM: Kimi   │  │  LLM: Kimi   │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Connector    │  │  Summarizer  │  │  Evaluator   │           │
│  │  关联分析师   │  │  综述生成师  │  │  质量评估师  │           │
│  │  LLM: Qwen   │  │  LLM: Qwen   │  │  LLM: Kimi   │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                      工具层 (Tool Layer)                          │
│                                                                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐ │
│  │CCF Database │ │ DBLP Search │ │  Semantic    │ │ arXiv API │ │
│  │ Query Tool  │ │    Tool     │ │  Scholar     │ │   Tool    │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘ │
│                                                                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐ │
│  │ PDF Parser  │ │  Keyword    │ │  Citation    │ │   File    │ │
│  │    Tool     │ │  Expander   │ │  Network     │ │  Manager  │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘ │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                     基础设施层 (Infrastructure Layer)             │
│                                                                   │
│  ┌─────────────┐ ┌─────────────┐ ┌──────────────────────────┐   │
│  │  LLM 网关   │ │ SQLAlchemy  │ │   配置管理               │   │
│  │ (4 Provider) │ │  数据库     │ │   (Pydantic Settings)    │   │
│  │  services/  │ │  database/  │ │   config/settings.py     │   │
│  └─────────────┘ └─────────────┘ └──────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 项目目录结构

```
MARS/
├── main.py                        # 入口文件，调用 CLI
├── requirements.txt               # 依赖列表
├── pyproject.toml                 # 构建配置
├── .env.example                   # 环境变量模板
├── README.md                      # 项目简介
│
├── docs/                          # 文档目录
│   ├── technical-guide.md         # 技术开发文档（中文，本文件）
│   ├── technical-guide_en.md      # Technical Development Guide (English)
│   ├── user-guide.md              # 使用手册（中文）
│   └── user-guide_en.md           # User Manual (English)
│
├── mars/                          # 主程序包
│   ├── __init__.py                # 包初始化（版本号）
│   ├── cli.py                     # CLI 命令定义 (Typer)
│   │
│   ├── config/                    # 配置管理
│   │   ├── __init__.py
│   │   └── settings.py            # Pydantic BaseSettings
│   │
│   ├── services/                  # 服务层
│   │   ├── __init__.py
│   │   └── llm_gateway.py         # LLM 网关（4 家供应商路由 + 自动回退）
│   │
│   ├── agents/                    # 6 个智能体
│   │   ├── __init__.py
│   │   ├── researcher.py          # 领域分析师
│   │   ├── searcher.py            # 论文检索师
│   │   ├── analyzer.py            # 深度分析师
│   │   ├── connector.py           # 关联分析师
│   │   ├── summarizer.py          # 综述生成师（双语输出）
│   │   └── evaluator.py           # 质量评估师
│   │
│   ├── data/                      # 静态数据文件
│   │   └── ccf_2025.json          # CCF 推荐期刊/会议列表 (JSON)
│   │
│   ├── tools/                     # 9 个工具
│   │   ├── __init__.py
│   │   ├── ccf_database.py        # CCF 排名数据库（从 JSON 加载）
│   │   ├── dblp_search.py         # DBLP 搜索
│   │   ├── semantic_scholar.py    # Semantic Scholar 搜索
│   │   ├── arxiv_api.py           # arXiv 搜索
│   │   ├── pdf_parser.py          # PDF 文本提取
│   │   ├── keyword_expander.py    # 关键词扩展
│   │   ├── citation_network.py    # 引用网络构建
│   │   └── file_manager.py        # 文件读写
│   │
│   ├── tasks/                     # 任务定义
│   │   ├── __init__.py
│   │   ├── task_definitions.py    # 8 个任务工厂函数
│   │   └── prompts/               # Prompt 模板文件 (txt)
│   │       ├── domain_analysis_task.txt
│   │       ├── paper_search_task.txt
│   │       └── ...（共 8 个）
│   │
│   ├── crews/                     # 4 个工作流
│   │   ├── __init__.py
│   │   ├── search_crew.py         # 基础检索（4 个任务）
│   │   ├── analysis_crew.py       # 深度分析
│   │   ├── connection_crew.py     # 关联分析
│   │   └── full_research_crew.py  # 完整研究（8 个任务）
│   │
│   ├── database/                  # 数据库
│   │   ├── __init__.py
│   │   └── models.py              # SQLAlchemy 模型
│   │
│   ├── utils/                     # 工具函数
│   │   ├── __init__.py
│   │   ├── llm_factory.py         # LLM 工厂（⚠️ 已废弃，v0.3.0 移除）
│   │   ├── logging_config.py      # 集中式日志配置（支持 run_id 注入）
│   │   └── retry.py               # 指数退避重试装饰器
│   │
│   ├── api/                       # Web API
│   │   ├── __init__.py
│   │   └── main.py                # FastAPI 应用（异步任务模式）
│   │
│   └── py.typed                   # PEP 561 类型标记文件
│
└── tests/                         # 测试
    ├── __init__.py
    ├── conftest.py                # 共享 Fixture 和 Mock
    ├── test_basic.py              # 基础组件测试
    ├── test_new_components.py     # 新组件测试（Settings、LLM Gateway、DB、API、Tasks）
    └── test_usage_features.py     # 使用功能测试
```

---

## 3. 核心模块设计

### 3.1 模块依赖关系

```
cli.py / api/main.py
    └── crews/ (search_crew, analysis_crew, connection_crew, full_research_crew)
          ├── agents/ (researcher, searcher, analyzer, connector, summarizer, evaluator)
          │     └── services/llm_gateway.py (get_llm_by_task → crewai.LLM)
          │     └── tools/ (各种工具)
          └── tasks/task_definitions.py (任务工厂)
                └── agents/ (智能体引用)

config/settings.py ──── 被所有模块引用
database/models.py ──── 被 API/CLI 引用
utils/logging_config.py ── 被 CLI/API 在启动时调用
utils/retry.py ─────────── 被工具层 HTTP 请求调用
```

### 3.2 设计原则

| 原则 | 实现 |
|------|------|
| **工厂模式** | `create_*_agent()`, `create_*_task()`, `get_llm()` |
| **单例模式** | `settings = MarsSettings()` 全局配置单例 |
| **策略模式** | LLM Provider 路由：`_PROVIDER_MAP` + `_AGENT_LLM_MAP` |
| **委托模式** | CrewAI 编排 Agent → Agent 调用 Tool |
| **关注点分离** | Agent 定义、Tool 实现、Task 描述、Crew 编排各自独立 |

---

## 4. 智能体（Agent）详细设计

### 4.1 智能体总览

每个 Agent 由以下属性定义：

```python
Agent(
    role="角色名称",           # 智能体角色
    goal="工作目标",           # 任务目标描述
    backstory="角色背景",       # 角色能力描述
    llm=get_llm_by_task(role), # 通过 LLM 网关获取
    tools=[...],               # 可用工具列表
    verbose=True,              # 输出执行日志
    allow_delegation=False,    # 不允许委托给其他 Agent
    max_iter=N,                # 最大迭代次数
)
```

### 4.2 各智能体详细设计

#### Researcher — 领域分析师

| 属性 | 值 |
|------|-----|
| 模块 | `mars/agents/researcher.py` |
| LLM | Qwen-Max（强推理能力） |
| 工具 | `CCFDatabaseQueryTool`, `KeywordExpanderTool` |
| max_iter | 5 |
| 输入 | 用户研究主题（中/英文） |
| 输出 | JSON：research_domain, sub_domains, keywords, recommended_venues |

#### Searcher — 论文检索师

| 属性 | 值 |
|------|-----|
| 模块 | `mars/agents/searcher.py` |
| LLM | Kimi（kimi-k2.5，长上下文） |
| 工具 | `DBLPSearchTool`, `SemanticScholarSearchTool`, `ArXivSearchTool`, `KeywordExpanderTool`, `FileWriterTool` |
| max_iter | 8 |
| 输入 | 领域分析结果 + 关键词 |
| 输出 | 结构化论文列表（20-50 篇），保存到 paper_search.json |

#### Analyzer — 深度分析师

| 属性 | 值 |
|------|-----|
| 模块 | `mars/agents/analyzer.py` |
| LLM | Kimi（kimi-k2.5，长上下文） |
| 工具 | `ArXivSearchTool`, `SemanticScholarSearchTool`, `PDFParserTool`, `FileWriterTool` |
| max_iter | 10 |
| 输入 | 论文列表（取 top 20） |
| 输出 | 论文深度分析报告（analysis_results.json） |

#### Connector — 关联分析师

| 属性 | 值 |
|------|-----|
| 模块 | `mars/agents/connector.py` |
| LLM | Qwen（qwen3.5-flash，推理能力） |
| 工具 | `CitationNetworkTool`, `SemanticScholarSearchTool`, `FileWriterTool` |
| max_iter | 8 |
| 输入 | 全部检索论文 |
| 输出 | 引用网络摘要、主题聚类、研究趋势（connection_analysis.json） |

#### Summarizer — 综述生成师

| 属性 | 值 |
|------|-----|
| 模块 | `mars/agents/summarizer.py` |
| LLM | Qwen（qwen3.5-flash，长文本生成） |
| 工具 | `FileWriterTool` |
| max_iter | 8 |
| 输入 | 所有分析结果 |
| 输出 | 英文综述（review_en.md，≥ 3000 字）+ 中文翻译（review_zh.md） |

#### Evaluator — 质量评估师

| 属性 | 值 |
|------|-----|
| 模块 | `mars/agents/evaluator.py` |
| LLM | Kimi（kimi-k2.5） |
| 工具 | `SemanticScholarSearchTool`, `FileWriterTool` |
| max_iter | 8 |
| 输入 | 深度分析的论文 |
| 输出 | 多维度评分（0-10）+ 审稿意见（quality_evaluation.json） |

---

## 5. 工具层（Tools）设计

### 5.1 工具总览

所有工具继承自 CrewAI 的 `BaseTool` 类，定义 `name`、`description` 和 `_run()` 方法。

| 工具 | 模块 | 外部依赖 | 需要 API Key |
|------|------|----------|-------------|
| CCFDatabaseQueryTool | `ccf_database.py` | 无（从 `mars/data/ccf_*.json` 加载，支持版本切换） | 否 |
| DBLPSearchTool | `dblp_search.py` | DBLP REST API | 否 |
| SemanticScholarSearchTool | `semantic_scholar.py` | S2 Graph API | 可选 |
| ArXivSearchTool | `arxiv_api.py` | arXiv Atom API | 否 |
| PDFParserTool | `pdf_parser.py` | PyMuPDF / PyPDF2 | 否 |
| KeywordExpanderTool | `keyword_expander.py` | LLM Provider | 是（LLM Key） |
| CitationNetworkTool | `citation_network.py` | S2 Graph API + NetworkX | 可选 |
| FileWriterTool | `file_manager.py` | 无 | 否 |
| FileReaderTool | `file_manager.py` | 无 | 否 |

### 5.2 工具输入输出格式

所有搜索工具接受 JSON 字符串作为输入：

```python
# 示例：DBLP 搜索工具输入
'{"query": "federated learning", "max_results": 20, "year_from": 2020, "year_to": 2024}'

# 示例：PDF 解析工具输入
'{"url": "https://arxiv.org/pdf/2301.12345.pdf", "max_pages": 20, "max_chars": 8000}'
```

### 5.3 外部 API 端点

| API | 端点 | 请求方式 | 速率限制 |
|-----|------|----------|----------|
| DBLP | `https://dblp.org/search/publ/api` | GET | 无限制 |
| Semantic Scholar | `https://api.semanticscholar.org/graph/v1/paper/search` | GET | 100/5min（无 Key） |
| arXiv | `https://export.arxiv.org/api/query` | GET | 合理使用 |

---

## 6. 任务（Task）定义

### 6.1 任务工厂模式

`mars/tasks/task_definitions.py` 提供 8 个任务工厂函数，Prompt 文本从
`mars/tasks/prompts/*.txt` 模板文件加载（使用 `str.format()` 占位符），
非开发人员可直接编辑 `.txt` 文件调整 Prompt 而无需修改 Python 代码：

```python
def create_domain_analysis_task(agent, topic, *, context=None) -> Task
def create_paper_search_task(agent, topic, max_papers=50, *, context=None) -> Task
def create_deep_analysis_task(agent, topic, papers_info="", limit=20, *, context=None) -> Task
def create_connection_analysis_task(agent, topic, papers_info="", *, context=None) -> Task
def create_english_review_task(agent, topic, *, context=None) -> Task     # 英文综述（≥ 3000 字）
def create_review_generation_task(agent, topic, *, context=None) -> Task  # 中文翻译综述
def create_quality_evaluation_task(agent, limit=20, *, context=None) -> Task
def create_full_research_synthesis_task(agent, topic, *, context=None) -> Task  # 综合研究报告
```

### 6.2 任务链上下文传递

任务之间通过 CrewAI 的 `context` 参数传递结果：

```
create_domain_analysis_task      ← 无前置依赖
create_paper_search_task         ← context = [domain_analysis_task]
create_deep_analysis_task        ← context = [paper_search_task]
create_connection_analysis_task  ← context = [paper_search_task]
create_quality_evaluation_task   ← context = [deep_analysis_task]
create_english_review_task       ← context = [全部前置任务]
create_review_generation_task    ← context = [全部前置任务 + english_review_task]
```

> **注意**：`create_review_generation_task` 负责将 `create_english_review_task` 生成的英文综述翻译成中文，而非重新生成综述内容。

---

## 7. 工作流（Crew）编排

### 7.1 基础检索流程 (search_crew)

```
Researcher ──(domain_analysis_task)──→ Searcher ──(paper_search_task)──→
Summarizer ──(english_review_task)──→ Summarizer ──(chinese_review_task)──→ 结果
```

- Agent 数量：3（Researcher、Searcher、Summarizer）
- Task 数量：4
- 执行模式：Sequential
- 输出：paper_search.json + review_en.md（英文综述 ≥ 3000 字）+ review_zh.md（中文翻译）+ domain_analysis.json

### 7.2 深度分析流程 (analysis_crew)

```
Analyzer ──(deep_analysis_task)──→ Evaluator ──(quality_evaluation_task)──→ 结果
```

- Agent 数量：2
- 执行模式：Sequential
- 输出：分析报告 + 质量评估

### 7.3 关联分析流程 (connection_crew)

```
Connector ──(connection_analysis_task)──→
Summarizer ──(chinese_review_task)──→ Summarizer ──(english_review_task)──→ 结果
```

- Agent 数量：2（Connector、Summarizer）
- Task 数量：3
- 执行模式：Sequential
- 输出：关联分析报告（connection_analysis.json）+ 中文综述 + 英文综述

### 7.4 完整研究流程 (full_research_crew)

```
Phase 1:  Researcher (domain_analysis_task)
          → Searcher (bulk_search_task)
Phase 2:  Analyzer (deep_analysis_task)
          → Connector (connection_analysis_task)
          → Evaluator (quality_evaluation_task)
Phase 3:  Summarizer (english_review_task → chinese_review_task)
```

- Agent 数量：6
- Task 数量：7
- 执行模式：Sequential（CrewAI 通过 context 实现逻辑依赖链）
- 输出：完整研究报告（domain_analysis.json + paper_search.json + analysis_results.json + connection_analysis.json + quality_evaluation.json + review_en.md + review_zh.md）

---

## 8. LLM 网关服务

### 8.1 模块位置

`mars/services/llm_gateway.py`

### 8.2 支持的 LLM 供应商

| 供应商 | 函数 | API 端点 | 默认模型 |
|--------|------|----------|----------|
| 阿里云（Qwen） | `get_qwen_llm()` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | qwen3.5-flash |
| DeepSeek | `get_deepseek_llm()` | `https://api.deepseek.com/v1` | deepseek-chat |
| 月之暗面（Kimi） | `get_kimi_llm()` | `https://api.moonshot.cn/v1` | kimi-k2.5 |
| 智谱 AI（GLM） | `get_glm_llm()` | `https://open.bigmodel.cn/api/paas/v4` | glm-4.7-flash |

- `get_qwen_llm()` / `get_deepseek_llm()` / `get_kimi_llm()` / `get_glm_llm()` 返回 `ChatOpenAI` 实例，供工具内部直接调用（如 `KeywordExpanderTool`）。
- `get_llm_by_task()` 返回 `crewai.LLM` 实例，供 CrewAI `Agent` 使用；模型字符串携带 LiteLLM provider 前缀：所有 OpenAI 兼容供应商（DashScope/Moonshot/Zhipu）统一使用 `openai/<model>` 前缀加 `base_url`（LiteLLM 通用兼容路径），DeepSeek 使用 `deepseek/<model>` 前缀（例如 `openai/qwen3.5-flash`、`openai/kimi-k2.5`、`deepseek/deepseek-chat`）。`openai/` 此处为 LiteLLM 路由前缀，与 OpenAI 官方 API 无关。

### 8.3 智能体 → LLM 映射

```python
_AGENT_LLM_MAP = {
    "researcher": ("qwen", None),    # 强推理 → Qwen (qwen3.5-flash)
    "searcher":   ("qwen", None),    # 长上下文检索综合 → Qwen
    "analyzer":   ("qwen", None),    # 长上下文深度分析 → Qwen
    "connector":  ("qwen", None),    # 关系推理 → Qwen
    "summarizer": ("qwen", None),    # 长文本生成 → Qwen
    "evaluator":  ("kimi", None),    # 评估任务 → Kimi (kimi-k2.5)
}
```

当首选供应商的 API Key 未配置时，网关自动回退到 `DEFAULT_LLM_PROVIDER`，再回退到任意可用供应商。

`RateLimitAwareLLM`（`crewai.LLM` 子类）为每个调用提供指数退避重试和自动切换备用供应商的限流保护，
用于 GLM 等免费模型场景。`call()` 方法使用 `*args, **kwargs` 签名以解耦 CrewAI 内部接口。`mars/utils/llm_factory.py`
为向后兼容层（v0.3.0 将移除），新代码应直接从 `mars.services.llm_gateway` 导入。

### 8.4 使用方式

```python
from mars.services.llm_gateway import get_llm, get_llm_by_task
from langchain_openai import ChatOpenAI
from crewai import LLM

# 通过供应商名获取 ChatOpenAI（用于工具内部直接调用 LLM）
llm: ChatOpenAI = get_llm(provider="qwen", temperature=0.3)

# 通过智能体角色获取 crewai.LLM（推荐，用于 CrewAI Agent）
llm: LLM = get_llm_by_task("researcher", temperature=0.3)
```

### 8.5 Agent max_iter 配置

所有 Agent 的 `max_iter` 值由 `settings.AGENT_MAX_ITER` 统一控制（默认 10），
可通过 `.env` 文件覆盖：`AGENT_MAX_ITER=15`

---

## 9. 数据库设计

### 9.1 模块位置

`mars/database/models.py`

### 9.2 数据模型

#### CCFVenue 表

```sql
CREATE TABLE ccf_venues (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        VARCHAR(50) NOT NULL UNIQUE,  -- 索引
    full_name   VARCHAR(256) NOT NULL,
    ccf_rank    VARCHAR(1) NOT NULL,          -- 索引 (A/B/C)
    venue_type  VARCHAR(20) NOT NULL,         -- conference/journal
    domains     TEXT NOT NULL,                -- 逗号分隔的领域列表
    dblp_url    VARCHAR(512) DEFAULT '',
    created_at  DATETIME DEFAULT NOW(),
    updated_at  DATETIME DEFAULT NOW()
);
```

#### Paper 表

```sql
CREATE TABLE papers (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    title                 VARCHAR(512) NOT NULL,  -- 索引
    authors               TEXT DEFAULT '',         -- JSON 数组
    venue                 VARCHAR(256) DEFAULT '',
    year                  INTEGER,                 -- 索引
    citation_count        INTEGER DEFAULT 0,
    doi                   VARCHAR(256) DEFAULT '',
    url                   VARCHAR(512) DEFAULT '',
    abstract              TEXT DEFAULT '',
    pdf_url               VARCHAR(512) DEFAULT '',
    source                VARCHAR(50) DEFAULT '',  -- dblp/semantic_scholar/arxiv
    semantic_scholar_id   VARCHAR(64) DEFAULT '',   -- 索引
    relevance_score       FLOAT DEFAULT 0.0,
    created_at            DATETIME DEFAULT NOW()
);
```

### 9.3 数据库操作

```python
from mars.database.models import init_db, get_session

# 初始化数据库（创建表）
init_db()

# 获取会话
session = get_session()
```

---

## 10. 配置管理

### 10.1 模块位置

`mars/config/settings.py`

### 10.2 配置类

```python
class MarsSettings(BaseSettings):
    model_config = {"env_file": str(_ENV_PATH), "env_file_encoding": "utf-8"}

    # LLM API Keys
    DASHSCOPE_API_KEY: str = ""
    DEEPSEEK_API_KEY: str = ""
    MOONSHOT_API_KEY: str = ""
    ZHIPU_API_KEY: str = ""

    # 模型配置
    QWEN_MODEL: str = "qwen3.5-flash"
    DEEPSEEK_MODEL: str = "deepseek-chat"
    DEEPSEEK_CODER_MODEL: str = "deepseek-coder"
    KIMI_MODEL: str = "kimi-k2.5"
    GLM_MODEL: str = "glm-4.7-flash"  # 免费模型，作为兜底

    # API 端点
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    MOONSHOT_BASE_URL: str = "https://api.moonshot.cn/v1"

    # 学术搜索
    SEMANTIC_SCHOLAR_API_KEY: str = ""
    MAX_PAPERS_PER_SEARCH: int = Field(default=50, gt=0)
    MAX_PAPERS_FOR_ANALYSIS: int = Field(default=20, gt=0)
    # arXiv 请求读取超时（秒），网络较慢时可适当增大
    ARXIV_SEARCH_TIMEOUT: int = Field(default=30, gt=0)

    # 应用设置
    DEFAULT_LLM_PROVIDER: str = "qwen"
    LOG_LEVEL: str = "INFO"
    OUTPUT_DIR: Path = Path("./output")
    AGENT_MAX_ITER: int = Field(default=10, ge=1)  # Agent 最大工具调用迭代次数

    # GLM 限流处理
    GLM_RATE_LIMIT_MAX_RETRIES: int = 3       # RateLimitError 重试次数
    GLM_RATE_LIMIT_RETRY_DELAY: float = 5.0   # 指数退避基础延迟（秒）

    # Crew 记忆（需要有效的 OpenAI 兼容嵌入 Key 才能启用）
    ENABLE_MEMORY: bool = False

    # 数据库
    DATABASE_URL: str = "sqlite:///./mars.db"

    # API 服务器
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
```

### 10.3 使用方式

```python
from mars.config import settings

# 所有模块均通过此单例访问配置
print(settings.DASHSCOPE_API_KEY)
print(settings.MAX_PAPERS_PER_SEARCH)
```

---

## 11. REST API 接口

### 11.1 基本信息

- **框架**：FastAPI 0.110.0+
- **服务器**：Uvicorn
- **启动方式**：`mars api` 或 `uvicorn mars.api.main:app`
- **交互式文档**：`http://localhost:8000/docs` (Swagger UI)

### 11.2 端点列表

所有工作流端点采用**异步任务模式**：`POST` 立即返回 `task_id`（HTTP 202），
通过 `GET /task/{task_id}` 轮询状态和结果。

| 方法 | 路径 | 请求体 | 说明 |
|------|------|--------|------|
| GET | `/health` | 无 | 健康检查 |
| GET | `/task/{task_id}` | 无 | 查询任务状态 |
| POST | `/search` | `SearchRequest` | 基础检索（异步） |
| POST | `/analyze` | `AnalyzeRequest` | 深度分析（异步） |
| POST | `/connect` | `ConnectRequest` | 关联分析（异步） |
| POST | `/full-research` | `FullResearchRequest` | 完整研究（异步） |

### 11.3 请求/响应模型

```python
# 请求（与旧版相同）
class SearchRequest(BaseModel):
    topic: str            # 研究主题
    max_results: int = 50 # 最大结果数 (1-200)

# POST 响应（异步模式）
class TaskAcceptedResponse(BaseModel):
    task_id: str          # 任务 ID（12 位 hex）
    status: str           # "pending"

# GET /task/{task_id} 响应
class TaskInfo(BaseModel):
    task_id: str
    status: TaskStatus    # "pending" | "running" | "success" | "failed"
    result: str           # 结果文本（仅在 success 时）
    error: str            # 错误信息（仅在 failed 时）
```

### 11.4 调用示例

```bash
# 提交基础检索任务
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"topic": "federated learning privacy", "max_results": 20}'
# → {"task_id":"a1b2c3d4e5f6","status":"pending"}

# 轮询任务状态
curl http://localhost:8000/task/a1b2c3d4e5f6
# → {"task_id":"a1b2c3d4e5f6","status":"running","result":"","error":""}
```

---

## 12. CLI 命令行接口

### 12.1 命令列表

| 命令 | 说明 | 示例 |
|------|------|------|
| `mars search <topic>` | 基础检索（输出双语综述） | `mars search "federated learning"` |
| `mars analyze <papers>` | 深度分析 | `mars analyze "Paper1 \| Paper2"` |
| `mars connect <papers> --topic <topic>` | 关联分析 | `mars connect "P1 \| P2" -t "ML"` |
| `mars full <topic>` | 完整研究 | `mars full "联邦学习隐私保护"` |
| `mars api` | 启动 API 服务器 | `mars api --port 8000` |
| `mars init-db` | 初始化数据库 | `mars init-db` |
| `mars check` | 检查配置状态 | `mars check` |

### 12.2 可选参数

- `--max-results / -n`：最大检索数量（默认 50）
- `--max-papers / -n`：最大分析论文数（默认 20）
- `--host / -h`：API 服务器地址（默认 0.0.0.0）
- `--port / -p`：API 服务器端口（默认 8000）
- `--topic / -t`：研究主题

---

## 13. 数据流与执行流程

### 13.1 完整研究流程数据流

```
用户输入: "federated learning with differential privacy"
          │
          ▼
┌──── Phase 1 ─────────────────────────────────────────────┐
│                                                           │
│  Researcher Agent                                        │
│  ├─ 调用 keyword_expander → 扩展关键词                   │
│  ├─ 调用 ccf_database_query → 推荐 CCF 期刊/会议        │
│  └─ 输出: domain_analysis.json                           │
│      {research_domain, sub_domains, keywords,            │
│       recommended_venues}                                 │
│                         │                                 │
│                         ▼                                 │
│  Searcher Agent                                          │
│  ├─ 调用 dblp_search → 检索 DBLP                        │
│  ├─ 调用 semantic_scholar_search → 检索 S2              │
│  ├─ 调用 arxiv_search → 检索 arXiv                      │
│  └─ 输出: paper_search.json (50 papers)                  │
│                                                           │
└──── Phase 2 ─────────────────────────────────────────────┐
│                                                           │
│  Analyzer Agent (top 20 papers)                          │
│  ├─ 调用 pdf_parser → 下载并提取 PDF 文本               │
│  ├─ 输出: analysis_results.json                          │
│                                                           │
│  Connector Agent (全部 50 papers)                        │
│  ├─ 调用 citation_network → 构建引用网络                 │
│  ├─ 输出: connection_analysis.json                       │
│                                                           │
│  Evaluator Agent (top 20 papers)                         │
│  ├─ 输出: quality_evaluation.json                        │
│                                                           │
└──── Phase 3 ─────────────────────────────────────────────┐
│                                                           │
│  Summarizer Agent                                        │
│  ├─ 生成英文综述（≥ 3000 字）→ review_en.md              │
│  └─ 翻译为中文综述 → review_zh.md                       │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

---

## 14. 测试体系

### 14.1 测试文件

| 文件 | 测试内容 | 说明 |
|------|----------|------|
| `tests/conftest.py` | 共享 Fixture 和 Mock | Mock 外部 API、数据库等 |
| `tests/test_basic.py` | 配置、工具、Agent/Crew 导入 | ~46 个测试 |
| `tests/test_new_components.py` | Settings、LLM Gateway、DB、API、Tasks | ~33 个测试 |
| `tests/test_usage_features.py` | 端到端使用功能 | ~26 个测试 |

**总计约 105 个测试，全部无需真实 API Key 即可通过。**

### 14.2 运行测试

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行全部测试
python -m pytest tests/ -v

# 运行指定测试
python -m pytest tests/test_basic.py -v
python -m pytest tests/test_new_components.py -v

# 生成覆盖率报告
python -m pytest tests/ --cov=mars --cov-report=html
```

### 14.3 测试特点

- 所有测试不需要真实 API Key（使用 Mock）
- 外部 HTTP 请求通过 `unittest.mock.patch` 模拟
- 数据库测试使用 `tmp_path` 临时目录

---

## 15. 技术栈与依赖

### 15.1 核心框架

| 组件 | 技术 | 版本 | 用途 |
|------|------|------|------|
| 多智能体框架 | CrewAI | ≥0.80.0 | Agent/Task/Crew 编排 |
| LLM 接口 | LangChain OpenAI | ≥0.2.0 | 统一 LLM 调用接口（工具层） |
| LLM 路由 | LiteLLM | ≥1.50.0 | CrewAI Agent LLM 路由与 Provider 回退 |
| CLI | Typer | ≥0.12.0 | 命令行界面 |
| Web API | FastAPI | ≥0.110.0 | REST API |
| 数据库 | SQLAlchemy | ≥2.0.0 | ORM |
| 配置 | Pydantic Settings | ≥2.1.0 | 类型化配置 |

### 15.2 LLM SDK

| SDK | 版本 | 对应 LLM |
|-----|------|----------|
| dashscope | ≥1.20.0 | Qwen (阿里云) |
| openai | ≥1.50.0 | DeepSeek, Kimi (OpenAI 兼容) |
| zhipuai | ≥2.1.0 | GLM (智谱 AI) |

### 15.3 学术搜索

| 库 | 版本 | 用途 |
|----|------|------|
| semanticscholar | ≥0.8.0 | Semantic Scholar API |
| scholarly | ≥1.7.11 | Google Scholar |
| requests | ≥2.31.0 | HTTP 请求 |
| httpx | ≥0.27.0 | 异步 HTTP |

### 15.4 数据处理

| 库 | 版本 | 用途 |
|----|------|------|
| PyMuPDF | ≥1.23.0 | PDF 文本提取 (fitz) |
| PyPDF2 | ≥3.0.0 | PDF 备用提取 |
| networkx | ≥3.2 | 引用网络分析 |
| pandas | ≥2.0.0 | 数据处理 |

---

## 16. 开发规范

### 16.1 代码风格

- 行长度：100 字符 (`black --line-length 100`)
- Linter：Ruff (`ruff check`)
- 类型检查：MyPy (`mypy mars/`)
- Python 版本：3.10+

### 16.2 新增 Agent 步骤

1. 在 `mars/agents/` 创建 `new_agent.py`
2. 实现 `create_new_agent()` 工厂函数
3. 在 `mars/services/llm_gateway.py` 的 `_AGENT_LLM_MAP` 添加映射
4. 在 `mars/tasks/prompts/` 添加 Prompt 模板文件（`.txt`），
   并在 `task_definitions.py` 中通过 `_load_prompt()` 加载
5. 在相关 Crew 中组装 Agent 和 Task
6. 在 `tests/` 中添加测试

### 16.3 新增 Tool 步骤

1. 在 `mars/tools/` 创建 `new_tool.py`
2. 继承 `BaseTool`，定义 `name`、`description`、`_run()` 方法
3. 在需要的 Agent 中添加工具实例到 `tools` 列表
4. 在 `tests/` 中添加 Mock 测试

---

## 17. 扩展指南

### 17.1 添加新的 LLM 供应商

```python
# 在 mars/services/llm_gateway.py 中

def get_new_provider_llm(model=None, temperature=0.3):
    return _openai_compatible_llm(
        model=model or settings.NEW_PROVIDER_MODEL,
        api_key=settings.NEW_PROVIDER_API_KEY,
        base_url="https://api.new-provider.com/v1",
        temperature=temperature,
    )

# 添加到 _PROVIDER_MAP
_PROVIDER_MAP["new_provider"] = get_new_provider_llm

# 在 settings.py 中添加配置
NEW_PROVIDER_API_KEY: str = ""
NEW_PROVIDER_MODEL: str = "model-name"
```

### 17.2 添加新的学术搜索源

```python
# 在 mars/tools/ 中创建新工具

class NewSearchTool(BaseTool):
    name: str = "new_search"
    description: str = "Search papers from NewSource"

    def _run(self, query: str) -> str:
        # 实现搜索逻辑
        pass
```

---

## 18. 已知限制与未来规划

### 18.1 当前限制

| 限制 | 说明 | 优先级 |
|------|------|--------|
| 无前端界面 | 仅 CLI + REST API，无 Web 前端 | 中 |
| 同步执行 | CrewAI 工作流为同步执行，长任务无进度反馈 | 中 |
| 无缓存机制 | 每次查询重新调用 API，无论文缓存 | 低 |
| 无身份认证 | API 端点无认证机制 | 低 |
| 无速率限制 | 可能超过外部 API 速率限制 | 低 |
| 无容器化部署 | 缺少 Docker/K8s 配置 | 中 |

### 18.2 未来规划

- [ ] 添加 Web 前端界面（React/Vue）
- [ ] 异步任务处理（Celery/RQ）
- [ ] ChromaDB 向量搜索集成
- [ ] Docker 容器化部署
- [ ] API 认证与速率限制
- [ ] 论文全文缓存机制
- [ ] 多语言支持优化
- [ ] Token 消耗追踪与成本监控

---

*文档最后更新：2026-03-30*
