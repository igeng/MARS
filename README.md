# 🚀 MARS - Multi-Agent Research System

**多智能体学术文献智能检索与分析系统**

MARS is an automated academic literature search and deep analysis system built on a multi-agent architecture. It leverages cutting-edge Chinese LLMs and the CrewAI framework to orchestrate specialized AI agents that collaborate to deliver comprehensive research intelligence—from journal recommendations and paper retrieval to deep analysis, relationship mapping, and literature review generation.

---

## 📐 Architecture Overview

MARS 采用四层分层架构，从用户接口到底层工具单向依赖，清晰解耦：

```
┌──────────────────────────────────────────────────────────────┐
│                  用户接口层 (Interface Layer)                  │
│                                                               │
│   CLI (Typer + Rich)          REST API (FastAPI + Uvicorn)   │
│   mars search/analyze/...     /search /analyze /full-research│
│   python main.py              Swagger UI: /docs              │
└────────────────────────┬─────────────────────────────────────┘
                         │  调用工作流入口函数 run_*()
┌────────────────────────▼─────────────────────────────────────┐
│               工作流编排层 (Crew / Orchestration Layer)        │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │  SearchCrew  │  │ AnalysisCrew │  │  FullResearchCrew  │  │
│  │  4 Tasks     │  │  2 Tasks     │  │  7 Tasks           │  │
│  └──────────────┘  └──────────────┘  └────────────────────┘  │
│          ↑ Sequential Process (CrewAI)                        │
│  ┌────────────────┐                                           │
│  │ ConnectionCrew │  任务定义集中于 mars/tasks/task_definitions│
│  │  3 Tasks       │                                           │
│  └────────────────┘                                           │
└────────────────────────┬─────────────────────────────────────┘
                         │  分配 Agent + Task
┌────────────────────────▼─────────────────────────────────────┐
│                   智能体层 (Agent Layer)                       │
│                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Researcher  │  │  Searcher   │  │  Analyzer   │          │
│  │  (Qwen)     │  │   (Kimi)    │  │   (Kimi)    │          │
│  │  领域分析   │  │  论文检索   │  │  深度解析   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  Connector  │  │ Summarizer  │  │  Evaluator  │          │
│  │  (Qwen)     │  │   (Qwen)    │  │   (Kimi)    │          │
│  │  关联分析   │  │  双语综述   │  │  质量评估   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│          ↑ LLM 网关 llm_gateway.py 统一管理供应商与降级         │
└────────────────────────┬─────────────────────────────────────┘
                         │  调用工具
┌────────────────────────▼─────────────────────────────────────┐
│                  工具与服务层 (Tools / Services Layer)         │
│                                                               │
│  学术数据源                    PDF & 文本处理                  │
│  ├─ DBLP API                  ├─ PDF Parser (PyMuPDF/PyPDF2) │
│  ├─ Semantic Scholar API      └─ Keyword Expander (LLM)      │
│  ├─ arXiv API                                                 │
│  └─ CCF Database (内置)       网络分析 & 存储                  │
│                               ├─ Citation Network (NetworkX) │
│  LLM 供应商（通过 LiteLLM）    ├─ File Manager                │
│  ├─ Qwen (DashScope)          └─ SQLite Database             │
│  ├─ DeepSeek                                                  │
│  ├─ Kimi (Moonshot)                                           │
│  └─ GLM (Zhipu)                                               │
└──────────────────────────────────────────────────────────────┘
```

### 关键设计决策

| 决策 | 原因 |
|------|------|
| 使用 CrewAI 作为 Multi-Agent 框架 | 内置 Agent/Task/Crew 抽象、工具调用、详细执行日志，快速搭建可观测的多智能体流程 |
| 全部使用 OpenAI 兼容 API | 通过 LiteLLM 统一接入 4 个国内 LLM，新增供应商只需一行配置 |
| 顺序执行（Sequential Process） | 学术研究各步骤存在依赖关系（检索→分析→综述），顺序执行保证上下文完整传递 |
| 每次运行独立时间戳目录 | 避免多次运行互相覆盖，完整保存每次研究过程，便于复现和对比 |
| Pydantic Settings 管理配置 | 类型安全、自动从 `.env` 加载、开箱即用的字段校验 |

---

## 🤖 Agents

| Agent | Role | LLM | Key Tools |
|-------|------|-----|-----------|
| **Researcher** | 领域分析师 – identifies research domain, recommends CCF-ranked venues | Qwen (qwen3.5-flash) | `ccf_database_query`, `keyword_expander` |
| **Searcher** | 论文检索师 – queries academic databases and ranks results | Kimi (kimi-k2.5) | `dblp_search`, `semantic_scholar_search`, `arxiv_search`, `keyword_expander` |
| **Analyzer** | 深度分析师 – downloads PDFs, extracts contributions & experiments | Kimi (kimi-k2.5) | `arxiv_search`, `pdf_parser`, `semantic_scholar_search` |
| **Connector** | 关联分析师 – builds citation networks, identifies clusters & trends | Qwen (qwen3.5-flash) | `citation_network_builder`, `semantic_scholar_search` |
| **Summarizer** | 综述生成师 – generates bilingual (EN + ZH) literature reviews | Qwen (qwen3.5-flash) | `file_writer` |
| **Evaluator** | 质量评估师 – scores papers on novelty, depth, validity, writing | Kimi (kimi-k2.5) | `semantic_scholar_search`, `file_writer` |

---

## 🔄 Workflows

### 1. Basic Search (基础检索)
```
User topic → Researcher (domain analysis) → Searcher (50 papers)
          → Summarizer (English review ≥ 3000 words → Chinese translation)
          → Output: paper_search.json + review_en.md + review_zh.md
```

### 2. Deep Analysis (深度分析)
```
Paper list → Analyzer (top 20 papers) → Evaluator → Analysis report + Quality scores
```

### 3. Connection Analysis (关联分析)
```
Multiple papers → Connector (citation network) → Summarizer
               → Network analysis + Bilingual review (EN + ZH)
```

### 4. Full Research (完整研究流程)
```
User topic
  → Researcher (domain analysis)
  → Searcher (50 papers)
  → Analyzer (top 20) + Connector (all 50) + Evaluator (top 20)
  → Summarizer (English review ≥ 3000 words → Chinese translation)
  → Output: 7 files including bilingual comprehensive review
```

---

## 🎯 Design Philosophy（设计理念）

MARS 的设计遵循以下核心原则：

### 1. 专业化智能体协作（Specialized Agent Collaboration）

MARS 摒弃"万能单一调用"思路，转而采用"分工明确、各司其职"的多智能体协作模式：6 个专业智能体分别负责领域分析、文献检索、深度解析、关联分析、综述生成和质量评估。每个 Agent 被赋予精准的角色定义、工具权限和输出格式要求，相互协作完成完整的学术研究流程，与真实研究团队的协作方式高度吻合。

### 2. LLM 提供商多样性与自动降级（Multi-Provider & Graceful Fallback）

为避免单一供应商依赖，MARS 通过统一的 LLM 网关（`llm_gateway.py`）支持 4 个国内主流大模型（Qwen、DeepSeek、Kimi、GLM），并按任务特性智能匹配最合适的模型。当某个供应商遇到限流或不可用时，系统自动以指数退避策略重试，并自动切换到备用供应商，保障系统持续可用。

### 3. 关注点严格分离（Strict Separation of Concerns）

代码库按职责划分为清晰的四层架构，各层间单向依赖，互不耦合：

| 层次 | 职责 | 关键模块 |
|------|------|----------|
| **工具层（Tools）** | 无状态的外部 API 封装，每个工具只做一件事 | `mars/tools/` |
| **智能体层（Agents）** | 具备角色定义与工具权限的 LLM 实体 | `mars/agents/` |
| **任务层（Tasks）** | 描述需要完成的具体工作及期望输出 | `mars/tasks/` |
| **工作流层（Crews）** | 组合 Agent 与 Task，编排完整流程 | `mars/crews/` |

这一设计使得添加新工具只需实现一个带有 `_run()` 方法的类，添加新工作流只需定义任务并组装 `Crew`，扩展成本极低。

### 4. 可观测性与可重现性（Observability & Reproducibility）

每次运行自动创建带时间戳的独立输出目录（`output/<workflow>_<timestamp>/`），其中包含：原始输入（`prompt.txt`）、完整运行日志（`run.log`）、所有中间与最终输出文件。任何一次研究过程都可完整追溯和复现，无需手动重定向日志。

### 5. 渐进式降级（Progressive Degradation）

关键依赖均设有回退机制：PDF 解析优先使用 PyMuPDF，失败则自动回退到 PyPDF2；LLM 供应商在限流时自动切换；学术搜索工具在单一 API 不可用时仍可通过其他数据源返回结果。系统在部分组件缺失的情况下依然能够完成核心功能。

### 6. 开放标准兼容（OpenAI-Compatible API Alignment）

所有 LLM 供应商均通过 OpenAI 兼容接口（via LiteLLM）统一接入，使得未来接入新供应商只需添加一组配置，无需修改业务逻辑代码。

---

## 🛠️ Tech Stack

- **Multi-agent Framework**: [CrewAI](https://github.com/joaomdmoura/crewAI)
- **LLMs**: Qwen (Alibaba Cloud) · DeepSeek · Kimi (Moonshot AI) · GLM (Zhipu AI)
- **LLM Routing**: LiteLLM (provider fallback & OpenAI-compatible proxy)
- **Academic APIs**: DBLP · Semantic Scholar · arXiv
- **PDF Processing**: PyMuPDF (fitz) / PyPDF2
- **Network Analysis**: NetworkX + PyVis
- **CLI**: Typer + Rich
- **Language**: Python 3.10+

---

## ⚡ Quick Start

### 1. Install

```bash
# Clone the repository
git clone https://github.com/igeng/MARS.git
cd MARS

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# Or install as a package
pip install -e .
```

### 2. Configure API Keys

```bash
cp .env.example .env
```

Edit `.env` and fill in your API keys:

```env
# Required: at least one LLM provider
DASHSCOPE_API_KEY=sk-xxxxxxxx       # Qwen (Alibaba Cloud)
DEEPSEEK_API_KEY=sk-xxxxxxxx        # DeepSeek
MOONSHOT_API_KEY=sk-xxxxxxxx        # Kimi (Moonshot AI)
ZHIPU_API_KEY=xxxxxxxx.xxxxxxxx     # GLM (Zhipu AI)

# Optional: increases Semantic Scholar rate limits
SEMANTIC_SCHOLAR_API_KEY=xxxxxxxx

# Customize defaults
DEFAULT_LLM_PROVIDER=qwen           # qwen | deepseek | kimi | glm
MAX_PAPERS_PER_SEARCH=50
MAX_PAPERS_FOR_ANALYSIS=20

# Optional: enable CrewAI memory (requires an OpenAI-compatible embedding key)
ENABLE_MEMORY=false
```

### 3. Run

#### CLI

```bash
# Basic search (outputs paper_search.json + bilingual reviews to output/<run>/)
mars search "federated learning with differential privacy"

# Deep analysis (provide paper titles/info)
mars analyze "Federated Learning: Challenges, Methods, and Future Directions | Communication-Efficient Learning of Deep Networks"

# Connection analysis
mars connect "paper1 | paper2 | paper3" --topic "federated learning"

# Full research workflow
mars full "联邦学习隐私保护技术"

# Validate configuration (check API keys & provider availability)
mars check
```

#### Python API

```python
from mars.crews.search_crew import run_search
from mars.crews.analysis_crew import run_analysis
from mars.crews.connection_crew import run_connection
from mars.crews.full_research_crew import run_full_research

# Basic search
result = run_search("graph neural network for recommendation")

# Full research
result = run_full_research("knowledge graph embedding methods")
print(result)
```

---

## 📁 Project Structure

```
MARS/
├── main.py                    # Entry point
├── requirements.txt           # Dependencies
├── pyproject.toml             # Build config
├── .env.example               # Environment variable template
│
├── mars/
│   ├── config/
│   │   └── settings.py        # Configuration from environment variables
│   │
│   ├── agents/
│   │   ├── researcher.py      # Researcher Agent (domain analysis)
│   │   ├── searcher.py        # Searcher Agent (paper retrieval)
│   │   ├── analyzer.py        # Analyzer Agent (deep analysis)
│   │   ├── connector.py       # Connector Agent (relationship analysis)
│   │   ├── summarizer.py      # Summarizer Agent (bilingual review)
│   │   └── evaluator.py       # Evaluator Agent (quality assessment)
│   │
│   ├── tools/
│   │   ├── ccf_database.py    # CCF ranking database
│   │   ├── dblp_search.py     # DBLP API search
│   │   ├── semantic_scholar.py # Semantic Scholar API
│   │   ├── arxiv_api.py       # arXiv API
│   │   ├── pdf_parser.py      # PDF download & text extraction
│   │   ├── keyword_expander.py # LLM-powered keyword expansion
│   │   ├── citation_network.py # Citation network builder
│   │   └── file_manager.py    # File I/O utilities
│   │
│   ├── tasks/
│   │   └── task_definitions.py # 7 task factory functions
│   │
│   ├── crews/
│   │   ├── search_crew.py     # Workflow 1: Basic search (4 tasks)
│   │   ├── analysis_crew.py   # Workflow 2: Deep analysis
│   │   ├── connection_crew.py # Workflow 3: Connection analysis
│   │   └── full_research_crew.py # Workflow 4: Full research (7 tasks)
│   │
│   ├── services/
│   │   └── llm_gateway.py     # LLM gateway (4 providers + fallback)
│   │
│   ├── utils/
│   │   ├── llm_factory.py     # LLM instance factory (compatibility layer)
│   │   ├── logging_config.py  # Centralised logging setup
│   │   └── retry.py           # Retry decorator with exponential back-off
│   │
│   ├── api/
│   │   └── main.py            # FastAPI application
│   │
│   └── cli.py                 # CLI commands (Typer)
│
└── tests/
    ├── conftest.py            # Shared fixtures & mocks
    ├── test_basic.py          # Basic component tests
    ├── test_new_components.py # Settings, LLM gateway, DB, API tests
    └── test_usage_features.py # End-to-end usage feature tests
```

---

## 🧪 Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
# 105 tests total across 4 test files (all pass without API keys)
```

---

## 🌐 Supported LLM Providers

| Provider | Models | API |
|----------|--------|-----|
| **Alibaba Cloud (Qwen)** | qwen3.5-flash, qwen-max, qwen-plus | DashScope (OpenAI-compatible) |
| **DeepSeek** | deepseek-chat, deepseek-coder | DeepSeek Open API (OpenAI-compatible) |
| **Moonshot AI (Kimi)** | kimi-k2.5, moonshot-v1-8k | Moonshot API (OpenAI-compatible) |
| **Zhipu AI (GLM)** | glm-4-plus, glm-4-air | Zhipu Open Platform (OpenAI-compatible) |

All providers are accessed via OpenAI-compatible APIs, making it easy to switch between them or add new providers.

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [Technical Development Guide](docs/technical-guide.md) | System architecture, module design, API reference, extension guide |
| [User Manual](docs/user-guide.md) | Deployment (Windows 10 & Ubuntu), CLI/API usage, troubleshooting |

---

## 📝 License

MIT License. See [LICENSE](LICENSE) for details.
