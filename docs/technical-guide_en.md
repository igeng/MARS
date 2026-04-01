# MARS Technical Development Guide

**Multi-Agent Research System — Intelligent Academic Literature Retrieval and Analysis System**

> Version: 0.1.0 | License: MIT | Python ≥ 3.10

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [System Architecture](#2-system-architecture)
3. [Core Module Design](#3-core-module-design)
4. [Agent Detailed Design](#4-agent-detailed-design)
5. [Tool Layer Design](#5-tool-layer-design)
6. [Task Definitions](#6-task-definitions)
7. [Crew Orchestration](#7-crew-orchestration)
8. [LLM Gateway Service](#8-llm-gateway-service)
9. [Database Design](#9-database-design)
10. [Configuration Management](#10-configuration-management)
11. [REST API Endpoints](#11-rest-api-endpoints)
12. [CLI Interface](#12-cli-interface)
13. [Data Flow and Execution Process](#13-data-flow-and-execution-process)
14. [Testing Framework](#14-testing-framework)
15. [Technology Stack and Dependencies](#15-technology-stack-and-dependencies)
16. [Development Guidelines](#16-development-guidelines)
17. [Extension Guide](#17-extension-guide)
18. [Known Limitations and Future Plans](#18-known-limitations-and-future-plans)

---

## 1. System Overview

### 1.1 Project Background

MARS (Multi-Agent Research System) is an intelligent academic literature retrieval and analysis system built on the CrewAI multi-agent framework. The system leverages 6 specialized AI agents working collaboratively to complete the full academic research workflow — from research domain analysis and paper retrieval to deep parsing and review generation.

### 1.2 Core Capabilities

| Capability | Description |
|------------|-------------|
| Domain Analysis | Automatically identifies research domains and sub-domains; recommends CCF-ranked journals/conferences |
| Paper Retrieval | Simultaneously queries three major academic databases: DBLP, Semantic Scholar, and arXiv |
| Deep Parsing | Extracts core contributions, research methods, experimental design, and conclusions from papers |
| Connection Analysis | Builds citation networks, identifies topic clusters and research trends |
| Review Generation | Automatically generates an English review (≥ 3000 words) and translates it to Chinese (bilingual output) |
| Quality Evaluation | Multi-dimensional academic quality assessment (novelty, technical depth, experimental validity, writing quality) |

### 1.3 Launch Modes

MARS provides three launch modes:

- **Command-Line (CLI) Mode**: Run via the `mars` command or `python main.py`; suitable for local research use
- **Web API Mode**: Start a FastAPI server via `mars api`; accessible through a browser or HTTP client
- **Python API Mode**: Directly `import` and call the various `run_*` functions in Python code; suitable for secondary development

> **Note**: MARS does not currently include a frontend web interface. Web API mode provides REST API endpoints that can be interacted with using Postman, curl, or a custom-built frontend. FastAPI includes an interactive documentation page (Swagger UI) accessible in the browser at `http://localhost:8000/docs`.

---

## 2. System Architecture

### 2.1 Layered Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                       Interface Layer                             │
│                                                                   │
│   ┌──────────────────────┐    ┌──────────────────────────────┐   │
│   │   CLI (Typer + Rich) │    │  REST API (FastAPI + Uvicorn)│   │
│   │   mars/cli.py        │    │  mars/api/main.py            │   │
│   │                      │    │  Swagger UI: /docs           │   │
│   │   Commands: search,  │    │  Endpoints: /search,         │   │
│   │   analyze, connect,  │    │  /analyze, /connect,         │   │
│   │   full, api,         │    │  /full-research, /health     │   │
│   │   init-db, check     │    │                              │   │
│   └──────────┬───────────┘    └───────────────┬──────────────┘   │
│              │                                │                   │
└──────────────┼────────────────────────────────┼───────────────────┘
               │                                │
               ▼                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                     Orchestration Layer                           │
│                                                                   │
│   ┌────────────────┐ ┌────────────────┐ ┌────────────────────┐   │
│   │  search_crew   │ │ analysis_crew  │ │  connection_crew   │   │
│   │  Basic Search  │ │  Deep Analysis │ │  Connection Anal.  │   │
│   └────────────────┘ └────────────────┘ └────────────────────┘   │
│                                                                   │
│   ┌──────────────────────────────────────────────────────────┐   │
│   │               full_research_crew                          │   │
│   │               Full Research Workflow (6 Agents)           │   │
│   └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│   CrewAI Process: Sequential (context-dependent execution)       │
│   Task factory: mars/tasks/task_definitions.py                   │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Agent Layer                                  │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Researcher   │  │   Searcher   │  │   Analyzer   │           │
│  │  Domain Anal. │  │ Paper Search │  │ Deep Analyst │           │
│  │  LLM: Qwen   │  │  LLM: Kimi   │  │  LLM: Kimi   │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Connector    │  │  Summarizer  │  │  Evaluator   │           │
│  │  Connection   │  │   Review     │  │   Quality    │           │
│  │  LLM: Qwen   │  │  LLM: Qwen   │  │  LLM: Kimi   │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Tool Layer                                   │
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
│                     Infrastructure Layer                          │
│                                                                   │
│  ┌─────────────┐ ┌─────────────┐ ┌──────────────────────────┐   │
│  │  LLM Gateway│ │ SQLAlchemy  │ │   Configuration Mgmt.    │   │
│  │ (4 Providers)│ │  Database   │ │   (Pydantic Settings)    │   │
│  │  services/  │ │  database/  │ │   config/settings.py     │   │
│  └─────────────┘ └─────────────┘ └──────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 Project Directory Structure

```
MARS/
├── main.py                        # Entry point, invokes CLI
├── requirements.txt               # Dependency list
├── pyproject.toml                 # Build configuration
├── .env.example                   # Environment variable template
├── README.md                      # Project overview
│
├── docs/                          # Documentation directory
│   ├── technical-guide.md         # Technical development guide (Chinese)
│   ├── technical-guide_en.md      # Technical Development Guide (English, this file)
│   ├── user-guide.md              # User manual (Chinese)
│   └── user-guide_en.md           # User Manual (English)
│
├── mars/                          # Main package
│   ├── __init__.py                # Package initialization (version)
│   ├── cli.py                     # CLI command definitions (Typer)
│   │
│   ├── config/                    # Configuration management
│   │   ├── __init__.py
│   │   └── settings.py            # Pydantic BaseSettings
│   │
│   ├── services/                  # Service layer
│   │   ├── __init__.py
│   │   └── llm_gateway.py         # LLM gateway (4-provider routing + auto-fallback)
│   │
│   ├── agents/                    # 6 agents
│   │   ├── __init__.py
│   │   ├── researcher.py          # Domain analyst
│   │   ├── searcher.py            # Paper retrieval specialist
│   │   ├── analyzer.py            # Deep analyst
│   │   ├── connector.py           # Connection analyst
│   │   ├── summarizer.py          # Review generator (bilingual output)
│   │   └── evaluator.py           # Quality evaluator
│   │
│   ├── tools/                     # 9 tools
│   │   ├── __init__.py
│   │   ├── ccf_database.py        # CCF ranking database
│   │   ├── dblp_search.py         # DBLP search
│   │   ├── semantic_scholar.py    # Semantic Scholar search
│   │   ├── arxiv_api.py           # arXiv search
│   │   ├── pdf_parser.py          # PDF text extraction
│   │   ├── keyword_expander.py    # Keyword expansion
│   │   ├── citation_network.py    # Citation network builder
│   │   └── file_manager.py        # File read/write
│   │
│   ├── tasks/                     # Task definitions
│   │   ├── __init__.py
│   │   └── task_definitions.py    # 7 task factory functions
│   │
│   ├── crews/                     # 4 workflows
│   │   ├── __init__.py
│   │   ├── search_crew.py         # Basic search (4 tasks)
│   │   ├── analysis_crew.py       # Deep analysis
│   │   ├── connection_crew.py     # Connection analysis
│   │   └── full_research_crew.py  # Full research (7 tasks)
│   │
│   ├── database/                  # Database
│   │   ├── __init__.py
│   │   └── models.py              # SQLAlchemy models
│   │
│   ├── utils/                     # Utility functions
│   │   ├── __init__.py
│   │   ├── llm_factory.py         # LLM factory (compatibility layer)
│   │   ├── logging_config.py      # Centralized logging configuration
│   │   └── retry.py               # Exponential backoff retry decorator
│   │
│   └── api/                       # Web API
│       ├── __init__.py
│       └── main.py                # FastAPI application
│
└── tests/                         # Tests
    ├── __init__.py
    ├── conftest.py                # Shared fixtures and mocks
    ├── test_basic.py              # Basic component tests
    ├── test_new_components.py     # New component tests (Settings, LLM Gateway, DB, API, Tasks)
    └── test_usage_features.py     # Usage feature tests
```

---

## 3. Core Module Design

### 3.1 Module Dependency Graph

```
cli.py / api/main.py
    └── crews/ (search_crew, analysis_crew, connection_crew, full_research_crew)
          ├── agents/ (researcher, searcher, analyzer, connector, summarizer, evaluator)
          │     └── services/llm_gateway.py (get_llm_by_task → crewai.LLM)
          │     └── tools/ (various tools)
          └── tasks/task_definitions.py (task factory)
                └── agents/ (agent references)

config/settings.py ──── referenced by all modules
database/models.py ──── referenced by API/CLI
utils/logging_config.py ── called by CLI/API at startup
utils/retry.py ─────────── called by tool layer HTTP requests
```

### 3.2 Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Factory Pattern** | `create_*_agent()`, `create_*_task()`, `get_llm()` |
| **Singleton Pattern** | `settings = MarsSettings()` — global configuration singleton |
| **Strategy Pattern** | LLM provider routing: `_PROVIDER_MAP` + `_AGENT_LLM_MAP` |
| **Delegation Pattern** | CrewAI orchestrates Agent → Agent invokes Tool |
| **Separation of Concerns** | Agent definitions, Tool implementations, Task descriptions, and Crew orchestration are each independent |

---

## 4. Agent Detailed Design

### 4.1 Agent Overview

Each Agent is defined by the following attributes:

```python
Agent(
    role="role name",              # Agent role
    goal="work objective",         # Task goal description
    backstory="role background",   # Role capability description
    llm=get_llm_by_task(role),     # Obtained via LLM gateway
    tools=[...],                   # List of available tools
    verbose=True,                  # Output execution logs
    allow_delegation=False,        # Do not delegate to other agents
    max_iter=N,                    # Maximum number of iterations
)
```

### 4.2 Individual Agent Details

#### Researcher — Domain Analyst

| Attribute | Value |
|-----------|-------|
| Module | `mars/agents/researcher.py` |
| LLM | Qwen-Max (strong reasoning capability) |
| Tools | `CCFDatabaseQueryTool`, `KeywordExpanderTool` |
| max_iter | 5 |
| Input | User research topic (Chinese/English) |
| Output | JSON: research_domain, sub_domains, keywords, recommended_venues |

#### Searcher — Paper Retrieval Specialist

| Attribute | Value |
|-----------|-------|
| Module | `mars/agents/searcher.py` |
| LLM | Kimi (kimi-k2.5, long context) |
| Tools | `DBLPSearchTool`, `SemanticScholarSearchTool`, `ArXivSearchTool`, `KeywordExpanderTool`, `FileWriterTool` |
| max_iter | 8 |
| Input | Domain analysis result + keywords |
| Output | Structured paper list (20–50 papers), saved to paper_search.json |

#### Analyzer — Deep Analyst

| Attribute | Value |
|-----------|-------|
| Module | `mars/agents/analyzer.py` |
| LLM | Kimi (kimi-k2.5, long context) |
| Tools | `ArXivSearchTool`, `SemanticScholarSearchTool`, `PDFParserTool`, `FileWriterTool` |
| max_iter | 10 |
| Input | Paper list (top 20) |
| Output | Deep paper analysis report (analysis_results.json) |

#### Connector — Connection Analyst

| Attribute | Value |
|-----------|-------|
| Module | `mars/agents/connector.py` |
| LLM | Qwen (qwen3.5-flash, reasoning capability) |
| Tools | `CitationNetworkTool`, `SemanticScholarSearchTool`, `FileWriterTool` |
| max_iter | 8 |
| Input | All retrieved papers |
| Output | Citation network summary, topic clusters, research trends (connection_analysis.json) |

#### Summarizer — Review Generator

| Attribute | Value |
|-----------|-------|
| Module | `mars/agents/summarizer.py` |
| LLM | Qwen (qwen3.5-flash, long-form text generation) |
| Tools | `FileWriterTool` |
| max_iter | 8 |
| Input | All analysis results |
| Output | English review (review_en.md, ≥ 3000 words) + Chinese translation (review_zh.md) |

#### Evaluator — Quality Evaluator

| Attribute | Value |
|-----------|-------|
| Module | `mars/agents/evaluator.py` |
| LLM | Kimi (kimi-k2.5) |
| Tools | `SemanticScholarSearchTool`, `FileWriterTool` |
| max_iter | 8 |
| Input | Papers from deep analysis |
| Output | Multi-dimensional scores (0–10) + review comments (quality_evaluation.json) |

---

## 5. Tool Layer Design

### 5.1 Tool Overview

All tools inherit from CrewAI's `BaseTool` class and define `name`, `description`, and `_run()` methods.

| Tool | Module | External Dependency | Requires API Key |
|------|--------|---------------------|------------------|
| CCFDatabaseQueryTool | `ccf_database.py` | None (built-in data) | No |
| DBLPSearchTool | `dblp_search.py` | DBLP REST API | No |
| SemanticScholarSearchTool | `semantic_scholar.py` | S2 Graph API | Optional |
| ArXivSearchTool | `arxiv_api.py` | arXiv Atom API | No |
| PDFParserTool | `pdf_parser.py` | PyMuPDF / PyPDF2 | No |
| KeywordExpanderTool | `keyword_expander.py` | LLM Provider | Yes (LLM Key) |
| CitationNetworkTool | `citation_network.py` | S2 Graph API + NetworkX | Optional |
| FileWriterTool | `file_manager.py` | None | No |
| FileReaderTool | `file_manager.py` | None | No |

### 5.2 Tool Input/Output Formats

All search tools accept a JSON string as input:

```python
# Example: DBLP search tool input
'{"query": "federated learning", "max_results": 20, "year_from": 2020, "year_to": 2024}'

# Example: PDF parser tool input
'{"url": "https://arxiv.org/pdf/2301.12345.pdf", "max_pages": 20, "max_chars": 8000}'
```

### 5.3 External API Endpoints

| API | Endpoint | Method | Rate Limit |
|-----|----------|--------|------------|
| DBLP | `https://dblp.org/search/publ/api` | GET | Unlimited |
| Semantic Scholar | `https://api.semanticscholar.org/graph/v1/paper/search` | GET | 100/5min (no key) |
| arXiv | `https://export.arxiv.org/api/query` | GET | Fair use |

---

## 6. Task Definitions

### 6.1 Task Factory Pattern

`mars/tasks/task_definitions.py` provides 7 task factory functions:

```python
def create_domain_analysis_task(agent, topic, *, context=None) -> Task
def create_paper_search_task(agent, topic, max_papers=50, *, context=None) -> Task
def create_deep_analysis_task(agent, topic, papers_info="", limit=20, *, context=None) -> Task
def create_connection_analysis_task(agent, topic, papers_info="", *, context=None) -> Task
def create_english_review_task(agent, topic, *, context=None) -> Task     # English review (≥ 3000 words)
def create_review_generation_task(agent, topic, *, context=None) -> Task  # Chinese translation review
def create_quality_evaluation_task(agent, limit=20, *, context=None) -> Task
```

### 6.2 Task Chain Context Passing

Tasks pass results to one another via CrewAI's `context` parameter:

```
create_domain_analysis_task      ← no prior dependencies
create_paper_search_task         ← context = [domain_analysis_task]
create_deep_analysis_task        ← context = [paper_search_task]
create_connection_analysis_task  ← context = [paper_search_task]
create_quality_evaluation_task   ← context = [deep_analysis_task]
create_english_review_task       ← context = [all prior tasks]
create_review_generation_task    ← context = [all prior tasks + english_review_task]
```

> **Note**: `create_review_generation_task` is responsible for translating the English review generated by `create_english_review_task` into Chinese — it does not regenerate the review content from scratch.

---

## 7. Crew Orchestration

### 7.1 Basic Search Workflow (search_crew)

```
Researcher ──(domain_analysis_task)──→ Searcher ──(paper_search_task)──→
Summarizer ──(english_review_task)──→ Summarizer ──(chinese_review_task)──→ Result
```

- Number of Agents: 3 (Researcher, Searcher, Summarizer)
- Number of Tasks: 4
- Execution Mode: Sequential
- Output: paper_search.json + review_en.md (English review ≥ 3000 words) + review_zh.md (Chinese translation) + domain_analysis.json

### 7.2 Deep Analysis Workflow (analysis_crew)

```
Analyzer ──(deep_analysis_task)──→ Evaluator ──(quality_evaluation_task)──→ Result
```

- Number of Agents: 2
- Execution Mode: Sequential
- Output: Analysis report + quality evaluation

### 7.3 Connection Analysis Workflow (connection_crew)

```
Connector ──(connection_analysis_task)──→
Summarizer ──(chinese_review_task)──→ Summarizer ──(english_review_task)──→ Result
```

- Number of Agents: 2 (Connector, Summarizer)
- Number of Tasks: 3
- Execution Mode: Sequential
- Output: Connection analysis report (connection_analysis.json) + Chinese review + English review

### 7.4 Full Research Workflow (full_research_crew)

```
Phase 1:  Researcher (domain_analysis_task)
          → Searcher (bulk_search_task)
Phase 2:  Analyzer (deep_analysis_task)
          → Connector (connection_analysis_task)
          → Evaluator (quality_evaluation_task)
Phase 3:  Summarizer (english_review_task → chinese_review_task)
```

- Number of Agents: 6
- Number of Tasks: 7
- Execution Mode: Sequential (CrewAI implements logical dependency chain via `context`)
- Output: Complete research report (domain_analysis.json + paper_search.json + analysis_results.json + connection_analysis.json + quality_evaluation.json + review_en.md + review_zh.md)

---

## 8. LLM Gateway Service

### 8.1 Module Location

`mars/services/llm_gateway.py`

### 8.2 Supported LLM Providers

| Provider | Function | API Endpoint | Default Model |
|----------|----------|--------------|---------------|
| Alibaba Cloud (Qwen) | `get_qwen_llm()` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | qwen3.5-flash |
| DeepSeek | `get_deepseek_llm()` | `https://api.deepseek.com/v1` | deepseek-chat |
| Moonshot (Kimi) | `get_kimi_llm()` | `https://api.moonshot.cn/v1` | kimi-k2.5 |
| Zhipu AI (GLM) | `get_glm_llm()` | `https://open.bigmodel.cn/api/paas/v4` | glm-4.7-flash |

- `get_qwen_llm()` / `get_deepseek_llm()` / `get_kimi_llm()` / `get_glm_llm()` return `ChatOpenAI` instances for direct use inside tools (e.g., `KeywordExpanderTool`).
- `get_llm_by_task()` returns a `crewai.LLM` instance for use by CrewAI `Agent`s; the model string carries a LiteLLM provider prefix: all OpenAI-compatible providers (DashScope/Moonshot/Zhipu) uniformly use the `openai/<model>` prefix with a `base_url` (LiteLLM generic compatibility path), while DeepSeek uses the `deepseek/<model>` prefix (e.g., `openai/qwen3.5-flash`, `openai/kimi-k2.5`, `deepseek/deepseek-chat`). The `openai/` here is a LiteLLM routing prefix and is unrelated to the official OpenAI API.

### 8.3 Agent → LLM Mapping

```python
_AGENT_LLM_MAP = {
    "researcher": ("qwen", None),    # Strong reasoning → Qwen (qwen3.5-flash)
    "searcher":   ("kimi", None),    # Long-context retrieval → Kimi (kimi-k2.5)
    "analyzer":   ("kimi", None),    # Long-context deep analysis → Kimi (kimi-k2.5)
    "connector":  ("qwen", None),    # Relational reasoning → Qwen (qwen3.5-flash)
    "summarizer": ("qwen", None),    # Long-form generation → Qwen (qwen3.5-flash)
    "evaluator":  ("kimi", None),    # Evaluation tasks → Kimi (kimi-k2.5)
}
```

When the preferred provider's API key is not configured, the gateway automatically falls back to `DEFAULT_LLM_PROVIDER`, then to any available provider.

### 8.4 Usage

```python
from mars.services.llm_gateway import get_llm, get_llm_by_task
from langchain_openai import ChatOpenAI
from crewai import LLM

# Get ChatOpenAI by provider name (for direct LLM calls inside tools)
llm: ChatOpenAI = get_llm(provider="qwen", temperature=0.3)

# Get crewai.LLM by agent role (recommended, for CrewAI Agent)
llm: LLM = get_llm_by_task("researcher", temperature=0.3)
```

---

## 9. Database Design

### 9.1 Module Location

`mars/database/models.py`

### 9.2 Data Models

#### CCFVenue Table

```sql
CREATE TABLE ccf_venues (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        VARCHAR(50) NOT NULL UNIQUE,  -- indexed
    full_name   VARCHAR(256) NOT NULL,
    ccf_rank    VARCHAR(1) NOT NULL,          -- indexed (A/B/C)
    venue_type  VARCHAR(20) NOT NULL,         -- conference/journal
    domains     TEXT NOT NULL,                -- comma-separated domain list
    dblp_url    VARCHAR(512) DEFAULT '',
    created_at  DATETIME DEFAULT NOW(),
    updated_at  DATETIME DEFAULT NOW()
);
```

#### Paper Table

```sql
CREATE TABLE papers (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    title                 VARCHAR(512) NOT NULL,  -- indexed
    authors               TEXT DEFAULT '',         -- JSON array
    venue                 VARCHAR(256) DEFAULT '',
    year                  INTEGER,                 -- indexed
    citation_count        INTEGER DEFAULT 0,
    doi                   VARCHAR(256) DEFAULT '',
    url                   VARCHAR(512) DEFAULT '',
    abstract              TEXT DEFAULT '',
    pdf_url               VARCHAR(512) DEFAULT '',
    source                VARCHAR(50) DEFAULT '',  -- dblp/semantic_scholar/arxiv
    semantic_scholar_id   VARCHAR(64) DEFAULT '',   -- indexed
    relevance_score       FLOAT DEFAULT 0.0,
    created_at            DATETIME DEFAULT NOW()
);
```

### 9.3 Database Operations

```python
from mars.database.models import init_db, get_session

# Initialize the database (create tables)
init_db()

# Get a session
session = get_session()
```

---

## 10. Configuration Management

### 10.1 Module Location

`mars/config/settings.py`

### 10.2 Configuration Class

```python
class MarsSettings(BaseSettings):
    model_config = {"env_file": str(_ENV_PATH), "env_file_encoding": "utf-8"}

    # LLM API Keys
    DASHSCOPE_API_KEY: str = ""
    DEEPSEEK_API_KEY: str = ""
    MOONSHOT_API_KEY: str = ""
    ZHIPU_API_KEY: str = ""

    # Model configuration
    QWEN_MODEL: str = "qwen3.5-flash"
    DEEPSEEK_MODEL: str = "deepseek-chat"
    DEEPSEEK_CODER_MODEL: str = "deepseek-coder"
    KIMI_MODEL: str = "kimi-k2.5"
    GLM_MODEL: str = "glm-4.7-flash"  # Free model, used as last resort fallback

    # API endpoints
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    MOONSHOT_BASE_URL: str = "https://api.moonshot.cn/v1"

    # Academic search
    SEMANTIC_SCHOLAR_API_KEY: str = ""
    MAX_PAPERS_PER_SEARCH: int = Field(default=50, gt=0)
    MAX_PAPERS_FOR_ANALYSIS: int = Field(default=20, gt=0)
    # arXiv request read timeout (seconds); increase if the network is slow
    ARXIV_SEARCH_TIMEOUT: int = Field(default=30, gt=0)

    # Application settings
    DEFAULT_LLM_PROVIDER: str = "qwen"
    LOG_LEVEL: str = "INFO"
    OUTPUT_DIR: Path = Path("./output")

    # GLM rate-limit handling
    GLM_RATE_LIMIT_MAX_RETRIES: int = 3       # Number of retries on RateLimitError
    GLM_RATE_LIMIT_RETRY_DELAY: float = 5.0   # Exponential backoff base delay (seconds)

    # Crew memory (requires a valid OpenAI-compatible embedding key to enable)
    ENABLE_MEMORY: bool = False

    # Database
    DATABASE_URL: str = "sqlite:///./mars.db"

    # API server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
```

### 10.3 Usage

```python
from mars.config import settings

# All modules access configuration through this singleton
print(settings.DASHSCOPE_API_KEY)
print(settings.MAX_PAPERS_PER_SEARCH)
```

---

## 11. REST API Endpoints

### 11.1 Basic Information

- **Framework**: FastAPI 0.110.0+
- **Server**: Uvicorn
- **Launch**: `mars api` or `uvicorn mars.api.main:app`
- **Interactive Docs**: `http://localhost:8000/docs` (Swagger UI)

### 11.2 Endpoint List

| Method | Path | Request Body | Description |
|--------|------|--------------|-------------|
| GET | `/health` | None | Health check |
| POST | `/search` | `SearchRequest` | Basic search |
| POST | `/analyze` | `AnalyzeRequest` | Deep analysis |
| POST | `/connect` | `ConnectRequest` | Connection analysis |
| POST | `/full-research` | `FullResearchRequest` | Full research |

### 11.3 Request/Response Models

```python
# Requests
class SearchRequest(BaseModel):
    topic: str            # Research topic
    max_results: int = 50 # Maximum number of results (1–200)

class AnalyzeRequest(BaseModel):
    papers_info: str      # Paper information
    max_papers: int = 20  # Maximum papers to analyze (1–100)

class ConnectRequest(BaseModel):
    papers_info: str      # Paper information
    topic: str            # Research topic

class FullResearchRequest(BaseModel):
    topic: str            # Research topic

# Response
class TaskResponse(BaseModel):
    status: str = "success"
    result: str           # Task result string
```

### 11.4 Usage Examples

```bash
# Basic search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"topic": "federated learning privacy", "max_results": 20}'

# Full research
curl -X POST http://localhost:8000/full-research \
  -H "Content-Type: application/json" \
  -d '{"topic": "graph neural network for recommendation"}'
```

---

## 12. CLI Interface

### 12.1 Command List

| Command | Description | Example |
|---------|-------------|---------|
| `mars search <topic>` | Basic search (outputs bilingual review) | `mars search "federated learning"` |
| `mars analyze <papers>` | Deep analysis | `mars analyze "Paper1 \| Paper2"` |
| `mars connect <papers> --topic <topic>` | Connection analysis | `mars connect "P1 \| P2" -t "ML"` |
| `mars full <topic>` | Full research | `mars full "federated learning privacy"` |
| `mars api` | Start API server | `mars api --port 8000` |
| `mars init-db` | Initialize database | `mars init-db` |
| `mars check` | Check configuration status | `mars check` |

### 12.2 Optional Parameters

- `--max-results / -n`: Maximum number of search results (default: 50)
- `--max-papers / -n`: Maximum number of papers to analyze (default: 20)
- `--host / -h`: API server address (default: 0.0.0.0)
- `--port / -p`: API server port (default: 8000)
- `--topic / -t`: Research topic

---

## 13. Data Flow and Execution Process

### 13.1 Full Research Workflow Data Flow

```
User input: "federated learning with differential privacy"
           │
           ▼
┌──── Phase 1 ─────────────────────────────────────────────┐
│                                                           │
│  Researcher Agent                                        │
│  ├─ calls keyword_expander → expands keywords            │
│  ├─ calls ccf_database_query → recommends CCF venues     │
│  └─ output: domain_analysis.json                         │
│      {research_domain, sub_domains, keywords,            │
│       recommended_venues}                                 │
│                         │                                 │
│                         ▼                                 │
│  Searcher Agent                                          │
│  ├─ calls dblp_search → retrieves from DBLP              │
│  ├─ calls semantic_scholar_search → retrieves from S2    │
│  ├─ calls arxiv_search → retrieves from arXiv            │
│  └─ output: paper_search.json (50 papers)                │
│                                                           │
└──── Phase 2 ─────────────────────────────────────────────┐
│                                                           │
│  Analyzer Agent (top 20 papers)                          │
│  ├─ calls pdf_parser → downloads and extracts PDF text   │
│  ├─ output: analysis_results.json                        │
│                                                           │
│  Connector Agent (all 50 papers)                         │
│  ├─ calls citation_network → builds citation network     │
│  ├─ output: connection_analysis.json                     │
│                                                           │
│  Evaluator Agent (top 20 papers)                         │
│  ├─ output: quality_evaluation.json                      │
│                                                           │
└──── Phase 3 ─────────────────────────────────────────────┐
│                                                           │
│  Summarizer Agent                                        │
│  ├─ generates English review (≥ 3000 words) → review_en.md│
│  └─ translates to Chinese review → review_zh.md         │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

---

## 14. Testing Framework

### 14.1 Test Files

| File | Test Content | Notes |
|------|--------------|-------|
| `tests/conftest.py` | Shared fixtures and mocks | Mocks external APIs, database, etc. |
| `tests/test_basic.py` | Configuration, tools, Agent/Crew imports | ~46 tests |
| `tests/test_new_components.py` | Settings, LLM Gateway, DB, API, Tasks | ~33 tests |
| `tests/test_usage_features.py` | End-to-end usage features | ~26 tests |

**Total: approximately 105 tests, all passable without real API keys.**

### 14.2 Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
python -m pytest tests/ -v

# Run specific tests
python -m pytest tests/test_basic.py -v
python -m pytest tests/test_new_components.py -v

# Generate coverage report
python -m pytest tests/ --cov=mars --cov-report=html
```

### 14.3 Testing Characteristics

- No real API keys are required (all external calls are mocked)
- External HTTP requests are simulated via `unittest.mock.patch`
- Database tests use the `tmp_path` temporary directory fixture

---

## 15. Technology Stack and Dependencies

### 15.1 Core Frameworks

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Multi-agent framework | CrewAI | ≥0.80.0 | Agent/Task/Crew orchestration |
| LLM interface | LangChain OpenAI | ≥0.2.0 | Unified LLM call interface (tool layer) |
| LLM routing | LiteLLM | ≥1.50.0 | CrewAI Agent LLM routing and provider fallback |
| CLI | Typer | ≥0.12.0 | Command-line interface |
| Web API | FastAPI | ≥0.110.0 | REST API |
| Database | SQLAlchemy | ≥2.0.0 | ORM |
| Configuration | Pydantic Settings | ≥2.1.0 | Typed configuration |

### 15.2 LLM SDKs

| SDK | Version | Corresponding LLM |
|-----|---------|-------------------|
| dashscope | ≥1.20.0 | Qwen (Alibaba Cloud) |
| openai | ≥1.50.0 | DeepSeek, Kimi (OpenAI-compatible) |
| zhipuai | ≥2.1.0 | GLM (Zhipu AI) |

### 15.3 Academic Search

| Library | Version | Purpose |
|---------|---------|---------|
| semanticscholar | ≥0.8.0 | Semantic Scholar API |
| scholarly | ≥1.7.11 | Google Scholar |
| requests | ≥2.31.0 | HTTP requests |
| httpx | ≥0.27.0 | Async HTTP |

### 15.4 Data Processing

| Library | Version | Purpose |
|---------|---------|---------|
| PyMuPDF | ≥1.23.0 | PDF text extraction (fitz) |
| PyPDF2 | ≥3.0.0 | PDF fallback extraction |
| networkx | ≥3.2 | Citation network analysis |
| pandas | ≥2.0.0 | Data processing |

---

## 16. Development Guidelines

### 16.1 Code Style

- Line length: 100 characters (`black --line-length 100`)
- Linter: Ruff (`ruff check`)
- Type checking: MyPy (`mypy mars/`)
- Python version: 3.10+

### 16.2 Steps to Add a New Agent

1. Create `new_agent.py` in `mars/agents/`
2. Implement the `create_new_agent()` factory function
3. Add the mapping to `_AGENT_LLM_MAP` in `mars/services/llm_gateway.py`
4. Add the corresponding task factory to `mars/tasks/task_definitions.py`
5. Assemble the Agent and Task in the relevant Crew
6. Add tests in `tests/`

### 16.3 Steps to Add a New Tool

1. Create `new_tool.py` in `mars/tools/`
2. Inherit from `BaseTool` and define `name`, `description`, and `_run()` methods
3. Add the tool instance to the `tools` list in the Agent(s) that need it
4. Add mock-based tests in `tests/`

---

## 17. Extension Guide

### 17.1 Adding a New LLM Provider

```python
# In mars/services/llm_gateway.py

def get_new_provider_llm(model=None, temperature=0.3):
    return _openai_compatible_llm(
        model=model or settings.NEW_PROVIDER_MODEL,
        api_key=settings.NEW_PROVIDER_API_KEY,
        base_url="https://api.new-provider.com/v1",
        temperature=temperature,
    )

# Add to _PROVIDER_MAP
_PROVIDER_MAP["new_provider"] = get_new_provider_llm

# Add configuration in settings.py
NEW_PROVIDER_API_KEY: str = ""
NEW_PROVIDER_MODEL: str = "model-name"
```

### 17.2 Adding a New Academic Search Source

```python
# Create a new tool in mars/tools/

class NewSearchTool(BaseTool):
    name: str = "new_search"
    description: str = "Search papers from NewSource"

    def _run(self, query: str) -> str:
        # Implement search logic
        pass
```

---

## 18. Known Limitations and Future Plans

### 18.1 Current Limitations

| Limitation | Description | Priority |
|------------|-------------|----------|
| No frontend interface | CLI + REST API only; no web frontend | Medium |
| Synchronous execution | CrewAI workflow is synchronous; no progress feedback for long tasks | Medium |
| No caching | Every query re-invokes external APIs; no paper cache | Low |
| No authentication | API endpoints have no authentication mechanism | Low |
| No rate limiting | May exceed external API rate limits | Low |
| No containerized deployment | Missing Docker/K8s configuration | Medium |

### 18.2 Future Plans

- [ ] Add web frontend interface (React/Vue)
- [ ] Asynchronous task processing (Celery/RQ)
- [ ] ChromaDB vector search integration
- [ ] Docker containerized deployment
- [ ] API authentication and rate limiting
- [ ] Paper full-text caching mechanism
- [ ] Multi-language support improvements
- [ ] Token usage tracking and cost monitoring

---

*Document last updated: 2026-03-30*
