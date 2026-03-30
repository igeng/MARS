# 🚀 MARS - Multi-Agent Research System

**多智能体学术文献智能检索与分析系统**

MARS is an automated academic literature search and deep analysis system built on a multi-agent architecture. It leverages cutting-edge Chinese LLMs and the CrewAI framework to orchestrate specialized AI agents that collaborate to deliver comprehensive research intelligence—from journal recommendations and paper retrieval to deep analysis, relationship mapping, and literature review generation.

---

## 📐 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    用户交互层 (User Interface)                │
│                  CLI (Typer) / Python API                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              编排层 (Orchestrator - CrewAI)                  │
│         Sequential & Parallel Task Execution                │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  智能体层 (Agent Layer)                       │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Researcher  │  │  Searcher   │  │  Analyzer   │         │
│  │  (Qwen)     │  │   (Kimi)    │  │   (Kimi)    │         │
│  │  领域分析   │  │  论文检索   │  │  深度解析   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Connector  │  │ Summarizer  │  │  Evaluator  │         │
│  │  (Qwen)     │  │   (Qwen)    │  │   (Kimi)    │         │
│  │  关联分析   │  │  双语综述   │  │  质量评估   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              工具层 (Tools / Services)                       │
│  DBLP API │ Semantic Scholar │ arXiv API │ CCF Database     │
│  PDF Parser │ Keyword Expander │ Citation Network Builder   │
└─────────────────────────────────────────────────────────────┘
```

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
