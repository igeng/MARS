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
│  │  (Qwen-Max) │  │ (DeepSeek)  │  │   (Kimi)    │         │
│  │  领域分析   │  │  论文检索   │  │  深度解析   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Connector  │  │ Summarizer  │  │  Evaluator  │         │
│  │  (GLM-4)    │  │  (Qwen-Max) │  │ (DeepSeek)  │         │
│  │  关联分析   │  │  综述生成   │  │  质量评估   │         │
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
| **Researcher** | 领域分析师 – identifies research domain, recommends CCF-ranked venues | Qwen-Max | `ccf_database_query`, `keyword_expander` |
| **Searcher** | 论文检索师 – queries academic databases and ranks results | DeepSeek-V3 | `dblp_search`, `semantic_scholar_search`, `arxiv_search` |
| **Analyzer** | 深度分析师 – downloads PDFs, extracts contributions & experiments | Kimi-Long-Context | `arxiv_search`, `pdf_parser`, `semantic_scholar_search` |
| **Connector** | 关联分析师 – builds citation networks, identifies clusters & trends | GLM-4-Plus | `citation_network_builder`, `semantic_scholar_search` |
| **Summarizer** | 综述生成师 – synthesizes analyses into a cohesive literature review | Qwen-Max | `file_writer` |
| **Evaluator** | 质量评估师 – scores papers on novelty, depth, validity, writing | DeepSeek-V3 | `semantic_scholar_search`, `file_writer` |

---

## 🔄 Workflows

### 1. Basic Search (基础检索)
```
User topic → Researcher → Searcher → Paper list
```

### 2. Deep Analysis (深度分析)
```
Paper list → Analyzer → Evaluator → Analysis report
```

### 3. Connection Analysis (关联分析)
```
Multiple papers → Connector → Summarizer → Network graph + Review
```

### 4. Full Research (完整研究流程)
```
User topic
  → Researcher (domain analysis)
  → Searcher (50 papers)
  → [Parallel] Analyzer (top 20) + Connector (all 50) + Evaluator (top 20)
  → Summarizer (comprehensive review ≥2000 words)
```

---

## 🛠️ Tech Stack

- **Multi-agent Framework**: [CrewAI](https://github.com/joaomdmoura/crewAI)
- **LLMs**: Qwen (Alibaba Cloud) · DeepSeek · Kimi (Moonshot AI) · GLM (Zhipu AI)
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
```

### 3. Run

#### CLI

```bash
# Basic search
mars search "federated learning with differential privacy"

# Deep analysis (provide paper titles/info)
mars analyze "Federated Learning: Challenges, Methods, and Future Directions | Communication-Efficient Learning of Deep Networks"

# Connection analysis
mars connect "paper1 | paper2 | paper3" --topic "federated learning"

# Full research workflow
mars full "联邦学习隐私保护技术"
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
│   │   ├── summarizer.py      # Summarizer Agent (review generation)
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
│   ├── crews/
│   │   ├── search_crew.py     # Workflow 1: Basic search
│   │   ├── analysis_crew.py   # Workflow 2: Deep analysis
│   │   ├── connection_crew.py # Workflow 3: Connection analysis
│   │   └── full_research_crew.py # Workflow 4: Full research
│   │
│   ├── utils/
│   │   └── llm_factory.py     # LLM instance factory
│   │
│   └── cli.py                 # CLI commands (Typer)
│
└── tests/
    └── test_basic.py          # Unit tests
```

---

## 🧪 Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## 🌐 Supported LLM Providers

| Provider | Models | API |
|----------|--------|-----|
| **Alibaba Cloud (Qwen)** | qwen-max, qwen-plus, qwen-turbo | DashScope (OpenAI-compatible) |
| **DeepSeek** | deepseek-chat, deepseek-coder | DeepSeek Open API (OpenAI-compatible) |
| **Moonshot AI (Kimi)** | moonshot-v1-8k, moonshot-v1-128k | Moonshot API (OpenAI-compatible) |
| **Zhipu AI (GLM)** | glm-4-plus, glm-4-air | Zhipu Open Platform (OpenAI-compatible) |

All providers are accessed via OpenAI-compatible APIs, making it easy to switch between them or add new providers.

---

## 📝 License

MIT License. See [LICENSE](LICENSE) for details.
