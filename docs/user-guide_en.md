# MARS User Manual

**Multi-Agent Research System — Intelligent Multi-Agent Academic Literature Retrieval and Analysis System**

> Version: 0.1.0 | Python ≥ 3.10

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [System Requirements](#2-system-requirements)
3. [Windows 10 Local Deployment](#3-windows-10-local-deployment)
   - [3.1–3.8 Standard Deployment (venv)](#31-install-python)
   - [3.9 Deployment with Conda + PyCharm](#39-deployment-with-conda--pycharm-alternative)
   - [3.10 Deployment with Git Bash (Recommended Terminal Option)](#310-deployment-with-git-bash-recommended-terminal-option)
4. [Ubuntu Cloud Deployment](#4-ubuntu-cloud-deployment)
5. [Configuration Reference](#5-configuration-reference)
6. [Command-Line (CLI) Usage Guide](#6-command-line-cli-usage-guide)
7. [Web API Usage Guide](#7-web-api-usage-guide)
8. [Python API Usage Guide](#8-python-api-usage-guide)
9. [Workflow Details](#9-workflow-details)
10. [Output Files Reference](#10-output-files-reference)
11. [Frequently Asked Questions & Troubleshooting](#11-frequently-asked-questions--troubleshooting)
12. [Appendix: API Key Acquisition Guide](#12-appendix-api-key-acquisition-guide)

---

## 1. System Overview

### 1.1 What is MARS?

MARS is an intelligent academic literature retrieval and analysis system built on a multi-agent architecture. It uses 6 specialized AI agents working in concert to help researchers automatically:

- 🔍 **Paper Retrieval**: Simultaneously searches three major academic databases — DBLP, Semantic Scholar, and arXiv
- 📊 **Deep Analysis**: Extracts core contributions, research methods, and experimental design from papers
- 🔗 **Relationship Analysis**: Builds citation networks and discovers research trends and hot topics
- 📝 **Review Generation**: Automatically generates an English literature review (≥ 3,000 words) with a Chinese translation (bilingual output)
- ⭐ **Quality Assessment**: Evaluates academic quality of papers across multiple dimensions

### 1.2 Usage Modes

MARS provides three usage modes:

| Mode | Description | Best For |
|------|-------------|----------|
| **Command Line (CLI)** | Use the `mars` command in a terminal | Quick queries, script automation |
| **Web API** | Start an HTTP server; access via browser or HTTP client | Team sharing, remote calls |
| **Python API** | Call directly from Python code | Custom development, integration with other systems |

> **Important**: MARS does not currently have a standalone web frontend. In Web API mode, you can interact with the system through the **Swagger UI interactive documentation** built into FastAPI (`http://localhost:8000/docs`).

---

## 2. System Requirements

### 2.1 Hardware Requirements

| Item | Minimum | Recommended |
|------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4 GB | 8 GB+ |
| Disk | 2 GB free space | 10 GB+ |
| Network | Internet access | Stable connection |

> Note: MARS calls cloud-hosted LLM APIs and does not require a local GPU.

### 2.2 Software Requirements

| Software | Version | Notes |
|----------|---------|-------|
| Python | ≥ 3.10 | Required |
| pip | Latest | Python package manager |
| Git | Any | For downloading the code (optional) |

### 2.3 API Key Requirements

At least **one** LLM provider API Key must be configured:

| Provider | API Key Environment Variable | How to Obtain |
|----------|------------------------------|---------------|
| Alibaba Cloud Qwen | `DASHSCOPE_API_KEY` | [Alibaba Cloud Bailian Platform](https://bailian.console.aliyun.com/) |
| DeepSeek | `DEEPSEEK_API_KEY` | [DeepSeek Open Platform](https://platform.deepseek.com/) |
| Moonshot AI Kimi | `MOONSHOT_API_KEY` | [Moonshot Open Platform](https://platform.moonshot.cn/) |
| Zhipu AI GLM | `ZHIPU_API_KEY` | [Zhipu AI Open Platform](https://open.bigmodel.cn/) |

**Recommended**: Configure API Keys for all 4 providers for the best experience (the system automatically selects the most suitable LLM for each task).

---

## 3. Windows 10 Local Deployment

### 3.1 Install Python

1. Visit the [Python official download page](https://www.python.org/downloads/)
2. Download the Windows installer for Python 3.10 or higher
3. Run the installer and **make sure to check "Add Python to PATH"**
4. Open Command Prompt (CMD) or PowerShell and verify the installation:

```powershell
python --version
# Should output Python 3.10.x or higher
```

### 3.2 Download MARS

**Option 1: Git Clone (Recommended)**

```powershell
git clone https://github.com/igeng/MARS.git
cd MARS
```

**Option 2: Direct Download**

1. Visit https://github.com/igeng/MARS
2. Click "Code" → "Download ZIP"
3. Extract to any directory
4. Navigate to that directory in Command Prompt:

```powershell
cd C:\path\to\MARS
```

### 3.3 Create a Virtual Environment

```powershell
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\activate

# Confirm activation ((.venv) should appear before the prompt)
(.venv) C:\path\to\MARS>
```

### 3.4 Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# Or install in development mode (recommended)
pip install -e .
```

> **Note**: PyMuPDF (`pymupdf`) may require additional Visual C++ Build Tools on Windows. If installation fails, try:
>
> ```powershell
> pip install pymupdf --no-build-isolation
> ```
>
> Or skip it (the system will automatically fall back to PyPDF2).

### 3.5 Configure API Keys

```powershell
# Copy the environment variable template
copy .env.example .env

# Edit the .env file with Notepad
notepad .env
```

Fill in your API Keys in the `.env` file:

```env
# Required: fill in at least one LLM provider API Key
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
MOONSHOT_API_KEY=sk-xxxxxxxxxxxxxxxx
ZHIPU_API_KEY=xxxxxxxx.xxxxxxxxxxxxxxxx

# Optional: increases Semantic Scholar rate limits
SEMANTIC_SCHOLAR_API_KEY=xxxxxxxxxxxxxxxx

# Default LLM provider (change to match the provider whose Key you configured)
DEFAULT_LLM_PROVIDER=qwen
```

### 3.6 Verify Installation

```powershell
# Run tests to confirm successful installation
python -m pytest tests/ -v

# Test that the CLI is available
mars --help
```

### 3.7 Getting Started

```powershell
# Check configuration status (recommended on first use)
mars check

# Basic search
mars search "federated learning with differential privacy"

# Start the API server
mars api --port 8000
# Then open http://localhost:8000/docs in your browser
```

### 3.8 Windows Notes

1. **Encoding Issues**: If terminal output shows garbled characters, run in PowerShell:
   ```powershell
   [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
   chcp 65001
   ```

2. **Path Separators**: Windows uses `\` as the path separator, but in Python it is recommended to use `/` or `Path` objects.

3. **Firewall**: If the API server cannot be accessed from other devices after starting, check whether Windows Firewall allows port 8000.

4. **Long Path Support**: If errors occur due to excessively long paths, enable Windows long path support:
   - Run `regedit`
   - Navigate to `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
   - Set `LongPathsEnabled` to `1`

### 3.9 Deployment with Conda + PyCharm (Alternative)

If your Windows 10 machine already has **Anaconda/Miniconda** and **PyCharm** installed, you can deploy MARS using the following approach without separately installing Python or using `venv`.

#### 3.9.1 Prerequisites

| Software | Notes |
|----------|-------|
| Anaconda or Miniconda | [Anaconda Download](https://www.anaconda.com/download) / [Miniconda Download](https://docs.conda.io/en/latest/miniconda.html) |
| PyCharm | Community Edition is sufficient, [PyCharm Download](https://www.jetbrains.com/pycharm/download/) |
| Git (optional) | For cloning the repository; you can also download the ZIP directly |

#### 3.9.2 Create a Conda Environment

Open **Anaconda Prompt** (search for "Anaconda Prompt" in the Start menu):

```powershell
# 1. Create a Conda environment named "mars" with Python 3.12
conda create -n mars python=3.12 -y

# 2. Activate the environment
conda activate mars

# 3. Verify the Python version
python --version
# Should output Python 3.12.x
```

> **Why Python 3.12?** MARS requires Python ≥ 3.10. Python 3.12 is stable and has good compatibility. You may also choose 3.10 or 3.11.

#### 3.9.3 Download the Project Code

```powershell
# Option 1: Git Clone (recommended)
git clone https://github.com/igeng/MARS.git
cd MARS

# Option 2: If Git is not installed, download the ZIP from GitHub, extract it, then navigate to the directory
cd C:\path\to\MARS
```

#### 3.9.4 Install Project Dependencies

In Anaconda Prompt with the `mars` environment activated:

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# Install MARS in development mode (recommended — this makes the mars command available)
pip install -e .
```

> **Conda User Tip**: All of MARS's dependencies can be installed via pip; `conda install` is not needed. Core packages like CrewAI may have older versions on conda-forge, so it is recommended to use pip exclusively.

If `pymupdf` installation fails, you can skip it — the system will automatically fall back to PyPDF2:
```powershell
pip install pymupdf --no-build-isolation
```

#### 3.9.5 Configure API Keys

```powershell
# Copy the environment variable template
copy .env.example .env

# Edit with Notepad (or edit directly in PyCharm)
notepad .env
```

Fill in your API Keys in `.env` (at least one LLM provider is required):

```env
# Fill in at least one
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
MOONSHOT_API_KEY=sk-xxxxxxxxxxxxxxxx
ZHIPU_API_KEY=xxxxxxxx.xxxxxxxxxxxxxxxx

# Change to the provider whose API Key you have configured
DEFAULT_LLM_PROVIDER=qwen
```

#### 3.9.6 Configure the Project in PyCharm

**Step 1: Open the Project**

1. Launch PyCharm
2. Select **File → Open**
3. Select the MARS project root directory (the folder containing `main.py`)
4. Click **OK** to open

**Step 2: Configure the Conda Interpreter**

1. Open **File → Settings** (or `Ctrl+Alt+S`)
2. Navigate to **Project: MARS → Python Interpreter**
3. Click the gear icon ⚙️ in the top right → **Add Interpreter → Add Local Interpreter**
4. Select **Conda Environment** in the left panel
5. Choose **Use existing environment**
6. Select `mars` from the dropdown (the Conda environment you just created)
7. Click **OK** to confirm

> If the `mars` environment does not appear in the dropdown, click the **...** button to manually locate the Conda executable, typically at:
> - Anaconda: `C:\Users\<your-username>\anaconda3\Scripts\conda.exe`
> - Miniconda: `C:\Users\<your-username>\miniconda3\Scripts\conda.exe`

**Step 3: Configure Environment Variables (Optional)**

When running code in PyCharm, `.env` files are loaded automatically by `pydantic-settings`. If additional configuration is needed:

1. Open **Run → Edit Configurations**
2. Select or create a Python configuration
3. Add variables in **Environment variables**, or specify the `.env` file path in the **EnvFile** plugin

> In most cases, no manual configuration is needed — MARS's `settings.py` automatically loads the `.env` file from the project root.

**Step 4: Configure the Terminal (Recommended)**

To have PyCharm's built-in terminal automatically use the Conda environment:

1. Open **File → Settings → Tools → Terminal**
2. Set the **Shell path** to:
   ```
   cmd.exe /K "C:\Users\<your-username>\anaconda3\Scripts\activate.bat mars"
   ```
   Or for Miniconda:
   ```
   cmd.exe /K "C:\Users\<your-username>\miniconda3\Scripts\activate.bat mars"
   ```
3. Click **OK**; reopen the Terminal panel to see the `(mars)` prefix

#### 3.9.7 Running MARS in PyCharm

**Option 1: Run CLI Commands via the Terminal Panel**

Click the **Terminal** tab at the bottom of PyCharm, then type commands directly:

```powershell
# Check configuration status
mars check

# Basic search
mars search "federated learning with differential privacy"

# Full research workflow
mars full "survey of federated learning privacy protection"

# Start the API server
mars api --port 8000
```

**Option 2: Right-Click to Run main.py**

1. Locate `main.py` in the project file tree
2. Right-click → **Run 'main'**
3. PyCharm will automatically create a run configuration on the first run

To pass command-line arguments:

1. Open **Run → Edit Configurations**
2. Select the `main` configuration
3. Enter parameters in the **Parameters** field, for example:
   ```
   search "graph neural network for recommendation"
   ```
4. Click **OK**, then click the run button ▶️

**Option 3: Configure Shortcut Run Configurations**

Create dedicated run configurations for frequently used commands:

1. **Run → Edit Configurations → + → Python**
2. Example configuration:

| Field | Value |
|-------|-------|
| Name | MARS - Full Research |
| Script path | `$ProjectFileDir$/main.py` |
| Parameters | `full "your research topic"` |
| Working directory | `$ProjectFileDir$` |
| Python interpreter | Select the `mars` Conda environment |

3. Click **OK**; afterwards you can quickly select and run it from the toolbar dropdown

#### 3.9.8 Running Tests in PyCharm

1. Right-click the `tests/` folder in the project file tree
2. Select **Run 'pytest in tests'**
3. Or run in the Terminal:

```powershell
python -m pytest tests/ -v
```

> All tests pass without an API Key (tool calls are mocked).

#### 3.9.9 Debugging in PyCharm

1. Click to the left of a line number in any `.py` file to set a breakpoint
2. Right-click `main.py` → **Debug 'main'** (or press `Shift+F9`)
3. The program will pause at the breakpoint so you can inspect variable values, the call stack, etc.

**Debugging the API Server**:

1. Create a run configuration with `api --port 8000` in the Parameters field
2. Run using Debug mode
3. Open `http://localhost:8000/docs` in a browser to trigger requests
4. Breakpoints will be hit in the corresponding route handler functions

#### 3.9.10 Common Issues

**Q: The `mars` command is not found in the Conda environment?**

Make sure you have installed in development mode within the Conda environment:
```powershell
conda activate mars
pip install -e .
```

Or use `python main.py` as a substitute for the `mars` command.

**Q: PyCharm cannot resolve module imports in the project?**

1. Confirm that the Conda `mars` environment is correctly selected as the Python Interpreter
2. Right-click the project root directory → **Mark Directory as → Sources Root**
3. Or run `pip install -e .` so that PyCharm can correctly resolve the `mars` package

**Q: PyCharm Terminal does not automatically activate the Conda environment?**

- Check that the Shell path in Settings → Tools → Terminal is configured correctly
- Or manually run `conda activate mars` in the Terminal each time

**Q: Dependency installation is very slow with "solving environment"?**

MARS dependencies are installed via pip; `conda install` is not needed. Use directly:
```powershell
conda activate mars
pip install -r requirements.txt
```

---

### 3.10 Deployment with Git Bash (Recommended Terminal Option)

Git Bash is a Unix-style terminal emulator bundled with **Git for Windows**. It provides a Bash experience on Windows that is nearly identical to Linux/macOS (commands like `ls`, `cd`, `source`, and `export` all work), making it ideal for developers who are comfortable with Linux command-line workflows or need cross-platform consistency when using MARS.

> **Use case**: You have Git for Windows installed and want to run MARS CLI commands on Windows 10 in a Linux-like way, without using Anaconda or PyCharm.

#### 3.10.1 Prerequisites Summary

Before getting started, confirm the following software is installed:

| Software | Version Requirement | Notes |
|----------|---------------------|-------|
| Windows 10 | 1903+ | Any Win10 version works |
| Git for Windows | ≥ 2.30 | Includes Git Bash — the core tool for this section |
| Python | ≥ 3.10 (3.10, 3.11, or 3.12 recommended) | Install from the Python official website; **do not** install only from the Microsoft Store |
| pip | Bundled with Python | Ensure `python -m pip` is callable from Git Bash |

---

#### 3.10.2 Install Git for Windows (to Get Git Bash)

1. Open a browser and visit **[https://git-scm.com/download/win](https://git-scm.com/download/win)**
2. Click **"Click here to download"** to download the latest 64-bit installer (`Git-x.xx.x-64-bit.exe`)
3. Run the installer; recommended options for each step are:

   | Installation Step | Recommended Option |
   |-------------------|--------------------|
   | Select Components | Keep defaults (includes Git Bash Here) |
   | Choosing the default editor | Choose Vim or Notepad++ (your preference) |
   | **Adjusting your PATH environment** | Select **"Git from the command line and also from 3rd-party software"** (recommended) |
   | Choosing HTTPS transport backend | Keep default (OpenSSL) |
   | **Configuring the line ending conversions** | Select **"Checkout as-is, commit Unix-style line endings"** (avoids CRLF issues) |
   | Configuring the terminal emulator | Select **"Use MinTTY"** (default; better interface) |
   | Remaining steps | Keep defaults, click Next → Install |

4. After installation, right-clicking on the desktop or any folder should show a **"Git Bash Here"** context menu item.

---

#### 3.10.3 Install Python (Ensure Git Bash Can Call It)

> If Python 3.10+ is already installed, skip to the verification step.

1. Visit **[https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)**
2. Download the **Windows installer (64-bit) for Python 3.10, 3.11, or 3.12** (MARS requires Python ≥ 3.10; all three versions work)
3. When running the installer, **make sure to check the following**:
   - ✅ **"Add Python 3.x to PATH"** (at the bottom of the page — very important!)
   - ✅ **"Install pip"** (checked by default)
4. Click **"Install Now"**

**Verify the Python installation in Git Bash**:

Open Git Bash (search for "Git Bash" in the Start menu, or right-click a folder and select "Git Bash Here") and type:

```bash
python --version
# Expected output: Python 3.12.x (or 3.10.x / 3.11.x)

python -m pip --version
# Expected output similar to: pip 23.x from C:\Users\<username>\AppData\...\pip (python 3.12)
```

> **Common Issue**: If you see `python: command not found`, Python was not added to PATH.
> Fix: Manually add the Python installation path to the system environment variables (typically `C:\Users\<username>\AppData\Local\Programs\Python\Python312\` and `C:\Users\<username>\AppData\Local\Programs\Python\Python312\Scripts\`), then **close and reopen Git Bash**.

---

#### 3.10.4 Clone the MARS Repository

In Git Bash, run:

```bash
# Navigate to the directory where you want to store the project (example: D:\Projects)
cd /d/Projects
# If the directory does not exist, create it first:
mkdir -p /d/Projects && cd /d/Projects

# Clone the repository
git clone https://github.com/igeng/MARS.git

# Enter the project directory
cd MARS

# View the project structure (confirm the clone succeeded)
ls -la
# You should see main.py, requirements.txt, mars/, docs/, etc.
```

> **Path Note**: In Git Bash, Windows paths are written as:
> - `C:\Users\username` → `/c/Users/username`
> - `D:\Projects` → `/d/Projects`

---

#### 3.10.5 Create a Python Virtual Environment

In Git Bash, from the MARS project root directory, run:

```bash
# Confirm you are in the MARS directory
pwd
# Expected output similar to: /d/Projects/MARS

# Create a virtual environment (named .venv, located in the project root)
python -m venv .venv

# Activate the virtual environment (Git Bash syntax, different from PowerShell)
source .venv/Scripts/activate

# Confirm activation: (.venv) should appear before the prompt
# (.venv) username@DESKTOP:/d/Projects/MARS $
```

> **Note**: In Git Bash, activate the virtual environment with `source .venv/Scripts/activate`, which differs slightly from Linux/macOS (`source .venv/bin/activate`) because Windows virtual environments place scripts in `Scripts/` rather than `bin/`.

**Verify the virtual environment is activated**:

```bash
# Check the current Python interpreter path (should point inside .venv)
which python
# Expected output similar to: /d/Projects/MARS/.venv/Scripts/python

python --version
# Expected output: Python 3.12.x
```

---

#### 3.10.6 Upgrade pip and Install Project Dependencies

After activating the virtual environment, run:

```bash
# Step 1: Upgrade pip to the latest version (avoids errors installing certain packages with an older pip)
python -m pip install --upgrade pip

# Step 2: Install all dependency packages
pip install -r requirements.txt
# This installs crewai, litellm, fastapi, pydantic-settings, pymupdf, and all other dependencies
# Estimated time: 3-10 minutes depending on network speed

# Step 3 (recommended): Install MARS itself in development mode
# This registers the mars command-line tool (i.e., mars search, mars full, etc.)
pip install -e .
```

**Verify that the dependency installation succeeded**:

```bash
# Check whether the mars command is available
mars --help
# Expected output: MARS command help with subcommands including search, analyze, connect, full, api, check

# Verify that core packages are installed
python -c "import crewai; print('crewai OK')"
python -c "import litellm; print('litellm OK')"
python -c "import fastapi; print('fastapi OK')"
```

> **If `pymupdf` installation fails** (this occasionally happens on Windows):
>
> ```bash
> # Option 1: Disable isolated build
> pip install pymupdf --no-build-isolation
>
> # Option 2: Skip installation (the system will automatically use PyPDF2 as a fallback)
> # Temporarily remove the pymupdf line from requirements.txt and install the rest
> grep -v "pymupdf" requirements.txt > req_no_mupdf.txt
> pip install -r req_no_mupdf.txt
> ```

---

#### 3.10.7 Configure API Keys

```bash
# Copy the environment variable template file
cp .env.example .env

# Open for editing with Windows Notepad (Windows apps can be called from Git Bash like this)
notepad .env

# Or use nano (if installed)
# nano .env

# Or open with VS Code (if installed)
# code .env
```

In the `.env` file that opens, fill in your API Keys. Below is the complete configuration reference:

```env
# ============================================================
# LLM API Key
# ============================================================

# DeepSeek
# Obtain at: https://platform.deepseek.com/ → API Keys
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Zhipu AI (GLM)
# Obtain at: https://open.bigmodel.cn/ → API Key (format: xxx.yyy)
ZHIPU_API_KEY=xxxxxxxxxxxxxxxx.xxxxxxxxxxxxxxxx

# ============================================================
# Optional: Academic Search API
# ============================================================

# Semantic Scholar (usable without configuration, but rate-limited)
# Obtain at: https://www.semanticscholar.org/product/api
SEMANTIC_SCHOLAR_API_KEY=

# ============================================================
# Application Configuration
# ============================================================

# Default LLM provider (change to the provider whose API Key you have configured)
# Options: qwen | deepseek | kimi | glm
DEFAULT_LLM_PROVIDER=qwen

# Maximum number of papers to retrieve and analyze
MAX_PAPERS_PER_SEARCH=50
MAX_PAPERS_FOR_ANALYSIS=20

# Whether to enable CrewAI memory (requires an OpenAI Embedding API key; disabled by default)
ENABLE_MEMORY=false
```

Save and close the file.

---

#### 3.10.8 Verify the Full Installation

Run MARS's built-in configuration check command:

```bash
mars check
```

Under normal conditions, the output looks like:

```
╭─────────────────────────────────────────────────────╮
│ 🚀 MARS - Multi-Agent Research System               │
│    Intelligent Multi-Agent Academic Literature      │
│    Retrieval and Analysis System                    │
╰─────────────────────────────────────────────────────╯

🔎 Checking system configuration...

          LLM Provider Status
┌──────────┬──────────────────────┬──────────────────────────────┐
│ Provider │ Environment Variable │ Status                       │
├──────────┼──────────────────────┼──────────────────────────────┤
│ qwen     │ DASHSCOPE_API_KEY    │ ✅ Configured (sk-x****xxxx) │
│ deepseek │ DEEPSEEK_API_KEY     │ ✅ Configured (sk-x****xxxx) │
│ kimi     │ MOONSHOT_API_KEY     │ ✅ Configured (sk-x****xxxx) │
│ glm      │ ZHIPU_API_KEY        │ ✅ Configured (xxxx****xxxx) │
└──────────┴──────────────────────┴──────────────────────────────┘

Default provider: qwen
Available providers: qwen, deepseek, kimi, glm

Semantic Scholar API Key: ⚠ Not configured (usable but rate-limited)

Output directory: D:\Projects\MARS\output
Database:         sqlite:///./mars.db

✅ System ready — 4 LLM providers available
```

If you see `✅ System ready`, you can start using MARS.

---

#### 3.10.9 Running CLI Commands in Git Bash

**Before starting, confirm the virtual environment is activated** (the prompt should have `(.venv)` at the front):

```bash
# If Git Bash was closed or a new window was opened, reactivate the virtual environment
cd /d/Projects/MARS
source .venv/Scripts/activate
```

**Basic Search (search)**

```bash
# Search for a research topic; automatically generates a paper list + bilingual review
mars search "federated learning with differential privacy"

# Chinese topics are also supported
mars search "federated learning privacy protection"

# Specify maximum number of results (default: 50)
mars search "graph neural network" --max-results 30
```

Output files are saved in the `output/search_<timestamp>/` directory and include:
- `prompt.txt` — original user input
- `run.log` — complete run log
- `paper_search.json` — paper list (with title, authors, year, citation count, etc.)
- `review_en.md` — English academic review (≥ 3,000 words)
- `review_zh.md` — Chinese review (high-quality translation of the English version)

**Deep Analysis (analyze)**

```bash
# Input a list of paper titles (separated by |) for deep parsing and quality assessment
mars analyze "Federated Learning: Challenges, Methods, and Future Directions | Communication-Efficient Learning of Deep Networks from Decentralized Data"

# Specify maximum number of papers to analyze (default: 20)
mars analyze "paper1 | paper2 | paper3" --max-papers 10
```

**Relationship Analysis (connect)**

```bash
# Analyze citation relationships and thematic connections among multiple papers
mars connect "Attention Is All You Need | BERT: Pre-training of Deep Bidirectional Transformers | GPT-3: Language Models are Few-Shot Learners" --topic "transformer models in NLP"
```

**Full Research Workflow (full)**

```bash
# One command executes: domain analysis → retrieval → deep parsing → relationship analysis → quality assessment → bilingual review
# This is the most complete research workflow; estimated time: 12-21 minutes
mars full "survey of knowledge graph embedding methods"

# English topic
mars full "survey of large language model alignment techniques"
```

**Start the Web API Server**

```bash
# Start the FastAPI server on local port 8000
mars api --port 8000

# After starting, open Swagger UI in a browser:
# http://localhost:8000/docs
```

**Run in the Background (nohup)**

```bash
# Run a full research task in the background; log output to bg.log
nohup mars full "topic here" > bg.log 2>&1 &
echo "Background PID: $!"

# Watch background task progress in real time
tail -f bg.log
```

---

#### 3.10.10 Git Bash-Specific Notes

**① Garbled Chinese Output**

Git Bash (MinTTY) has good UTF-8 support and garbled text is usually not an issue. If problems occur, set the following in Git Bash:

```bash
# Set terminal encoding to UTF-8
export LANG=zh_CN.UTF-8
export PYTHONIOENCODING=utf-8

# Or add these two lines to ~/.bashrc to take effect automatically on startup
echo 'export LANG=zh_CN.UTF-8' >> ~/.bashrc
echo 'export PYTHONIOENCODING=utf-8' >> ~/.bashrc
source ~/.bashrc
```

**② `mars` Command Not Found (`command not found`)**

Cause: Virtual environment is not activated, or the project was not installed in development mode.

```bash
# Fix: activate the virtual environment first, then install
source .venv/Scripts/activate
pip install -e .

# Verify
which mars
# Should output: /d/Projects/MARS/.venv/Scripts/mars
```

**③ Spaces in the Python Path**

If the project path contains spaces (e.g., `D:\My Projects\MARS`), wrap it in quotes in Git Bash:

```bash
cd "/d/My Projects/MARS"
source .venv/Scripts/activate
```

It is recommended to place the project in a path without spaces (e.g., `/d/Projects/MARS`) to avoid potential compatibility issues.

**④ `pip` Still Points to Global Python After Activating the Virtual Environment**

```bash
# Check which pip is being used
which pip
# Should output the .venv path, e.g.: /d/Projects/MARS/.venv/Scripts/pip

# If it points to the wrong location, use explicitly:
python -m pip install -r requirements.txt
python -m pip install -e .
```

**⑤ Virtual Environment Must Be Reactivated Every Time Git Bash Opens**

Add a shortcut function to `~/.bashrc`:

```bash
# Edit ~/.bashrc in Git Bash
echo 'alias mars-activate="cd /d/Projects/MARS && source .venv/Scripts/activate"' >> ~/.bashrc
source ~/.bashrc

# From now on, just run the following each time Git Bash opens:
mars-activate
```

**⑥ Slow `pip install`**

To speed up downloads, use a mirror source:

```bash
# One-time install using the Aliyun mirror
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# Or permanently configure the mirror (written to the pip config file)
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
```

**⑦ Long Path Error (`WinError 206`)**

If you encounter errors about paths being too long, enable Windows long path support:

Run PowerShell as Administrator (search for PowerShell in the Start menu, right-click "Run as administrator") and execute:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

No restart is needed; the change takes effect the next time Git Bash is opened.

---

#### 3.10.11 Quick Reference: Complete Deployment Steps

The following is the minimal command sequence to go from zero to a running system (assumes Git and Python are already installed):

```bash
# 1. Open Git Bash and navigate to your working directory
cd /d/Projects

# 2. Clone the project
git clone https://github.com/igeng/MARS.git
cd MARS

# 3. Create and activate the virtual environment
python -m venv .venv
source .venv/Scripts/activate

# 4. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 5. Configure API Keys
cp .env.example .env
notepad .env          # Fill in at least one LLM API Key, save and exit

# 6. Verify configuration
mars check

# 7. Start using MARS!
mars search "your research topic"
```

---

## 4. Ubuntu Cloud Deployment

### 4.1 System Preparation

```bash
# Update the system
sudo apt update && sudo apt upgrade -y

# Install Python and required tools
sudo apt install -y python3.10 python3.10-venv python3-pip git curl

# Verify the Python version
python3 --version
# Should output Python 3.10.x or higher
```

> If you are on an older Ubuntu version (e.g., 20.04), Python 3.10 may not be in the default repository:
>
> ```bash
> sudo add-apt-repository ppa:deadsnakes/ppa
> sudo apt update
> sudo apt install -y python3.10 python3.10-venv python3.10-dev
> ```

### 4.2 Download MARS

```bash
# Clone the repository
git clone https://github.com/igeng/MARS.git
cd MARS
```

### 4.3 Create a Virtual Environment

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Confirm activation
(.venv) user@server:~/MARS$
```

### 4.4 Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install system-level dependencies (may be required by PyMuPDF)
sudo apt install -y libmupdf-dev

# Install project dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### 4.5 Configure API Keys

```bash
# Copy the environment variable template
cp .env.example .env

# Edit the .env file
nano .env
# Or use vim: vim .env
```

Fill in the API Keys (see the Windows section for details).

### 4.6 Verify Installation

```bash
# Run tests
python -m pytest tests/ -v

# Test the CLI
mars --help
```

### 4.7 Use CLI Mode

```bash
# Basic search
mars search "deep learning for medical image analysis"

# Full research (run in the background)
nohup mars full "federated learning privacy protection" > research_output.log 2>&1 &
```

### 4.8 Deploy the API Server

#### Option 1: Run Directly

```bash
# Run in the foreground
mars api --host 0.0.0.0 --port 8000

# Run in the background
nohup mars api --host 0.0.0.0 --port 8000 > api.log 2>&1 &
```

#### Option 2: Use a systemd Service (Recommended for Production)

Create a service file:

```bash
sudo nano /etc/systemd/system/mars-api.service
```

Enter the following content (adjust paths as needed):

```ini
[Unit]
Description=MARS API Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/MARS
Environment="PATH=/home/ubuntu/MARS/.venv/bin"
EnvironmentFile=/home/ubuntu/MARS/.env
ExecStart=/home/ubuntu/MARS/.venv/bin/uvicorn mars.api.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
# Reload systemd
sudo systemctl daemon-reload

# Start the service
sudo systemctl start mars-api

# Enable auto-start on boot
sudo systemctl enable mars-api

# Check status
sudo systemctl status mars-api

# View logs
sudo journalctl -u mars-api -f
```

#### Option 3: Use Nginx Reverse Proxy (Recommended for Production)

Install Nginx:

```bash
sudo apt install -y nginx
```

Create an Nginx configuration:

```bash
sudo nano /etc/nginx/sites-available/mars
```

Enter:

```nginx
server {
    listen 80;
    server_name your-server-ip-or-domain;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Long-running requests need a longer timeout
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
```

Enable the configuration:

```bash
# Create a symbolic link
sudo ln -s /etc/nginx/sites-available/mars /etc/nginx/sites-enabled/

# Remove the default configuration (optional)
sudo rm /etc/nginx/sites-enabled/default

# Test the configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
```

### 4.9 Firewall Configuration

```bash
# Allow the HTTP port
sudo ufw allow 80/tcp

# If accessing port 8000 directly
sudo ufw allow 8000/tcp

# Check firewall status
sudo ufw status
```

### 4.10 Security Recommendations

1. **Do not expose the .env file**: Ensure `.env` is not in a publicly accessible directory
2. **Use HTTPS**: Configure an SSL certificate (Let's Encrypt) for production environments
3. **Restrict access**: Use a firewall to limit API access by source
4. **Keep up to date**: Regularly update the system and dependency packages

---

## 5. Configuration Reference

### 5.1 Complete List of Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DASHSCOPE_API_KEY` | ⚡ | (empty) | Alibaba Cloud Qwen API Key |
| `DEEPSEEK_API_KEY` | ⚡ | (empty) | DeepSeek API Key |
| `MOONSHOT_API_KEY` | ⚡ | (empty) | Kimi API Key |
| `ZHIPU_API_KEY` | ⚡ | (empty) | GLM API Key |
| `SEMANTIC_SCHOLAR_API_KEY` | No | (empty) | Semantic Scholar API Key |
| `DEFAULT_LLM_PROVIDER` | No | qwen | Default LLM provider |
| `QWEN_MODEL` | No | qwen3.5-flash | Qwen model name |
| `DEEPSEEK_MODEL` | No | deepseek-chat | DeepSeek model name |
| `KIMI_MODEL` | No | kimi-k2.5 | Kimi model name |
| `GLM_MODEL` | No | glm-4.7-flash | GLM model name (free model, used as fallback) |
| `MAX_PAPERS_PER_SEARCH` | No | 50 | Maximum papers to retrieve per search |
| `MAX_PAPERS_FOR_ANALYSIS` | No | 20 | Maximum papers for deep analysis |
| `ARXIV_SEARCH_TIMEOUT` | No | 30 | arXiv API request timeout in seconds |
| `OUTPUT_DIR` | No | ./output | Root directory for output files |
| `ENABLE_MEMORY` | No | false | Whether to enable CrewAI memory (requires an OpenAI-compatible embedding Key) |
| `GLM_RATE_LIMIT_MAX_RETRIES` | No | 3 | GLM rate-limit retry count |
| `GLM_RATE_LIMIT_RETRY_DELAY` | No | 5.0 | GLM rate-limit base backoff delay (seconds) |
| `DATABASE_URL` | No | sqlite:///./mars.db | Database connection URL |
| `LOG_LEVEL` | No | INFO | Log level |
| `API_HOST` | No | 0.0.0.0 | API server host address |
| `API_PORT` | No | 8000 | API server port |

> ⚡ At least one LLM provider API Key must be configured. All 4 are recommended for the best experience.

### 5.2 Using a Single LLM Provider

If you only have an API Key for one LLM, set `DEFAULT_LLM_PROVIDER` to the corresponding provider name:

```env
# Example: only a DeepSeek Key
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
DEFAULT_LLM_PROVIDER=deepseek
```

> Note: With this configuration, all 6 Agents will use the same LLM, which may affect output quality.

---

## 6. Command-Line (CLI) Usage Guide

### 6.1 View Help

```bash
mars --help
mars search --help
mars full --help
```

### 6.2 Basic Search

Search for academic papers on a specified topic:

```bash
# English topic
mars search "federated learning with differential privacy"

# Chinese topic
mars search "graph neural network recommendation system"

# Limit the number of results
mars search "deep reinforcement learning" --max-results 20
mars search "deep reinforcement learning" -n 20
```

**Output**: Paper list with title, authors, publication venue/conference, year, citation count, and URL. Each run generates 6 files in the `output/search_<timestamp>/` directory: `prompt.txt`, `run.log`, `domain_analysis.json`, `paper_search.json`, `review_en.md` (English review ≥ 3,000 words), and `review_zh.md` (Chinese translation).

### 6.3 Deep Analysis

Perform a deep content analysis of specified papers:

```bash
# Separate multiple papers with |
mars analyze "Federated Learning: Challenges, Methods, and Future Directions | Communication-Efficient Learning of Deep Networks"

# Limit the number of papers to analyze
mars analyze "paper1 | paper2 | paper3" --max-papers 5
```

**Output**: Core contributions, research methods, experimental design, conclusions, and more for each paper.

### 6.4 Relationship Analysis

Analyze citation relationships and thematic connections among multiple papers:

```bash
mars connect "paper1 | paper2 | paper3" --topic "federated learning"
mars connect "paper1 | paper2" -t "federated learning"
```

**Output**: Citation network summary, topic clusters, research trends, key papers + literature review.

### 6.5 Full Research (Recommended)

Execute the complete end-to-end research workflow:

```bash
# English topic
mars full "knowledge graph embedding methods"

# Chinese topic
mars full "federated learning privacy protection techniques"
```

**Execution Stages**:
1. Domain Analysis → recommends CCF journals/conferences (`domain_analysis.json`)
2. Multi-database Retrieval → retrieves up to 50 relevant papers (`paper_search.json`)
3. Deep Analysis → extracts core content from the top 20 papers (`analysis_results.json`)
4. Relationship Analysis → builds citation network, identifies trends (`connection_analysis.json`)
5. Quality Assessment → multi-dimensional scoring (`quality_evaluation.json`)
6. Review Generation → generates English review ≥ 3,000 words (`review_en.md`) → translates to Chinese (`review_zh.md`)

**Output**: A complete research report saved to 7 files in the `output/full_<timestamp>/` directory.

### 6.6 Start the API Server

```bash
# Default startup (0.0.0.0:8000)
mars api

# Custom host and port
mars api --host 127.0.0.1 --port 3000
mars api -h 0.0.0.0 -p 9000
```

After starting, open `http://localhost:8000/docs` in a browser to see the interactive API documentation.

### 6.7 Initialize the Database

```bash
mars init-db
```

Creates the SQLite database and table schema (run before first use).

### 6.8 Check System Configuration

```bash
mars check
```

Checks the configuration status of all LLM provider API Keys, and displays available providers, the output directory, the database path, and other information to help diagnose configuration issues.

---

## 7. Web API Usage Guide

### 7.1 Start the Server

```bash
mars api --port 8000
```

### 7.2 Swagger UI (Browser-Based Interaction)

After starting, open in a browser: **http://localhost:8000/docs**

This is the interactive documentation interface built into FastAPI. You can directly in the browser:
1. View all API endpoints
2. Fill in parameters and send requests
3. View response results

![Swagger UI Usage Instructions]
1. Click the endpoint you want to use (e.g., POST /search)
2. Click "Try it out"
3. Fill in parameters in the request body
4. Click "Execute"
5. View the response results below

### 7.3 Using curl

> **Async API mode (v0.2.0)**: All workflow endpoints use a "submit-then-poll" pattern.
> `POST` returns a `task_id` immediately; poll `GET /task/{task_id}` for status and results.
> Long HTTP timeouts for slow workflows are no longer necessary.

#### Health Check

```bash
curl http://localhost:8000/health
# Returns: {"status": "ok"}
```

#### Basic Search

```bash
# 1. Submit task → returns task_id immediately
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"topic": "federated learning privacy", "max_results": 20}'
# Returns: {"task_id":"a1b2c3d4e5f6","status":"pending"}  (HTTP 202)

# 2. Poll for status
curl http://localhost:8000/task/a1b2c3d4e5f6
# Returns: {"task_id":"a1b2c3d4e5f6","status":"running","result":"","error":""}
# On completion: {"task_id":"...","status":"success","result":"...","error":""}
```

#### Deep Analysis

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"papers_info": "Paper Title 1 | Paper Title 2", "max_papers": 10}'
```

#### Relationship Analysis

```bash
curl -X POST http://localhost:8000/connect \
  -H "Content-Type: application/json" \
  -d '{"papers_info": "Paper1 | Paper2 | Paper3", "topic": "federated learning"}'
```

#### Full Research

```bash
curl -X POST http://localhost:8000/full-research \
  -H "Content-Type: application/json" \
  -d '{"topic": "knowledge graph embedding methods"}'
```

> **Tip**: All workflow endpoints return HTTP 202 (Accepted) + `task_id`.
> Use `GET /task/{task_id}` to poll; `status` transitions: `pending` → `running` → `success` / `failed`.

### 7.4 Using Python requests

```python
import requests
import time

BASE = "http://localhost:8000"

# 1. Submit task
resp = requests.post(
    f"{BASE}/search",
    json={"topic": "deep learning for NLP", "max_results": 20}
)
task = resp.json()
task_id = task["task_id"]
print(f"Task submitted: {task_id}")

# 2. Poll until done
while True:
    resp = requests.get(f"{BASE}/task/{task_id}")
    status = resp.json()
    if status["status"] in ("success", "failed"):
        break
    print(f"Status: {status['status']}...")
    time.sleep(5)

if status["status"] == "success":
    print(status["result"])
else:
    print(f"Failed: {status['error']}")
```

---

## 8. Python API Usage Guide

### 8.1 Call Directly from Python

```python
# Basic search
from mars.crews.search_crew import run_search
result = run_search("graph neural network for recommendation")
print(result)

# Deep analysis
from mars.crews.analysis_crew import run_analysis
result = run_analysis("Paper1 | Paper2", max_papers=10)
print(result)

# Relationship analysis
from mars.crews.connection_crew import run_connection
result = run_connection("Paper1 | Paper2", topic="federated learning")
print(result)

# Full research
from mars.crews.full_research_crew import run_full_research
result = run_full_research("federated learning privacy protection techniques")
print(result)
```

### 8.2 Custom Crew

```python
from mars.agents.researcher import create_researcher_agent
from mars.agents.searcher import create_searcher_agent
from mars.tasks.task_definitions import (
    create_domain_analysis_task,
    create_paper_search_task,
)
from crewai import Crew, Process

# Create agents
researcher = create_researcher_agent()
searcher = create_searcher_agent()

# Create tasks
topic = "federated learning"
task1 = create_domain_analysis_task(researcher, topic)
task2 = create_paper_search_task(searcher, topic, max_papers=30, context=[task1])

# Assemble the crew
crew = Crew(
    agents=[researcher, searcher],
    tasks=[task1, task2],
    process=Process.sequential,
    verbose=True,
)

# Execute
result = crew.kickoff()
print(result)
```

---

## 9. Workflow Details

### 9.1 Basic Search Workflow

```
Input: Research topic (e.g., "federated learning")
  │
  ├─ Step 1: Researcher Agent
  │   ├─ Expands keywords using keyword_expander
  │   ├─ Queries the CCF database for recommended conferences/journals
  │   └─ Output: domain_analysis.json (domain analysis report)
  │
  ├─ Step 2: Searcher Agent
  │   ├─ Searches DBLP using the expanded keywords
  │   ├─ Searches Semantic Scholar (filtered by citation count)
  │   ├─ Searches arXiv (latest preprints)
  │   ├─ Merges and deduplicates, sorts by relevance
  │   └─ Output: paper_search.json (up to 50 papers)
  │
  ├─ Step 3: Summarizer Agent (English review)
  │   ├─ Synthesizes domain analysis and search results
  │   └─ Output: review_en.md (English literature review ≥ 3,000 words)
  │
  └─ Step 4: Summarizer Agent (Chinese translation)
      ├─ Translates the English review into high-quality Chinese
      └─ Output: review_zh.md (Chinese literature review)
```

**Estimated time**: approximately 3–6 minutes (including bilingual review generation)

### 9.2 Deep Analysis Workflow

```
Input: List of paper information
  │
  ├─ Step 1: Analyzer Agent
  │   ├─ Retrieves detailed paper information and PDFs
  │   ├─ Extracts core contributions, research methods, experimental design
  │   └─ Output: deep analysis report
  │
  └─ Step 2: Evaluator Agent
      ├─ Evaluates originality (0–10)
      ├─ Evaluates technical depth (0–10)
      ├─ Evaluates experimental validity (0–10)
      ├─ Evaluates writing quality (0–10)
      └─ Output: quality assessment report
```

**Estimated time**: approximately 3–5 minutes

### 9.3 Relationship Analysis Workflow

```
Input: Paper information + research topic
  │
  ├─ Step 1: Connector Agent
  │   ├─ Builds a citation network
  │   ├─ Performs topic cluster analysis
  │   ├─ Identifies research trends
  │   └─ Output: relationship analysis report
  │
  └─ Step 2: Summarizer Agent
      ├─ Integrates all analysis results
      ├─ Writes a domain literature review
      └─ Output: literature review report (Markdown)
```

**Estimated time**: approximately 3–5 minutes

### 9.4 Full Research Workflow

```
Input: Research topic
  │
  ├─ Phase 1: Retrieval (2–3 minutes)
  │   ├─ Researcher → Domain analysis (domain_analysis.json)
  │   └─ Searcher → Paper retrieval (paper_search.json, 50 papers)
  │
  ├─ Phase 2: Analysis (5–10 minutes)
  │   ├─ Analyzer → Deep analysis (analysis_results.json, Top 20)
  │   ├─ Connector → Relationship analysis (connection_analysis.json, all 50 papers)
  │   └─ Evaluator → Quality assessment (quality_evaluation.json, Top 20)
  │
  └─ Phase 3: Synthesis (5–8 minutes)
      ├─ Summarizer → English review (review_en.md, ≥ 3,000 words)
      └─ Summarizer → Chinese translation (review_zh.md)
```

**Total estimated time**: approximately 12–21 minutes

---

## 10. Output Files Reference

Each run generates an independent subdirectory under `output/<workflow>_<timestamp>/` to prevent results from different runs from overwriting each other.

| File | Description | Generated by |
|------|-------------|--------------|
| `prompt.txt` | Original topic/question entered by the user | All workflows |
| `run.log` | Complete log for this run | All workflows |
| `domain_analysis.json` | Domain analysis results (research domain, keywords, recommended journals) | search, full |
| `paper_search.json` | Paper retrieval results list | search, full |
| `analysis_results.json` | Deep paper analysis results | analyze, full |
| `connection_analysis.json` | Citation network and topic clusters | connect, full |
| `quality_evaluation.json` | Paper quality assessment scores | analyze, full |
| `review_en.md` | English academic literature review (≥ 3,000 words) | search, connect, full |
| `review_zh.md` | Chinese literature review (high-quality translation of the English version) | search, connect, full |

---

## 11. Frequently Asked Questions & Troubleshooting

### Q1: PyMuPDF (pymupdf) Installation Fails

**Windows**:
```powershell
pip install pymupdf --no-build-isolation
```

**Ubuntu**:
```bash
sudo apt install -y libmupdf-dev
pip install pymupdf
```

If installation still fails, you can skip it — the system will automatically fall back to PyPDF2.

### Q2: "Unknown LLM provider" Error

Ensure that `DEFAULT_LLM_PROVIDER` in `.env` is set to a provider for which an API Key is configured:
```env
DEFAULT_LLM_PROVIDER=deepseek  # Change to the provider whose API Key you have
```

### Q3: API Call Timeout

Since v0.2.0, all workflow APIs are asynchronous — `POST` returns a `task_id` immediately
and the result is retrieved via `GET /task/{task_id}` polling. Long HTTP timeouts are no longer needed.

```bash
# Submit the job
curl -X POST http://localhost:8000/full-research \
  -H "Content-Type: application/json" \
  -d '{"topic": "..."}'
# → {"task_id": "abc123", "status": "pending"}

# Poll every 5–10 seconds until done
curl http://localhost:8000/task/abc123
# → status: "pending" → "running" → "success" / "failed"
```

### Q4: Paper Search Returns No Results

1. Check that your network connection is working
2. Try more general search terms
3. DBLP and arXiv do not require API Keys; verify they are accessible
4. Semantic Scholar has rate limits (100 requests per 5 minutes); configuring `SEMANTIC_SCHOLAR_API_KEY` increases the limit

### Q5: Garbled Chinese Characters in Terminal Output

**Windows PowerShell**:
```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001
```

**CMD**:
```cmd
chcp 65001
```

### Q6: Port Already in Use

```bash
# Find the process using the port
# Linux/Mac:
lsof -i :8000
# Windows:
netstat -ano | findstr :8000

# Use a different port
mars api --port 9000
```

### Q7: How to Use Only One LLM Provider?

Edit the `.env` file:
```env
DEEPSEEK_API_KEY=sk-your-key
DEFAULT_LLM_PROVIDER=deepseek
```

This makes all Agents use DeepSeek.

### Q8: How to View the Agent Execution Process?

All Agents have `verbose=True` enabled by default, so the execution process is output to the terminal in real time. Each run generates a `run.log` file in the `output/<workflow>_<timestamp>/` directory that saves the complete log — no manual redirection is needed.

To save output manually:

```bash
# Save output to a file
mars full "topic" 2>&1 | tee research.log
```

### Q9: mars Command Not Found

Ensure the virtual environment is activated and the project is installed in development mode:
```bash
source .venv/bin/activate  # Windows venv: .venv\Scripts\activate
pip install -e .
```

Conda users:
```powershell
conda activate mars
pip install -e .
```

Or use `python main.py` directly:
```bash
python main.py search "topic"
```

### Q10: SQLite Database Lock Error

This may occur with concurrent multi-process access. For production environments, PostgreSQL is recommended:
```env
DATABASE_URL=postgresql://user:password@localhost:5432/mars
```

### Q11: How to Quickly Verify the System is Ready?

Use the `mars check` command to check the configuration status of all API Keys:
```bash
mars check
```
The output shows the configuration status of each provider (✅ Configured / ❌ Not configured) and the list of available providers.

---

## 12. SurGE Benchmark Evaluation

MARS supports standardized evaluation on the **[SurGE](https://github.com/oneal2000/SurGE)** benchmark (SIGIR 2026):

```bash
# Generate survey using SurGE corpus
mars full --mode surge "federated learning"

# Generate + auto-evaluate all dimensions
mars full --mode surge --eval "federated learning"

# Compare against SurGE baselines
mars benchmark --action compare

# Single survey evaluation
mars evaluate --survey review_en.md "topic"
```

---

## 13. Appendix: API Key Acquisition Guide

### 12.1 Alibaba Cloud Qwen (DashScope)

1. Visit the [Alibaba Cloud Bailian Platform](https://bailian.console.aliyun.com/)
2. Log in with your Alibaba Cloud account
3. Navigate to the "API-KEY Management" page
4. Click "Create New API-KEY"
5. Copy the Key into `.env` as `DASHSCOPE_API_KEY`

**Free Quota**: New users typically receive a certain amount of free API calls

### 12.2 DeepSeek

1. Visit the [DeepSeek Open Platform](https://platform.deepseek.com/)
2. Register and log in
3. Navigate to the "API Keys" page
4. Create a new API Key
5. Copy it into `.env` as `DEEPSEEK_API_KEY`

**Free Quota**: A certain amount of free quota is available after registration

### 12.3 Moonshot AI Kimi

1. Visit the [Moonshot Open Platform](https://platform.moonshot.cn/)
2. Register and log in
3. Navigate to the "API Key Management" page
4. Create an API Key
5. Copy it into `.env` as `MOONSHOT_API_KEY`

**Free Quota**: New users receive a free trial quota

### 12.4 Zhipu AI GLM

1. Visit the [Zhipu AI Open Platform](https://open.bigmodel.cn/)
2. Register and log in
3. Navigate to the "API Key" page
4. Create an API Key (format: `xxx.yyy`)
5. Copy it into `.env` as `ZHIPU_API_KEY`

**Free Quota**: New users receive a free API call quota

### 12.5 Semantic Scholar (Optional)

1. Visit the [Semantic Scholar API](https://www.semanticscholar.org/product/api)
2. Apply for an API Key
3. Copy it into `.env` as `SEMANTIC_SCHOLAR_API_KEY`

**Purpose**: Increases the API rate limit (from 100 requests per 5 minutes to a higher limit); not required.

---

*Document last updated: 2026-03-30*
