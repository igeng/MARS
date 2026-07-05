# MARS 使用手册

**Multi-Agent Research System — 多智能体学术文献智能检索与分析系统**

> 版本：0.1.0 | Python ≥ 3.10

---

## 目录

1. [系统简介](#1-系统简介)
2. [系统要求](#2-系统要求)
3. [Windows 10 本地部署](#3-windows-10-本地部署)
   - [3.1–3.8 标准部署（venv）](#31-安装-python)
   - [3.9 使用 Conda + PyCharm 部署](#39-使用-conda--pycharm-部署替代方案)
   - [3.10 使用 Git Bash 部署（推荐终端方案）](#310-使用-git-bash-部署推荐终端方案)
4. [Ubuntu 云端部署](#4-ubuntu-云端部署)
5. [配置说明](#5-配置说明)
6. [命令行（CLI）使用指南](#6-命令行cli使用指南)
7. [Web API 使用指南](#7-web-api-使用指南)
8. [Python API 使用指南](#8-python-api-使用指南)
9. [工作流详解](#9-工作流详解)
10. [输出文件说明](#10-输出文件说明)
11. [常见问题与故障排除](#11-常见问题与故障排除)
12. [附录：API Key 获取指南](#12-附录api-key-获取指南)

---

## 1. 系统简介

### 1.1 MARS 是什么？

MARS 是一个基于多智能体架构的学术文献智能检索与分析系统。它使用 6 个专门的 AI 智能体协同工作，帮助研究者自动完成：

- 🔍 **论文检索**：从 DBLP、Semantic Scholar、arXiv 三大学术数据库同时搜索
- 📊 **深度分析**：提取论文的核心贡献、研究方法、实验设计
- 🔗 **关联分析**：构建引用网络，发现研究趋势和热点
- 📝 **综述生成**：自动生成 ≥ 3000 字英文综述并翻译成中文（双语输出）
- ⭐ **质量评估**：多维度评估论文学术质量

### 1.2 启动方式

MARS 提供三种使用方式：

| 方式 | 说明 | 适合场景 |
|------|------|----------|
| **命令行（CLI）** | 在终端中使用 `mars` 命令 | 快速查询、脚本自动化 |
| **Web API** | 启动 HTTP 服务器，通过浏览器或 HTTP 客户端访问 | 团队共享、远程调用 |
| **Python API** | 在 Python 代码中直接调用 | 二次开发、集成到其他系统 |

> **重要提示**：MARS 目前没有独立的网页前端界面。Web API 模式下，可以通过 FastAPI 自带的 **Swagger UI 交互式文档**（`http://localhost:8000/docs`）在浏览器中进行操作。

---

## 2. 系统要求

### 2.1 硬件要求

| 配置项 | 最低要求 | 推荐配置 |
|--------|----------|----------|
| CPU | 2 核 | 4 核+ |
| 内存 | 4 GB | 8 GB+ |
| 磁盘 | 2 GB 可用空间 | 10 GB+ |
| 网络 | 可访问互联网 | 稳定网络连接 |

> 注：MARS 调用云端 LLM API，不需要本地 GPU。

### 2.2 软件要求

| 软件 | 版本 | 说明 |
|------|------|------|
| Python | ≥ 3.10 | 必需 |
| pip | 最新版 | Python 包管理器 |
| Git | 任意版本 | 下载代码（可选） |

### 2.3 API Key 要求

至少需要配置 **一个** LLM 供应商的 API Key：

| 供应商 | API Key 环境变量 | 获取方式 |
|--------|-----------------|----------|
| 阿里云 Qwen | `DASHSCOPE_API_KEY` | [阿里云百炼平台](https://bailian.console.aliyun.com/) |
| DeepSeek | `DEEPSEEK_API_KEY` | [DeepSeek 开放平台](https://platform.deepseek.com/) |
| 月之暗面 Kimi | `MOONSHOT_API_KEY` | [Moonshot 开放平台](https://platform.moonshot.cn/) |
| 智谱 AI GLM | `ZHIPU_API_KEY` | [智谱 AI 开放平台](https://open.bigmodel.cn/) |

**推荐**：配置全部 4 个供应商的 Key，以获得最佳体验（系统会根据任务自动选择最合适的 LLM）。

---

## 3. Windows 10 本地部署

### 3.1 安装 Python

1. 访问 [Python 官网下载页](https://www.python.org/downloads/)
2. 下载 Python 3.10 或更高版本的 Windows 安装包
3. 运行安装程序，**务必勾选 "Add Python to PATH"**
4. 打开命令提示符（CMD）或 PowerShell，验证安装：

```powershell
python --version
# 应输出 Python 3.10.x 或更高
```

### 3.2 下载 MARS

**方式一：Git 克隆（推荐）**

```powershell
git clone https://github.com/igeng/MARS.git
cd MARS
```

**方式二：直接下载**

1. 访问 https://github.com/igeng/MARS
2. 点击 "Code" → "Download ZIP"
3. 解压到任意目录
4. 在命令提示符中进入该目录：

```powershell
cd C:\path\to\MARS
```

### 3.3 创建虚拟环境

```powershell
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
.venv\Scripts\activate

# 确认已激活（提示符前出现 (.venv)）
(.venv) C:\path\to\MARS>
```

### 3.4 安装依赖

```powershell
# 升级 pip
python -m pip install --upgrade pip

# 安装项目依赖
pip install -r requirements.txt

# 或者以开发模式安装（推荐）
pip install -e .
```

> **注意**：PyMuPDF (`pymupdf`) 在 Windows 上可能需要额外安装 Visual C++ Build Tools。如果安装失败，可以尝试：
>
> ```powershell
> pip install pymupdf --no-build-isolation
> ```
>
> 或者跳过它（系统会自动回退到 PyPDF2）。

### 3.5 配置 API Key

```powershell
# 复制环境变量模板
copy .env.example .env

# 用记事本编辑 .env 文件
notepad .env
```

在 `.env` 文件中填入你的 API Key：

```env
# 必需：至少填写一个 LLM 供应商的 API Key
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
MOONSHOT_API_KEY=sk-xxxxxxxxxxxxxxxx
ZHIPU_API_KEY=xxxxxxxx.xxxxxxxxxxxxxxxx

# 可选：提高 Semantic Scholar 速率限制
SEMANTIC_SCHOLAR_API_KEY=xxxxxxxxxxxxxxxx

# 默认 LLM 供应商（如果只配置了一个 Key，改为对应的供应商）
DEFAULT_LLM_PROVIDER=qwen
```

### 3.6 验证安装

```powershell
# 运行测试确认安装成功
python -m pytest tests/ -v

# 测试 CLI 是否可用
mars --help
```

### 3.7 开始使用

```powershell
# 检查配置状态（推荐首次使用时运行）
mars check

# 基础检索
mars search "federated learning with differential privacy"

# 启动 API 服务器
mars api --port 8000
# 然后在浏览器中打开 http://localhost:8000/docs
```

### 3.8 Windows 注意事项

1. **编码问题**：如果终端输出乱码，在 PowerShell 中执行：
   ```powershell
   [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
   chcp 65001
   ```

2. **路径分隔符**：Windows 使用 `\` 作为路径分隔符，但 Python 中建议使用 `/` 或 `Path` 对象。

3. **防火墙**：如果启动 API 服务器后无法从其他设备访问，检查 Windows 防火墙是否允许端口 8000。

4. **长路径支持**：如果路径过长导致错误，启用 Windows 长路径支持：
   - 运行 `regedit`
   - 导航到 `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
   - 设置 `LongPathsEnabled` 为 `1`

### 3.9 使用 Conda + PyCharm 部署（替代方案）

如果你的 Windows 10 电脑已经安装了 **Anaconda/Miniconda** 和 **PyCharm**，可以使用以下方式部署 MARS，无需单独安装 Python 或使用 `venv`。

#### 3.9.1 前置条件

| 软件 | 说明 |
|------|------|
| Anaconda 或 Miniconda | [Anaconda 下载](https://www.anaconda.com/download) / [Miniconda 下载](https://docs.conda.io/en/latest/miniconda.html) |
| PyCharm | Community 版即可，[PyCharm 下载](https://www.jetbrains.com/pycharm/download/) |
| Git（可选） | 用于克隆仓库，也可以直接下载 ZIP |

#### 3.9.2 使用 Conda 创建环境

打开 **Anaconda Prompt**（在开始菜单中搜索 "Anaconda Prompt"）：

```powershell
# 1. 创建名为 mars 的 Conda 环境，指定 Python 3.12
conda create -n mars python=3.12 -y

# 2. 激活环境
conda activate mars

# 3. 验证 Python 版本
python --version
# 应输出 Python 3.12.x
```

> **为什么选择 Python 3.12？** MARS 需要 Python ≥ 3.10。Python 3.12 稳定且兼容性好，推荐使用。你也可以选择 3.10 或 3.11。

#### 3.9.3 下载项目代码

```powershell
# 方式一：Git 克隆（推荐）
git clone https://github.com/igeng/MARS.git
cd MARS

# 方式二：如果没有 Git，从 GitHub 下载 ZIP 并解压后进入目录
cd C:\path\to\MARS
```

#### 3.9.4 安装项目依赖

在已激活 `mars` 环境的 Anaconda Prompt 中：

```powershell
# 升级 pip
python -m pip install --upgrade pip

# 安装全部依赖
pip install -r requirements.txt

# 以开发模式安装项目（推荐，这样 mars 命令可用）
pip install -e .
```

> **Conda 用户提示**：MARS 的依赖全部通过 pip 安装即可，不需要用 `conda install`。CrewAI 等核心包在 conda-forge 上可能版本较旧，建议统一使用 pip。

如果 `pymupdf` 安装失败，可以跳过——系统会自动回退到 PyPDF2：
```powershell
pip install pymupdf --no-build-isolation
```

#### 3.9.5 配置 API Key

```powershell
# 复制环境变量模板
copy .env.example .env

# 用记事本编辑（或在 PyCharm 中直接编辑）
notepad .env
```

在 `.env` 中填入你的 API Key（至少需要一个 LLM 供应商）：

```env
# 至少填写一个
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
MOONSHOT_API_KEY=sk-xxxxxxxxxxxxxxxx
ZHIPU_API_KEY=xxxxxxxx.xxxxxxxxxxxxxxxx

# 改为你已配置 Key 的供应商
DEFAULT_LLM_PROVIDER=qwen
```

#### 3.9.6 在 PyCharm 中配置项目

**步骤 1：打开项目**

1. 启动 PyCharm
2. 选择 **File → Open**
3. 选择 MARS 项目根目录（包含 `main.py` 的文件夹）
4. 点击 **OK** 打开

**步骤 2：配置 Conda 解释器**

1. 打开 **File → Settings**（或 `Ctrl+Alt+S`）
2. 导航到 **Project: MARS → Python Interpreter**
3. 点击右上角齿轮图标 ⚙️ → **Add Interpreter → Add Local Interpreter**
4. 在左侧选择 **Conda Environment**
5. 选择 **Use existing environment**
6. 在下拉列表中选择 `mars`（即刚才创建的 Conda 环境）
7. 点击 **OK** 确认

> 如果下拉列表中没有看到 `mars` 环境，点击 **...** 按钮手动定位 Conda 可执行文件路径，通常在：
> - Anaconda: `C:\Users\<你的用户名>\anaconda3\Scripts\conda.exe`
> - Miniconda: `C:\Users\<你的用户名>\miniconda3\Scripts\conda.exe`

**步骤 3：配置环境变量（可选）**

PyCharm 中运行代码时，`.env` 文件会被 `pydantic-settings` 自动加载。如果需要额外配置：

1. 打开 **Run → Edit Configurations**
2. 选择或新建一个 Python 配置
3. 在 **Environment variables** 中可以添加变量，或者在 **EnvFile** 插件中指定 `.env` 文件路径

> 一般情况下无需手动配置——MARS 的 `settings.py` 会自动从项目根目录加载 `.env` 文件。

**步骤 4：配置 Terminal（推荐）**

让 PyCharm 内置终端自动使用 Conda 环境：

1. 打开 **File → Settings → Tools → Terminal**
2. 将 **Shell path** 设置为：
   ```
   cmd.exe /K "C:\Users\<你的用户名>\anaconda3\Scripts\activate.bat mars"
   ```
   或者对于 Miniconda：
   ```
   cmd.exe /K "C:\Users\<你的用户名>\miniconda3\Scripts\activate.bat mars"
   ```
3. 点击 **OK**，重新打开 Terminal 面板即可看到 `(mars)` 前缀

#### 3.9.7 在 PyCharm 中运行 MARS

**方式一：通过 Terminal 面板运行 CLI 命令**

点击 PyCharm 底部的 **Terminal** 标签，在终端中直接输入：

```powershell
# 检查配置状态
mars check

# 基础检索
mars search "federated learning with differential privacy"

# 完整研究流程
mars full "联邦学习隐私保护技术"

# 启动 API 服务器
mars api --port 8000
```

**方式二：右键运行 main.py**

1. 在项目文件树中找到 `main.py`
2. 右键点击 → **Run 'main'**
3. 首次运行时 PyCharm 会自动创建运行配置

如需传入命令行参数：

1. 打开 **Run → Edit Configurations**
2. 选择 `main` 配置
3. 在 **Parameters** 中填入参数，例如：
   ```
   search "graph neural network for recommendation"
   ```
4. 点击 **OK**，然后点击运行按钮 ▶️

**方式三：配置快捷运行配置**

为常用命令创建专门的运行配置：

1. **Run → Edit Configurations → + → Python**
2. 配置示例：

| 配置项 | 值 |
|--------|-----|
| Name | MARS - Full Research |
| Script path | `$ProjectFileDir$/main.py` |
| Parameters | `full "你的研究主题"` |
| Working directory | `$ProjectFileDir$` |
| Python interpreter | 选择 `mars` Conda 环境 |

3. 点击 **OK**，之后可以从工具栏的下拉菜单快速选择并运行

#### 3.9.8 在 PyCharm 中运行测试

1. 在项目文件树中右键点击 `tests/` 文件夹
2. 选择 **Run 'pytest in tests'**
3. 或者在 Terminal 中运行：

```powershell
python -m pytest tests/ -v
```

> 所有测试无需 API Key 即可通过（工具调用已 mock）。

#### 3.9.9 在 PyCharm 中调试

1. 在任意 `.py` 文件中点击行号左侧设置断点
2. 右键点击 `main.py` → **Debug 'main'**（或按 `Shift+F9`）
3. 程序会在断点处暂停，你可以查看变量值、调用栈等

**调试 API 服务器**：

1. 创建运行配置，Parameters 填写 `api --port 8000`
2. 使用 Debug 模式运行
3. 在浏览器中访问 `http://localhost:8000/docs` 触发请求
4. 断点会在对应的路由处理函数中命中

#### 3.9.10 常见问题

**Q: Conda 环境中 `mars` 命令找不到？**

确保已在 Conda 环境中以开发模式安装：
```powershell
conda activate mars
pip install -e .
```

或直接使用 `python main.py` 代替 `mars` 命令。

**Q: PyCharm 无法识别项目中的模块导入？**

1. 确认已正确选择 Conda `mars` 环境作为 Python Interpreter
2. 右键点击项目根目录 → **Mark Directory as → Sources Root**
3. 或者执行 `pip install -e .`，让 PyCharm 能正确解析 `mars` 包

**Q: PyCharm Terminal 没有自动激活 Conda 环境？**

- 检查 Settings → Tools → Terminal 中的 Shell path 配置是否正确
- 或者每次在 Terminal 中手动执行 `conda activate mars`

**Q: 安装依赖时出现 "solving environment" 很慢？**

MARS 的依赖通过 pip 安装，不需要 `conda install`。直接使用：
```powershell
conda activate mars
pip install -r requirements.txt
```

---

### 3.10 使用 Git Bash 部署（推荐终端方案）

Git Bash 是 **Git for Windows** 附带的 Unix 风格终端模拟器，它在 Windows 上提供了与 Linux/macOS 几乎一致的 Bash 体验（`ls`、`cd`、`source`、`export` 等命令全部可用），非常适合习惯 Linux 命令行或需要跨平台统一工作流的开发者使用 MARS。

> **适用场景**：你已安装 Git for Windows，希望用接近 Linux 的方式在 Win10 上运行 MARS CLI 命令，不想使用 Anaconda 或 PyCharm。

#### 3.10.1 前置条件汇总

在开始之前，请确认以下软件已安装：

| 软件 | 版本要求 | 说明 |
|------|----------|------|
| Windows 10 | 1903+ | 任何 Win10 版本均可 |
| Git for Windows | ≥ 2.30 | 包含 Git Bash，是本节核心工具 |
| Python | ≥ 3.10（推荐 3.10、3.11 或 3.12） | 推荐从 Python 官网安装，**不要** 只从 Microsoft Store 安装 |
| pip | 随 Python 自带 | 确保能在 Git Bash 中调用 `python -m pip` |

---

#### 3.10.2 安装 Git for Windows（获取 Git Bash）

1. 打开浏览器，访问 **[https://git-scm.com/download/win](https://git-scm.com/download/win)**
2. 点击 **"Click here to download"** 下载最新 64 位安装包（`Git-x.xx.x-64-bit.exe`）
3. 运行安装程序，在各步骤中推荐选项如下：

   | 安装步骤 | 推荐选项 |
   |----------|----------|
   | Select Components | 保持默认（包含 Git Bash Here） |
   | Choosing the default editor | 选 Vim 或 Notepad++（随意） |
   | **Adjusting your PATH environment** | 选 **"Git from the command line and also from 3rd-party software"**（推荐） |
   | Choosing HTTPS transport backend | 保持默认（OpenSSL） |
   | **Configuring the line ending conversions** | 选 **"Checkout as-is, commit Unix-style line endings"**（避免 CRLF 问题） |
   | Configuring the terminal emulator | 选 **"Use MinTTY"**（默认，界面更好看） |
   | 其余步骤 | 保持默认，点击 Next → Install |

4. 安装完成后，在桌面或任意文件夹内右键，应能看到 **"Git Bash Here"** 菜单项。

---

#### 3.10.3 安装 Python（确保 Git Bash 可调用）

> 如果已安装 Python 3.10+，可跳到验证步骤。

1. 访问 **[https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)**
2. 下载 **Python 3.10、3.11 或 3.12 的 Windows installer (64-bit)**（MARS 要求 Python ≥ 3.10，三个版本均可使用）
3. 运行安装程序时，**必须勾选以下选项**：
   - ✅ **"Add Python 3.x to PATH"**（页面底部，非常重要！）
   - ✅ **"Install pip"**（默认已勾选）
4. 点击 **"Install Now"**

**在 Git Bash 中验证 Python 安装**：

打开 Git Bash（在开始菜单搜索 "Git Bash" 或右键文件夹选 "Git Bash Here"），输入：

```bash
python --version
# 期望输出：Python 3.12.x（或 3.10.x / 3.11.x）

python -m pip --version
# 期望输出类似：pip 23.x from C:\Users\<用户名>\AppData\...\pip (python 3.12)
```

> **常见问题**：如果提示 `python: command not found`，说明 Python 未添加到 PATH。
> 解决方法：在系统环境变量中手动添加 Python 安装路径（通常为 `C:\Users\<用户名>\AppData\Local\Programs\Python\Python312\` 和 `C:\Users\<用户名>\AppData\Local\Programs\Python\Python312\Scripts\`），然后**关闭并重新打开 Git Bash**。

---

#### 3.10.4 克隆 MARS 仓库

在 Git Bash 中执行：

```bash
# 进入你想存放项目的目录（示例：D 盘的 Projects 文件夹）
cd /d/Projects
# 如果该目录不存在，先创建：
mkdir -p /d/Projects && cd /d/Projects

# 克隆仓库
git clone https://github.com/igeng/MARS.git

# 进入项目目录
cd MARS

# 查看项目结构（确认克隆成功）
ls -la
# 应能看到 main.py、requirements.txt、mars/、docs/ 等文件和目录
```

> **路径说明**：Git Bash 中 Windows 路径的写法：
> - `C:\Users\用户名` → `/c/Users/用户名`
> - `D:\Projects` → `/d/Projects`

---

#### 3.10.5 创建 Python 虚拟环境

在 Git Bash 中，于 MARS 项目根目录执行：

```bash
# 确认当前在 MARS 目录中
pwd
# 期望输出类似：/d/Projects/MARS

# 创建虚拟环境（名为 .venv，位于项目根目录下）
python -m venv .venv

# 激活虚拟环境（Git Bash 语法，不同于 PowerShell）
source .venv/Scripts/activate

# 确认激活成功：命令提示符前应出现 (.venv)
# (.venv) username@DESKTOP:/d/Projects/MARS $
```

> **注意**：Git Bash 中激活虚拟环境使用 `source .venv/Scripts/activate`，与 Linux/macOS 下的 `source .venv/bin/activate` 路径略有不同（Windows 虚拟环境的脚本在 `Scripts/` 而非 `bin/` 下）。

**验证虚拟环境已激活**：

```bash
# 查看当前使用的 Python 解释器路径（应指向 .venv 内部）
which python
# 期望输出类似：/d/Projects/MARS/.venv/Scripts/python

python --version
# 期望输出：Python 3.12.x
```

---

#### 3.10.6 升级 pip 并安装项目依赖

激活虚拟环境后，执行：

```bash
# 第一步：升级 pip 到最新版（避免旧版 pip 安装某些包时报错）
python -m pip install --upgrade pip

# 第二步：安装所有依赖包
pip install -r requirements.txt
# 这将安装 crewai、litellm、fastapi、pydantic-settings、pymupdf 等全部依赖
# 预计耗时 3-10 分钟，取决于网络速度

# 第三步（推荐）：以开发模式安装 MARS 本身
# 这一步会注册 mars 命令行工具（即 mars search、mars full 等命令）
pip install -e .
```

**验证依赖安装是否成功**：

```bash
# 检查 mars 命令是否可用
mars --help
# 期望输出：显示 MARS 命令帮助信息，包含 search、analyze、connect、full、api、check 等子命令

# 验证核心包已安装
python -c "import crewai; print('crewai OK')"
python -c "import litellm; print('litellm OK')"
python -c "import fastapi; print('fastapi OK')"
```

> **若 `pymupdf` 安装失败**（Windows 上偶有此情况）：
>
> ```bash
> # 方法一：禁用隔离编译
> pip install pymupdf --no-build-isolation
>
> # 方法二：跳过安装（系统会自动使用 PyPDF2 作为备选）
> # 从 requirements.txt 临时移除 pymupdf 行后再安装其余依赖
> grep -v "pymupdf" requirements.txt > /tmp/req_no_mupdf.txt
> pip install -r /tmp/req_no_mupdf.txt
> ```

---

#### 3.10.7 配置 API Key

```bash
# 复制环境变量模板文件
cp .env.example .env

# 用 Windows 记事本打开编辑（Git Bash 中可以这样调用 Windows 应用）
notepad .env

# 或者用 nano（如果安装了）
# nano .env

# 或者直接用 VS Code 打开（如果已安装）
# code .env
```

在打开的 `.env` 文件中，填入你的 API Key。以下是完整的配置项说明：

```env
# ============================================================
# LLM 供应商 API Key（至少填写一个，推荐全部填写）
# ============================================================

# 阿里云 Qwen（通义千问）
# 获取地址：https://bailian.console.aliyun.com/ → API-KEY 管理
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# DeepSeek
# 获取地址：https://platform.deepseek.com/ → API Keys
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 月之暗面 Kimi（Moonshot AI）
# 获取地址：https://platform.moonshot.cn/ → API Key 管理
MOONSHOT_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 智谱 AI（GLM）
# 获取地址：https://open.bigmodel.cn/ → API Key（格式为 xxx.yyy）
ZHIPU_API_KEY=xxxxxxxxxxxxxxxx.xxxxxxxxxxxxxxxx

# ============================================================
# 可选：学术搜索 API
# ============================================================

# Semantic Scholar（不配置也能使用，但有速率限制）
# 获取地址：https://www.semanticscholar.org/product/api
SEMANTIC_SCHOLAR_API_KEY=

# ============================================================
# 应用配置
# ============================================================

# 默认 LLM 供应商（改为你已配置 API Key 的供应商）
# 可选值：qwen | deepseek | kimi | glm
DEFAULT_LLM_PROVIDER=qwen

# 检索和分析的最大论文数量
MAX_PAPERS_PER_SEARCH=50
MAX_PAPERS_FOR_ANALYSIS=20

# 是否启用 CrewAI 记忆功能（需要 OpenAI Embedding API，默认关闭）
ENABLE_MEMORY=false
```

保存并关闭文件。

---

#### 3.10.8 验证整体安装

运行 MARS 内置的配置检查命令：

```bash
mars check
```

正常情况下，输出类似：

```
╭─────────────────────────────────────────────────────╮
│ 🚀 MARS - Multi-Agent Research System               │
│    多智能体学术文献智能检索与分析系统                │
╰─────────────────────────────────────────────────────╯

🔎 检查系统配置...

          LLM 供应商状态
┌──────────┬──────────────────────┬──────────────────────────────┐
│ 供应商   │ 环境变量             │ 状态                         │
├──────────┼──────────────────────┼──────────────────────────────┤
│ qwen     │ DASHSCOPE_API_KEY    │ ✅ 已配置 (sk-x****xxxx)     │
│ deepseek │ DEEPSEEK_API_KEY     │ ✅ 已配置 (sk-x****xxxx)     │
│ kimi     │ MOONSHOT_API_KEY     │ ✅ 已配置 (sk-x****xxxx)     │
│ glm      │ ZHIPU_API_KEY        │ ✅ 已配置 (xxxx****xxxx)     │
└──────────┴──────────────────────┴──────────────────────────────┘

默认供应商: qwen
可用供应商: qwen, deepseek, kimi, glm

Semantic Scholar API Key: ⚠ 未配置（可使用，但速率受限）

输出目录: D:\Projects\MARS\output
数据库:   sqlite:///./mars.db

✅ 系统就绪 — 可使用 4 个 LLM 供应商
```

如果看到 `✅ 系统就绪`，即可开始使用 MARS。

---

#### 3.10.9 在 Git Bash 中运行 CLI 命令

**启动前确认虚拟环境已激活**（提示符前有 `(.venv)` 字样）：

```bash
# 如果关闭了 Git Bash 或打开了新窗口，需要重新激活虚拟环境
cd /d/Projects/MARS
source .venv/Scripts/activate
```

**基础检索（search）**

```bash
# 检索某一研究主题，自动生成论文列表 + 双语综述
mars search "federated learning with differential privacy"

# 中文主题同样支持
mars search "联邦学习隐私保护"

# 指定最大检索数量（默认 50）
mars search "graph neural network" --max-results 30
```

输出文件保存在 `output/search_<时间戳>/` 目录下，包含：
- `prompt.txt` — 原始输入
- `run.log` — 完整运行日志
- `paper_search.json` — 论文列表（含标题、作者、年份、引用数等）
- `review_en.md` — 英文学术综述（≥ 3000 字）
- `review_zh.md` — 中文综述（英文版的高质量翻译）

**深度分析（analyze）**

```bash
# 输入论文标题列表（用 | 分隔），进行深度解析和质量评估
mars analyze "Federated Learning: Challenges, Methods, and Future Directions | Communication-Efficient Learning of Deep Networks from Decentralized Data"

# 指定最大分析论文数（默认 20）
mars analyze "paper1 | paper2 | paper3" --max-papers 10
```

**关联分析（connect）**

```bash
# 分析多篇论文之间的引用关系和主题关联
mars connect "Attention Is All You Need | BERT: Pre-training of Deep Bidirectional Transformers | GPT-3: Language Models are Few-Shot Learners" --topic "transformer models in NLP"
```

**完整研究流程（full）**

```bash
# 一键执行：领域分析 → 检索 → 深度解析 → 关联分析 → 质量评估 → 双语综述
# 这是最完整的研究工作流，预计耗时 12-21 分钟
mars full "知识图谱嵌入方法综述"

# 英文主题
mars full "survey of large language model alignment techniques"
```

**启动 Web API 服务器**

```bash
# 在本地 8000 端口启动 FastAPI 服务器
mars api --port 8000

# 启动后在浏览器中打开 Swagger UI：
# http://localhost:8000/docs
```

**在后台运行（nohup）**

```bash
# 将完整研究任务放后台运行，日志写入 bg.log
nohup mars full "topic here" > bg.log 2>&1 &
echo "Background PID: $!"

# 实时查看后台任务进度
tail -f bg.log
```

---

#### 3.10.10 Git Bash 专项注意事项

**① 中文输出乱码问题**

Git Bash（MinTTY）对 UTF-8 支持良好，通常不会出现乱码。但如果遇到问题，可以在 Git Bash 中设置：

```bash
# 设置终端编码为 UTF-8
export LANG=zh_CN.UTF-8
export PYTHONIOENCODING=utf-8

# 或者将这两行加入 ~/.bashrc，每次启动自动生效
echo 'export LANG=zh_CN.UTF-8' >> ~/.bashrc
echo 'export PYTHONIOENCODING=utf-8' >> ~/.bashrc
source ~/.bashrc
```

**② `mars` 命令找不到（`command not found`）**

原因：虚拟环境未激活，或未以开发模式安装项目。

```bash
# 解决方法：先激活虚拟环境，再安装
source .venv/Scripts/activate
pip install -e .

# 验证
which mars
# 应输出：/d/Projects/MARS/.venv/Scripts/mars
```

**③ Python 路径中有空格**

如果项目路径包含空格（如 `D:\My Projects\MARS`），在 Git Bash 中需要加引号：

```bash
cd "/d/My Projects/MARS"
source .venv/Scripts/activate
```

建议将项目放在无空格的路径下（如 `/d/Projects/MARS`），避免潜在的兼容性问题。

**④ 虚拟环境激活后 `pip` 仍指向全局 Python**

```bash
# 检查 pip 指向
which pip
# 应输出 .venv 路径，如：/d/Projects/MARS/.venv/Scripts/pip

# 如果指向错误，强制使用：
python -m pip install -r requirements.txt
python -m pip install -e .
```

**⑤ 每次打开 Git Bash 都需要重新激活虚拟环境**

可以在 `~/.bashrc` 中添加快捷函数：

```bash
# 在 Git Bash 中编辑 ~/.bashrc
echo 'alias mars-activate="cd /d/Projects/MARS && source .venv/Scripts/activate"' >> ~/.bashrc
source ~/.bashrc

# 以后每次打开 Git Bash 只需运行：
mars-activate
```

**⑥ `pip install` 速度慢**

在中国大陆地区，可以使用国内镜像源加速：

```bash
# 使用阿里云镜像一次性安装
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 或者永久配置镜像（写入 pip 配置文件）
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
```

**⑦ 长路径错误（`WinError 206`）**

如果出现文件路径过长的错误，需要启用 Windows 长路径支持：

以管理员身份运行 PowerShell（在开始菜单搜索 PowerShell，右键"以管理员身份运行"），执行：

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

修改后无需重启，再次打开 Git Bash 即可生效。

---

#### 3.10.11 完整部署流程快速参考

以下是从零开始到成功运行的最简命令序列（假设 Git 和 Python 已安装）：

```bash
# 1. 打开 Git Bash，进入工作目录
cd /d/Projects

# 2. 克隆项目
git clone https://github.com/igeng/MARS.git
cd MARS

# 3. 创建并激活虚拟环境
python -m venv .venv
source .venv/Scripts/activate

# 4. 安装依赖
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 5. 配置 API Key
cp .env.example .env
notepad .env          # 填入至少一个 LLM API Key，保存退出

# 6. 验证配置
mars check

# 7. 开始使用！
mars search "your research topic"
```

---

## 4. Ubuntu 云端部署

### 4.1 系统准备

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装 Python 和必要工具
sudo apt install -y python3.10 python3.10-venv python3-pip git curl

# 验证 Python 版本
python3 --version
# 应输出 Python 3.10.x 或更高
```

> 如果 Ubuntu 版本较低（如 20.04），Python 3.10 可能不在默认仓库中：
>
> ```bash
> sudo add-apt-repository ppa:deadsnakes/ppa
> sudo apt update
> sudo apt install -y python3.10 python3.10-venv python3.10-dev
> ```

### 4.2 下载 MARS

```bash
# 克隆仓库
git clone https://github.com/igeng/MARS.git
cd MARS
```

### 4.3 创建虚拟环境

```bash
# 创建虚拟环境
python3 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate

# 确认已激活
(.venv) user@server:~/MARS$
```

### 4.4 安装依赖

```bash
# 升级 pip
pip install --upgrade pip

# 安装系统级依赖（PyMuPDF 可能需要）
sudo apt install -y libmupdf-dev

# 安装项目依赖
pip install -r requirements.txt

# 或以开发模式安装
pip install -e .
```

### 4.5 配置 API Key

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件
nano .env
# 或者使用 vim: vim .env
```

填入 API Key（同 Windows 部分的说明）。

### 4.6 验证安装

```bash
# 运行测试
python -m pytest tests/ -v

# 测试 CLI
mars --help
```

### 4.7 使用 CLI 模式

```bash
# 基础检索
mars search "deep learning for medical image analysis"

# 完整研究（后台运行）
nohup mars full "联邦学习隐私保护技术" > research_output.log 2>&1 &
```

### 4.8 部署 API 服务器

#### 方式一：直接运行

```bash
# 前台运行
mars api --host 0.0.0.0 --port 8000

# 后台运行
nohup mars api --host 0.0.0.0 --port 8000 > api.log 2>&1 &
```

#### 方式二：使用 systemd 服务（推荐生产环境）

创建服务文件：

```bash
sudo nano /etc/systemd/system/mars-api.service
```

写入以下内容（根据实际路径修改）：

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

启用并启动服务：

```bash
# 重新加载 systemd
sudo systemctl daemon-reload

# 启动服务
sudo systemctl start mars-api

# 设置开机自启
sudo systemctl enable mars-api

# 查看状态
sudo systemctl status mars-api

# 查看日志
sudo journalctl -u mars-api -f
```

#### 方式三：使用 Nginx 反向代理（推荐生产环境）

安装 Nginx：

```bash
sudo apt install -y nginx
```

创建 Nginx 配置：

```bash
sudo nano /etc/nginx/sites-available/mars
```

写入：

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

        # 长时间运行的请求需要较长超时
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
```

启用配置：

```bash
# 创建符号链接
sudo ln -s /etc/nginx/sites-available/mars /etc/nginx/sites-enabled/

# 删除默认配置（可选）
sudo rm /etc/nginx/sites-enabled/default

# 测试配置
sudo nginx -t

# 重启 Nginx
sudo systemctl restart nginx
```

### 4.9 防火墙配置

```bash
# 允许 HTTP 端口
sudo ufw allow 80/tcp

# 如果直接访问 8000 端口
sudo ufw allow 8000/tcp

# 查看防火墙状态
sudo ufw status
```

### 4.10 安全建议

1. **不要暴露 .env 文件**：确保 `.env` 不在公开目录中
2. **使用 HTTPS**：生产环境建议配置 SSL 证书（Let's Encrypt）
3. **限制访问**：使用防火墙限制 API 访问来源
4. **定期更新**：保持系统和依赖包更新

---

## 5. 配置说明

### 5.1 环境变量完整列表

| 变量名 | 必需 | 默认值 | 说明 |
|--------|------|--------|------|
| `DASHSCOPE_API_KEY` | ⚡ | (空) | 阿里云 Qwen API Key |
| `DEEPSEEK_API_KEY` | ⚡ | (空) | DeepSeek API Key |
| `MOONSHOT_API_KEY` | ⚡ | (空) | Kimi API Key |
| `ZHIPU_API_KEY` | ⚡ | (空) | GLM API Key |
| `SEMANTIC_SCHOLAR_API_KEY` | 否 | (空) | Semantic Scholar API Key |
| `DEFAULT_LLM_PROVIDER` | 否 | qwen | 默认 LLM 供应商 |
| `QWEN_MODEL` | 否 | qwen3.5-flash | Qwen 模型名称 |
| `DEEPSEEK_MODEL` | 否 | deepseek-chat | DeepSeek 模型名称 |
| `KIMI_MODEL` | 否 | kimi-k2.5 | Kimi 模型名称 |
| `GLM_MODEL` | 否 | glm-4.7-flash | GLM 模型名称（免费模型，作为兜底） |
| `MAX_PAPERS_PER_SEARCH` | 否 | 50 | 每次检索最大论文数 |
| `MAX_PAPERS_FOR_ANALYSIS` | 否 | 20 | 深度分析最大论文数 |
| `ARXIV_SEARCH_TIMEOUT` | 否 | 30 | arXiv API 请求超时秒数 |
| `OUTPUT_DIR` | 否 | ./output | 输出文件根目录 |
| `ENABLE_MEMORY` | 否 | false | 是否启用 CrewAI 记忆（需要 OpenAI 兼容嵌入 Key） |
| `GLM_RATE_LIMIT_MAX_RETRIES` | 否 | 3 | GLM 限流重试次数 |
| `GLM_RATE_LIMIT_RETRY_DELAY` | 否 | 5.0 | GLM 限流退避基础延迟（秒） |
| `DATABASE_URL` | 否 | sqlite:///./mars.db | 数据库连接 URL |
| `LOG_LEVEL` | 否 | INFO | 日志级别 |
| `API_HOST` | 否 | 0.0.0.0 | API 服务器地址 |
| `API_PORT` | 否 | 8000 | API 服务器端口 |

> ⚡ 至少需要配置一个 LLM 供应商的 API Key。建议配置全部 4 个以获得最佳体验。

### 5.2 仅使用单个 LLM 供应商

如果只有一个 LLM 的 API Key，需要修改 `DEFAULT_LLM_PROVIDER` 为对应的供应商名称：

```env
# 例如只有 DeepSeek 的 Key
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
DEFAULT_LLM_PROVIDER=deepseek
```

> 注意：在此配置下，所有 6 个 Agent 都会使用同一个 LLM，可能影响效果。

---

## 6. 命令行（CLI）使用指南

### 6.1 查看帮助

```bash
mars --help
mars search --help
mars full --help
```

### 6.2 基础检索

搜索指定主题的学术论文：

```bash
# 英文主题
mars search "federated learning with differential privacy"

# 中文主题
mars search "图神经网络推荐系统"

# 限制返回数量
mars search "deep reinforcement learning" --max-results 20
mars search "deep reinforcement learning" -n 20
```

**输出**：论文列表，包含标题、作者、发表期刊/会议、年份、引用次数、URL。每次运行在 `output/search_<时间戳>/` 目录下生成6份文件：`prompt.txt`、`run.log`、`domain_analysis.json`、`paper_search.json`、`review_en.md`（英文综述 ≥ 3000 字）、`review_zh.md`（中文翻译）。

### 6.3 深度分析

对指定论文进行深度内容分析：

```bash
# 用 | 分隔多篇论文
mars analyze "Federated Learning: Challenges, Methods, and Future Directions | Communication-Efficient Learning of Deep Networks"

# 限制分析论文数量
mars analyze "paper1 | paper2 | paper3" --max-papers 5
```

**输出**：每篇论文的核心贡献、研究方法、实验设计、结论等。

### 6.4 关联分析

分析多篇论文之间的引用关系和主题关联：

```bash
mars connect "paper1 | paper2 | paper3" --topic "federated learning"
mars connect "paper1 | paper2" -t "联邦学习"
```

**输出**：引用网络摘要、主题聚类、研究趋势、核心论文 + 文献综述。

### 6.5 完整研究（推荐）

执行完整的端到端研究流程：

```bash
# 英文
mars full "knowledge graph embedding methods"

# 中文
mars full "联邦学习隐私保护技术"
```

**执行流程**：
1. 领域分析 → 推荐 CCF 期刊/会议（domain_analysis.json）
2. 多数据库检索 → 获取 50 篇相关论文（paper_search.json）
3. 深度分析 → 提取 Top 20 论文的核心内容（analysis_results.json）
4. 关联分析 → 构建引用网络，识别趋势（connection_analysis.json）
5. 质量评估 → 多维度评分（quality_evaluation.json）
6. 综述生成 → 生成英文综述 ≥ 3000 字（review_en.md）→ 翻译为中文综述（review_zh.md）

**输出**：完整的研究报告，保存到 `output/full_<时间戳>/` 目录下的7份文件。

### 6.6 启动 API 服务器

```bash
# 默认启动（0.0.0.0:8000）
mars api

# 自定义地址和端口
mars api --host 127.0.0.1 --port 3000
mars api -h 0.0.0.0 -p 9000
```

启动后在浏览器中打开 `http://localhost:8000/docs` 可以看到交互式 API 文档。

### 6.7 初始化数据库

```bash
mars init-db
```

创建 SQLite 数据库和表结构（首次使用前执行）。

### 6.8 检查系统配置

```bash
mars check
```

检查所有 LLM 供应商 API Key 的配置状态，显示可用供应商、输出目录、数据库路径等信息，方便排查配置问题。

---

## 7. Web API 使用指南

### 7.1 启动服务器

```bash
mars api --port 8000
```

### 7.2 Swagger UI（浏览器操作）

启动后在浏览器中打开：**http://localhost:8000/docs**

这是 FastAPI 自带的交互式文档界面，可以直接在浏览器中：
1. 查看所有 API 端点
2. 填写参数并发送请求
3. 查看响应结果

![Swagger UI 使用说明]
1. 点击要使用的端点（如 POST /search）
2. 点击 "Try it out"
3. 在请求体中填写参数
4. 点击 "Execute"
5. 查看下方的响应结果

### 7.3 使用 curl 调用

> **API 异步模式（v0.2.0）**：所有工作流端点采用"提交-轮询"模式。
> `POST` 请求立即返回 `task_id`，随后通过 `GET /task/{task_id}` 查询状态和结果。
> 不再需要为长时间运行的工作流设置 HTTP 超时。

#### 健康检查

```bash
curl http://localhost:8000/health
# 返回: {"status": "ok"}
```

#### 基础检索

```bash
# 1. 提交任务 → 立即返回 task_id
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"topic": "federated learning privacy", "max_results": 20}'
# 返回: {"task_id":"a1b2c3d4e5f6","status":"pending"}  (HTTP 202)

# 2. 轮询任务状态
curl http://localhost:8000/task/a1b2c3d4e5f6
# 返回: {"task_id":"a1b2c3d4e5f6","status":"running","result":"","error":""}
# 完成后: {"task_id":"a1b2c3d4e5f6","status":"success","result":"...","error":""}
```

#### 深度分析

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"papers_info": "Paper Title 1 | Paper Title 2", "max_papers": 10}'
```

#### 关联分析

```bash
curl -X POST http://localhost:8000/connect \
  -H "Content-Type: application/json" \
  -d '{"papers_info": "Paper1 | Paper2 | Paper3", "topic": "federated learning"}'
```

#### 完整研究

```bash
curl -X POST http://localhost:8000/full-research \
  -H "Content-Type: application/json" \
  -d '{"topic": "knowledge graph embedding methods"}'
```

> **提示**：所有工作流端点统一返回 HTTP 202 (Accepted) + `task_id`。
> 使用 `GET /task/{task_id}` 轮询结果，`status` 字段取值：`pending` → `running` → `success` / `failed`。

### 7.4 使用 Python requests 调用

```python
import requests
import time

BASE = "http://localhost:8000"

# 1. 提交任务
resp = requests.post(
    f"{BASE}/search",
    json={"topic": "deep learning for NLP", "max_results": 20}
)
task = resp.json()
task_id = task["task_id"]
print(f"任务已提交: {task_id}")

# 2. 轮询直到完成
while True:
    resp = requests.get(f"{BASE}/task/{task_id}")
    status = resp.json()
    if status["status"] in ("success", "failed"):
        break
    print(f"状态: {status['status']}...")
    time.sleep(5)

if status["status"] == "success":
    print(status["result"])
else:
    print(f"失败: {status['error']}")
```

---

## 8. Python API 使用指南

### 8.1 直接在 Python 中调用

```python
# 基础检索
from mars.crews.search_crew import run_search
result = run_search("graph neural network for recommendation")
print(result)

# 深度分析
from mars.crews.analysis_crew import run_analysis
result = run_analysis("Paper1 | Paper2", max_papers=10)
print(result)

# 关联分析
from mars.crews.connection_crew import run_connection
result = run_connection("Paper1 | Paper2", topic="federated learning")
print(result)

# 完整研究
from mars.crews.full_research_crew import run_full_research
result = run_full_research("联邦学习隐私保护技术")
print(result)
```

### 8.2 自定义 Crew

```python
from mars.agents.researcher import create_researcher_agent
from mars.agents.searcher import create_searcher_agent
from mars.tasks.task_definitions import (
    create_domain_analysis_task,
    create_paper_search_task,
)
from crewai import Crew, Process

# 创建 Agent
researcher = create_researcher_agent()
searcher = create_searcher_agent()

# 创建 Task
topic = "federated learning"
task1 = create_domain_analysis_task(researcher, topic)
task2 = create_paper_search_task(searcher, topic, max_papers=30, context=[task1])

# 组装 Crew
crew = Crew(
    agents=[researcher, searcher],
    tasks=[task1, task2],
    process=Process.sequential,
    verbose=True,
)

# 执行
result = crew.kickoff()
print(result)
```

---

## 9. 工作流详解

### 9.1 基础检索流程

```
输入: 研究主题（如 "federated learning"）
  │
  ├─ Step 1: Researcher Agent
  │   ├─ 使用 keyword_expander 扩展关键词
  │   ├─ 查询 CCF 数据库推荐会议/期刊
  │   └─ 输出: domain_analysis.json（领域分析报告）
  │
  ├─ Step 2: Searcher Agent
  │   ├─ 使用扩展后的关键词在 DBLP 检索
  │   ├─ 在 Semantic Scholar 检索（按引用量过滤）
  │   ├─ 在 arXiv 检索（最新预印本）
  │   ├─ 合并去重，按相关性排序
  │   └─ 输出: paper_search.json（最多 50 篇）
  │
  ├─ Step 3: Summarizer Agent（英文综述）
  │   ├─ 综合领域分析和检索结果
  │   └─ 输出: review_en.md（英文文献综述 ≥ 3000 字）
  │
  └─ Step 4: Summarizer Agent（中文翻译）
      ├─ 将英文综述翻译为高质量中文
      └─ 输出: review_zh.md（中文文献综述）
```

**耗时**：约 3-6 分钟（含双语综述生成）

### 9.2 深度分析流程

```
输入: 论文信息列表
  │
  ├─ Step 1: Analyzer Agent
  │   ├─ 获取论文详细信息和 PDF
  │   ├─ 提取核心贡献、研究方法、实验设计
  │   └─ 输出: 深度分析报告
  │
  └─ Step 2: Evaluator Agent
      ├─ 评估创新性 (0-10)
      ├─ 评估技术深度 (0-10)
      ├─ 评估实验有效性 (0-10)
      ├─ 评估写作质量 (0-10)
      └─ 输出: 质量评估报告
```

**耗时**：约 3-5 分钟

### 9.3 关联分析流程

```
输入: 论文信息 + 研究主题
  │
  ├─ Step 1: Connector Agent
  │   ├─ 构建引用网络
  │   ├─ 主题聚类分析
  │   ├─ 识别研究趋势
  │   └─ 输出: 关联分析报告
  │
  └─ Step 2: Summarizer Agent
      ├─ 整合所有分析结果
      ├─ 撰写领域综述
      └─ 输出: 文献综述报告（Markdown）
```

**耗时**：约 3-5 分钟

### 9.4 完整研究流程

```
输入: 研究主题
  │
  ├─ Phase 1: 检索（2-3 分钟）
  │   ├─ Researcher → 领域分析（domain_analysis.json）
  │   └─ Searcher → 论文检索（paper_search.json，50 篇）
  │
  ├─ Phase 2: 分析（5-10 分钟）
  │   ├─ Analyzer → 深度分析（analysis_results.json，Top 20）
  │   ├─ Connector → 关联分析（connection_analysis.json，全部 50 篇）
  │   └─ Evaluator → 质量评估（quality_evaluation.json，Top 20）
  │
  └─ Phase 3: 综合（5-8 分钟）
      ├─ Summarizer → 英文综述（review_en.md，≥ 3000 字）
      └─ Summarizer → 中文翻译（review_zh.md）
```

**总耗时**：约 12-21 分钟

---

## 10. 输出文件说明

每次运行会在 `output/<workflow>_<时间戳>/` 目录下生成独立的子目录，避免不同次运行的结果相互覆盖。

| 文件 | 说明 | 生成工作流 |
|------|------|------------|
| `prompt.txt` | 用户输入的原始主题/问题 | 所有工作流 |
| `run.log` | 本次运行的完整日志 | 所有工作流 |
| `domain_analysis.json` | 领域分析结果（研究领域、关键词、推荐期刊） | search、full |
| `paper_search.json` | 论文检索结果列表 | search、full |
| `analysis_results.json` | 论文深度分析结果 | analyze、full |
| `connection_analysis.json` | 引用网络和主题聚类 | connect、full |
| `quality_evaluation.json` | 论文质量评估评分 | analyze、full |
| `review_en.md` | 英文学术综述报告（≥ 3000 字） | search、connect、full |
| `review_zh.md` | 中文综述报告（英文版的高质量翻译） | search、connect、full |

---

## 11. 常见问题与故障排除

### Q1: 安装 PyMuPDF (pymupdf) 失败

**Windows**：
```powershell
pip install pymupdf --no-build-isolation
```

**Ubuntu**：
```bash
sudo apt install -y libmupdf-dev
pip install pymupdf
```

如果仍然失败，可以跳过安装——系统会自动回退到 PyPDF2。

### Q2: 提示 "Unknown LLM provider"

确保 `.env` 中 `DEFAULT_LLM_PROVIDER` 设置为已配置 API Key 的供应商：
```env
DEFAULT_LLM_PROVIDER=deepseek  # 改为你有 API Key 的供应商
```

### Q3: API 调用返回超时

自 v0.2.0 起，所有工作流 API 采用异步模式——`POST` 立即返回 `task_id`，不再阻塞等待结果。
只需通过 `GET /task/{task_id}` 轮询状态即可，无需设置 HTTP 长超时。

```bash
# 旧方式（不再需要）：curl --max-time 1200 -X POST ...
# 新方式：先 POST 获取 task_id，再 GET 轮询
curl -X POST http://localhost:8000/full-research \
  -H "Content-Type: application/json" \
  -d '{"topic": "..."}'
# → {"task_id": "abc123", "status": "pending"}

curl http://localhost:8000/task/abc123
# → 每 5-10 秒查询一次，直到 status 变为 "success" 或 "failed"
```

### Q4: 论文检索结果为空

1. 检查网络连接是否正常
2. 尝试更通用的搜索词
3. DBLP 和 arXiv 不需要 API Key，检查是否能正常访问
4. Semantic Scholar 有速率限制（100 次/5 分钟），配置 `SEMANTIC_SCHOLAR_API_KEY` 可提高限制

### Q5: 终端输出中文乱码

**Windows PowerShell**：
```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001
```

**CMD**：
```cmd
chcp 65001
```

### Q6: 端口被占用

```bash
# 查看占用端口的进程
# Linux/Mac:
lsof -i :8000
# Windows:
netstat -ano | findstr :8000

# 使用其他端口
mars api --port 9000
```

### Q7: 如何只使用一个 LLM 供应商？

修改 `.env` 文件：
```env
DEEPSEEK_API_KEY=sk-your-key
DEFAULT_LLM_PROVIDER=deepseek
```

这样所有 Agent 都会使用 DeepSeek。

### Q8: 如何查看 Agent 的执行过程？

所有 Agent 默认启用 `verbose=True`，执行过程会实时输出到终端。每次运行会在 `output/<workflow>_<时间戳>/` 下生成 `run.log` 文件保存完整日志，无需额外重定向。

如需手动保存：

```bash
# 将输出保存到文件
mars full "topic" 2>&1 | tee research.log
```

### Q9: mars 命令找不到

确保已激活虚拟环境，并以开发模式安装：
```bash
source .venv/bin/activate  # Windows venv: .venv\Scripts\activate
pip install -e .
```

Conda 用户：
```powershell
conda activate mars
pip install -e .
```

或直接使用 `python main.py`：
```bash
python main.py search "topic"
```

### Q10: SQLite 数据库锁定错误

在多进程并发访问时可能出现。对于生产环境，建议使用 PostgreSQL：
```env
DATABASE_URL=postgresql://user:password@localhost:5432/mars
```

### Q11: 如何快速验证系统是否就绪？

使用 `mars check` 命令检查所有 API Key 的配置状态：
```bash
mars check
```
输出会显示每个供应商的配置状态（✅ 已配置 / ❌ 未配置）及可用供应商列表。

---

## 12. 附录：API Key 获取指南

### 12.1 阿里云 Qwen (DashScope)

1. 访问 [阿里云百炼平台](https://bailian.console.aliyun.com/)
2. 使用阿里云账号登录
3. 进入「API-KEY 管理」页面
4. 点击「创建新的 API-KEY」
5. 复制 Key 到 `.env` 文件的 `DASHSCOPE_API_KEY`

**免费额度**：新用户通常有一定的免费调用额度

### 12.2 DeepSeek

1. 访问 [DeepSeek 开放平台](https://platform.deepseek.com/)
2. 注册并登录
3. 进入「API Keys」页面
4. 创建新的 API Key
5. 复制到 `.env` 文件的 `DEEPSEEK_API_KEY`

**免费额度**：注册后有一定免费额度

### 12.3 月之暗面 Kimi (Moonshot)

1. 访问 [Moonshot 开放平台](https://platform.moonshot.cn/)
2. 注册并登录
3. 进入「API Key 管理」页面
4. 创建 API Key
5. 复制到 `.env` 文件的 `MOONSHOT_API_KEY`

**免费额度**：新用户有免费试用额度

### 12.4 智谱 AI GLM

1. 访问 [智谱 AI 开放平台](https://open.bigmodel.cn/)
2. 注册并登录
3. 进入「API Key」页面
4. 创建 API Key（格式为 `xxx.yyy`）
5. 复制到 `.env` 文件的 `ZHIPU_API_KEY`

**免费额度**：新用户有免费调用额度

### 12.5 Semantic Scholar（可选）

1. 访问 [Semantic Scholar API](https://www.semanticscholar.org/product/api)
2. 申请 API Key
3. 复制到 `.env` 文件的 `SEMANTIC_SCHOLAR_API_KEY`

**作用**：提高 API 速率限制（从 100 次/5 分钟到更高），非必需。

---

*文档最后更新：2026-03-30*
