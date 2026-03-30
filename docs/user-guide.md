# MARS 使用手册

**Multi-Agent Research System — 多智能体学术文献智能检索与分析系统**

> 版本：0.1.0 | Python ≥ 3.10

---

## 目录

1. [系统简介](#1-系统简介)
2. [系统要求](#2-系统要求)
3. [Windows 10 本地部署](#3-windows-10-本地部署)
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
- 📝 **综述生成**：自动撰写领域综述报告
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
| `QWEN_MODEL` | 否 | qwen-max | Qwen 模型名称 |
| `DEEPSEEK_MODEL` | 否 | deepseek-chat | DeepSeek 模型名称 |
| `KIMI_MODEL` | 否 | moonshot-v1-8k | Kimi 模型名称 |
| `GLM_MODEL` | 否 | glm-4-plus | GLM 模型名称 |
| `MAX_PAPERS_PER_SEARCH` | 否 | 50 | 每次检索最大论文数 |
| `MAX_PAPERS_FOR_ANALYSIS` | 否 | 20 | 深度分析最大论文数 |
| `OUTPUT_DIR` | 否 | ./output | 输出文件目录 |
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

**输出**：论文列表，包含标题、作者、发表期刊/会议、年份、引用次数、URL。

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
1. 领域分析 → 推荐 CCF 期刊/会议
2. 多数据库检索 → 获取 50 篇相关论文
3. 深度分析 → 提取 Top 20 论文的核心内容
4. 关联分析 → 构建引用网络，识别趋势
5. 质量评估 → 多维度评分
6. 综述生成 → 撰写 ≥2000 字的综述报告

**输出**：完整的研究报告，保存到 `output/` 目录。

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

#### 健康检查

```bash
curl http://localhost:8000/health
# 返回: {"status": "ok"}
```

#### 基础检索

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"topic": "federated learning privacy", "max_results": 20}'
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

> **注意**：完整研究流程可能需要几分钟到十几分钟，请耐心等待响应。

### 7.4 使用 Python requests 调用

```python
import requests

# 基础检索
response = requests.post(
    "http://localhost:8000/search",
    json={"topic": "deep learning for NLP", "max_results": 20}
)
print(response.json())

# 完整研究
response = requests.post(
    "http://localhost:8000/full-research",
    json={"topic": "federated learning privacy"}
)
result = response.json()
print(result["result"])
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
  │   └─ 输出: 领域分析报告（JSON）
  │
  └─ Step 2: Searcher Agent
      ├─ 使用扩展后的关键词在 DBLP 检索
      ├─ 在 Semantic Scholar 检索（按引用量过滤）
      ├─ 在 arXiv 检索（最新预印本）
      ├─ 合并去重，按相关性排序
      └─ 输出: 论文列表（最多 50 篇）
```

**耗时**：约 1-3 分钟

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
  │   ├─ Researcher → 领域分析
  │   └─ Searcher → 论文检索（50 篇）
  │
  ├─ Phase 2: 分析（5-10 分钟）
  │   ├─ Analyzer → 深度分析（Top 20）
  │   ├─ Connector → 关联分析（全部 50 篇）
  │   └─ Evaluator → 质量评估（Top 20）
  │
  └─ Phase 3: 综合（3-5 分钟）
      └─ Summarizer → 综合综述（≥2000 字）
```

**总耗时**：约 10-20 分钟

---

## 10. 输出文件说明

完整研究流程会在 `output/` 目录生成以下文件：

| 文件 | 说明 | 生成阶段 |
|------|------|----------|
| `analysis_results.json` | 论文深度分析结果 | Phase 2 (Analyzer) |
| `connection_analysis.json` | 引用网络和主题聚类 | Phase 2 (Connector) |
| `quality_evaluation.json` | 论文质量评估评分 | Phase 2 (Evaluator) |
| `literature_review.md` | 文献综述报告 | Phase 3 (Summarizer) |
| `comprehensive_review.md` | 完整研究综述 | Phase 3 (Summarizer) |

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

完整研究流程可能需要 10-20 分钟。使用 curl 时增加超时时间：
```bash
curl --max-time 1200 -X POST http://localhost:8000/full-research ...
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

所有 Agent 默认启用 `verbose=True`，执行过程会实时输出到终端。如需保存日志：

```bash
# 将输出保存到文件
mars full "topic" 2>&1 | tee research.log
```

### Q9: mars 命令找不到

确保已激活虚拟环境，并以开发模式安装：
```bash
source .venv/bin/activate  # Windows: .venv\Scripts\activate
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
