"""
MARS CLI - Command Line Interface

Usage:
    mars search "federated learning privacy"
    mars analyze "paper1 title | paper2 title | ..."
    mars connect "paper IDs or titles" --topic "federated learning"
    mars full "federated learning with differential privacy"
    mars api                          # Start FastAPI server
    mars init-db                      # Initialise the database
    mars check                        # Validate configuration
"""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# On Windows the default console encoding is often GBK/cp936, which cannot
# encode emoji and other Unicode characters outside its range, raising
# UnicodeEncodeError.  Reconfigure stdout/stderr to UTF-8 (with replacement
# for any remaining unencodable bytes) so that Rich can render the banner
# and all output without crashing.  The `hasattr` guard makes this safe on
# non-CPython runtimes where `reconfigure` may be absent.
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(
    name="mars",
    help="MARS - Multi-Agent Research System | 多智能体学术文献智能检索与分析系统",
    add_completion=False,
)
console = Console(legacy_windows=False)


def _print_banner() -> None:
    banner = Text()
    banner.append("🚀 MARS", style="bold cyan")
    banner.append(" - Multi-Agent Research System\n", style="bold white")
    banner.append("   多智能体学术文献智能检索与分析系统", style="dim white")
    console.print(Panel(banner, border_style="cyan"))


def _startup() -> None:
    """Common startup tasks: logging, output dir, database."""
    from mars.utils.logging_config import setup_logging
    from mars.config import settings

    setup_logging()
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@app.command("search")
def search_command(
    topic: str = typer.Argument(..., help="研究主题（中文或英文）"),
    max_results: int = typer.Option(
        50, "--max-results", "-n", help="最大检索论文数量"
    ),
) -> None:
    """
    基础检索流程：领域分析 → 论文检索 → 返回论文列表
    """
    _print_banner()
    _startup()
    console.print(f"\n[bold green]🔍 开始检索：[/bold green] {topic}\n")

    from mars.crews.search_crew import run_search

    try:
        result = run_search(topic, max_results=max_results)
        console.print("\n[bold cyan]📋 检索结果：[/bold cyan]")
        console.print(result)
    except Exception as exc:
        console.print(f"[bold red]❌ 检索失败：{exc}[/bold red]")
        raise typer.Exit(1) from exc


@app.command("analyze")
def analyze_command(
    papers: str = typer.Argument(
        ...,
        help="论文信息（标题列表，用 | 分隔，或包含URL的描述）",
    ),
    max_papers: int = typer.Option(
        20, "--max-papers", "-n", help="最大深度分析论文数量"
    ),
) -> None:
    """
    深度分析流程：论文获取 → 深度解析 → 质量评估 → 返回分析报告
    """
    _print_banner()
    _startup()
    console.print(f"\n[bold green]🔬 开始深度分析...[/bold green]\n")

    from mars.crews.analysis_crew import run_analysis

    try:
        result = run_analysis(papers, max_papers=max_papers)
        console.print("\n[bold cyan]📊 分析报告：[/bold cyan]")
        console.print(result)
    except Exception as exc:
        console.print(f"[bold red]❌ 分析失败：{exc}[/bold red]")
        raise typer.Exit(1) from exc


@app.command("connect")
def connect_command(
    papers: str = typer.Argument(
        ..., help="论文信息（标题或ID列表，用 | 分隔）"
    ),
    topic: str = typer.Option(
        ..., "--topic", "-t", help="研究主题（用于综述生成）"
    ),
) -> None:
    """
    关联分析流程：引用分析 → 相似度计算 → 综述生成 → 返回关联图谱和综述
    """
    _print_banner()
    _startup()
    console.print(f"\n[bold green]🔗 开始关联分析...[/bold green]\n")

    from mars.crews.connection_crew import run_connection

    try:
        result = run_connection(papers, topic=topic)
        console.print("\n[bold cyan]🗺️  关联分析报告：[/bold cyan]")
        console.print(result)
    except Exception as exc:
        console.print(f"[bold red]❌ 关联分析失败：{exc}[/bold red]")
        raise typer.Exit(1) from exc


@app.command("full")
def full_research_command(
    topic: str = typer.Argument(..., help="研究主题（中文或英文）"),
) -> None:
    """
    完整研究流程：领域分析 → 批量检索 → 深度解析 + 关联分析 + 质量评估 → 综述生成
    """
    _print_banner()
    _startup()
    console.print(f"\n[bold green]🚀 启动完整研究流程：[/bold green] {topic}\n")
    console.print(
        "[dim]此流程将依次执行：领域分析 → 论文检索 → 深度解析 → "
        "关联分析 → 质量评估 → 综述生成[/dim]\n"
    )

    from mars.crews.full_research_crew import run_full_research

    try:
        result = run_full_research(topic)
        console.print("\n[bold cyan]📚 完整研究报告：[/bold cyan]")
        console.print(result)
    except Exception as exc:
        console.print(f"[bold red]❌ 研究流程失败：{exc}[/bold red]")
        raise typer.Exit(1) from exc


@app.command("api")
def api_command(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="绑定主机地址"),
    port: int = typer.Option(8000, "--port", "-p", help="端口号"),
) -> None:
    """
    启动 FastAPI REST API 服务器
    """
    _print_banner()
    console.print(
        f"\n[bold green]🌐 启动 API 服务器：[/bold green] http://{host}:{port}\n"
    )

    import uvicorn

    uvicorn.run("mars.api.main:app", host=host, port=port, reload=False)


@app.command("init-db")
def init_db_command() -> None:
    """
    初始化数据库表结构
    """
    _print_banner()
    _startup()
    console.print("\n[bold green]🗄️  初始化数据库...[/bold green]\n")

    from mars.database.models import init_db

    init_db()
    console.print("[bold cyan]✅ 数据库初始化完成[/bold cyan]")


@app.command("check")
def check_command() -> None:
    """
    检查系统配置状态：API Key、LLM 供应商、输出目录等
    """
    _print_banner()
    console.print("\n[bold green]🔎 检查系统配置...[/bold green]\n")

    from mars.config import settings
    from mars.services.llm_gateway import get_available_providers

    # --- LLM provider status ---
    provider_info = {
        "qwen": ("DASHSCOPE_API_KEY", settings.DASHSCOPE_API_KEY),
        "deepseek": ("DEEPSEEK_API_KEY", settings.DEEPSEEK_API_KEY),
        "kimi": ("MOONSHOT_API_KEY", settings.MOONSHOT_API_KEY),
        "glm": ("ZHIPU_API_KEY", settings.ZHIPU_API_KEY),
    }

    table = Table(title="LLM 供应商状态")
    table.add_column("供应商", style="cyan")
    table.add_column("环境变量", style="white")
    table.add_column("状态", style="bold")

    available = get_available_providers()
    for name, (env_var, key_val) in provider_info.items():
        if key_val:
            masked = key_val[:4] + "****" + key_val[-4:] if len(key_val) > 8 else "****"
            status = f"[green]✅ 已配置 ({masked})[/green]"
        else:
            status = "[red]❌ 未配置[/red]"
        table.add_row(name, env_var, status)

    console.print(table)
    console.print(f"\n默认供应商: [cyan]{settings.DEFAULT_LLM_PROVIDER}[/cyan]")
    console.print(f"可用供应商: [cyan]{', '.join(available) if available else '无'}[/cyan]")

    # --- Academic search API ---
    console.print()
    s2_status = (
        "[green]✅ 已配置[/green]"
        if settings.SEMANTIC_SCHOLAR_API_KEY
        else "[yellow]⚠ 未配置（可使用，但速率受限）[/yellow]"
    )
    console.print(f"Semantic Scholar API Key: {s2_status}")

    # --- Paths ---
    console.print()
    console.print(f"输出目录: [cyan]{settings.OUTPUT_DIR.resolve()}[/cyan]")
    console.print(f"数据库:   [cyan]{settings.DATABASE_URL}[/cyan]")

    # --- Summary ---
    console.print()
    if available:
        console.print(
            f"[bold green]✅ 系统就绪 — "
            f"可使用 {len(available)} 个 LLM 供应商[/bold green]"
        )
    else:
        console.print(
            "[bold red]❌ 系统尚未就绪 — "
            "请在 .env 文件中至少配置一个 LLM API Key[/bold red]"
        )
        raise typer.Exit(1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
