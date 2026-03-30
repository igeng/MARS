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

import datetime
import sys
from pathlib import Path
from typing import IO, Optional

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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _TeeWriter:
    """Wraps a primary text stream and mirrors all writes to a secondary one.

    Passes ``isatty()`` and ``fileno()`` through to the primary stream so
    that downstream code (e.g. Rich's auto-detection) behaves as if the
    terminal is still attached.
    """

    def __init__(self, primary: IO[str], secondary: IO[str]) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, data: str) -> int:
        n = self._primary.write(data)
        try:
            self._secondary.write(data)
        except Exception:
            pass
        return n

    def flush(self) -> None:
        self._primary.flush()
        try:
            self._secondary.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        return getattr(self._primary, "isatty", lambda: False)()

    def fileno(self) -> int:
        return self._primary.fileno()

    # Forward any attribute lookups not handled above to the primary stream
    def __getattr__(self, name: str):
        return getattr(self._primary, name)


def _create_run_dir(prefix: str) -> Path:
    """Return (and create) a timestamped sub-folder inside OUTPUT_DIR."""
    from mars.config import settings

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = settings.OUTPUT_DIR / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _print_banner() -> None:
    banner = Text()
    banner.append("🚀 MARS", style="bold cyan")
    banner.append(" - Multi-Agent Research System\n", style="bold white")
    banner.append("   多智能体学术文献智能检索与分析系统", style="dim white")
    console.print(Panel(banner, border_style="cyan"))


def _save_result(result: str, prefix: str) -> None:
    """Save *result* to a timestamped Markdown file in the output directory."""
    from mars.config import settings

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.md"
    file_path = settings.OUTPUT_DIR / filename
    try:
        file_path.write_text(result, encoding="utf-8")
        console.print(f"\n[dim]💾 结果已保存至：{file_path.resolve()}[/dim]")
    except OSError as exc:
        console.print(f"[yellow]⚠ 结果保存失败：{exc}[/yellow]")


def _startup(run_dir: Optional[Path] = None) -> None:
    """Common startup: redirect OUTPUT_DIR to *run_dir*, configure logging."""
    from mars.utils.logging_config import setup_logging
    from mars.config import settings

    if run_dir is not None:
        settings.OUTPUT_DIR = run_dir
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command("search")
def search_command(
    topic: str = typer.Argument(..., help="研究主题（中文或英文）"),
    max_results: int = typer.Option(
        50, "--max-results", "-n", help="最大检索论文数量"
    ),
) -> None:
    """
    基础检索流程：领域分析 → 论文检索 → 中/英文献综述 → 保存5份输出文件
    """
    from mars.config import settings

    # Create per-run output folder
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = _create_run_dir("search")

    # Open the unified run log (captures both our prints and CrewAI verbose)
    log_path = run_dir / "run.log"
    log_fh = open(log_path, "w", encoding="utf-8", errors="replace")

    # Redirect sys.stdout / sys.stderr through the tee so that everything
    # printed to the terminal is also written to run.log
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeWriter(_orig_stdout, log_fh)
    sys.stderr = _TeeWriter(_orig_stderr, log_fh)

    # Reinitialise the module-level Rich console to use the tee so that our
    # own formatted output is also captured.
    global console
    console = Console(file=sys.stdout, legacy_windows=False)

    try:
        # Save prompt
        (run_dir / "prompt.txt").write_text(topic, encoding="utf-8")

        _print_banner()
        _startup(run_dir)          # sets OUTPUT_DIR = run_dir and calls setup_logging

        console.print(f"\n[bold green]🔍 开始检索：[/bold green] {topic}\n")
        console.print(
            f"[dim]📁 本次运行输出目录：{run_dir.resolve()}[/dim]\n"
            f"[dim]预计生成5份文件：prompt.txt、run.log、paper_search.json、"
            f"review_zh.md、review_en.md[/dim]\n"
        )

        from mars.crews.search_crew import run_search

        result = run_search(topic, max_results=max_results)
        console.print("\n[bold cyan]📋 检索完成[/bold cyan]")
        console.print(f"\n[dim]✅ 所有输出文件已保存至：{run_dir.resolve()}[/dim]")

    except Exception as exc:
        console.print(f"[bold red]❌ 检索失败：{exc}[/bold red]")
        raise typer.Exit(1) from exc
    finally:
        log_fh.flush()
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr
        log_fh.close()
        # Restore console to a plain terminal console
        console = Console(legacy_windows=False)


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
    from mars.config import settings

    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = _create_run_dir("analyze")
    log_path = run_dir / "run.log"
    log_fh = open(log_path, "w", encoding="utf-8", errors="replace")
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeWriter(_orig_stdout, log_fh)
    sys.stderr = _TeeWriter(_orig_stderr, log_fh)
    global console
    console = Console(file=sys.stdout, legacy_windows=False)

    try:
        (run_dir / "prompt.txt").write_text(papers, encoding="utf-8")
        _print_banner()
        _startup(run_dir)
        console.print(f"\n[bold green]🔬 开始深度分析...[/bold green]\n")
        console.print(f"[dim]📁 本次运行输出目录：{run_dir.resolve()}[/dim]\n")

        from mars.crews.analysis_crew import run_analysis

        result = run_analysis(papers, max_papers=max_papers)
        console.print("\n[bold cyan]📊 分析报告：[/bold cyan]")
        console.print(result)
        _save_result(result, "analysis_report")
    except Exception as exc:
        console.print(f"[bold red]❌ 分析失败：{exc}[/bold red]")
        raise typer.Exit(1) from exc
    finally:
        log_fh.flush()
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr
        log_fh.close()
        console = Console(legacy_windows=False)


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
    关联分析流程：引用分析 → 相似度计算 → 中/英综述生成 → 返回关联图谱和综述
    """
    from mars.config import settings

    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = _create_run_dir("connect")
    log_path = run_dir / "run.log"
    log_fh = open(log_path, "w", encoding="utf-8", errors="replace")
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeWriter(_orig_stdout, log_fh)
    sys.stderr = _TeeWriter(_orig_stderr, log_fh)
    global console
    console = Console(file=sys.stdout, legacy_windows=False)

    try:
        (run_dir / "prompt.txt").write_text(
            f"papers: {papers}\ntopic: {topic}", encoding="utf-8"
        )
        _print_banner()
        _startup(run_dir)
        console.print(f"\n[bold green]🔗 开始关联分析...[/bold green]\n")
        console.print(f"[dim]📁 本次运行输出目录：{run_dir.resolve()}[/dim]\n")

        from mars.crews.connection_crew import run_connection

        result = run_connection(papers, topic=topic)
        console.print("\n[bold cyan]🗺️  关联分析报告：[/bold cyan]")
        console.print(result)
        _save_result(result, "connection_report")
    except Exception as exc:
        console.print(f"[bold red]❌ 关联分析失败：{exc}[/bold red]")
        raise typer.Exit(1) from exc
    finally:
        log_fh.flush()
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr
        log_fh.close()
        console = Console(legacy_windows=False)


@app.command("full")
def full_research_command(
    topic: str = typer.Argument(..., help="研究主题（中文或英文）"),
) -> None:
    """
    完整研究流程：领域分析 → 批量检索 → 深度解析 + 关联分析 + 质量评估 → 中/英综述生成
    """
    from mars.config import settings

    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = _create_run_dir("full")
    log_path = run_dir / "run.log"
    log_fh = open(log_path, "w", encoding="utf-8", errors="replace")
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeWriter(_orig_stdout, log_fh)
    sys.stderr = _TeeWriter(_orig_stderr, log_fh)
    global console
    console = Console(file=sys.stdout, legacy_windows=False)

    try:
        (run_dir / "prompt.txt").write_text(topic, encoding="utf-8")
        _print_banner()
        _startup(run_dir)
        console.print(f"\n[bold green]🚀 启动完整研究流程：[/bold green] {topic}\n")
        console.print(
            "[dim]此流程将依次执行：领域分析 → 论文检索 → 深度解析 → "
            "关联分析 → 质量评估 → 中/英综述生成[/dim]\n"
        )
        console.print(f"[dim]📁 本次运行输出目录：{run_dir.resolve()}[/dim]\n")

        from mars.crews.full_research_crew import run_full_research

        result = run_full_research(topic)
        console.print("\n[bold cyan]📚 完整研究报告：[/bold cyan]")
        console.print(result)
        _save_result(result, "full_research_report")
    except Exception as exc:
        console.print(f"[bold red]❌ 研究流程失败：{exc}[/bold red]")
        raise typer.Exit(1) from exc
    finally:
        log_fh.flush()
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr
        log_fh.close()
        console = Console(legacy_windows=False)


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
