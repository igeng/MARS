"""
MARS CLI - Command Line Interface

Usage:
    mars search "federated learning privacy"
    mars analyze "paper1 title | paper2 title | ..."
    mars connect "paper IDs or titles" --topic "federated learning"
    mars full "federated learning with differential privacy"
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

app = typer.Typer(
    name="mars",
    help="MARS - Multi-Agent Research System | 多智能体学术文献智能检索与分析系统",
    add_completion=False,
)
console = Console()


def _print_banner() -> None:
    banner = Text()
    banner.append("🚀 MARS", style="bold cyan")
    banner.append(" - Multi-Agent Research System\n", style="bold white")
    banner.append("   多智能体学术文献智能检索与分析系统", style="dim white")
    console.print(Panel(banner, border_style="cyan"))


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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
