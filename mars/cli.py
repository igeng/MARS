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

import contextlib
import datetime
import sys
from pathlib import Path
from typing import IO, Callable, Generator, List, Optional, Tuple

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

    def writable(self) -> bool:
        return getattr(self._primary, "writable", lambda: False)()

    def readable(self) -> bool:
        return getattr(self._primary, "readable", lambda: False)()

    def seekable(self) -> bool:
        return getattr(self._primary, "seekable", lambda: False)()

    def close(self) -> None:
        """Close only the secondary stream; keep the primary alive."""
        try:
            self._secondary.close()
        except Exception:
            pass

    # Forward any attribute lookups not handled above to the primary stream
    def __getattr__(self, name: str):
        return getattr(self._primary, name)


@contextlib.contextmanager
def _run_context(
    run_dir: Path,
) -> Generator[Tuple[Console, IO[str]], None, None]:
    """Context manager that sets up per-run logging and stdout/stderr capture.

    Opens ``run_dir/run.log``, replaces ``sys.stdout`` and ``sys.stderr``
    with :class:`_TeeWriter` instances, creates a fresh :class:`Console`
    writing to the tee, and restores everything on exit.

    Yields:
        A ``(console, log_fh)`` tuple where *console* is bound to the tee
        and *log_fh* is the open log-file handle (already passed to the
        logging subsystem).
    """
    from mars.utils.logging_config import setup_logging

    log_path = run_dir / "run.log"
    with open(log_path, "w", encoding="utf-8", errors="replace") as log_fh:
        orig_stdout, orig_stderr = sys.stdout, sys.stderr
        sys.stdout = _TeeWriter(orig_stdout, log_fh)
        sys.stderr = _TeeWriter(orig_stderr, log_fh)
        run_console = Console(file=sys.stdout, legacy_windows=False)
        setup_logging(log_stream=log_fh)
        try:
            yield run_console, log_fh
        finally:
            log_fh.flush()
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr


def _create_run_dir(prefix: str) -> Path:
    """Return (and create) a timestamped sub-folder inside OUTPUT_DIR."""
    from mars.config import settings

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = settings.OUTPUT_DIR / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _print_banner(con: Console) -> None:
    banner = Text()
    banner.append("🚀 MARS", style="bold cyan")
    banner.append(" - Multi-Agent Research System\n", style="bold white")
    banner.append("   多智能体学术文献智能检索与分析系统", style="dim white")
    con.print(Panel(banner, border_style="cyan"))


def _save_result(result: str, prefix: str, con: Console) -> None:
    """Save *result* to a timestamped Markdown file in the output directory."""
    from mars.config import settings

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.md"
    file_path = settings.OUTPUT_DIR / filename
    try:
        file_path.write_text(result, encoding="utf-8")
        con.print(f"\n[dim]💾 结果已保存至：{file_path.resolve()}[/dim]")
    except OSError as exc:
        con.print(f"[yellow]⚠ 结果保存失败：{exc}[/yellow]")


def _startup(run_dir: Optional[Path] = None) -> None:
    """Common startup: redirect OUTPUT_DIR to *run_dir*, configure logging."""
    from mars.config import settings

    if run_dir is not None:
        settings.OUTPUT_DIR = run_dir
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Post-generation evaluation
# ---------------------------------------------------------------------------

def _run_evaluation_after_generation(con: Console, run_dir: Path, topic: str) -> None:
    """Run full evaluation on the generated survey and print results."""
    review_path = run_dir / "review_en.md"
    if not review_path.is_file():
        con.print("[yellow]⚠ review_en.md 不存在，跳过评估。[/yellow]")
        return

    con.print(f"\n[bold yellow]━━━ 全维度评估 ━━━[/bold yellow]")

    try:
        from mars.evaluation.full_eval import run_full_evaluation, save_eval_report

        eval_results = run_full_evaluation(
            topic=topic,
            survey_path=review_path,
        )
        save_eval_report(eval_results, run_dir)
        _print_eval_summary(con, eval_results)
    except Exception as exc:
        con.print(f"[red]❌ 评估失败：{exc}[/red]")


def _print_eval_summary(con: Console, results: dict) -> None:
    """Print a compact evaluation summary."""
    m = results.get("metrics", {})

    # Citation F1
    cit = m.get("citation", {})
    if cit and "f1" in cit:
        con.print(f"\n[bold]📊 Citation Metrics[/bold]")
        con.print(f"  Recall: {cit['recall']:.2%}  Precision: {cit['precision']:.2%}  F1: {cit['f1']:.2%}")
        con.print(f"  Refs: {cit['matched_refs']}/{cit['ground_truth_refs']} matched")

    # LLM Judge
    judge = m.get("llm_judge_12dim", {})
    if isinstance(judge, dict) and "overall_score" in judge:
        con.print(f"\n[bold]🎓 LLM-as-Judge (12-dim)[/bold]")
        con.print(f"  Overall: {judge['overall_score']}/10")
        for cat in ["core_quality", "writing_quality", "content_depth"]:
            cdata = judge.get(cat, {})
            if "category_score" in cdata:
                con.print(f"  {cat}: {cdata['category_score']}/10")

    # Hallucination
    hallu = m.get("hallucination", {})
    if isinstance(hallu, dict) and "total_citations" in hallu:
        con.print(f"\n[bold]🔍 Hallucination[/bold]")
        con.print(f"  Total: {hallu['total_citations']}  Verified: {hallu.get('verified',0)}  Fabrication: {hallu.get('fabrication_rate',0):.1%}")

    # SurGE official
    surge = m.get("surge_official", {})
    if isinstance(surge, dict) and surge:
        con.print(f"\n[bold]📐 SurGE Official[/bold]")
        for k, v in surge.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, (int, float)):
                        con.print(f"  {k}.{sub_k}: {sub_v:.4f}")

    # Baseline comparison
    baselines = results.get("baselines", {})
    if baselines and not baselines.get("error"):
        con.print(f"\n[bold]⚔️ Baseline Comparison[/bold]")
        con.print(f"  {'Method':<15} {'F1':>8} {'Recall':>8} {'Precision':>8}")
        for name in ["Naive", "ID", "Autosurvey"]:
            if name in baselines:
                b = baselines[name]
                con.print(f"  {name:<15} {b['avg_f1']:>7.4f} {b['avg_recall']:>7.4f} {b['avg_precision']:>7.4f}")

    con.print(f"\n[dim]📄 完整报告: {results.get('survey_path', 'N/A').replace('review_en.md', 'full_eval_report.json')}[/dim]")


# ---------------------------------------------------------------------------
# Shared workflow runner
# ---------------------------------------------------------------------------

_OutputFile = Tuple[str, str, str]  # (filename, description, format)


def _run_workflow(
    prefix: str,
    prompt_text: str,
    run_label: str,
    workflow_fn: Callable[[], str | None],
    *,
    expected_files: List[_OutputFile] | None = None,
    success_msg_extra: str = "",
    save_result_as: str | None = None,
    eval_after: bool = False,
) -> None:
    """Generic workflow runner shared by all CLI commands.

    Args:
        prefix: Run-dir prefix (e.g. ``"search"``, ``"analyze"``).
        prompt_text: Content for ``prompt.txt``.
        run_label: Emoji + label shown in the "starting…" line.
        workflow_fn: Zero-argument callable that executes the workflow and
            returns an optional result string.  If the callable needs
            topic / papers / etc. it should close over them.
        expected_files: Optional list of ``(filename, description, format)``
            tuples to display before the run.
        success_msg_extra: Extra text appended after the success line.
        save_result_as: If provided and *workflow_fn* returns a non-empty
            string, write it to ``<save_result_as>.md`` inside *run_dir*.
    """
    from mars.config import settings
    from mars.utils.logging_config import set_run_id

    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = _create_run_dir(prefix)
    _startup(run_dir)

    # Inject the run directory name as the logging run_id so every log line
    # is tagged with the active run.
    set_run_id(run_dir.name)

    with _run_context(run_dir) as (con, _log_fh):
        (run_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
        _print_banner(con)
        con.print(f"\n[bold green]{run_label}[/bold green]\n")
        con.print(f"[dim]📁 本次运行输出目录：{run_dir.resolve()}[/dim]\n")

        if expected_files:
            file_table = Table(
                title="📋 本次运行预计生成以下文件",
                show_header=True,
                header_style="bold cyan",
            )
            file_table.add_column("文件名", style="cyan", no_wrap=True)
            file_table.add_column("内容说明", style="white")
            file_table.add_column("格式", style="dim")
            for fname, desc, fmt in expected_files:
                file_table.add_row(fname, desc, fmt)
            con.print(file_table)
            con.print()

        try:
            result = workflow_fn()

            if success_msg_extra:
                con.print(f"\n[bold cyan]{success_msg_extra}[/bold cyan]")

            con.print(
                f"\n[dim]所有输出文件已保存至：{run_dir.resolve()}[/dim]"
            )

            if save_result_as and result:
                report_path = run_dir / f"{save_result_as}.md"
                try:
                    report_path.write_text(result, encoding="utf-8")
                    con.print(f"\n[dim]💾 结果已保存至：{report_path.resolve()}[/dim]")
                except OSError as exc:
                    con.print(f"[yellow]⚠ 结果保存失败：{exc}[/yellow]")

        except Exception as exc:
            con.print(f"[bold red]❌ 失败：{exc}[/bold red]")
            raise typer.Exit(1) from exc

        # Post-generation evaluation (if --eval flag)
        if eval_after:
            _run_evaluation_after_generation(con, run_dir, prompt_text)


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
    _run_workflow(
        prefix="search",
        prompt_text=topic,
        run_label="🔍 开始检索：" + topic,
        workflow_fn=lambda: _call_search(topic, max_results),
        expected_files=[
            ("prompt.txt",       "输入的研究主题",                         "文本"),
            ("run.log",          "完整运行日志",                           "文本"),
            ("paper_search.json","检索结果：论文完整元数据",               "JSON"),
            ("review_zh.md",     "中文学术文献综述",                       "Markdown"),
            ("review_en.md",     "English literature review",              "Markdown"),
        ],
        success_msg_extra="✅ 检索完成",
    )


def _call_search(topic: str, max_results: int) -> None:
    from mars.crews.search_crew import run_search
    run_search(topic, max_results=max_results)
    return None


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
    _run_workflow(
        prefix="analyze",
        prompt_text=papers,
        run_label="🔬 开始深度分析...",
        workflow_fn=lambda: _call_analyze(papers, max_papers),
        expected_files=[
            ("prompt.txt",        "输入的论文信息",      "文本"),
            ("run.log",           "完整运行日志",        "文本"),
            ("analysis_report.md","深度分析报告",        "Markdown"),
        ],
        success_msg_extra="✅ 分析完成",
        save_result_as="analysis_report",
    )


def _call_analyze(papers: str, max_papers: int) -> str:
    from mars.crews.analysis_crew import run_analysis
    return run_analysis(papers, max_papers=max_papers)


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
    _run_workflow(
        prefix="connect",
        prompt_text=f"papers: {papers}\ntopic: {topic}",
        run_label="🔗 开始关联分析...",
        workflow_fn=lambda: _call_connect(papers, topic),
        success_msg_extra="✅ 关联分析完成",
    )


def _call_connect(papers: str, topic: str) -> str:
    from mars.crews.connection_crew import run_connection
    return run_connection(papers, topic=topic)


@app.command("full")
def full_research_command(
    topic: str = typer.Argument(..., help="研究主题（中文或英文）"),
    mode: str = typer.Option(
        "online", "--mode", "-m", help="检索模式: online (实时API) | surge (SurGE语料库)"
    ),
    eval_flag: bool = typer.Option(
        False, "--eval", help="生成后自动运行全维度评估（含 SurGE 指标 + baseline 对比）"
    ),
) -> None:
    """
    完整研究流程：领域分析 → 批量检索 → 深度解析 + 关联分析 + 质量评估 → 中/英综述生成
    """
    _run_workflow(
        prefix="full",
        prompt_text=topic,
        run_label=f"🚀 启动完整研究流程（{mode}模式）：" + topic,
        workflow_fn=lambda: _call_full(topic, mode),
        eval_after=eval_flag,
        expected_files=[
            ("prompt.txt",                   "输入的研究主题",                                  "文本"),
            ("run.log",                      "完整运行日志",                                    "文本"),
            ("domain_analysis.json",         "领域分析：子领域 / 关键词 / 推荐期刊会议",        "JSON"),
            ("paper_search.json",            "检索结果：50篇论文完整元数据",                    "JSON"),
            ("connection_analysis.json",     "关联网络结构化数据：引用图 / 聚类 / 趋势",        "JSON"),
            ("connection_analysis_report.md","论文关联关系深度分析报告（2000+字）",             "Markdown"),
            ("analysis_report.md",           "逐篇论文质量评估报告（多维评分 + 改进建议）",     "Markdown"),
            ("review_en.md",                 "英文学术文献综述（3000+字）",                     "Markdown"),
            ("review_zh.md",                 "中文学术文献综述（英文版完整翻译）",              "Markdown"),
            ("full_research_report.md",      "综合研究报告：统计 + 质量排名 + 洞见 + 导读指南", "Markdown"),
        ],
        success_msg_extra="✅ 完整研究流程已完成",
    )


def _call_full(topic: str, mode: str = "online") -> None:
    from mars.crews.full_research_crew import run_full_research
    run_full_research(topic, corpus_mode=mode)
    return None


@app.command("api")
def api_command(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="绑定主机地址"),
    port: int = typer.Option(8000, "--port", "-p", help="端口号"),
) -> None:
    """
    启动 FastAPI REST API 服务器
    """
    _print_banner(console)
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
    _print_banner(console)
    _startup()
    console.print("\n[bold green]🗄️  初始化数据库...[/bold green]\n")

    from mars.database.models import init_db

    init_db()
    console.print("[bold cyan]✅ 数据库初始化完成[/bold cyan]")


@app.command("evaluate")
def evaluate_command(
    topic: str = typer.Argument(..., help="研究主题"),
    survey_path: str = typer.Option(
        ..., "--survey", "-s", help="待评估的综述文件路径（Markdown）"
    ),
    topic_id: str = typer.Option(
        "", "--topic-id", "-t", help="SurGE topic ID（不指定则自动匹配）"
    ),
    surge_dir: str = typer.Option(
        "", "--surge-dir", help="SurGE data 目录路径（如 D:/.../SurGE/data），直接从官方数据集加载 205 个 topic"
    ),
) -> None:
    """
    评估已生成的文献综述质量：SurGE benchmark + LLM-as-Judge + 幻觉检测
    """
    from mars.config import settings
    from mars.evaluation.surge_eval import SurGEEvaluator
    from mars.evaluation.llm_judge import LLMJudge
    from mars.evaluation.hallucination_checker import HallucinationChecker

    survey_path_obj = Path(survey_path)
    if not survey_path_obj.is_file():
        console.print(f"[red]❌ 文件不存在：{survey_path}[/red]")
        raise typer.Exit(1)

    survey_text = survey_path_obj.read_text(encoding="utf-8")

    _print_banner(console)
    console.print(f"\n[bold green]📊 评估综述质量：[/bold green] {topic}\n")

    # 1. SurGE benchmark
    console.print("[bold]1/3 SurGE Benchmark 评估...[/bold]")
    if surge_dir:
        evaluator = SurGEEvaluator.from_surge(surge_dir)
        console.print(f"   从 SurGE 加载: {len(evaluator.topic_ids)} 个 topic")
    else:
        evaluator = SurGEEvaluator()
    if not topic_id:
        # Auto-match: find topic id that best matches the topic string
        for tid, tname in evaluator.topics.items():
            if topic.lower() in tname.lower() or tname.lower() in topic.lower():
                topic_id = tid
                break
    if not topic_id:
        topic_id = list(evaluator.topic_ids)[0]
        console.print(f"[yellow]⚠ 未匹配到 topic，使用默认: {topic_id}[/yellow]")

    try:
        surge_result = evaluator.evaluate(topic_id, survey_text)
        console.print(f"   Recall:    {surge_result.recall:.2%}")
        console.print(f"   Precision: {surge_result.precision:.2%}")
        console.print(f"   F1:        {surge_result.f1:.2%}")
        console.print(f"   Matched:   {surge_result.matched_ref_count}/{surge_result.ground_truth_ref_count}")
    except KeyError as exc:
        console.print(f"[yellow]⚠ SurGE 评估跳过：{exc}[/yellow]")
        surge_result = None

    # 2. LLM-as-Judge (12-dimension)
    console.print("\n[bold]2/3 LLM-as-Judge 评分 (12维)...[/bold]")
    try:
        judge = LLMJudge()
        judge_result = judge.evaluate(survey_text, topic)
        d = judge_result.to_flat_dict()
        console.print("   [bold cyan]Core Quality (60%)[/bold cyan]")
        console.print(f"     Citation Coverage:     {d['citation_coverage']}/10")
        console.print(f"     Citation Accuracy:     {d['citation_accuracy']}/10")
        console.print(f"     Synthesis Quality:     {d['synthesis_quality']}/10")
        console.print(f"     Organization:          {d['organization']}/10")
        console.print(f"     → Score: {d['core_quality']}/10")
        console.print("   [bold cyan]Writing Quality (20%)[/bold cyan]")
        console.print(f"     Readability:           {d['readability']}/10")
        console.print(f"     Academic Rigor:        {d['academic_rigor']}/10")
        console.print(f"     Clarity:               {d['clarity']}/10")
        console.print(f"     Coherence:             {d['coherence']}/10")
        console.print(f"     → Score: {d['writing_quality']}/10")
        console.print("   [bold cyan]Content Depth (20%)[/bold cyan]")
        console.print(f"     Comprehensiveness:     {d['comprehensiveness']}/10")
        console.print(f"     Critical Analysis:     {d['critical_analysis']}/10")
        console.print(f"     Novelty / Insights:    {d['novelty_insights']}/10")
        console.print(f"     Future Directions:     {d['future_directions']}/10")
        console.print(f"     → Score: {d['content_depth']}/10")
        console.print(f"   [bold]Overall: {judge_result.overall_score}/10[/bold]")
        console.print(f"   {judge_result.overall_comment}")
    except Exception as exc:
        console.print(f"[yellow]⚠ LLM Judge 评估跳过：{exc}[/yellow]")
        judge_result = None

    # 3. Hallucination check
    console.print("\n[bold]3/3 引用幻觉检测...[/bold]")
    try:
        checker = HallucinationChecker()
        hallu_report = checker.check(survey_text, [])
        console.print(f"   Total citations:   {hallu_report.total_citations}")
        console.print(f"   Verified:           {hallu_report.verified}")
        console.print(f"   Fabricated:         {hallu_report.fabricated}")
        console.print(f"   Unchecked:          {hallu_report.unchecked}")
        console.print(f"   [bold]Fabrication rate: {hallu_report.to_dict()['fabrication_rate']:.2%}[/bold]")
    except Exception as exc:
        console.print(f"[yellow]⚠ 幻觉检测跳过：{exc}[/yellow]")
        hallu_report = None

    # Save report
    report = {
        "topic": topic,
        "topic_id": topic_id,
        "surge": surge_result.to_dict() if surge_result else None,
        "llm_judge": judge_result.to_dict() if judge_result else None,
        "hallucination": hallu_report.to_dict() if hallu_report else None,
    }
    import json
    report_path = settings.OUTPUT_DIR / f"evaluation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\n[dim]💾 评估报告已保存至：{report_path.resolve()}[/dim]")


@app.command("benchmark")
def benchmark_command(
    surge_dir: str = typer.Option(
        "D:/PersonalResearch/PapersWS/Paper-MARS/Ref/SurGE/data",
        "--surge-dir", help="SurGE data 目录路径",
    ),
    action: str = typer.Option(
        "compare", "--action", "-a", help="compare (加载baseline+对比) | run (运行MARS+评估) | all (全部)"
    ),
    topics: str = typer.Option(
        "", "--topics", help="逗号分隔的 topic ID 列表（默认全部41个测试topic）"
    ),
) -> None:
    """
    SurGE Benchmark: MARS vs AutoSurvey / ID / Naive 在 41 个 topic 上的对比评估
    """
    from mars.evaluation.benchmark_runner import BenchmarkRunner

    _print_banner(console)
    console.print("\n[bold green]📊 SurGE Benchmark[/bold green]\n")

    runner = BenchmarkRunner(surge_dir=surge_dir)

    # Parse topics
    topic_list = None
    if topics:
        topic_list = [t.strip() for t in topics.split(",") if t.strip()]

    if action in ("compare", "all"):
        console.print("[bold]加载 SurGE Baselines...[/bold]")
        runner.load_baselines()
        for name, br in runner.results.items():
            console.print(
                f"  {name}: {br.num_topics} topics, "
                f"Recall={br.avg_recall:.4f}, F1={br.avg_f1:.4f}"
            )

    if action in ("run", "all"):
        console.print("\n[bold]运行 MARS (surge mode)...[/bold]")
        runner.run_mars(corpus_mode="surge", topics=topic_list)
        mars_result = runner.results.get("MARS")
        if mars_result:
            console.print(
                f"\n[bold cyan]MARS: {mars_result.num_topics} topics, "
                f"Recall={mars_result.avg_recall:.4f}, F1={mars_result.avg_f1:.4f}[/bold cyan]"
            )

    if runner.results:
        console.print("\n[bold]Comparison Table:[/bold]")
        console.print(runner.compare())
        out = runner.save_results()
        console.print(f"\n[dim]💾 结果保存至: {out}[/dim]")


# ---------------------------------------------------------------------------
# Schedule subcommand group
# ---------------------------------------------------------------------------

schedule_app = typer.Typer(help="定时监控新论文", add_completion=False)
app.add_typer(schedule_app, name="schedule")


@schedule_app.command("add")
def schedule_add(
    topic: str = typer.Argument(..., help="研究主题"),
    interval_hours: int = typer.Option(24, "--every", "-e", help="间隔小时数"),
    at_time: str = typer.Option("09:00", "--at", help="每天执行时间 (HH:MM)"),
    max_papers: int = typer.Option(50, "--max", "-n", help="每次检索论文数"),
) -> None:
    """添加定时研究主题"""
    from mars.scheduler import Scheduler

    sched = Scheduler()
    sched.add(topic, interval_hours=interval_hours, at_time=at_time, max_papers=max_papers)
    console.print(f"[green]✅ 已添加: '{topic}' — 每 {interval_hours}h 执行[/green]")


@schedule_app.command("remove")
def schedule_remove(
    topic: str = typer.Argument(..., help="研究主题"),
) -> None:
    """删除定时研究主题"""
    from mars.scheduler import Scheduler

    sched = Scheduler()
    if sched.remove(topic):
        console.print(f"[green]✅ 已删除: '{topic}'[/green]")
    else:
        console.print(f"[yellow]⚠ 未找到: '{topic}'[/yellow]")


@schedule_app.command("list")
def schedule_list() -> None:
    """列出所有定时研究主题"""
    from mars.scheduler import Scheduler

    sched = Scheduler()
    entries = sched.list_all()
    if not entries:
        console.print("[dim]暂无定时任务。使用 'mars schedule add <topic>' 添加。[/dim]")
        return

    table = Table(title="定时研究任务")
    table.add_column("主题", style="cyan")
    table.add_column("间隔", style="white")
    table.add_column("上次执行", style="dim")
    table.add_column("状态", style="bold")
    for e in entries:
        table.add_row(
            e.topic[:60],
            f"每 {e.interval_hours}h @ {e.at_time}",
            e.last_run[:16] if e.last_run else "从未",
            "[green]启用[/green]" if e.enabled else "[red]禁用[/red]",
        )
    console.print(table)


@schedule_app.command("run")
def schedule_run() -> None:
    """手动触发所有到期的定时任务"""
    from mars.scheduler import Scheduler

    _print_banner(console)
    console.print("[bold green]⏰ 执行定时任务...[/bold green]\n")
    sched = Scheduler()
    results = sched.run_due()
    if not results:
        console.print("[dim]无到期任务。[/dim]")
    for r in results:
        icon = "✅" if r["status"] == "success" else "❌"
        console.print(f"  {icon} {r['topic']}")


@schedule_app.command("serve")
def schedule_serve(
    check_interval: int = typer.Option(60, "--interval", "-i", help="检查间隔（秒）"),
) -> None:
    """启动后台调度守护进程（Ctrl+C 停止）"""
    from mars.scheduler import Scheduler

    _print_banner(console)
    console.print(
        f"[bold green]🔄 启动调度守护进程...[/bold green] "
        f"(每 {check_interval}s 检查一次)"
    )
    console.print("[dim]按 Ctrl+C 停止[/dim]\n")
    sched = Scheduler()
    sched.serve(check_interval=check_interval)


@app.command("latex")
def latex_command(
    input_path: str = typer.Argument(..., help="输入的 Markdown 综述文件路径"),
    output_path: str = typer.Option("", "--output", "-o", help="输出 LaTeX 文件路径"),
    title: str = typer.Option("Automated Literature Review", "--title", help="文档标题"),
    author: str = typer.Option("MARS", "--author", help="作者"),
) -> None:
    """
    将 Markdown 综述转换为 LaTeX 学术论文格式
    """
    from mars.utils.latex_converter import md_to_latex

    input_file = Path(input_path)
    if not input_file.is_file():
        console.print(f"[red]❌ 文件不存在：{input_path}[/red]")
        raise typer.Exit(1)

    md_text = input_file.read_text(encoding="utf-8")
    latex = md_to_latex(md_text, title=title, author=author)

    out = Path(output_path) if output_path else input_file.with_suffix(".tex")
    out.write_text(latex, encoding="utf-8")
    console.print(f"[green]✅ LaTeX 已生成：{out.resolve()}[/green]")


@app.command("check")
def check_command() -> None:
    """
    检查系统配置状态：API Key、LLM 供应商、输出目录等
    """
    _print_banner(console)
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
