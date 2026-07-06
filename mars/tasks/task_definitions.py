"""
Reusable task factory functions for all 6 MARS agents.

Each ``create_*_task`` function returns a ``crewai.Task`` that can be
plugged into any Crew.  Keeping task definitions separate from crew
assembly makes it easy to compose new workflows.

Prompt text lives in ``mars/tasks/prompts/*.txt`` so non-developers can
iterate on the wording without touching Python code.  Templates use
Python :meth:`str.format` placeholders (``{topic}``, ``{max_papers}``,
``{limit}``, ``{papers_block}``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from crewai import Agent, Task

# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load_prompt(name: str) -> tuple[str, str]:
    """Return ``(description, expected_output)`` for template *name*.

    Each ``.txt`` file contains the description block, a ``---`` separator
    line, and an ``expected_output: |`` block.  The return values still
    contain ``{placeholder}`` markers — callers must format them.
    """
    path = _PROMPTS_DIR / f"{name}.txt"
    if not path.is_file():
        # Graceful fallback: return empty strings so the calling factory
        # function can provide a default or detect the issue.
        return "", ""

    raw = path.read_text(encoding="utf-8")
    parts = raw.split("\n---\n", 1)
    description = parts[0].strip()
    expected = ""
    if len(parts) == 2:
        # Strip the "expected_output: |" header
        expected_lines = parts[1].strip().split("\n", 1)
        expected = expected_lines[1].strip() if len(expected_lines) == 2 else ""
    return description, expected


# ---------------------------------------------------------------------------
# 1. Researcher Agent tasks
# ---------------------------------------------------------------------------

def create_domain_analysis_task(
    agent: Agent,
    topic: str,
    *,
    context: list[Task] | None = None,
) -> Task:
    """Task: analyse a research topic, identify CCF venues."""
    desc, exp = _load_prompt("domain_analysis_task")
    return Task(
        description=desc.format(topic=topic),
        expected_output=exp.format(topic=topic),
        agent=agent,
        context=context or [],
    )


# ---------------------------------------------------------------------------
# 2. Searcher Agent tasks
# ---------------------------------------------------------------------------

def create_paper_search_task(
    agent: Agent,
    topic: str,
    max_papers: int = 50,
    *,
    pool_size: int | None = None,
    context: list[Task] | None = None,
) -> Task:
    """Task: search academic databases for papers on a topic."""
    from mars.config import settings
    pool = pool_size or settings.SEARCH_POOL_SIZE
    desc, exp = _load_prompt("paper_search_task")
    return Task(
        description=desc.format(topic=topic, max_papers=max_papers, pool_size=pool),
        expected_output=exp.format(max_papers=max_papers),
        agent=agent,
        context=context or [],
    )


# ---------------------------------------------------------------------------
# 3. Analyzer Agent tasks
# ---------------------------------------------------------------------------

def create_deep_analysis_task(
    agent: Agent,
    topic: str,
    papers_info: str = "",
    limit: int = 20,
    *,
    context: list[Task] | None = None,
) -> Task:
    """Task: deeply analyse papers — contributions, methods, experiments."""
    papers_block = f"\n\n{papers_info}" if papers_info else ""
    desc, exp = _load_prompt("deep_analysis_task")
    return Task(
        description=desc.format(topic=topic, limit=limit, papers_block=papers_block),
        expected_output=exp.format(limit=limit),
        agent=agent,
        context=context or [],
    )


# ---------------------------------------------------------------------------
# 4. Connector Agent tasks
# ---------------------------------------------------------------------------

def create_connection_analysis_task(
    agent: Agent,
    topic: str,
    papers_info: str = "",
    *,
    context: list[Task] | None = None,
) -> Task:
    """Task: build citation network, topic clusters, research trends; save JSON + Markdown."""
    papers_block = f"\n\n{papers_info}" if papers_info else ""
    desc, exp = _load_prompt("connection_analysis_task")
    return Task(
        description=desc.format(topic=topic, papers_block=papers_block),
        expected_output=exp,
        agent=agent,
        context=context or [],
    )


# ---------------------------------------------------------------------------
# 5. Summarizer Agent tasks
# ---------------------------------------------------------------------------

def create_review_generation_task(
    agent: Agent,
    topic: str,
    *,
    context: list[Task] | None = None,
) -> Task:
    """Task: translate the English literature review into Chinese."""
    desc, exp = _load_prompt("review_generation_task")
    return Task(
        description=desc.format(topic=topic),
        expected_output=exp,
        agent=agent,
        context=context or [],
    )


def create_english_review_task(
    agent: Agent,
    topic: str,
    *,
    context: list[Task] | None = None,
) -> Task:
    """Task: synthesise a rigorous English academic literature review."""
    desc, exp = _load_prompt("english_review_task")
    return Task(
        description=desc.format(topic=topic),
        expected_output=exp,
        agent=agent,
        context=context or [],
    )


# ---------------------------------------------------------------------------
# 5a. Outline generation task (EXP-1.3)
# ---------------------------------------------------------------------------

def create_outline_generation_task(
    agent: Agent,
    topic: str,
    *,
    context: list[Task] | None = None,
) -> Task:
    """Task: generate a validated hierarchical outline before writing the full survey."""
    desc, exp = _load_prompt("outline_generation_task")
    return Task(
        description=desc.format(topic=topic),
        expected_output=exp,
        agent=agent,
        context=context or [],
    )


# ---------------------------------------------------------------------------
# 5b. Refinement task (iterative improvement loop)
# ---------------------------------------------------------------------------

def create_refinement_task(
    agent: Agent,
    topic: str,
    previous_survey: str,
    evaluation_feedback: str,
    output_filename: str = "review_en.md",
    *,
    context: list[Task] | None = None,
) -> Task:
    """Task: revise a survey draft based on LLM-as-Judge evaluation feedback.

    This is the core of the iterative refinement loop (EXP-1.1).
    """
    desc, exp = _load_prompt("refinement_task")
    return Task(
        description=desc.format(
            topic=topic,
            previous_survey=previous_survey,
            evaluation_feedback=evaluation_feedback,
            output_filename=output_filename,
        ),
        expected_output=exp,
        agent=agent,
        context=context or [],
    )


# ---------------------------------------------------------------------------
# 6. Evaluator Agent tasks
# ---------------------------------------------------------------------------

def create_quality_evaluation_task(
    agent: Agent,
    limit: int = 20,
    *,
    context: list[Task] | None = None,
) -> Task:
    """Task: multi-dimensional quality evaluation of papers, producing comprehensive analysis_report.md."""
    desc, exp = _load_prompt("quality_evaluation_task")
    return Task(
        description=desc.format(limit=limit),
        expected_output=exp,
        agent=agent,
        context=context or [],
    )


# ---------------------------------------------------------------------------
# 7. Synthesizer tasks (full research workflow only)
# ---------------------------------------------------------------------------

def create_full_research_synthesis_task(
    agent: Agent,
    topic: str,
    *,
    context: list[Task] | None = None,
) -> Task:
    """Task: generate a comprehensive final synthesis report for the full research workflow.

    This is distinct from review_zh.md (Chinese translation) — it provides an
    executive-level synthesis combining statistics, quality rankings, connection
    insights, and actionable future directions in a single authoritative document.
    """
    desc, exp = _load_prompt("full_research_synthesis_task")
    return Task(
        description=desc.format(topic=topic),
        expected_output=exp,
        agent=agent,
        context=context or [],
    )
