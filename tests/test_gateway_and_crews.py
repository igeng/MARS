"""
Tests for LLM gateway and crew orchestration.
"""

from __future__ import annotations

import pytest


# ============================================================================
# get_llm_by_task tests
# ============================================================================

class TestLLMProvider:
    def test_valid_roles_return_llm(self) -> None:
        from mars.services.llm_gateway import get_llm_by_task

        for role in ["researcher", "searcher", "analyzer", "connector", "summarizer", "evaluator"]:
            llm = get_llm_by_task(role)
            assert llm is not None, f"get_llm_by_task({role!r}) returned None"
            assert "deepseek" in llm.model

    def test_invalid_role_raises(self) -> None:
        from mars.services.llm_gateway import get_llm_by_task

        with pytest.raises(ValueError, match="Unknown agent role"):
            get_llm_by_task("janitor")


# ============================================================================
# Crew orchestration tests
# ============================================================================

class TestCrewOrchestration:
    def test_full_research_crew_sequential(self) -> None:
        from mars.crews.full_research_crew import create_full_research_crew

        crew = create_full_research_crew("test topic")
        assert len(crew.agents) == 6
        assert len(crew.tasks) == 8

    def test_full_research_crew_task_order(self) -> None:
        from mars.crews.full_research_crew import create_full_research_crew
        from crewai import Process

        crew = create_full_research_crew("test topic")
        assert crew.process == Process.sequential

    def test_search_crew_creation(self) -> None:
        from mars.crews.search_crew import create_search_crew

        crew = create_search_crew("federated learning", max_results=10)
        assert len(crew.agents) >= 3
        assert len(crew.tasks) >= 3

    def test_analysis_crew_creation(self) -> None:
        from mars.crews.analysis_crew import create_analysis_crew

        crew = create_analysis_crew("paper1 | paper2", topic="ML")
        assert crew is not None

    def test_connection_crew_creation(self) -> None:
        from mars.crews.connection_crew import create_connection_crew

        crew = create_connection_crew("p1 | p2", topic="security")
        assert crew is not None


# ============================================================================
# CCF data loading tests
# ============================================================================

class TestCCFDataLoading:
    def test_ccf_database_is_non_empty(self) -> None:
        from mars.tools.ccf_database import CCF_DATABASE
        assert len(CCF_DATABASE) > 10

    def test_load_ccf_data_fallback_on_missing_file(self) -> None:
        from mars.tools.ccf_database import _load_ccf_data
        data = _load_ccf_data(version="9999")
        assert len(data) > 0

    def test_ccf_query_tool_returns_results(self) -> None:
        from mars.tools.ccf_database import CCFDatabaseQueryTool
        tool = CCFDatabaseQueryTool()
        result = tool._run("machine learning")
        assert "ICML" in result or "NeurIPS" in result or "Found" in result


# ============================================================================
# Prompt template loading tests
# ============================================================================

class TestPromptTemplates:
    def test_all_templates_load(self) -> None:
        from mars.tasks.task_definitions import _load_prompt

        templates = [
            "domain_analysis_task", "paper_search_task", "deep_analysis_task",
            "connection_analysis_task", "english_review_task",
            "review_generation_task", "quality_evaluation_task",
            "full_research_synthesis_task", "refinement_task",
            "outline_generation_task",
        ]
        for name in templates:
            desc, exp = _load_prompt(name)
            assert desc, f"Template '{name}' has empty description"

    def test_format_substitution_works(self) -> None:
        from mars.tasks.task_definitions import _load_prompt

        desc, _ = _load_prompt("paper_search_task")
        formatted = desc.format(topic="federated learning", max_papers=42, pool_size=200)
        assert "federated learning" in formatted
        assert "42" in formatted
        assert "200" in formatted
