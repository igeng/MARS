"""
Tests for LLM gateway fallback logic and crew orchestration.

These tests use heavy mocking so they never touch the network.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# RateLimitAwareLLM tests
# ============================================================================

class TestRateLimitAwareLLM:
    """Verify retry / fallback behaviour of RateLimitAwareLLM without
    touching a real LLM backend."""

    def test_construction_with_api_key(self) -> None:
        from mars.services.llm_gateway import RateLimitAwareLLM

        llm = RateLimitAwareLLM(
            model="openai/test-model",
            api_key="sk-test",
            base_url="https://test.example.com/v1",
        )
        assert llm.model == "openai/test-model"

    def test_construction_without_api_key(self) -> None:
        from mars.services.llm_gateway import RateLimitAwareLLM

        llm = RateLimitAwareLLM(
            model="openai/test-model",
            base_url="https://test.example.com/v1",
            is_litellm=True,
        )
        assert llm.model == "openai/test-model"

    def test_successful_call_no_retry(self) -> None:
        from mars.services.llm_gateway import RateLimitAwareLLM

        llm = RateLimitAwareLLM(
            model="openai/test-model",
            api_key="sk-test",
            base_url="https://test.example.com/v1",
        )
        # Patch the parent class call to return a fake response
        with patch.object(
            type(llm).__bases__[0], "call", return_value="ok"
        ) as mock_super:
            result = llm.call("messages", "tools", "callbacks", "functions")
            assert result == "ok"
            mock_super.assert_called_once()

    def test_retries_on_rate_limit_then_succeeds(self) -> None:
        from mars.services.llm_gateway import RateLimitAwareLLM
        from litellm.exceptions import RateLimitError

        llm = RateLimitAwareLLM(
            model="openai/test-model",
            api_key="sk-test",
            base_url="https://test.example.com/v1",
            max_retries=3,
            retry_base_delay=0.001,  # tiny delay to keep tests fast
        )

        call_count = [0]

        def _side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise RateLimitError("rate limited", llm_provider="test", model="test-model")
            return "success after retry"

        with patch.object(type(llm).__bases__[0], "call", side_effect=_side_effect):
            result = llm.call("msg")
            assert result == "success after retry"
            assert call_count[0] == 3  # 2 failures + 1 success

    def test_falls_back_after_exhausting_retries(self) -> None:
        from mars.services.llm_gateway import RateLimitAwareLLM
        from litellm.exceptions import RateLimitError

        fallback = MagicMock()
        fallback.model = "openai/fallback-model"
        fallback.call.return_value = "fallback result"

        llm = RateLimitAwareLLM(
            model="openai/primary-model",
            api_key="sk-test",
            base_url="https://test.example.com/v1",
            fallback_llms=[fallback],
            max_retries=1,
            retry_base_delay=0.001,
        )

        with patch.object(
            type(llm).__bases__[0], "call",
            side_effect=RateLimitError("rate limited", llm_provider="test", model="test-model"),
        ):
            result = llm.call("msg")
            assert result == "fallback result"
            fallback.call.assert_called_once()

    def test_raises_when_all_fallbacks_exhausted(self) -> None:
        from mars.services.llm_gateway import RateLimitAwareLLM
        from litellm.exceptions import RateLimitError

        fallback = MagicMock()
        fallback.model = "openai/fallback-model"
        fallback.call.side_effect = RateLimitError(
            "fallback exhausted", llm_provider="test", model="test-model"
        )

        llm = RateLimitAwareLLM(
            model="openai/primary-model",
            api_key="sk-test",
            base_url="https://test.example.com/v1",
            fallback_llms=[fallback],
            max_retries=1,
            retry_base_delay=0.001,
        )

        with patch.object(
            type(llm).__bases__[0], "call",
            side_effect=RateLimitError("rate limited", llm_provider="test", model="test-model"),
        ):
            with pytest.raises(RateLimitError):
                llm.call("msg")

    def test_call_passes_args_and_kwargs_through(self) -> None:
        from mars.services.llm_gateway import RateLimitAwareLLM

        llm = RateLimitAwareLLM(
            model="openai/test-model",
            api_key="sk-test",
            base_url="https://test.example.com/v1",
        )

        with patch.object(type(llm).__bases__[0], "call", return_value="ok") as mock_super:
            llm.call("a", "b", extra="c", foo="bar")
            mock_super.assert_called_once_with("a", "b", extra="c", foo="bar")


# ============================================================================
# get_llm_by_task provider resolution tests
# ============================================================================

class TestLLMProviderResolution:
    """Ensure that get_llm_by_task returns the right provider and falls back
    when keys are missing."""

    def test_valid_roles_return_llm(self) -> None:
        from mars.services.llm_gateway import get_llm_by_task

        for role in ["researcher", "searcher", "analyzer", "connector", "summarizer", "evaluator"]:
            llm = get_llm_by_task(role)
            assert llm is not None, f"get_llm_by_task({role!r}) returned None"

    def test_invalid_role_raises(self) -> None:
        from mars.services.llm_gateway import get_llm_by_task

        with pytest.raises(ValueError, match="Unknown agent role"):
            get_llm_by_task("janitor")

    def test_glm_fallback_construction(self) -> None:
        from mars.services.llm_gateway import _get_glm_with_fallbacks

        llm = _get_glm_with_fallbacks()
        assert llm is not None
        # When no GLM key is configured, should still construct via LiteLLM path
        assert "glm" in llm.model.lower() or "openai" in llm.model.lower()


# ============================================================================
# Crew orchestration tests
# ============================================================================

class TestCrewOrchestration:
    """Verify that crew factory functions wire agents + tasks correctly."""

    def test_full_research_crew_sequential(self) -> None:
        from mars.crews.full_research_crew import create_full_research_crew

        crew = create_full_research_crew("test topic")
        assert len(crew.agents) == 6
        assert len(crew.tasks) == 8

    def test_full_research_crew_task_order(self) -> None:
        from mars.crews.full_research_crew import create_full_research_crew
        from crewai import Process

        crew = create_full_research_crew("test topic")
        # Verify process mode is sequential
        assert crew.process == Process.sequential

    def test_search_crew_creation(self) -> None:
        from mars.crews.search_crew import create_search_crew

        crew = create_search_crew("federated learning", max_results=10)
        assert len(crew.agents) >= 3  # researcher, searcher, summarizer
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
    """Verify that CCF data loads from JSON and falls back correctly."""

    def test_ccf_database_is_non_empty(self) -> None:
        from mars.tools.ccf_database import CCF_DATABASE
        assert len(CCF_DATABASE) > 10, "CCF database should contain >10 venues"

    def test_load_ccf_data_fallback_on_missing_file(self) -> None:
        from mars.tools.ccf_database import _load_ccf_data
        data = _load_ccf_data(version="9999")  # nonexistent version
        assert len(data) > 0, "Fallback data should be non-empty"

    def test_ccf_query_tool_returns_results(self) -> None:
        from mars.tools.ccf_database import CCFDatabaseQueryTool

        tool = CCFDatabaseQueryTool()
        result = tool._run("machine learning")
        assert "ICML" in result or "NeurIPS" in result or "Found" in result


# ============================================================================
# Prompt template loading tests
# ============================================================================

class TestPromptTemplates:
    """Verify that prompt templates load correctly."""

    def test_all_templates_load(self) -> None:
        from mars.tasks.task_definitions import _load_prompt

        templates = [
            "domain_analysis_task",
            "paper_search_task",
            "deep_analysis_task",
            "connection_analysis_task",
            "english_review_task",
            "review_generation_task",
            "quality_evaluation_task",
            "full_research_synthesis_task",
        ]
        for name in templates:
            desc, exp = _load_prompt(name)
            assert desc, f"Template '{name}' has empty description"
            # Some templates may not have expected_output — that's ok

    def test_format_substitution_works(self) -> None:
        from mars.tasks.task_definitions import _load_prompt

        desc, _ = _load_prompt("paper_search_task")
        formatted = desc.format(topic="federated learning", max_papers=42, pool_size=200)
        assert "federated learning" in formatted
        assert "42" in formatted
        assert "200" in formatted
