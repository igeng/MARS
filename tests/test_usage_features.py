"""
Tests for features added to close the "normal usage" gap:
- Structured logging (setup_logging)
- Retry decorator (retry_on_network_error)
- LLM provider fallback (get_available_providers, _resolve_provider)
- CLI check command
- API error handlers
"""

from __future__ import annotations

import logging
import time
from unittest.mock import MagicMock, patch

import pytest
import requests


# ---------------------------------------------------------------------------
# Logging configuration tests
# ---------------------------------------------------------------------------

class TestLoggingConfig:
    def test_setup_logging_callable(self) -> None:
        from mars.utils.logging_config import setup_logging
        assert callable(setup_logging)

    def test_setup_logging_configures_root_logger(self) -> None:
        import mars.utils.logging_config as lc
        # Reset so we can test fresh
        lc._CONFIGURED = False
        try:
            lc.setup_logging()
            root = logging.getLogger()
            assert root.level <= logging.INFO
        finally:
            lc._CONFIGURED = False

    def test_setup_logging_idempotent(self) -> None:
        import mars.utils.logging_config as lc
        lc._CONFIGURED = False
        try:
            lc.setup_logging()
            handler_count = len(logging.getLogger().handlers)
            lc.setup_logging()  # second call should be a no-op
            assert len(logging.getLogger().handlers) == handler_count
        finally:
            lc._CONFIGURED = False


# ---------------------------------------------------------------------------
# Retry decorator tests
# ---------------------------------------------------------------------------

class TestRetryDecorator:
    def test_no_retry_on_success(self) -> None:
        from mars.utils.retry import retry_on_network_error

        call_count = 0

        @retry_on_network_error(max_retries=3, base_delay=0.01)
        def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = succeed()
        assert result == "ok"
        assert call_count == 1

    def test_retries_on_request_exception(self) -> None:
        from mars.utils.retry import retry_on_network_error

        call_count = 0

        @retry_on_network_error(max_retries=2, base_delay=0.01)
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise requests.ConnectionError("network down")
            return "recovered"

        result = flaky()
        assert result == "recovered"
        assert call_count == 3

    def test_raises_after_max_retries(self) -> None:
        from mars.utils.retry import retry_on_network_error

        @retry_on_network_error(max_retries=2, base_delay=0.01)
        def always_fail():
            raise requests.ConnectionError("always fails")

        with pytest.raises(requests.ConnectionError, match="always fails"):
            always_fail()

    def test_no_retry_on_non_request_exception(self) -> None:
        from mars.utils.retry import retry_on_network_error

        @retry_on_network_error(max_retries=3, base_delay=0.01)
        def type_error():
            raise TypeError("not a network error")

        with pytest.raises(TypeError, match="not a network error"):
            type_error()


# ---------------------------------------------------------------------------
# LLM provider fallback tests
# ---------------------------------------------------------------------------

class TestLLMFallback:
    def test_get_available_providers_empty_keys(self) -> None:
        """With all keys empty, no providers should be available."""
        from mars.services.llm_gateway import get_available_providers
        from mars.config import settings

        orig = {
            "DASHSCOPE_API_KEY": settings.DASHSCOPE_API_KEY,
            "DEEPSEEK_API_KEY": settings.DEEPSEEK_API_KEY,
            "MOONSHOT_API_KEY": settings.MOONSHOT_API_KEY,
            "ZHIPU_API_KEY": settings.ZHIPU_API_KEY,
        }
        try:
            settings.DASHSCOPE_API_KEY = ""
            settings.DEEPSEEK_API_KEY = ""
            settings.MOONSHOT_API_KEY = ""
            settings.ZHIPU_API_KEY = ""
            assert get_available_providers() == []
        finally:
            for k, v in orig.items():
                setattr(settings, k, v)

    def test_get_available_providers_with_one_key(self) -> None:
        from mars.services.llm_gateway import get_available_providers
        from mars.config import settings

        orig = {
            "DASHSCOPE_API_KEY": settings.DASHSCOPE_API_KEY,
            "DEEPSEEK_API_KEY": settings.DEEPSEEK_API_KEY,
            "MOONSHOT_API_KEY": settings.MOONSHOT_API_KEY,
            "ZHIPU_API_KEY": settings.ZHIPU_API_KEY,
        }
        try:
            settings.DASHSCOPE_API_KEY = ""
            settings.DEEPSEEK_API_KEY = "sk-test-key"
            settings.MOONSHOT_API_KEY = ""
            settings.ZHIPU_API_KEY = ""
            providers = get_available_providers()
            assert "deepseek" in providers
            assert "qwen" not in providers
        finally:
            for k, v in orig.items():
                setattr(settings, k, v)

    def test_fallback_to_available_provider(self) -> None:
        """When preferred provider has no key, fall back to one that does."""
        from mars.services.llm_gateway import get_llm
        from mars.config import settings

        orig = {
            "DASHSCOPE_API_KEY": settings.DASHSCOPE_API_KEY,
            "DEEPSEEK_API_KEY": settings.DEEPSEEK_API_KEY,
            "MOONSHOT_API_KEY": settings.MOONSHOT_API_KEY,
            "ZHIPU_API_KEY": settings.ZHIPU_API_KEY,
        }
        try:
            settings.DASHSCOPE_API_KEY = ""
            settings.DEEPSEEK_API_KEY = "sk-test-key"
            settings.MOONSHOT_API_KEY = ""
            settings.ZHIPU_API_KEY = ""
            # Request qwen, but only deepseek has a key
            llm = get_llm(provider="qwen")
            assert llm is not None
        finally:
            for k, v in orig.items():
                setattr(settings, k, v)

    def test_get_llm_by_task_with_fallback(self) -> None:
        """All agent roles should work with only one provider key."""
        from mars.services.llm_gateway import get_llm_by_task
        from mars.config import settings

        orig = {
            "DASHSCOPE_API_KEY": settings.DASHSCOPE_API_KEY,
            "DEEPSEEK_API_KEY": settings.DEEPSEEK_API_KEY,
            "MOONSHOT_API_KEY": settings.MOONSHOT_API_KEY,
            "ZHIPU_API_KEY": settings.ZHIPU_API_KEY,
        }
        try:
            settings.DASHSCOPE_API_KEY = ""
            settings.DEEPSEEK_API_KEY = "sk-test-key"
            settings.MOONSHOT_API_KEY = ""
            settings.ZHIPU_API_KEY = ""
            # All roles should succeed via fallback
            for role in ("researcher", "searcher", "analyzer",
                         "connector", "summarizer", "evaluator"):
                llm = get_llm_by_task(role)
                assert llm is not None
        finally:
            for k, v in orig.items():
                setattr(settings, k, v)


# ---------------------------------------------------------------------------
# CLI check command tests
# ---------------------------------------------------------------------------

class TestCLICheckCommand:
    def test_check_command_exists(self) -> None:
        from mars.cli import check_command
        assert callable(check_command)

    def test_check_command_runs_without_error(self) -> None:
        """Smoke test: check command should not crash."""
        from typer.testing import CliRunner
        from mars.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["check"])
        # May exit 1 if no keys, but should not crash
        assert result.exit_code in (0, 1)
        assert "LLM" in result.output or "供应商" in result.output


# ---------------------------------------------------------------------------
# API error handler tests
# ---------------------------------------------------------------------------

class TestAPIErrorHandlers:
    def test_health_still_works(self) -> None:
        from fastapi.testclient import TestClient
        from mars.api.main import app

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_value_error_returns_400(self) -> None:
        from fastapi.testclient import TestClient
        from mars.api.main import create_app

        app = create_app()

        @app.get("/test-value-error")
        def raise_value_error():
            raise ValueError("bad input")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/test-value-error")
        assert resp.status_code == 400
        data = resp.json()
        assert data["status"] == "error"
        assert "bad input" in data["result"]

    def test_unhandled_exception_returns_500(self) -> None:
        from fastapi.testclient import TestClient
        from mars.api.main import create_app

        app = create_app()

        @app.get("/test-crash")
        def raise_runtime():
            raise RuntimeError("unexpected")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/test-crash")
        assert resp.status_code == 500
        data = resp.json()
        assert data["status"] == "error"

    def test_api_app_has_lifespan(self) -> None:
        """The app should have a lifespan (startup/shutdown handler)."""
        from mars.api.main import app
        assert app.router.lifespan_context is not None


# ---------------------------------------------------------------------------
# Retry applied to tools tests
# ---------------------------------------------------------------------------

class TestToolRetry:
    def test_dblp_search_uses_retry(self) -> None:
        """DBLP search should retry on network errors."""
        from mars.tools.dblp_search import DBLPSearchTool

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise requests.ConnectionError("network down")
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"result": {"hits": {"hit": []}}}
            mock_resp.raise_for_status.return_value = None
            return mock_resp

        tool = DBLPSearchTool()
        with patch("mars.tools.dblp_search.requests.get", side_effect=side_effect):
            result = tool._run('{"query": "test"}')

        assert call_count == 3
        assert "No results found" in result

    def test_semantic_scholar_uses_retry(self) -> None:
        """Semantic Scholar search should retry on network errors."""
        from mars.tools.semantic_scholar import SemanticScholarSearchTool

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise requests.ConnectionError("timeout")
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"data": []}
            mock_resp.raise_for_status.return_value = None
            return mock_resp

        tool = SemanticScholarSearchTool()
        with patch("mars.tools.semantic_scholar.requests.get", side_effect=side_effect):
            result = tool._run('{"query": "test"}')

        assert call_count == 2
        assert "No results found" in result

    def test_arxiv_uses_retry(self) -> None:
        """arXiv search should retry on network errors."""
        from mars.tools.arxiv_api import ArXivSearchTool

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise requests.ConnectionError("dns failure")
            mock_resp = MagicMock()
            mock_resp.text = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
            mock_resp.raise_for_status.return_value = None
            return mock_resp

        tool = ArXivSearchTool()
        with patch("mars.tools.arxiv_api.requests.get", side_effect=side_effect):
            result = tool._run('{"query": "test"}')

        assert call_count == 2
        assert "No arXiv results found" in result
