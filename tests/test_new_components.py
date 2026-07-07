"""
Tests for new components added during the code review:
- Pydantic BaseSettings config
- LLM Gateway service (get_llm_by_task)
- Database models (SQLAlchemy)
- FastAPI application
- Task definitions module
"""

from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# Settings (Pydantic BaseSettings) tests
# ---------------------------------------------------------------------------

class TestPydanticSettings:
    def test_settings_is_base_settings(self) -> None:
        from pydantic_settings import BaseSettings
        from mars.config.settings import MarsSettings
        assert issubclass(MarsSettings, BaseSettings)

    def test_settings_singleton(self) -> None:
        from mars.config.settings import settings
        from mars.config import settings as settings2
        assert settings is settings2

    def test_settings_has_database_url(self) -> None:
        from mars.config.settings import settings
        assert hasattr(settings, "DATABASE_URL")
        assert "sqlite" in settings.DATABASE_URL or "postgres" in settings.DATABASE_URL

    def test_settings_has_api_host(self) -> None:
        from mars.config.settings import settings
        assert hasattr(settings, "API_HOST")
        assert hasattr(settings, "API_PORT")

    def test_settings_max_papers_positive(self) -> None:
        from mars.config.settings import settings
        assert settings.MAX_PAPERS_PER_SEARCH > 0
        assert settings.MAX_PAPERS_FOR_ANALYSIS > 0


# ---------------------------------------------------------------------------
# LLM Gateway tests
# ---------------------------------------------------------------------------

class TestLLMGateway:
    def test_get_llm_by_task_researcher(self) -> None:
        from mars.services.llm_gateway import get_llm_by_task
        llm = get_llm_by_task("researcher")
        assert llm is not None

    def test_get_llm_by_task_searcher(self) -> None:
        from mars.services.llm_gateway import get_llm_by_task
        llm = get_llm_by_task("searcher")
        assert llm is not None

    def test_get_llm_by_task_analyzer(self) -> None:
        from mars.services.llm_gateway import get_llm_by_task
        llm = get_llm_by_task("analyzer")
        assert llm is not None

    def test_get_llm_by_task_connector(self) -> None:
        from mars.services.llm_gateway import get_llm_by_task
        llm = get_llm_by_task("connector")
        assert llm is not None

    def test_get_llm_by_task_summarizer(self) -> None:
        from mars.services.llm_gateway import get_llm_by_task
        llm = get_llm_by_task("summarizer")
        assert llm is not None

    def test_get_llm_by_task_evaluator(self) -> None:
        from mars.services.llm_gateway import get_llm_by_task
        llm = get_llm_by_task("evaluator")
        assert llm is not None

    def test_get_llm_by_task_invalid_role(self) -> None:
        from mars.services.llm_gateway import get_llm_by_task
        with pytest.raises(ValueError, match="Unknown agent role"):
            get_llm_by_task("nonexistent_role")

    def test_get_llm_returns_object(self) -> None:
        from mars.services.llm_gateway import get_llm
        llm = get_llm()
        assert llm is not None

    def test_re_export_compatibility(self) -> None:
        """utils/llm_factory should re-export from services/llm_gateway."""
        from mars.utils.llm_factory import get_llm_by_task, get_llm
        assert callable(get_llm_by_task)
        assert callable(get_llm)


# ---------------------------------------------------------------------------
# Database models tests
# ---------------------------------------------------------------------------

class TestDatabaseModels:
    def test_ccf_venue_model_exists(self) -> None:
        from mars.database.models import CCFVenue
        assert CCFVenue.__tablename__ == "ccf_venues"

    def test_ccf_venue_columns(self) -> None:
        from mars.database.models import CCFVenue
        column_names = [c.name for c in CCFVenue.__table__.columns]
        assert "name" in column_names
        assert "full_name" in column_names
        assert "ccf_rank" in column_names
        assert "venue_type" in column_names
        assert "domains" in column_names
        assert "dblp_url" in column_names

    def test_paper_model_exists(self) -> None:
        from mars.database.models import Paper
        assert Paper.__tablename__ == "papers"

    def test_paper_columns(self) -> None:
        from mars.database.models import Paper
        column_names = [c.name for c in Paper.__table__.columns]
        assert "title" in column_names
        assert "authors" in column_names
        assert "venue" in column_names
        assert "year" in column_names
        assert "citation_count" in column_names
        assert "semantic_scholar_id" in column_names

    def test_init_db_creates_tables(self, tmp_path) -> None:
        from mars.database.models import Base, init_db, get_engine
        from mars.config.settings import settings

        original_url = settings.DATABASE_URL
        settings.DATABASE_URL = f"sqlite:///{tmp_path / 'test.db'}"
        # Reset cached engine
        import mars.database.models as db_mod
        db_mod._engine = None
        db_mod._SessionLocal = None
        try:
            init_db()
            engine = get_engine()
            from sqlalchemy import inspect
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            assert "ccf_venues" in tables
            assert "papers" in tables
        finally:
            settings.DATABASE_URL = original_url
            db_mod._engine = None
            db_mod._SessionLocal = None


# ---------------------------------------------------------------------------
# FastAPI application tests
# ---------------------------------------------------------------------------

class TestFastAPIApp:
    def test_app_is_importable(self) -> None:
        from mars.api.main import app
        assert app is not None

    def test_create_app_factory(self) -> None:
        from mars.api.main import create_app
        app = create_app()
        assert app.title == "MARS API"

    def test_health_endpoint(self) -> None:
        from fastapi.testclient import TestClient
        from mars.api.main import app

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_cors_headers(self) -> None:
        from fastapi.testclient import TestClient
        from mars.api.main import app

        client = TestClient(app)
        resp = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.status_code == 200

    def test_search_request_schema(self) -> None:
        from mars.api.main import SearchRequest
        req = SearchRequest(topic="federated learning")
        assert req.topic == "federated learning"
        assert req.max_results == 50

    def test_analyze_request_schema(self) -> None:
        from mars.api.main import AnalyzeRequest
        req = AnalyzeRequest(papers_info="paper1 | paper2")
        assert req.papers_info == "paper1 | paper2"
        assert req.max_papers == 20
        assert req.topic == ""

    def test_analyze_request_schema_with_topic(self) -> None:
        from mars.api.main import AnalyzeRequest
        req = AnalyzeRequest(papers_info="paper1", topic="federated learning")
        assert req.topic == "federated learning"

    def test_full_research_request_schema(self) -> None:
        from mars.api.main import FullResearchRequest
        req = FullResearchRequest(topic="machine learning")
        assert req.topic == "machine learning"

    def test_connect_request_schema(self) -> None:
        from mars.api.main import ConnectRequest
        req = ConnectRequest(papers_info="id1 | id2", topic="ML")
        assert req.papers_info == "id1 | id2"
        assert req.topic == "ML"


# ---------------------------------------------------------------------------
# Task definitions tests
# ---------------------------------------------------------------------------

class TestTaskDefinitions:
    def test_create_domain_analysis_task(self) -> None:
        from mars.tasks.task_definitions import create_domain_analysis_task
        assert callable(create_domain_analysis_task)

    def test_create_paper_search_task(self) -> None:
        from mars.tasks.task_definitions import create_paper_search_task
        assert callable(create_paper_search_task)

    def test_create_deep_analysis_task(self) -> None:
        from mars.tasks.task_definitions import create_deep_analysis_task
        assert callable(create_deep_analysis_task)

    def test_create_connection_analysis_task(self) -> None:
        from mars.tasks.task_definitions import create_connection_analysis_task
        assert callable(create_connection_analysis_task)

    def test_create_review_generation_task(self) -> None:
        from mars.tasks.task_definitions import create_review_generation_task
        assert callable(create_review_generation_task)

    def test_create_quality_evaluation_task(self) -> None:
        from mars.tasks.task_definitions import create_quality_evaluation_task
        assert callable(create_quality_evaluation_task)


# ---------------------------------------------------------------------------
# Code review fixes verification tests
# ---------------------------------------------------------------------------

class TestCodeReviewFixes:
    def test_analysis_crew_accepts_topic(self) -> None:
        """analysis_crew.create_analysis_crew should accept a topic arg."""
        from mars.crews.analysis_crew import create_analysis_crew
        import inspect
        sig = inspect.signature(create_analysis_crew)
        assert "topic" in sig.parameters

    def test_analysis_crew_run_accepts_topic(self) -> None:
        """run_analysis should accept a topic arg."""
        from mars.crews.analysis_crew import run_analysis
        import inspect
        sig = inspect.signature(run_analysis)
        assert "topic" in sig.parameters

    def test_full_research_crew_uses_task_factory(self) -> None:
        """full_research_crew should import create_review_generation_task."""
        import mars.crews.full_research_crew as frc
        assert hasattr(frc, "create_review_generation_task")

    def test_conftest_output_dir_fixture_exists(self) -> None:
        """conftest.py should provide an output_dir fixture."""
        import tests.conftest as conftest
        assert hasattr(conftest, "output_dir")
        assert callable(conftest.output_dir)

    def test_file_write_uses_output_dir_fixture(self, output_dir) -> None:
        """Verify the conftest output_dir fixture works correctly."""
        from mars.config import settings
        assert settings.OUTPUT_DIR == output_dir
