"""
FastAPI application for MARS.

Provides a REST API interface alongside the CLI.
Workflows are executed asynchronously via background threads:
``POST`` returns a ``task_id`` immediately; poll ``GET /task/{task_id}``
for status and results.
"""

from __future__ import annotations

import logging
import threading
import uuid
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from mars.config.settings import settings
from mars.utils.logging_config import setup_logging, set_run_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-memory task store
# ---------------------------------------------------------------------------

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class TaskInfo(BaseModel):
    task_id: str
    status: TaskStatus
    result: str = ""
    error: str = ""


_task_store: dict[str, TaskInfo] = {}
_store_lock = threading.Lock()


def _create_task() -> TaskInfo:
    task_id = uuid.uuid4().hex[:12]
    info = TaskInfo(task_id=task_id, status=TaskStatus.PENDING)
    with _store_lock:
        _task_store[task_id] = info
    return info


def _run_in_background(task: TaskInfo, target: Any, *args: Any) -> None:
    """Execute *target(*args)* in a daemon thread and update *task* status."""

    def _wrapper() -> None:
        run_id = f"api_{task.task_id}"
        set_run_id(run_id)
        try:
            task.status = TaskStatus.RUNNING
            logger.info("Task %s started.", task.task_id)
            result = target(*args)
            task.result = str(result) if result else ""
            task.status = TaskStatus.SUCCESS
            logger.info("Task %s completed successfully.", task.task_id)
        except Exception as exc:
            task.error = str(exc)
            task.status = TaskStatus.FAILED
            logger.exception("Task %s failed.", task.task_id)
        finally:
            set_run_id(None)

    t = threading.Thread(target=_wrapper, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    """Request body for the /search endpoint."""
    topic: str = Field(..., min_length=1, description="研究主题")
    max_results: int = Field(
        default=50, gt=0, le=200, description="最大检索论文数量"
    )


class AnalyzeRequest(BaseModel):
    """Request body for the /analyze endpoint."""
    papers_info: str = Field(
        ..., min_length=1, description="论文信息（标题列表或描述）"
    )
    topic: str = Field(
        default="", description="研究主题（可选，提供时分析更有针对性）"
    )
    max_papers: int = Field(
        default=20, gt=0, le=100, description="最大深度分析论文数量"
    )


class ConnectRequest(BaseModel):
    """Request body for the /connect endpoint."""
    papers_info: str = Field(
        ..., min_length=1, description="论文信息（标题或ID列表）"
    )
    topic: str = Field(..., min_length=1, description="研究主题")


class FullResearchRequest(BaseModel):
    """Request body for the /full-research endpoint."""
    topic: str = Field(..., min_length=1, description="研究主题")


class TaskAcceptedResponse(BaseModel):
    """Returned when a workflow is accepted for background processing."""
    task_id: str
    status: str = "pending"


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown logic)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Run one-time setup when the application starts."""
    setup_logging()
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("MARS API server starting – output dir: %s", settings.OUTPUT_DIR)
    yield


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="MARS API",
        description="Multi-Agent Research System – 多智能体学术文献智能检索与分析系统",
        version="0.1.0",
        lifespan=_lifespan,
    )

    # ---- CORS (restrict allow_origins in production) ----
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- Global exception handlers ----
    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "result": f"Internal server error: {exc}",
            },
        )

    @app.exception_handler(ValueError)
    async def _value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=400,
            content={"status": "error", "result": str(exc)},
        )

    # ---- Health check ----
    @app.get("/health")
    def health_check():
        return {"status": "ok"}

    # ---- Task status polling ----
    @app.get("/task/{task_id}", response_model=TaskInfo)
    def get_task_status(task_id: str):
        """Poll for the status and result of an async workflow task."""
        with _store_lock:
            info = _task_store.get(task_id)
        if info is None:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        return info

    # ---- Search ----
    @app.post("/search", response_model=TaskAcceptedResponse, status_code=202)
    def search_papers(req: SearchRequest):
        from mars.crews.search_crew import run_search

        task = _create_task()
        _run_in_background(task, run_search, req.topic, req.max_results)
        return TaskAcceptedResponse(task_id=task.task_id)

    # ---- Analysis ----
    @app.post("/analyze", response_model=TaskAcceptedResponse, status_code=202)
    def analyze_papers(req: AnalyzeRequest):
        from mars.crews.analysis_crew import run_analysis

        task = _create_task()
        _run_in_background(
            task, run_analysis, req.papers_info, req.topic, req.max_papers
        )
        return TaskAcceptedResponse(task_id=task.task_id)

    # ---- Connection ----
    @app.post("/connect", response_model=TaskAcceptedResponse, status_code=202)
    def connect_papers(req: ConnectRequest):
        from mars.crews.connection_crew import run_connection

        task = _create_task()
        _run_in_background(task, run_connection, req.papers_info, req.topic)
        return TaskAcceptedResponse(task_id=task.task_id)

    # ---- Full research ----
    @app.post("/full-research", response_model=TaskAcceptedResponse, status_code=202)
    def full_research(req: FullResearchRequest):
        from mars.crews.full_research_crew import run_full_research

        task = _create_task()
        _run_in_background(task, run_full_research, req.topic)
        return TaskAcceptedResponse(task_id=task.task_id)

    return app


# Convenience: module-level ``app`` object so ``uvicorn mars.api.main:app``
# works out of the box.
app = create_app()
