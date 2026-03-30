"""
FastAPI application for MARS.

Provides a REST API interface alongside the CLI.
Supports search, analysis, connection, and full-research workflows.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mars.config.settings import settings

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


class TaskResponse(BaseModel):
    """Generic response wrapper."""
    status: str = "success"
    result: str = ""


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="MARS API",
        description="Multi-Agent Research System – 多智能体学术文献智能检索与分析系统",
        version="0.1.0",
    )

    # ---- CORS ----
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- Health check ----
    @app.get("/health")
    def health_check():
        return {"status": "ok"}

    # ---- Search ----
    @app.post("/search", response_model=TaskResponse)
    def search_papers(req: SearchRequest):
        from mars.crews.search_crew import run_search
        result = run_search(req.topic, max_results=req.max_results)
        return TaskResponse(result=result)

    # ---- Analysis ----
    @app.post("/analyze", response_model=TaskResponse)
    def analyze_papers(req: AnalyzeRequest):
        from mars.crews.analysis_crew import run_analysis
        result = run_analysis(req.papers_info, max_papers=req.max_papers)
        return TaskResponse(result=result)

    # ---- Connection ----
    @app.post("/connect", response_model=TaskResponse)
    def connect_papers(req: ConnectRequest):
        from mars.crews.connection_crew import run_connection
        result = run_connection(req.papers_info, topic=req.topic)
        return TaskResponse(result=result)

    # ---- Full research ----
    @app.post("/full-research", response_model=TaskResponse)
    def full_research(req: FullResearchRequest):
        from mars.crews.full_research_crew import run_full_research
        result = run_full_research(req.topic)
        return TaskResponse(result=result)

    return app


# Convenience: module-level ``app`` object so ``uvicorn mars.api.main:app``
# works out of the box.
app = create_app()
