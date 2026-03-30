"""
SQLAlchemy models for MARS database.

Defines the CCF venue table and a paper-cache table so that
previously retrieved papers can be served from a local store.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from mars.config.settings import settings


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# CCF Venue model
# ---------------------------------------------------------------------------

class CCFVenue(Base):
    """CCF recommended international academic venue (conference / journal)."""

    __tablename__ = "ccf_venues"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False, unique=True, index=True)
    full_name = Column(String(256), nullable=False)
    ccf_rank = Column(String(1), nullable=False, index=True)  # A / B / C
    venue_type = Column(String(20), nullable=False)  # conference / journal
    domains = Column(Text, nullable=False)  # comma-separated domain list
    dblp_url = Column(String(512), default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<CCFVenue [{self.ccf_rank}] {self.name}>"


# ---------------------------------------------------------------------------
# Paper cache model
# ---------------------------------------------------------------------------

class Paper(Base):
    """Cached academic paper metadata."""

    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(512), nullable=False, index=True)
    authors = Column(Text, default="")         # JSON list of author names
    venue = Column(String(256), default="")
    year = Column(Integer, nullable=True, index=True)
    citation_count = Column(Integer, default=0)
    doi = Column(String(256), default="")
    url = Column(String(512), default="")
    abstract = Column(Text, default="")
    pdf_url = Column(String(512), default="")
    source = Column(String(50), default="")    # dblp / semantic_scholar / arxiv
    semantic_scholar_id = Column(String(64), default="", index=True)
    relevance_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<Paper {self.title[:50]}>"


# ---------------------------------------------------------------------------
# Engine / session helpers
# ---------------------------------------------------------------------------

_engine = None
_SessionLocal = None


def get_engine():
    """Return (and cache) the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(settings.DATABASE_URL, echo=False)
    return _engine


def get_session() -> Session:
    """Return a new SQLAlchemy session."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine())
    return _SessionLocal()


def init_db() -> None:
    """Create all tables (idempotent)."""
    Base.metadata.create_all(bind=get_engine())
