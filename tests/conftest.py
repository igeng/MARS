"""
Shared pytest fixtures for MARS tests.

All fixtures here are automatically available to every test module
without requiring an explicit import.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mars.config.settings import settings


@pytest.fixture()
def output_dir(tmp_path: Path):
    """Temporarily redirect ``settings.OUTPUT_DIR`` to a temp directory."""
    original = settings.OUTPUT_DIR
    settings.OUTPUT_DIR = tmp_path
    yield tmp_path
    settings.OUTPUT_DIR = original
