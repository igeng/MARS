"""
File manager tool.

Handles reading and writing of research output files
(JSON results, Markdown reports, etc.) to the configured output directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from crewai.tools import BaseTool

from mars.config import settings


def _ensure_output_dir() -> Path:
    out = settings.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out


class FileWriterTool(BaseTool):
    """Write content to a file in the output directory."""

    name: str = "file_writer"
    description: str = (
        "Write content to a file in the output directory. "
        "Input should be a JSON string with keys: "
        "'filename' (required), "
        "'content' (required, string), "
        "'mode' (optional, 'w' or 'a', default 'w'). "
        "Returns the absolute path of the written file."
    )

    def _run(self, query_json: str) -> str:
        try:
            params: dict[str, Any] = json.loads(query_json)
        except json.JSONDecodeError:
            return "Error: Input must be valid JSON."

        filename: str = params.get("filename", "")
        content: str = params.get("content", "")
        mode: str = params.get("mode", "w")

        if not filename:
            return "Error: 'filename' is required."

        out_dir = _ensure_output_dir()
        # Security: prevent path traversal
        safe_name = Path(filename).name
        file_path = out_dir / safe_name

        try:
            with open(file_path, mode, encoding="utf-8") as f:
                f.write(content)
            return f"File written successfully: {file_path}"
        except OSError as exc:
            return f"Failed to write file: {exc}"


class FileReaderTool(BaseTool):
    """Read content from a file in the output directory."""

    name: str = "file_reader"
    description: str = (
        "Read content from a file in the output directory. "
        "Input should be a JSON string with key: "
        "'filename' (required). "
        "Returns the file content as a string."
    )

    def _run(self, query_json: str) -> str:
        try:
            params: dict[str, Any] = json.loads(query_json)
        except json.JSONDecodeError:
            params = {"filename": query_json}

        filename: str = params.get("filename", "")
        if not filename:
            return "Error: 'filename' is required."

        out_dir = _ensure_output_dir()
        safe_name = Path(filename).name
        file_path = out_dir / safe_name

        if not file_path.exists():
            return f"File not found: {safe_name}"

        try:
            return file_path.read_text(encoding="utf-8")
        except OSError as exc:
            return f"Failed to read file: {exc}"
