"""
File manager tool.

Handles reading and writing of research output files
(JSON results, Markdown reports, etc.) to the configured output directory.
"""

from __future__ import annotations

from pathlib import Path

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
        "Parameters: 'filename' (required, str), "
        "'content' (required, str), "
        "'mode' (optional, 'w' or 'a', default 'w'). "
        "Returns the absolute path of the written file."
    )

    def _run(self, filename: str, content: str, mode: str = "w") -> str:
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
        "Parameters: 'filename' (required, str). "
        "Returns the file content as a string."
    )

    def _run(self, filename: str) -> str:
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
