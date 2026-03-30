"""
PDF parser tool.

Downloads a PDF from a URL and extracts text content using PyMuPDF (fitz).
Falls back to PyPDF2 if fitz is unavailable.
"""

from __future__ import annotations

import io
import json
import re
from typing import Any

import requests
from crewai.tools import BaseTool

from mars.utils.retry import retry_on_network_error

MAX_PAGES = 20  # limit pages to avoid excessive token use
MAX_CHARS = 8000  # max characters returned


@retry_on_network_error(max_retries=2)
def _download_pdf(url: str) -> bytes:
    resp = requests.get(url, timeout=30, stream=True)
    resp.raise_for_status()
    return resp.content


class PDFParserTool(BaseTool):
    """Download and extract text from a PDF file."""

    name: str = "pdf_parser"
    description: str = (
        "Download a PDF from a URL and extract its text content. "
        "Input should be a JSON string with keys: "
        "'url' (required, direct PDF URL), "
        "'max_pages' (optional, default 20), "
        "'max_chars' (optional, default 8000). "
        "Returns extracted text from the PDF."
    )

    def _run(self, query_json: str) -> str:
        try:
            params: dict[str, Any] = json.loads(query_json)
        except json.JSONDecodeError:
            params = {"url": query_json}

        url: str = params.get("url", "")
        max_pages: int = int(params.get("max_pages", MAX_PAGES))
        max_chars: int = int(params.get("max_chars", MAX_CHARS))

        if not url:
            return "Error: 'url' parameter is required."

        try:
            pdf_bytes = _download_pdf(url)
        except requests.RequestException as exc:
            return f"Failed to download PDF from '{url}': {exc}"

        text = _extract_text_fitz(pdf_bytes, max_pages) or _extract_text_pypdf2(
            pdf_bytes, max_pages
        )

        if not text:
            return f"Could not extract text from PDF at '{url}'."

        # Trim to max_chars
        if len(text) > max_chars:
            text = text[:max_chars] + "\n[... truncated ...]"

        return f"Extracted text from {url}:\n\n{text}"


def _extract_text_fitz(pdf_bytes: bytes, max_pages: int) -> str:
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = min(len(doc), max_pages)
        parts: list[str] = []
        for i in range(pages):
            parts.append(doc[i].get_text())
        return "\n".join(parts)
    except Exception:
        return ""


def _extract_text_pypdf2(pdf_bytes: bytes, max_pages: int) -> str:
    try:
        import PyPDF2

        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        pages = min(len(reader.pages), max_pages)
        parts: list[str] = []
        for i in range(pages):
            page_text = reader.pages[i].extract_text() or ""
            parts.append(page_text)
        return "\n".join(parts)
    except Exception:
        return ""
