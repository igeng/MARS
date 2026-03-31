"""
arXiv API tool.

Fetches paper metadata and open-access PDF URLs from arXiv.

API docs: https://arxiv.org/help/api/
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from typing import Any, Optional
from urllib.parse import quote_plus

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, model_validator

from mars.config.settings import settings
from mars.utils.retry import retry_on_network_error

ARXIV_SEARCH_URL = "https://export.arxiv.org/api/query"
ARXIV_NS = "http://www.w3.org/2005/Atom"
DEFAULT_MAX_RESULTS = 10


class ArXivSearchToolSchema(BaseModel):
    """Input schema for ArXivSearchTool.

    Accepts either a pre-encoded ``query_json`` string or individual fields.
    """

    query_json: str = Field(
        description=(
            "JSON string with keys: 'query' (required, can include field "
            "prefixes like ti:, au:, abs:), "
            "'max_results' (optional, default 10)."
        )
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_direct_args(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        if "query_json" not in values and "query" in values:
            payload: dict[str, Any] = {"query": values["query"]}
            if values.get("max_results") is not None:
                payload["max_results"] = values["max_results"]
            return {"query_json": json.dumps(payload)}
        return values


@retry_on_network_error(max_retries=3)
def _arxiv_get(params: dict) -> requests.Response:
    response = requests.get(ARXIV_SEARCH_URL, params=params, timeout=settings.ARXIV_SEARCH_TIMEOUT)
    response.raise_for_status()
    return response


class ArXivSearchTool(BaseTool):
    """Search arXiv for papers and retrieve metadata + PDF links."""

    name: str = "arxiv_search"
    description: str = (
        "Search arXiv for academic papers. "
        "Input should be a JSON string with keys: "
        "'query' (required, can include field prefixes like ti:, au:, abs:), "
        "'max_results' (optional, default 10). "
        "Returns paper metadata including abstract and PDF URL."
    )
    args_schema: type[ArXivSearchToolSchema] = ArXivSearchToolSchema

    def _run(self, query_json: str) -> str:
        try:
            params: dict[str, Any] = json.loads(query_json)
        except json.JSONDecodeError:
            params = {"query": query_json}

        query: str = params.get("query", "")
        max_results: int = int(params.get("max_results", DEFAULT_MAX_RESULTS))

        if not query:
            return "Error: 'query' parameter is required."

        api_params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        try:
            response = _arxiv_get(api_params)
        except requests.RequestException as exc:
            return f"arXiv API request failed: {exc}"

        root = ET.fromstring(response.text)
        entries = root.findall(f"{{{ARXIV_NS}}}entry")

        if not entries:
            return f"No arXiv results found for query: '{query}'."

        papers: list[dict[str, Any]] = []
        for entry in entries:
            def _text(tag: str) -> str:
                el = entry.find(f"{{{ARXIV_NS}}}{tag}")
                return el.text.strip() if el is not None and el.text else ""

            arxiv_id_raw = _text("id")
            # Extract short ID like 2301.12345
            match = re.search(r"abs/([^v]+)", arxiv_id_raw)
            arxiv_id = match.group(1) if match else arxiv_id_raw

            authors = [
                a.find(f"{{{ARXIV_NS}}}name").text.strip()
                for a in entry.findall(f"{{{ARXIV_NS}}}author")
                if a.find(f"{{{ARXIV_NS}}}name") is not None
            ]

            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            papers.append(
                {
                    "arxiv_id": arxiv_id,
                    "title": _text("title").replace("\n", " "),
                    "authors": authors,
                    "published": _text("published")[:10],
                    "abstract": _text("summary").replace("\n", " ")[:400],
                    "pdf_url": pdf_url,
                }
            )

        lines = [f"arXiv results for '{query}' ({len(papers)} papers):\n"]
        for i, p in enumerate(papers, 1):
            lines.append(
                f"{i}. [{p['arxiv_id']}] {p['title']} ({p['published'][:4]})\n"
                f"   Authors: {', '.join(p['authors'])}\n"
                f"   Abstract: {p['abstract']}...\n"
                f"   PDF: {p['pdf_url']}"
            )
        return "\n".join(lines)
