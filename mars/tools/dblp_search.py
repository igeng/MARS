"""
DBLP search tool.

Uses the public DBLP REST API (no API key required) to search for
academic papers by keyword / venue / author.

API reference: https://dblp.org/faq/How+to+use+the+dblp+search+API.html
"""

from __future__ import annotations

import json
from typing import Any

import requests
from crewai.tools import BaseTool

from mars.utils.retry import retry_on_network_error

DBLP_SEARCH_URL = "https://dblp.org/search/publ/api"
DEFAULT_MAX_RESULTS = 20


@retry_on_network_error(max_retries=3)
def _dblp_get(params: dict) -> requests.Response:
    response = requests.get(DBLP_SEARCH_URL, params=params, timeout=15)
    response.raise_for_status()
    return response


class DBLPSearchTool(BaseTool):
    """Search for papers in the DBLP database."""

    name: str = "dblp_search"
    description: str = (
        "Search for academic papers using the DBLP API. "
        "Input should be a JSON string with keys: "
        "'query' (required, search keywords), "
        "'max_results' (optional, integer, default 20), "
        "'year_from' (optional, int), "
        "'year_to' (optional, int). "
        "Returns a list of papers with title, authors, venue, year, and URL."
    )

    def _run(self, query_json: str) -> str:
        try:
            params: dict[str, Any] = json.loads(query_json)
        except json.JSONDecodeError:
            # Treat plain string as query
            params = {"query": query_json}

        query: str = params.get("query", "")
        max_results: int = int(params.get("max_results", DEFAULT_MAX_RESULTS))
        year_from: int | None = params.get("year_from")
        year_to: int | None = params.get("year_to")

        if not query:
            return "Error: 'query' parameter is required."

        api_params = {
            "q": query,
            "format": "json",
            "h": max_results,
            "f": 0,  # first result index
        }

        try:
            response = _dblp_get(api_params)
            data = response.json()
        except requests.RequestException as exc:
            return f"DBLP API request failed: {exc}"

        hits = data.get("result", {}).get("hits", {}).get("hit", [])
        if not hits:
            return f"No results found in DBLP for query: '{query}'."

        papers: list[dict[str, Any]] = []
        for hit in hits:
            info = hit.get("info", {})
            year_val = info.get("year")

            # Optional year range filter
            if year_from is not None and year_val is not None:
                if int(year_val) < year_from:
                    continue
            if year_to is not None and year_val is not None:
                if int(year_val) > year_to:
                    continue

            # Authors can be a dict (single) or list of dicts
            authors_raw = info.get("authors", {}).get("author", [])
            if isinstance(authors_raw, dict):
                authors_raw = [authors_raw]
            authors = [a.get("text", "") for a in authors_raw]

            papers.append(
                {
                    "title": info.get("title", ""),
                    "authors": authors,
                    "venue": info.get("venue", ""),
                    "year": year_val,
                    "doi": info.get("doi", ""),
                    "url": info.get("url", ""),
                    "type": info.get("type", ""),
                }
            )

        if not papers:
            return f"No results found in DBLP matching year filter for query: '{query}'."

        lines = [f"DBLP search results for '{query}' ({len(papers)} papers):\n"]
        for i, p in enumerate(papers, 1):
            lines.append(
                f"{i}. {p['title']}\n"
                f"   Authors: {', '.join(p['authors'])}\n"
                f"   Venue: {p['venue']}  Year: {p['year']}\n"
                f"   URL: {p['url']}"
            )
        return "\n".join(lines)
