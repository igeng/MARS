"""
Semantic Scholar search tool.

Uses the official Semantic Scholar Academic Graph API.
An optional API key (SEMANTIC_SCHOLAR_API_KEY) increases rate limits.

API docs: https://api.semanticscholar.org/api-docs/
"""

from __future__ import annotations

import json
from typing import Any

import requests
from crewai.tools import BaseTool

from mars.config import settings

SS_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SS_FIELDS = (
    "paperId,title,authors,year,venue,citationCount,"
    "externalIds,abstract,openAccessPdf,url"
)
DEFAULT_MAX_RESULTS = 20


class SemanticScholarSearchTool(BaseTool):
    """Search for papers using the Semantic Scholar API."""

    name: str = "semantic_scholar_search"
    description: str = (
        "Search for academic papers using the Semantic Scholar API. "
        "Input should be a JSON string with keys: "
        "'query' (required), "
        "'max_results' (optional, default 20), "
        "'year_from' (optional, int), "
        "'year_to' (optional, int), "
        "'min_citations' (optional, int). "
        "Returns papers with title, authors, venue, year, citation count, "
        "abstract, and PDF URL when available."
    )

    def _run(self, query_json: str) -> str:
        try:
            params: dict[str, Any] = json.loads(query_json)
        except json.JSONDecodeError:
            params = {"query": query_json}

        query: str = params.get("query", "")
        max_results: int = int(params.get("max_results", DEFAULT_MAX_RESULTS))
        year_from: int | None = params.get("year_from")
        year_to: int | None = params.get("year_to")
        min_citations: int | None = params.get("min_citations")

        if not query:
            return "Error: 'query' parameter is required."

        headers: dict[str, str] = {}
        if settings.SEMANTIC_SCHOLAR_API_KEY:
            headers["x-api-key"] = settings.SEMANTIC_SCHOLAR_API_KEY

        api_params: dict[str, Any] = {
            "query": query,
            "limit": max_results,
            "fields": SS_FIELDS,
        }
        if year_from or year_to:
            year_range = ""
            if year_from:
                year_range += str(year_from)
            year_range += "-"
            if year_to:
                year_range += str(year_to)
            api_params["year"] = year_range

        try:
            response = requests.get(
                SS_SEARCH_URL, params=api_params, headers=headers, timeout=15
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            return f"Semantic Scholar API request failed: {exc}"

        papers_raw: list[dict[str, Any]] = data.get("data", [])
        if not papers_raw:
            return f"No results found in Semantic Scholar for query: '{query}'."

        papers: list[dict[str, Any]] = []
        for p in papers_raw:
            citations = p.get("citationCount", 0) or 0
            if min_citations is not None and citations < min_citations:
                continue

            authors = [a.get("name", "") for a in p.get("authors", [])]
            pdf_url = ""
            if p.get("openAccessPdf"):
                pdf_url = p["openAccessPdf"].get("url", "")

            papers.append(
                {
                    "paper_id": p.get("paperId", ""),
                    "title": p.get("title", ""),
                    "authors": authors,
                    "year": p.get("year"),
                    "venue": p.get("venue", ""),
                    "citation_count": citations,
                    "abstract": (p.get("abstract") or "")[:300],
                    "pdf_url": pdf_url,
                    "url": p.get("url", ""),
                }
            )

        if not papers:
            return (
                f"No results matched filters in Semantic Scholar for query: '{query}'."
            )

        lines = [
            f"Semantic Scholar results for '{query}' ({len(papers)} papers):\n"
        ]
        for i, p in enumerate(papers, 1):
            lines.append(
                f"{i}. {p['title']} ({p['year']})\n"
                f"   Authors: {', '.join(p['authors'])}\n"
                f"   Venue: {p['venue']}  Citations: {p['citation_count']}\n"
                f"   Abstract: {p['abstract']}...\n"
                f"   PDF: {p['pdf_url'] or 'N/A'}  URL: {p['url']}"
            )
        return "\n".join(lines)
