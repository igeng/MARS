"""
Citation network builder tool.

Fetches the citation graph for a set of papers using the
Semantic Scholar API and builds a NetworkX graph for analysis.
"""

from __future__ import annotations

import json
from typing import Any

import requests
from crewai.tools import BaseTool

from mars.config import settings
from mars.utils.retry import retry_on_network_error

SS_PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
SS_REFS_FIELDS = "references.paperId,references.title,references.year,references.citationCount"
SS_CITES_FIELDS = "citations.paperId,citations.title,citations.year,citations.citationCount"
DEFAULT_MAX_REFS = 30


@retry_on_network_error(max_retries=2)
def _ss_paper_get(url: str, params: dict, headers: dict) -> requests.Response:
    response = requests.get(url, params=params, headers=headers, timeout=15)
    response.raise_for_status()
    return response


class CitationNetworkTool(BaseTool):
    """Build a citation network for a list of paper IDs."""

    name: str = "citation_network_builder"
    description: str = (
        "Build a citation network for a list of Semantic Scholar paper IDs. "
        "Input should be a JSON string with keys: "
        "'paper_ids' (required, list of Semantic Scholar paper IDs), "
        "'max_refs_per_paper' (optional, default 30). "
        "Returns a description of the citation network including nodes and key edges."
    )

    def _run(self, query_json: str) -> str:
        try:
            params: dict[str, Any] = json.loads(query_json)
        except json.JSONDecodeError:
            return "Error: Input must be valid JSON."

        paper_ids: list[str] = params.get("paper_ids", [])
        max_refs: int = int(params.get("max_refs_per_paper", DEFAULT_MAX_REFS))

        if not paper_ids:
            return "Error: 'paper_ids' list is required."

        headers: dict[str, str] = {}
        if settings.SEMANTIC_SCHOLAR_API_KEY:
            headers["x-api-key"] = settings.SEMANTIC_SCHOLAR_API_KEY

        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, str]] = []

        for pid in paper_ids:
            # Fetch references
            try:
                resp = _ss_paper_get(
                    SS_PAPER_URL.format(paper_id=pid),
                    params={"fields": SS_REFS_FIELDS},
                    headers=headers,
                )
                data = resp.json()
            except requests.RequestException as exc:
                continue

            nodes[pid] = {
                "paper_id": pid,
                "title": data.get("title", "Unknown"),
            }

            for ref in (data.get("references") or [])[:max_refs]:
                ref_id = ref.get("paperId")
                if not ref_id:
                    continue
                if ref_id not in nodes:
                    nodes[ref_id] = {
                        "paper_id": ref_id,
                        "title": ref.get("title", "Unknown"),
                        "year": ref.get("year"),
                        "citation_count": ref.get("citationCount", 0),
                    }
                edges.append({"from": pid, "to": ref_id, "type": "cites"})

        # Try to build NetworkX summary
        summary = _build_network_summary(nodes, edges)

        lines = [
            f"Citation network for {len(paper_ids)} seed paper(s):",
            f"  Total nodes: {len(nodes)}",
            f"  Total edges: {len(edges)}",
            "",
            summary,
        ]
        return "\n".join(lines)


def _build_network_summary(
    nodes: dict[str, Any], edges: list[dict[str, str]]
) -> str:
    try:
        import networkx as nx

        G = nx.DiGraph()
        for nid, ndata in nodes.items():
            G.add_node(nid, **ndata)
        for e in edges:
            G.add_edge(e["from"], e["to"])

        # Top cited nodes (highest in-degree)
        in_degrees = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)[:5]
        top_cited = [
            f"  {nodes.get(nid, {}).get('title', nid)!r} (cited {deg} times)"
            for nid, deg in in_degrees
        ]
        return "Top cited papers in network:\n" + "\n".join(top_cited)
    except ImportError:
        return "(NetworkX not available for network analysis)"
    except Exception as exc:
        return f"(Network analysis error: {exc})"
