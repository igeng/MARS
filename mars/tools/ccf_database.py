"""
CCF (China Computer Federation) ranking database tool.

Provides a static snapshot of the CCF recommended international academic
venues list.  In a production deployment this can be backed by a PostgreSQL
table or an up-to-date YAML/JSON file.
"""

from __future__ import annotations

from typing import Any

from crewai.tools import BaseTool
from pydantic import Field

# ---------------------------------------------------------------------------
# Static CCF data (A/B/C ranked venues, computer science domains)
# This is a representative subset – extend as needed.
# ---------------------------------------------------------------------------

CCF_DATABASE: list[dict[str, Any]] = [
    # ---- A-rank conferences ----
    {
        "name": "CVPR",
        "full_name": "IEEE Conference on Computer Vision and Pattern Recognition",
        "ccf_rank": "A",
        "type": "conference",
        "domains": ["computer vision", "image recognition", "deep learning"],
        "dblp_url": "https://dblp.org/db/conf/cvpr/",
    },
    {
        "name": "ICML",
        "full_name": "International Conference on Machine Learning",
        "ccf_rank": "A",
        "type": "conference",
        "domains": [
            "machine learning",
            "federated learning",
            "optimization",
            "deep learning",
        ],
        "dblp_url": "https://dblp.org/db/conf/icml/",
    },
    {
        "name": "NeurIPS",
        "full_name": "Neural Information Processing Systems",
        "ccf_rank": "A",
        "type": "conference",
        "domains": [
            "deep learning",
            "neural network",
            "reinforcement learning",
            "generative model",
        ],
        "dblp_url": "https://dblp.org/db/conf/nips/",
    },
    {
        "name": "ICLR",
        "full_name": "International Conference on Learning Representations",
        "ccf_rank": "A",
        "type": "conference",
        "domains": ["representation learning", "deep learning", "generative model"],
        "dblp_url": "https://dblp.org/db/conf/iclr/",
    },
    {
        "name": "ACL",
        "full_name": "Annual Meeting of the Association for Computational Linguistics",
        "ccf_rank": "A",
        "type": "conference",
        "domains": [
            "natural language processing",
            "text generation",
            "machine translation",
        ],
        "dblp_url": "https://dblp.org/db/conf/acl/",
    },
    {
        "name": "EMNLP",
        "full_name": "Empirical Methods in Natural Language Processing",
        "ccf_rank": "B",
        "type": "conference",
        "domains": [
            "natural language processing",
            "information extraction",
            "sentiment analysis",
        ],
        "dblp_url": "https://dblp.org/db/conf/emnlp/",
    },
    {
        "name": "SIGKDD",
        "full_name": "ACM SIGKDD Conference on Knowledge Discovery and Data Mining",
        "ccf_rank": "A",
        "type": "conference",
        "domains": ["data mining", "knowledge discovery", "graph neural network"],
        "dblp_url": "https://dblp.org/db/conf/kdd/",
    },
    {
        "name": "SIGMOD",
        "full_name": "ACM SIGMOD International Conference on Management of Data",
        "ccf_rank": "A",
        "type": "conference",
        "domains": ["database", "data management", "query optimization"],
        "dblp_url": "https://dblp.org/db/conf/sigmod/",
    },
    {
        "name": "VLDB",
        "full_name": "International Conference on Very Large Data Bases",
        "ccf_rank": "A",
        "type": "conference",
        "domains": ["database", "distributed database", "storage system"],
        "dblp_url": "https://dblp.org/db/conf/vldb/",
    },
    {
        "name": "SOSP",
        "full_name": "ACM Symposium on Operating Systems Principles",
        "ccf_rank": "A",
        "type": "conference",
        "domains": ["operating system", "distributed system", "storage"],
        "dblp_url": "https://dblp.org/db/conf/sosp/",
    },
    {
        "name": "OSDI",
        "full_name": "USENIX Symposium on Operating Systems Design and Implementation",
        "ccf_rank": "A",
        "type": "conference",
        "domains": ["operating system", "distributed system", "systems"],
        "dblp_url": "https://dblp.org/db/conf/osdi/",
    },
    {
        "name": "USENIX Security",
        "full_name": "USENIX Security Symposium",
        "ccf_rank": "A",
        "type": "conference",
        "domains": [
            "security",
            "privacy",
            "cryptography",
            "differential privacy",
        ],
        "dblp_url": "https://dblp.org/db/conf/uss/",
    },
    {
        "name": "CCS",
        "full_name": "ACM Conference on Computer and Communications Security",
        "ccf_rank": "A",
        "type": "conference",
        "domains": ["security", "privacy", "federated learning privacy"],
        "dblp_url": "https://dblp.org/db/conf/ccs/",
    },
    # ---- A-rank journals ----
    {
        "name": "TPAMI",
        "full_name": "IEEE Transactions on Pattern Analysis and Machine Intelligence",
        "ccf_rank": "A",
        "type": "journal",
        "domains": [
            "computer vision",
            "pattern recognition",
            "machine learning",
        ],
        "dblp_url": "https://dblp.org/db/journals/pami/",
    },
    {
        "name": "IJCV",
        "full_name": "International Journal of Computer Vision",
        "ccf_rank": "A",
        "type": "journal",
        "domains": ["computer vision", "3D reconstruction"],
        "dblp_url": "https://dblp.org/db/journals/ijcv/",
    },
    {
        "name": "AI",
        "full_name": "Artificial Intelligence",
        "ccf_rank": "A",
        "type": "journal",
        "domains": [
            "artificial intelligence",
            "machine learning",
            "knowledge representation",
        ],
        "dblp_url": "https://dblp.org/db/journals/ai/",
    },
    {
        "name": "JMLR",
        "full_name": "Journal of Machine Learning Research",
        "ccf_rank": "A",
        "type": "journal",
        "domains": [
            "machine learning",
            "statistical learning",
            "federated learning",
        ],
        "dblp_url": "https://dblp.org/db/journals/jmlr/",
    },
    # ---- More A/B-rank conferences ----
    {
        "name": "ICCV",
        "full_name": "International Conference on Computer Vision",
        "ccf_rank": "A",
        "type": "conference",
        "domains": ["computer vision", "object detection", "segmentation"],
        "dblp_url": "https://dblp.org/db/conf/iccv/",
    },
    {
        "name": "ECCV",
        "full_name": "European Conference on Computer Vision",
        "ccf_rank": "B",
        "type": "conference",
        "domains": ["computer vision", "image processing"],
        "dblp_url": "https://dblp.org/db/conf/eccv/",
    },
    {
        "name": "AAAI",
        "full_name": "AAAI Conference on Artificial Intelligence",
        "ccf_rank": "A",
        "type": "conference",
        "domains": [
            "artificial intelligence",
            "machine learning",
            "planning",
            "knowledge representation",
        ],
        "dblp_url": "https://dblp.org/db/conf/aaai/",
    },
    {
        "name": "IJCAI",
        "full_name": "International Joint Conference on Artificial Intelligence",
        "ccf_rank": "A",
        "type": "conference",
        "domains": ["artificial intelligence", "multi-agent systems", "planning"],
        "dblp_url": "https://dblp.org/db/conf/ijcai/",
    },
    {
        "name": "WWW",
        "full_name": "The Web Conference",
        "ccf_rank": "A",
        "type": "conference",
        "domains": [
            "web mining",
            "recommendation system",
            "knowledge graph",
            "graph neural network",
        ],
        "dblp_url": "https://dblp.org/db/conf/www/",
    },
    {
        "name": "INFOCOM",
        "full_name": "IEEE International Conference on Computer Communications",
        "ccf_rank": "A",
        "type": "conference",
        "domains": ["networking", "distributed system", "federated learning"],
        "dblp_url": "https://dblp.org/db/conf/infocom/",
    },
]


class CCFDatabaseQueryTool(BaseTool):
    """Query the CCF recommended venues database."""

    name: str = "ccf_database_query"
    description: str = (
        "Query the CCF (China Computer Federation) recommended international "
        "academic venues database. "
        "Input should be a research domain or keyword string. "
        "Returns matching conferences/journals with their CCF rank, type, and DBLP URL."
    )

    def _run(self, query: str) -> str:
        query_lower = query.lower()
        results: list[dict[str, Any]] = []

        for venue in CCF_DATABASE:
            # Match against name, full_name, or any domain keyword
            if query_lower in venue["name"].lower() or query_lower in venue[
                "full_name"
            ].lower():
                results.append(venue)
                continue
            for domain in venue["domains"]:
                if query_lower in domain.lower() or domain.lower() in query_lower:
                    results.append(venue)
                    break

        if not results:
            return f"No CCF venues found matching '{query}'."

        lines = [f"Found {len(results)} venue(s) matching '{query}':\n"]
        for v in results:
            lines.append(
                f"- [{v['ccf_rank']}] {v['name']} ({v['type'].upper()}): "
                f"{v['full_name']}\n"
                f"  Domains: {', '.join(v['domains'])}\n"
                f"  DBLP: {v['dblp_url']}"
            )
        return "\n".join(lines)
