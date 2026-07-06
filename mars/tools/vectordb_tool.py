"""
VectorDB tool — ChromaDB-backed semantic search for MARS papers.

Provides two tools:

- **IndexPapersTool**: Embed and index paper metadata into a vector store.
- **SearchPapersTool**: Semantic search over indexed papers, returning the most
  relevant chunks.

Used by the Summarizer agent for RAG-enhanced grounded generation (EXP-1.2):
before writing each survey section, the agent queries the vector store to
retrieve the most relevant paper content, then generates paragraphs grounded
in actual retrieved text rather than LLM memory alone.
"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from mars.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding helper — tries OpenAI-compatible API first, falls back to
# sentence-transformers (all-MiniLM-L6-v2).
# ---------------------------------------------------------------------------

_EMBED_FN = None


def _get_embed_fn():
    """Return a callable ``fn(texts: list[str]) -> list[list[float]]``."""
    global _EMBED_FN
    if _EMBED_FN is not None:
        return _EMBED_FN

    # Try OpenAI-compatible embedding API
    try:
        import os
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            def _openai_embed(texts: list[str]) -> list[list[float]]:
                resp = client.embeddings.create(
                    model="text-embedding-3-small", input=texts,
                )
                return [d.embedding for d in resp.data]

            _EMBED_FN = _openai_embed
            logger.info("Using OpenAI text-embedding-3-small for vector store.")
            return _EMBED_FN
    except Exception:
        pass

    # Fallback: sentence-transformers (local, no API key needed)
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")

        def _st_embed(texts: list[str]) -> list[list[float]]:
            embeddings = model.encode(texts, show_progress_bar=False)
            return embeddings.tolist()

        _EMBED_FN = _st_embed
        logger.info("Using all-MiniLM-L6-v2 (local) for vector store.")
        return _EMBED_FN
    except Exception:
        logger.warning(
            "No embedding backend available. "
            "Install sentence-transformers: pip install sentence-transformers"
        )
        # Return a dummy embedder that returns zero vectors
        def _dummy(texts: list[str]) -> list[list[float]]:
            return [[0.0] * 384 for _ in texts]
        _EMBED_FN = _dummy
        return _EMBED_FN


# ---------------------------------------------------------------------------
# Shared ChromaDB client
# ---------------------------------------------------------------------------

_collection = None


def _get_collection() -> Any:
    """Return (or create) the persistent ChromaDB collection for MARS papers."""
    global _collection
    if _collection is not None:
        return _collection

    persist_dir = str(settings.OUTPUT_DIR / "chroma_db")
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    _collection = client.get_or_create_collection(
        name="mars_papers",
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("ChromaDB collection ready at %s", persist_dir)
    return _collection


# ---------------------------------------------------------------------------
# Tool: Index Papers
# ---------------------------------------------------------------------------

class IndexPapersInput(BaseModel):
    papers_json: str = Field(
        default="",
        description="JSON string: list of paper dicts, each with 'title' (required), "
                    "'abstract', 'contributions', 'methodology', 'results', 'year', 'authors'.",
    )


class IndexPapersTool(BaseTool):
    """Index paper metadata into the vector store for later semantic search."""

    name: str = "index_papers"
    description: str = (
        "Index paper metadata into the vector store. "
        "Input: JSON string of [{title, abstract, contributions, methodology, results, year, authors}, ...]. "
        "Call this BEFORE writing any survey content. Returns the number of papers indexed."
    )
    args_schema: type[BaseModel] = IndexPapersInput

    def _run(self, papers_json: str = "") -> str:
        if not papers_json:
            return "Error: papers_json is required."

        try:
            papers = json.loads(papers_json)
        except json.JSONDecodeError as exc:
            return f"Error: invalid JSON — {exc}"

        if not isinstance(papers, list):
            return "Error: expected a JSON array of paper objects."

        embed_fn = _get_embed_fn()
        collection = _get_collection()

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for i, paper in enumerate(papers):
            title = paper.get("title", f"Untitled-{i}")
            # Build a rich text chunk for embedding
            chunk_parts = [title]
            for field in ("abstract", "contributions", "methodology", "results"):
                val = paper.get(field, "")
                if val:
                    chunk_parts.append(val[:2000])  # truncate long fields
            chunk_text = "\n\n".join(chunk_parts)

            paper_id = f"paper_{i}_{title[:40].replace(' ', '_')}"
            ids.append(paper_id)
            documents.append(chunk_text)
            metadatas.append({
                "title": title[:200],
                "authors": str(paper.get("authors", ""))[:200],
                "year": str(paper.get("year", "")),
                "paper_index": i,
            })

        if not ids:
            return "No papers to index."

        # Embed and upsert
        try:
            embeddings = embed_fn(documents)
            collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            logger.info("Indexed %d papers into vector store.", len(ids))
            return f"Successfully indexed {len(ids)} papers into the vector store."
        except Exception as exc:
            logger.error("Failed to index papers: %s", exc)
            return f"Error indexing papers: {exc}"


# ---------------------------------------------------------------------------
# Tool: Search Papers
# ---------------------------------------------------------------------------

class SearchPapersInput(BaseModel):
    query: str = Field(..., description="Natural language search query.")
    top_k: int = Field(default=5, description="Number of results to return (1-20).")


class SearchPapersTool(BaseTool):
    """Semantic search over indexed papers using the vector store."""

    name: str = "search_papers"
    description: str = (
        "Search indexed papers using semantic similarity. "
        "Parameters: 'query' (str, required), 'top_k' (int, default 5). "
        "Returns the most relevant paper chunks with metadata. "
        "Use this BEFORE writing each section to ground your claims in real papers."
    )
    args_schema: type[BaseModel] = SearchPapersInput

    def _run(self, query: str, top_k: int = 5) -> str:
        if not query:
            return "Error: 'query' is required."

        top_k = max(1, min(top_k, 20))
        embed_fn = _get_embed_fn()
        collection = _get_collection()

        try:
            query_embedding = embed_fn([query])[0]
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            return f"Error querying vector store: {exc}"

        if not results or not results.get("ids") or not results["ids"][0]:
            return "No matching papers found in vector store. Try indexing papers first with index_papers."

        lines: list[str] = [f"Top {top_k} papers matching '{query}':\n"]
        for i in range(len(results["ids"][0])):
            doc_id = results["ids"][0][i]
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            document = results["documents"][0][i] if results["documents"] else ""
            distance = results["distances"][0][i] if results.get("distances") else 1.0

            title = metadata.get("title", "Unknown")
            authors = metadata.get("authors", "")
            year = metadata.get("year", "")
            # Truncate document for readability
            snippet = document[:800] + ("..." if len(document) > 800 else "")

            lines.append(
                f"[{i+1}] **{title}** ({year})\n"
                f"    Authors: {authors}\n"
                f"    Relevance: {1.0 - distance:.2f}\n"
                f"    Content: {snippet}\n"
            )

        return "\n".join(lines)
