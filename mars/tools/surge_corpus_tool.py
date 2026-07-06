"""
SurGE corpus retrieval tool — searches the 1.09M-paper SurGE corpus locally.

Uses TF-IDF (scikit-learn) over title + abstract for fast similarity search.
Returns paper metadata with ``doc_id`` — the same IDs used in SurGE's ground-truth
``all_cites`` lists, enabling direct Recall/Precision/F1 computation.

.. important::
    This tool reads SurGE **data** (corpus.json). It does NOT import, copy, or
    adapt any SurGE **code** (src/evaluator.py, src/informationFuncs.py, etc.).
    All retrieval logic is independently implemented.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from crewai.tools import BaseTool

from mars.config import settings
from mars.evaluation.surge_adapter import SurGEAdapter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Index cache path
# ---------------------------------------------------------------------------

_INDEX_CACHE_DIR = Path("data/surge/index")


def _get_cache_paths() -> tuple[Path, Path, Path]:
    """Return (index_dir, vectorizer_path, corpus_meta_path)."""
    idx_dir = _INDEX_CACHE_DIR
    idx_dir.mkdir(parents=True, exist_ok=True)
    return (
        idx_dir,
        idx_dir / "tfidf_vectorizer.pkl",
        idx_dir / "corpus_metadata.pkl",
    )


# ---------------------------------------------------------------------------
# Corpus loader + index builder
# ---------------------------------------------------------------------------

# Module-level cache — built once, reused across all tool instances.
_tfidf_matrix = None
_vectorizer = None
_corpus_meta: List[Dict[str, Any]] = []


def _load_or_build_index(corpus_path: str, force_rebuild: bool = False) -> int:
    """Ensure the TF-IDF index is loaded. Returns number of indexed papers."""
    global _tfidf_matrix, _vectorizer, _corpus_meta

    if _tfidf_matrix is not None and not force_rebuild:
        return len(_corpus_meta)

    idx_dir, vec_path, meta_path = _get_cache_paths()

    # Try loading cached index
    if not force_rebuild and vec_path.is_file() and meta_path.is_file():
        logger.info("Loading cached TF-IDF index from %s ...", idx_dir)
        try:
            _vectorizer = pickle.loads(vec_path.read_bytes())
            _tfidf_matrix = pickle.loads(
                (idx_dir / "tfidf_matrix.pkl").read_bytes()
            )
            _corpus_meta = pickle.loads(meta_path.read_bytes())
            logger.info(
                "Cached index loaded: %d papers, %d features.",
                len(_corpus_meta),
                _tfidf_matrix.shape[1],
            )
            return len(_corpus_meta)
        except Exception as exc:
            logger.warning("Failed to load cached index: %s — rebuilding.", exc)

    # Build index from SurGE corpus
    logger.info("Building TF-IDF index from %s ...", corpus_path)
    t0 = time.time()

    if not os.path.isfile(corpus_path):
        raise FileNotFoundError(
            f"SurGE corpus not found at {corpus_path}. "
            f"Download it from SurGE Google Drive or set SURGE_CORPUS_PATH."
        )

    corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    papers: List[Dict[str, Any]] = []
    if isinstance(corpus, dict):
        papers = [{"doc_id": int(k), **v} for k, v in corpus.items()]
    elif isinstance(corpus, list):
        papers = corpus

    if not papers:
        logger.warning("Empty corpus — TF-IDF index will be empty.")
        return 0

    # Build document texts: title + first 600 chars of abstract
    documents: List[str] = []
    for p in papers:
        title = p.get("Title", p.get("title", ""))
        abstract = p.get("Abstract", p.get("abstract", ""))[:600]
        authors = " ".join(p.get("Authors", p.get("authors", [])))[:200]
        doc_text = f"{title}. {abstract}. {authors}"
        documents.append(doc_text)

    from sklearn.feature_extraction.text import TfidfVectorizer

    _vectorizer = TfidfVectorizer(
        max_features=80000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    _tfidf_matrix = _vectorizer.fit_transform(documents)
    _corpus_meta = papers

    elapsed = time.time() - t0
    logger.info(
        "TF-IDF index built: %d papers, %d features, %.1fs.",
        len(papers),
        _tfidf_matrix.shape[1],
        elapsed,
    )

    # Cache to disk
    try:
        vec_path.write_bytes(pickle.dumps(_vectorizer))
        (idx_dir / "tfidf_matrix.pkl").write_bytes(pickle.dumps(_tfidf_matrix))
        meta_path.write_bytes(pickle.dumps(_corpus_meta))
        logger.info("TF-IDF index cached to %s", idx_dir)
    except Exception as exc:
        logger.warning("Failed to cache index: %s", exc)

    return len(papers)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class SurgeCorpusSearchTool(BaseTool):
    """Search the SurGE 1.09M-paper corpus and return results with doc_id.

    doc_id is the key field — it maps directly to SurGE ground-truth all_cites
    lists, enabling precise Recall/Precision/F1 computation.

    This tool does NOT import or adapt any SurGE ``src/`` code.
    """

    name: str = "surge_corpus_search"
    description: str = (
        "Search the SurGE arXiv paper corpus (1.09M papers) using keyword/TF-IDF. "
        "Parameters: 'query' (required, str), 'max_results' (optional, int, default 20). "
        "Returns: JSON list of {doc_id, title, authors, year, abstract}. "
        "doc_id is critical — it maps to SurGE ground-truth citations."
    )

    def _run(self, query: str, max_results: int = 20) -> str:
        if not query:
            return json.dumps({"error": "query is required"})

        # Determine corpus path
        corpus_path = os.environ.get(
            "SURGE_CORPUS_PATH",
            "D:/PersonalResearch/PapersWS/Paper-MARS/Ref/SurGE/data/corpus.json",
        )

        # Ensure index is loaded
        try:
            n_papers = _load_or_build_index(corpus_path)
        except FileNotFoundError as exc:
            return json.dumps({"error": str(exc)})

        if n_papers == 0:
            return json.dumps({"error": "corpus is empty", "count": 0})

        # Transform query and compute cosine similarity
        query_vec = _vectorizer.transform([query])
        from sklearn.metrics.pairwise import cosine_similarity
        scores = cosine_similarity(query_vec, _tfidf_matrix).flatten()

        # Top-K
        top_k = min(max_results, n_papers)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < 0.01:  # skip near-zero matches
                continue
            paper = _corpus_meta[idx]
            doc_id = paper.get("doc_id", idx)
            results.append({
                "doc_id": doc_id,
                "title": paper.get("Title", paper.get("title", "")),
                "authors": paper.get("Authors", paper.get("authors", [])),
                "year": paper.get("Year", paper.get("year", "")),
                "abstract": (paper.get("Abstract", paper.get("abstract", "")))[:500],
                "category": paper.get("Category", paper.get("category", "")),
                "score": round(score, 4),
            })

        return json.dumps({
            "query": query,
            "count": len(results),
            "results": results,
        }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Builder helper — call once to pre-build and cache the index
# ---------------------------------------------------------------------------

def build_surge_index(corpus_path: str | None = None) -> int:
    """Pre-build and cache the SurGE TF-IDF index.

    Call this once after downloading corpus.json. Subsequent runs load the
    cached index in <1 second.
    """
    path = corpus_path or os.environ.get(
        "SURGE_CORPUS_PATH",
        "D:/PersonalResearch/PapersWS/Paper-MARS/Ref/SurGE/data/corpus.json",
    )
    return _load_or_build_index(path, force_rebuild=True)
