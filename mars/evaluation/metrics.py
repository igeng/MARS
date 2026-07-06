"""
Core evaluation metrics for automated survey generation.

Implements the standardized metrics used across the ASG literature:
Citation Recall, Citation Precision, Citation F1 (CQF1), and coverage ratios.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Set, Tuple


def extract_cited_paper_titles(text: str) -> List[str]:
    """Extract paper titles from a generated survey's reference section.

    Handles two common reference formats:

    1. ``[N] Title. Authors, Venue, Year.`` (standard academic)
    2. ``[N] Title`` (SurGE baseline format — title only)
    """
    titles: List[str] = []

    # Find reference section boundaries
    ref_section = re.search(
        r'(?:^##\s*(?:References?|参考文献))',
        text, re.MULTILINE | re.IGNORECASE,
    )
    search_text = text[ref_section.start():] if ref_section else text

    # Pattern 1: [N] Title. Authors, Venue... (verbose format)
    pattern1 = re.compile(
        r'^\[\d+\]\s+(.+?)\.\s+(?:[A-Z][a-z]+(?:[\s.][A-Z][a-z]+)*)',
        re.MULTILINE,
    )
    for match in pattern1.finditer(search_text):
        title = match.group(1).strip()
        if len(title) > 5:
            titles.append(title)

    # Pattern 2: [N] Title (bare format — SurGE baselines)
    pattern2 = re.compile(
        r'^\[\d+\]\s+(.+?)$',
        re.MULTILINE,
    )
    for match in pattern2.finditer(search_text):
        title = match.group(1).strip()
        if len(title) > 5 and title not in titles:
            titles.append(title)

    return titles


def extract_doi_or_arxiv(text: str) -> Set[str]:
    """Extract DOI and arXiv IDs from a survey text for paper identification."""
    ids: Set[str] = set()

    # DOI pattern
    doi_pattern = re.compile(r'\b10\.\d{4,}/[^\s\]]+')
    ids.update(doi_pattern.findall(text))

    # arXiv ID pattern
    arxiv_pattern = re.compile(r'arxiv[:\s]+(\d{4}\.\d{4,}(?:v\d+)?)', re.IGNORECASE)
    ids.update(arxiv_pattern.findall(text))

    return ids


def normalize_title(title: str) -> str:
    """Normalize a paper title for fuzzy comparison.

    Lowercases, strips punctuation, collapses whitespace.
    """
    title = title.lower()
    title = re.sub(r'[^a-z0-9\s]', '', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title


def compute_citation_metrics(
    generated_refs: List[str],
    ground_truth_refs: List[str],
    *,
    fuzzy: bool = True,
) -> dict:
    """Compute Recall, Precision, and F1 for citation coverage.

    Args:
        generated_refs: Paper titles/IDs cited in the generated survey.
        ground_truth_refs: Paper titles/IDs from the human-written reference survey.
        fuzzy: When True, use normalized-title matching (default).
               When False, require exact string match.

    Returns:
        Dict with keys: recall, precision, f1, generated_count,
        ground_truth_count, matched_count.
    """
    if fuzzy:
        gen_set = {normalize_title(t) for t in generated_refs}
        gt_set = {normalize_title(t) for t in ground_truth_refs}
    else:
        gen_set = set(generated_refs)
        gt_set = set(ground_truth_refs)

    matched = gen_set & gt_set
    matched_count = len(matched)
    gen_count = len(gen_set)
    gt_count = len(gt_set)

    recall = matched_count / gt_count if gt_count > 0 else 0.0
    precision = matched_count / gen_count if gen_count > 0 else 0.0
    f1 = (
        2 * recall * precision / (recall + precision)
        if (recall + precision) > 0
        else 0.0
    )

    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "generated_count": gen_count,
        "ground_truth_count": gt_count,
        "matched_count": matched_count,
    }


def compute_citation_metrics_from_doc_ids(
    generated_doc_ids: List[str],
    ground_truth_doc_ids: List[str],
) -> dict:
    """Compute Recall, Precision, F1 using exact doc_id matching.

    This is the preferred method for SurGE mode, where both MARS citations
    and ground-truth ``all_cites`` carry SurGE corpus doc_ids, enabling
    precise set-intersection matching without title ambiguity.

    Args:
        generated_doc_ids: doc_ids cited by the generated survey.
        ground_truth_doc_ids: doc_ids from SurGE ground-truth all_cites.

    Returns:
        Dict with recall, precision, f1, counts.
    """
    gen_set = set(str(d) for d in generated_doc_ids if d)
    gt_set = set(str(d) for d in ground_truth_doc_ids if d)

    matched = gen_set & gt_set
    gen_count = len(gen_set)
    gt_count = len(gt_set)
    matched_count = len(matched)

    recall = matched_count / gt_count if gt_count > 0 else 0.0
    precision = matched_count / gen_count if gen_count > 0 else 0.0
    f1 = (
        2 * recall * precision / (recall + precision)
        if (recall + precision) > 0
        else 0.0
    )

    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "generated_count": gen_count,
        "ground_truth_count": gt_count,
        "matched_count": matched_count,
    }


def extract_doc_ids_from_paper_search(
    output_dir: str | Path,
    corpus_path: str | None = None,
) -> set:
    """Extract all doc_ids from a paper_search.json file.

    When the paper entries lack a ``doc_id`` field, falls back to searching
    the SurGE corpus by paper title via TF-IDF to resolve doc_ids.

    Args:
        output_dir: Directory containing paper_search.json, or path to the file.
        corpus_path: Path to SurGE corpus.json for title→doc_id resolution.
            Auto-detected from SURGE_CORPUS_PATH env var if None.

    Returns:
        Set of doc_id strings.
    """
    import json as _json
    import os as _os

    path = Path(output_dir) / "paper_search.json" if not str(output_dir).endswith(".json") else Path(output_dir)
    if not path.is_file():
        return set()

    try:
        data = _json.loads(path.read_text(encoding="utf-8"))
    except (_json.JSONDecodeError, OSError):
        return set()

    papers = data if isinstance(data, list) else list(data.values()) if isinstance(data, dict) else []
    doc_ids = set()

    for p in papers:
        if not isinstance(p, dict):
            continue
        did = p.get("doc_id")
        if did is not None:
            doc_ids.add(str(did))

    # If no doc_ids found in JSON, try title-based corpus lookup
    if not doc_ids and corpus_path is None:
        corpus_path = _os.environ.get(
            "SURGE_CORPUS_PATH",
            "D:/PersonalResearch/PapersWS/Paper-MARS/Ref/SurGE/data/corpus.json",
        )

    if not doc_ids and corpus_path and Path(corpus_path).is_file():
        doc_ids = _resolve_doc_ids_by_title(papers, corpus_path)

    return doc_ids


def _resolve_doc_ids_by_title(
    papers: list[dict],
    corpus_path: str,
) -> set:
    """Resolve doc_ids by searching the SurGE corpus for each paper title.

    Uses the pre-built TF-IDF index from surge_corpus_tool for fast lookup.
    """
    try:
        from mars.tools.surge_corpus_tool import SurgeCorpusSearchTool
        import json as _json

        tool = SurgeCorpusSearchTool()
        doc_ids = set()

        for p in papers:
            title = p.get("title", "")
            if not title:
                continue
            result_str = tool._run(title, max_results=1)
            try:
                result = _json.loads(result_str)
                results = result.get("results", [])
                if results:
                    did = results[0].get("doc_id")
                    if did is not None:
                        doc_ids.add(str(did))
            except _json.JSONDecodeError:
                pass

        return doc_ids
    except Exception:
        return set()


def compute_coverage_ratio(
    generated_subtopics: Set[str],
    ground_truth_subtopics: Set[str],
) -> float:
    """Compute topic coverage: what fraction of ground-truth subtopics are covered."""
    if not ground_truth_subtopics:
        return 0.0
    return len(generated_subtopics & ground_truth_subtopics) / len(ground_truth_subtopics)


# ---------------------------------------------------------------------------
# Text similarity metrics (ROUGE-L approximation via longest common substring)
# ---------------------------------------------------------------------------


def _lcs_length(a: str, b: str) -> int:
    """Longest common subsequence length (DP, O(mn))."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l(generated: str, reference: str) -> float:
    """Compute ROUGE-L F1 (LCS-based) for two texts."""
    gen_tokens = generated.lower().split()
    ref_tokens = reference.lower().split()
    lcs = _lcs_length(gen_tokens, ref_tokens)

    recall = lcs / len(ref_tokens) if ref_tokens else 0.0
    precision = lcs / len(gen_tokens) if gen_tokens else 0.0

    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)
