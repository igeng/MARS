"""
Core evaluation metrics for automated survey generation.

Implements the standardized metrics used across the ASG literature:
Citation Recall, Citation Precision, Citation F1 (CQF1), and coverage ratios.
"""

from __future__ import annotations

import re
from typing import List, Set, Tuple


def extract_cited_paper_titles(text: str) -> List[str]:
    """Extract paper titles from a generated survey's reference section.

    Looks for patterns like ``[N] Title. Authors. Venue, Year.`` in the
    reference list and returns the cleaned title strings.
    """
    titles: List[str] = []
    # Match reference entries: [N] Title. Authors...
    ref_pattern = re.compile(
        r'\[\d+\]\s+(.+?)(?:\.\s+(?:[A-Z][a-z]+|[A-Z]\.))',
        re.MULTILINE,
    )
    for match in ref_pattern.finditer(text):
        titles.append(match.group(1).strip())
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
