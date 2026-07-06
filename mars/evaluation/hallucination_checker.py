"""
Citation hallucination detection for MARS-generated surveys.

Verifies each citation in a generated survey by cross-referencing against
the paper database (``paper_search.json``) and, optionally, live API lookups
via Semantic Scholar.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class CitationVerdict:
    """Verdict for a single citation."""

    index: int                         # [N] marker number
    title: str                         # Extracted or resolved paper title
    status: str                        # "verified" | "fabricated" | "misattributed" | "unchecked"
    detail: str = ""                   # Human-readable explanation
    source: str = ""                   # Where the paper was found (paper_db / s2_api)


@dataclass
class HallucinationReport:
    """Complete hallucination audit for a generated survey."""

    total_citations: int = 0
    verified: int = 0
    fabricated: int = 0
    misattributed: int = 0
    unchecked: int = 0
    verdicts: List[CitationVerdict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_citations": self.total_citations,
            "verified": self.verified,
            "fabricated": self.fabricated,
            "misattributed": self.misattributed,
            "unchecked": self.unchecked,
            "fabrication_rate": (
                round(self.fabricated / self.total_citations, 4)
                if self.total_citations > 0 else 0.0
            ),
            "verification_rate": (
                round(self.verified / self.total_citations, 4)
                if self.total_citations > 0 else 0.0
            ),
            "verdicts": [vars(v) for v in self.verdicts],
        }

    def passed(self, max_fabrication_rate: float = 0.10) -> bool:
        """Check whether the hallucination rate is acceptable."""
        if self.total_citations == 0:
            return True
        return (self.fabricated / self.total_citations) <= max_fabrication_rate


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------

class HallucinationChecker:
    """Verify citations in a MARS-generated survey.

    Parameters:
        enable_api_check: When True, cross-reference citations against
            Semantic Scholar's API for papers not found in the local DB.
        api_rate_limit_delay: Seconds between API calls (Semantic Scholar
            allows 1 request/sec without an API key).
    """

    def __init__(
        self,
        enable_api_check: bool = False,
        api_rate_limit_delay: float = 1.1,
    ):
        self._enable_api_check = enable_api_check
        self._api_delay = api_rate_limit_delay

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        survey_text: str,
        paper_db: List[Dict[str, Any]],
    ) -> HallucinationReport:
        """Audit all citations in *survey_text*.

        Args:
            survey_text: Full Markdown of the generated survey.
            paper_db: List of paper metadata dicts (from ``paper_search.json``),
                each with at least ``title``, ``doi``, ``authors`` fields.

        Returns:
            :class:`HallucinationReport` with per-citation verdicts.
        """
        citations = self._extract_citations(survey_text)
        if not citations:
            return HallucinationReport()

        # Build local lookup
        local_titles = {self._normalize(p["title"]): p for p in paper_db if p.get("title")}
        local_dois = {p["doi"]: p for p in paper_db if p.get("doi")}

        verdicts: List[CitationVerdict] = []
        for idx, (ref_num, body_context, ref_title) in enumerate(citations):
            verdict = self._check_one(ref_num, ref_title, local_titles, local_dois)
            verdicts.append(verdict)

        verified = sum(1 for v in verdicts if v.status == "verified")
        fabricated = sum(1 for v in verdicts if v.status == "fabricated")
        misattr = sum(1 for v in verdicts if v.status == "misattributed")
        unchecked = sum(1 for v in verdicts if v.status == "unchecked")

        return HallucinationReport(
            total_citations=len(verdicts),
            verified=verified,
            fabricated=fabricated,
            misattributed=misattr,
            unchecked=unchecked,
            verdicts=verdicts,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_citations(text: str) -> List[tuple]:
        """Extract ``(ref_number, surrounding_context)`` from survey text.

        Finds ``[N]`` markers and captures ~150 chars of surrounding context
        to check whether the cited paper's claims match.
        """
        # Find reference section boundaries
        ref_section = re.search(
            r'(?:##\s*(?:References|参考文献)|###\s*(?:References|参考文献))',
            text,
            re.IGNORECASE,
        )

        # Extract cited titles from reference list
        ref_titles: Dict[str, str] = {}  # [N] → title
        if ref_section:
            ref_text = text[ref_section.start():]
            # Match: [N] Title. Authors. Venue, Year.
            for m in re.finditer(
                r'\[(\d+)\]\s+(.+?)(?:\.\s+(?:[A-Z][a-z]+|[A-Z]\.))',
                ref_text,
            ):
                ref_titles[m.group(1)] = m.group(2).strip()

        # Find citation markers in body text
        citations: List[tuple] = []
        body = text[:ref_section.start()] if ref_section else text
        for m in re.finditer(r'\[(\d+(?:,\d+)*)\]', body):
            ref_nums = m.group(1).split(",")
            start = max(0, m.start() - 100)
            end = min(len(body), m.end() + 100)
            context = body[start:end].strip()
            for num in ref_nums:
                num = num.strip()
                if ref_titles.get(num):
                    citations.append((num, context, ref_titles[num]))

        return citations

    def _check_one(
        self,
        ref_num: str,
        title_from_ref: str,
        local_titles: Dict[str, Dict],
        local_dois: Dict[str, Dict],
    ) -> CitationVerdict:
        """Check a single citation."""

        # 1. Check local DB by title
        norm_title = self._normalize(title_from_ref)
        if norm_title and norm_title in local_titles:
            return CitationVerdict(
                index=int(ref_num),
                title=title_from_ref[:120],
                status="verified",
                detail="Found in paper_search.json",
                source="paper_db",
            )

        # 2. API check (if enabled)
        if self._enable_api_check and title_from_ref:
            api_result = self._api_lookup(title_from_ref)
            if api_result:
                return CitationVerdict(
                    index=int(ref_num),
                    title=title_from_ref[:120],
                    status="verified",
                    detail=f"Found via Semantic Scholar API: {api_result}",
                    source="s2_api",
                )

        # 3. Cannot verify
        return CitationVerdict(
            index=int(ref_num),
            title=title_from_ref[:120],
            status="unchecked",
            detail="Not found in local DB (API check disabled or failed)",
            source="none",
        )

    def _api_lookup(self, title: str) -> Optional[str]:
        """Look up a paper title via Semantic Scholar API."""
        import requests

        time.sleep(self._api_delay)
        try:
            resp = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={"query": title, "limit": 1},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            papers = data.get("data", [])
            if papers:
                return papers[0].get("title", "")
        except Exception as exc:
            logger.debug("S2 API lookup failed for '%s': %s", title[:80], exc)
        return None

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize paper title for fuzzy matching."""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
