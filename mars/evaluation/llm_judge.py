"""
LLM-as-Judge quality evaluator for MARS-generated surveys.

Uses an independent LLM (GPT-4 / Claude) to score a generated survey across
6 dimensions adapted from the Agentic AutoSurvey 12-dimension framework.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring rubric (6 dimensions, total 100 points)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """\
You are an expert academic survey evaluator.  Your task is to assess the
quality of an AI-generated academic literature survey against a rigorous,
multi-dimensional rubric.

You MUST respond with ONLY a valid JSON object — no preamble, no commentary.

The JSON object must have exactly this structure:

{
  "citation_coverage": {score (1-10), "reasoning": "..."},
  "citation_accuracy": {score (1-10), "reasoning": "..."},
  "content_comprehensiveness": {score (1-10), "reasoning": "..."},
  "structural_coherence": {score (1-10), "reasoning": "..."},
  "critical_analysis_depth": {score (1-10), "reasoning": "..."},
  "writing_quality": {score (1-10), "reasoning": "..."},
  "overall_comment": "<2-3 sentence overall assessment>"
}

Scoring guide for each dimension:

1. citation_coverage (1-10):
   - 9-10: Cites virtually every key foundational + recent paper in the area.
   - 7-8:  Covers most key papers; a few notable omissions.
   - 5-6:  Covers ~half the expected key papers.
   - 3-4:  Major gaps; many key papers missing.
   - 1-2:  Nearly empty reference list or entirely irrelevant papers.

2. citation_accuracy (1-10):
   - 9-10: Every citation maps to a real paper and correctly supports its claim.
   - 7-8:  Most citations accurate; 1-2 minor issues.
   - 5-6:  Several citations suspicious or misplaced.
   - 3-4:  Many citations appear fabricated or irrelevant.
   - 1-2:  Most citations are hallucinated or entirely wrong.

3. content_comprehensiveness (1-10):
   - 9-10: Covers all major sub-areas, methods, datasets, and evaluation paradigms.
   - 7-8:  Covers most sub-areas; one or two aspects underdeveloped.
   - 5-6:  Several important sub-areas missing or very thin.
   - 3-4:  Only covers one or two narrow aspects.
   - 1-2:  Barely scratches the surface of the topic.

4. structural_coherence (1-10):
   - 9-10: Clear logical flow (intro→background→methods→challenges→future).
           Sections well-proportioned, smooth transitions.
   - 7-8:  Generally logical; minor organizational issues.
   - 5-6:  Some sections disjointed; flow interrupted in places.
   - 3-4:  Poor organization; sections do not build on each other.
   - 1-2:  No discernible structure; stream-of-consciousness.

5. critical_analysis_depth (1-10):
   - 9-10: Goes beyond summarization; compares/contrasts methods, identifies
           patterns, explains WHY certain approaches succeed, synthesizes
           novel insights.
   - 7-8:  Some synthesis and comparison, but largely descriptive.
   - 5-6:  Mostly list-like descriptions of papers; minimal analysis.
   - 3-4:  Pure summarization without any critical perspective.
   - 1-2:  No analysis whatsoever; may just list paper titles.

6. writing_quality (1-10):
   - 9-10: Academic prose of publishable quality; precise, concise, appropriate
           register, no grammatical errors.
   - 7-8:  Good academic writing; minor phrasing issues.
   - 5-6:  Readable but informal in places; some grammatical errors.
   - 3-4:  Awkward phrasing, grammatical errors throughout.
   - 1-2:  Barely readable; severe language problems.
"""

_JUDGE_USER_TEMPLATE = """\
Please evaluate the following AI-generated academic literature survey.

**Research Topic**: {topic}

**Survey Content**:
{survey_text}

Evaluate according to the 6-dimension rubric and return ONLY the JSON object."""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class JudgeResult:
    """6-dimension quality scores for a generated survey."""

    citation_coverage: int
    citation_accuracy: int
    content_comprehensiveness: int
    structural_coherence: int
    critical_analysis_depth: int
    writing_quality: int
    overall_comment: str = ""
    overall_score: float = 0.0
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "citation_coverage": self.citation_coverage,
            "citation_accuracy": self.citation_accuracy,
            "content_comprehensiveness": self.content_comprehensiveness,
            "structural_coherence": self.structural_coherence,
            "critical_analysis_depth": self.critical_analysis_depth,
            "writing_quality": self.writing_quality,
            "overall_score": self.overall_score,
            "overall_comment": self.overall_comment,
        }

    def passed(self, threshold: float = 7.0) -> bool:
        """Check whether the overall score meets the quality threshold."""
        return self.overall_score >= threshold


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

class LLMJudge:
    """Score a MARS-generated survey using an independent LLM as judge.

    Parameters:
        model: LiteLLM model string (e.g., ``"openai/gpt-4o"``,
            ``"anthropic/claude-sonnet-4-20250514"``).
        api_key: Provider API key.  When ``None``, reads from the
            appropriate environment variable.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self._model = model
        self._api_key = api_key
        self._base_url = base_url

    def evaluate(
        self,
        generated_survey: str,
        topic: str,
        *,
        max_chars: int = 16000,
    ) -> JudgeResult:
        """Score *generated_survey* across the 6-dimension rubric.

        The survey text is truncated to *max_chars* to stay within the
        judge LLM's context window.
        """
        survey_text = generated_survey[:max_chars]

        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": _JUDGE_USER_TEMPLATE.format(
                topic=topic,
                survey_text=survey_text,
            )},
        ]

        raw = self._call_llm(messages)
        parsed = self._parse_response(raw)

        overall = round(sum([
            parsed["citation_coverage"],
            parsed["citation_accuracy"],
            parsed["content_comprehensiveness"],
            parsed["structural_coherence"],
            parsed["critical_analysis_depth"],
            parsed["writing_quality"],
        ]) / 6, 1)

        return JudgeResult(
            citation_coverage=parsed["citation_coverage"],
            citation_accuracy=parsed["citation_accuracy"],
            content_comprehensiveness=parsed["content_comprehensiveness"],
            structural_coherence=parsed["structural_coherence"],
            critical_analysis_depth=parsed["critical_analysis_depth"],
            writing_quality=parsed["writing_quality"],
            overall_comment=parsed.get("overall_comment", ""),
            overall_score=overall,
            raw_response=raw,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Send messages to the judge LLM and return the raw response string."""
        try:
            import litellm
            response = litellm.completion(
                model=self._model,
                messages=messages,
                api_key=self._api_key,
                api_base=self._base_url,
                temperature=0.0,  # deterministic scoring
                max_tokens=2000,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("LLM Judge call failed: %s", exc)
            # Return a valid JSON skeleton so downstream code doesn't crash
            return json.dumps({
                "citation_coverage": {"score": 0, "reasoning": f"Judge LLM error: {exc}"},
                "citation_accuracy": {"score": 0, "reasoning": ""},
                "content_comprehensiveness": {"score": 0, "reasoning": ""},
                "structural_coherence": {"score": 0, "reasoning": ""},
                "critical_analysis_depth": {"score": 0, "reasoning": ""},
                "writing_quality": {"score": 0, "reasoning": ""},
                "overall_comment": "Evaluation failed — judge LLM unavailable.",
            })

    @staticmethod
    def _parse_response(raw: str) -> Dict[str, Any]:
        """Parse the judge LLM's JSON response into a flat dict.

        Handles the nested ``{"citation_coverage": {"score": 8, ...}, ...}``
        format returned by the judge.
        """
        # Extract JSON block (may be wrapped in ```json fences)
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if not json_match:
            logger.warning("LLM Judge returned non-JSON response: %s", raw[:200])
            return _FALLBACK_SCORES

        try:
            data = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM Judge JSON: %s", raw[:200])
            return _FALLBACK_SCORES

        # Flatten nested {dim: {score: N, reasoning: "..."}} → {dim: N}
        dims = [
            "citation_coverage", "citation_accuracy",
            "content_comprehensiveness", "structural_coherence",
            "critical_analysis_depth", "writing_quality",
        ]
        result: Dict[str, Any] = {}
        for dim in dims:
            if isinstance(data.get(dim), dict):
                result[dim] = int(data[dim].get("score", 0))
            elif isinstance(data.get(dim), (int, float)):
                result[dim] = int(data[dim])
            else:
                result[dim] = 0

        result["overall_comment"] = data.get("overall_comment", "")
        return result


_FALLBACK_SCORES: Dict[str, Any] = {
    "citation_coverage": 0,
    "citation_accuracy": 0,
    "content_comprehensiveness": 0,
    "structural_coherence": 0,
    "critical_analysis_depth": 0,
    "writing_quality": 0,
    "overall_comment": "Evaluation failed — could not parse judge response.",
}
