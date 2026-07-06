"""
LLM-as-Judge quality evaluator for MARS-generated surveys.

Uses an independent LLM (GPT-4 / Claude) to score a generated survey across
the full 12-dimension rubric from Agentic AutoSurvey (Liu et al., 2025).

Dimensions (3 categories × 4 dimensions each):

- **Core Quality (60%)**: citation_coverage, citation_accuracy,
  synthesis_quality, organization
- **Writing Quality (20%)**: readability, academic_rigor, clarity, coherence
- **Content Depth (20%)**: comprehensiveness, critical_analysis,
  novelty_insights, future_directions
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 12-dimension scoring rubric (Agentic AutoSurvey, 2025)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """\
You are an expert academic survey evaluator. Your task is to assess the
quality of an AI-generated academic literature survey against a rigorous,
12-dimension rubric derived from the Agentic AutoSurvey framework.

You MUST respond with ONLY a valid JSON object — no preamble, no commentary.

The JSON object must have exactly this structure:

{
  "citation_coverage":    {"score": 0, "reasoning": "..."},
  "citation_accuracy":    {"score": 0, "reasoning": "..."},
  "synthesis_quality":    {"score": 0, "reasoning": "..."},
  "organization":         {"score": 0, "reasoning": "..."},
  "readability":          {"score": 0, "reasoning": "..."},
  "academic_rigor":       {"score": 0, "reasoning": "..."},
  "clarity":              {"score": 0, "reasoning": "..."},
  "coherence":            {"score": 0, "reasoning": "..."},
  "comprehensiveness":    {"score": 0, "reasoning": "..."},
  "critical_analysis":    {"score": 0, "reasoning": "..."},
  "novelty_insights":     {"score": 0, "reasoning": "..."},
  "future_directions":    {"score": 0, "reasoning": "..."},
  "overall_comment": "<2-3 sentence overall assessment>"
}

Each dimension is scored 1-10 (integer).

--- Core Quality (60% weight) ---

1. citation_coverage (1-10):
   - 9-10: Cites virtually every key foundational + recent paper in the area.
   - 7-8:  Covers most key papers; a few notable omissions.
   - 5-6:  Covers ~half the expected key papers; several gaps.
   - 3-4:  Major gaps; many expected papers missing.
   - 1-2:  Nearly empty or entirely irrelevant reference list.

2. citation_accuracy (1-10):
   - 9-10: Every citation is real, correctly placed, and supports the claim.
   - 7-8:  Most citations accurate; 1-2 minor misplacements or errors.
   - 5-6:  Several citations suspicious, misplaced, or unsupported.
   - 3-4:  Many citations appear fabricated or irrelevant to the claims.
   - 1-2:  Most citations are hallucinated or entirely wrong.

3. synthesis_quality (1-10):
   - 9-10: Deep cross-paper synthesis; identifies patterns, contradictions,
           and connections across works; goes far beyond listing papers.
   - 7-8:  Good synthesis in most sections; some sections read as paper lists.
   - 5-6:  Partial synthesis; mostly sequential paper summaries.
   - 3-4:  Lists papers with minimal connection between them.
   - 1-2:  No synthesis whatsoever; each paper described in isolation.

4. organization (1-10):
   - 9-10: Flawless hierarchical structure; sections naturally build on each
           other; logical progression from broad to specific.
   - 7-8:  Generally well-organized; minor structural issues in 1-2 sections.
   - 5-6:  Adequate structure but some sections disjointed or misplaced.
   - 3-4:  Poor organization; sections do not flow logically.
   - 1-2:  No discernible organization; stream-of-consciousness.

--- Writing Quality (20% weight) ---

5. readability (1-10):
   - 9-10: Effortlessly readable; appropriate sentence variety; natural flow.
   - 7-8:  Readable throughout; occasional awkward sentences.
   - 5-6:  Readable but dense or choppy in places.
   - 3-4:  Difficult to follow; long, convoluted sentences.
   - 1-2:  Extremely difficult to read; requires re-reading to understand.

6. academic_rigor (1-10):
   - 9-10: Impeccable academic standards; precise terminology, proper hedging,
           balanced treatment of competing views.
   - 7-8:  Good academic standards; minor lapses in precision or balance.
   - 5-6:  Adequate but informal in places; some unsupported assertions.
   - 3-4:  Lacks academic rigor; overclaims, insufficient hedging.
   - 1-2:  Non-academic style; opinionated, unsupported, or promotional.

7. clarity (1-10):
   - 9-10: Crystal-clear exposition; complex ideas explained accessibly;
           terminology defined on first use.
   - 7-8:  Clear throughout; one or two concepts could be better explained.
   - 5-6:  Generally clear but some passages ambiguous or jargon-heavy.
   - 3-4:  Many unclear passages; readers would struggle to follow arguments.
   - 1-2:  Pervasive ambiguity; core ideas are incomprehensible.

8. coherence (1-10):
   - 9-10: Seamless transitions between sections and paragraphs; strong
           thematic threads throughout; reads as one unified work.
   - 7-8:  Good coherence; occasional abrupt transitions.
   - 5-6:  Adequate but some sections feel disconnected from the whole.
   - 3-4:  Sections read as independent essays; weak connective tissue.
   - 1-2:  No coherence; paragraphs are randomly ordered.

--- Content Depth (20% weight) ---

9. comprehensiveness (1-10):
   - 9-10: Exhaustive coverage of all major sub-areas, methods, datasets,
           evaluation paradigms, and application domains.
   - 7-8:  Covers most sub-areas; one or two aspects underdeveloped.
   - 5-6:  Several important sub-areas missing or only briefly mentioned.
   - 3-4:  Only covers one or two narrow aspects of the topic.
   - 1-2:  Barely scratches the surface; misses most of the field.

10. critical_analysis (1-10):
    - 9-10: Penetrating analysis; identifies fundamental limitations,
            assumptions, and trade-offs; evaluates methodological soundness.
    - 7-8:  Good critical perspective; identifies major strengths/weaknesses.
    - 5-6:  Some critical commentary but largely descriptive.
    - 3-4:  Minimal critical analysis; mostly accepts papers at face value.
    - 1-2:  No critical perspective; purely descriptive summaries.

11. novelty_insights (1-10):
    - 9-10: Offers original insights beyond the cited papers; identifies
            non-obvious connections; proposes novel taxonomies or frameworks.
    - 7-8:  Some original observations; goes beyond paper content in places.
    - 5-6:  Occasional insights but mostly restates existing work.
    - 3-4:  Few original ideas; essentially a compilation of paper summaries.
    - 1-2:  No novel insights; purely derivative.

12. future_directions (1-10):
    - 9-10: Compelling, concrete, and actionable future research directions;
            identifies specific open problems with clear motivation.
    - 7-8:  Good future directions with reasonable specificity.
    - 5-6:  Generic future directions; could apply to many fields.
    - 3-4:  Vague or obvious directions; lacks specificity.
    - 1-2:  No future directions or entirely meaningless ones.

--- Scoring weights ---
Overall score = (citation_coverage + citation_accuracy + synthesis_quality + organization) / 4 * 0.6
              + (readability + academic_rigor + clarity + coherence) / 4 * 0.2
              + (comprehensiveness + critical_analysis + novelty_insights + future_directions) / 4 * 0.2
"""

_JUDGE_USER_TEMPLATE = """\
Please evaluate the following AI-generated academic literature survey.

**Research Topic**: {topic}

**Survey Content**:
{survey_text}

Evaluate according to the 12-dimension rubric and return ONLY the JSON object."""

# ---------------------------------------------------------------------------
# All 12 dimension keys
# ---------------------------------------------------------------------------

_DIM_KEYS = [
    # Core Quality (60%)
    "citation_coverage",
    "citation_accuracy",
    "synthesis_quality",
    "organization",
    # Writing Quality (20%)
    "readability",
    "academic_rigor",
    "clarity",
    "coherence",
    # Content Depth (20%)
    "comprehensiveness",
    "critical_analysis",
    "novelty_insights",
    "future_directions",
]

_DIM_LABELS: Dict[str, str] = {
    "citation_coverage": "Citation Coverage",
    "citation_accuracy": "Citation Accuracy",
    "synthesis_quality": "Synthesis Quality",
    "organization": "Organization",
    "readability": "Readability",
    "academic_rigor": "Academic Rigor",
    "clarity": "Clarity",
    "coherence": "Coherence",
    "comprehensiveness": "Comprehensiveness",
    "critical_analysis": "Critical Analysis",
    "novelty_insights": "Novelty / Insights",
    "future_directions": "Future Directions",
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class JudgeResult:
    """12-dimension quality scores for a generated survey."""

    # Core Quality
    citation_coverage: int = 0
    citation_accuracy: int = 0
    synthesis_quality: int = 0
    organization: int = 0

    # Writing Quality
    readability: int = 0
    academic_rigor: int = 0
    clarity: int = 0
    coherence: int = 0

    # Content Depth
    comprehensiveness: int = 0
    critical_analysis: int = 0
    novelty_insights: int = 0
    future_directions: int = 0

    overall_comment: str = ""
    overall_score: float = 0.0
    core_quality_score: float = 0.0
    writing_quality_score: float = 0.0
    content_depth_score: float = 0.0
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "core_quality": {
                "citation_coverage": self.citation_coverage,
                "citation_accuracy": self.citation_accuracy,
                "synthesis_quality": self.synthesis_quality,
                "organization": self.organization,
                "category_score": round(self.core_quality_score, 1),
            },
            "writing_quality": {
                "readability": self.readability,
                "academic_rigor": self.academic_rigor,
                "clarity": self.clarity,
                "coherence": self.coherence,
                "category_score": round(self.writing_quality_score, 1),
            },
            "content_depth": {
                "comprehensiveness": self.comprehensiveness,
                "critical_analysis": self.critical_analysis,
                "novelty_insights": self.novelty_insights,
                "future_directions": self.future_directions,
                "category_score": round(self.content_depth_score, 1),
            },
            "overall_score": self.overall_score,
            "overall_comment": self.overall_comment,
        }

    def passed(self, threshold: float = 7.0) -> bool:
        return self.overall_score >= threshold

    def to_flat_dict(self) -> Dict[str, Any]:
        """Flat dict for CSV/table output."""
        return {
            "citation_coverage": self.citation_coverage,
            "citation_accuracy": self.citation_accuracy,
            "synthesis_quality": self.synthesis_quality,
            "organization": self.organization,
            "readability": self.readability,
            "academic_rigor": self.academic_rigor,
            "clarity": self.clarity,
            "coherence": self.coherence,
            "comprehensiveness": self.comprehensiveness,
            "critical_analysis": self.critical_analysis,
            "novelty_insights": self.novelty_insights,
            "future_directions": self.future_directions,
            "overall_score": self.overall_score,
            "core_quality": round(self.core_quality_score, 1),
            "writing_quality": round(self.writing_quality_score, 1),
            "content_depth": round(self.content_depth_score, 1),
        }


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
        """Score *generated_survey* across the 12-dimension rubric."""
        survey_text = generated_survey[:max_chars]

        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": _JUDGE_USER_TEMPLATE.format(
                topic=topic, survey_text=survey_text,
            )},
        ]

        raw = self._call_llm(messages)
        parsed = self._parse_response(raw)

        # Compute weighted scores
        core_quality = sum(
            parsed[k] for k in _DIM_KEYS[:4]
        ) / 4
        writing_quality = sum(
            parsed[k] for k in _DIM_KEYS[4:8]
        ) / 4
        content_depth = sum(
            parsed[k] for k in _DIM_KEYS[8:12]
        ) / 4
        overall = round(
            core_quality * 0.6 + writing_quality * 0.2 + content_depth * 0.2, 1,
        )

        return JudgeResult(
            citation_coverage=parsed["citation_coverage"],
            citation_accuracy=parsed["citation_accuracy"],
            synthesis_quality=parsed["synthesis_quality"],
            organization=parsed["organization"],
            readability=parsed["readability"],
            academic_rigor=parsed["academic_rigor"],
            clarity=parsed["clarity"],
            coherence=parsed["coherence"],
            comprehensiveness=parsed["comprehensiveness"],
            critical_analysis=parsed["critical_analysis"],
            novelty_insights=parsed["novelty_insights"],
            future_directions=parsed["future_directions"],
            overall_comment=parsed.get("overall_comment", ""),
            overall_score=overall,
            core_quality_score=round(core_quality, 1),
            writing_quality_score=round(writing_quality, 1),
            content_depth_score=round(content_depth, 1),
            raw_response=raw,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        try:
            import litellm
            response = litellm.completion(
                model=self._model,
                messages=messages,
                api_key=self._api_key,
                api_base=self._base_url,
                temperature=0.0,
                max_tokens=3000,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("LLM Judge call failed: %s", exc)
            return json.dumps({
                k: {"score": 0, "reasoning": f"Judge LLM error: {exc}"}
                for k in _DIM_KEYS
            } | {"overall_comment": "Evaluation failed — judge LLM unavailable."})

    @staticmethod
    def _parse_response(raw: str) -> Dict[str, Any]:
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if not json_match:
            logger.warning("LLM Judge returned non-JSON: %s", raw[:200])
            return _fallback()

        try:
            data = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            logger.warning("Failed to parse Judge JSON: %s", raw[:200])
            return _fallback()

        result: Dict[str, Any] = {}
        for dim in _DIM_KEYS:
            if isinstance(data.get(dim), dict):
                result[dim] = int(data[dim].get("score", 0))
            elif isinstance(data.get(dim), (int, float)):
                result[dim] = int(data[dim])
            else:
                result[dim] = 0

        result["overall_comment"] = data.get("overall_comment", "")
        return result


def _fallback() -> Dict[str, Any]:
    return {
        dim: 0 for dim in _DIM_KEYS
    } | {"overall_comment": "Evaluation failed — could not parse judge response."}
