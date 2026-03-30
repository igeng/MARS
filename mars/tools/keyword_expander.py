"""
Keyword expander tool.

Uses the configured LLM to generate synonyms, related terms, and
Chinese/English translations for a given research keyword.
This broadens search coverage when querying academic databases.
"""

from __future__ import annotations

import json
from typing import Any

from crewai.tools import BaseTool

from mars.utils.llm_factory import get_llm


class KeywordExpanderTool(BaseTool):
    """Expand a research keyword into related terms using an LLM."""

    name: str = "keyword_expander"
    description: str = (
        "Expand research keywords into synonyms and related terms to improve "
        "academic search coverage. "
        "Input should be a JSON string with keys: "
        "'keyword' (required), "
        "'provider' (optional, LLM provider: qwen/deepseek/kimi/glm). "
        "Returns a list of expanded keywords."
    )

    def _run(self, query_json: str) -> str:
        try:
            params: dict[str, Any] = json.loads(query_json)
        except json.JSONDecodeError:
            params = {"keyword": query_json}

        keyword: str = params.get("keyword", "")
        provider: str | None = params.get("provider")

        if not keyword:
            return "Error: 'keyword' parameter is required."

        prompt = (
            f"You are a research keyword expert. "
            f"Given the research keyword: '{keyword}', generate:\n"
            f"1. 5-8 synonyms or closely related terms in English\n"
            f"2. 3-5 broader/narrower related concepts\n"
            f"3. Common abbreviations if any\n"
            f"4. Chinese translation(s)\n\n"
            f"Return ONLY a JSON object with keys: "
            f"'synonyms', 'related_concepts', 'abbreviations', 'chinese'.\n"
            f"Each value is a list of strings."
        )

        try:
            llm = get_llm(provider=provider)
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            # Try to parse JSON from the response
            json_match = _extract_json(content)
            if json_match:
                data = json.loads(json_match)
                all_terms: list[str] = []
                for key in ("synonyms", "related_concepts", "abbreviations", "chinese"):
                    all_terms.extend(data.get(key, []))

                lines = [f"Expanded keywords for '{keyword}':\n"]
                lines.append(f"Synonyms: {', '.join(data.get('synonyms', []))}")
                lines.append(
                    f"Related concepts: {', '.join(data.get('related_concepts', []))}"
                )
                lines.append(
                    f"Abbreviations: {', '.join(data.get('abbreviations', []))}"
                )
                lines.append(f"Chinese: {', '.join(data.get('chinese', []))}")
                lines.append(f"\nAll terms: {all_terms}")
                return "\n".join(lines)
            else:
                return f"Keyword expansion for '{keyword}':\n{content}"
        except Exception as exc:
            return f"Keyword expansion failed: {exc}"


def _extract_json(text: str) -> str | None:
    """Extract the first JSON object from a text string."""
    import re

    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None
