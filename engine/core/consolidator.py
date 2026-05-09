"""Consolidator — assembles the final answer from sub-task results.

One LLM call. Plain text output (not JSON) — small models are more fluent
in natural language than structured formats for final answers.

Context given to the model (minimal):
  - Original query
  - Relevant soul section (if any matched)
  - User preferences (if any)
  - One-line summary per sub-task result
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are Sisyphean. Answer the user's question directly and specifically using the results below. "
    "Extract exact facts, numbers, and names from the results — do not paraphrase vaguely. "
    "If soul guidance is given, follow it strictly. "
    "Be terse. No filler. No hollow openers. No '/think'."
)


async def consolidate(
    query: str,
    soul_section: str,
    user_prefs: str,
    results: list[dict],
    client,
) -> str:
    """Produce the final plain-text answer from all sub-task results.

    Falls back to a best-effort answer from available results on failure.
    """
    parts: list[str] = [f"User: {query[:200]}"]

    if soul_section:
        parts.append(f"Soul guidance (follow this):\n{soul_section[:300]}")

    if user_prefs:
        parts.append(f"User preferences:\n{user_prefs[:200]}")

    # Collect result summaries — skip outer tool placeholders (no result yet)
    summaries = [
        r.get("summary", "") or r.get("result", "")
        for r in results
        if r.get("result") is not None and not r.get("outer")
    ]
    if summaries:
        parts.append("Results:\n" + "\n".join(f"- {s[:500]}" for s in summaries))

    prompt = "\n\n".join(p for p in parts if p.strip())

    try:
        result = await client.generate(
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=512,
            temperature=0.3,
            stream=False,
            thinking=False,
            # No response_format — plain text is more reliable for final answers
        )
        text = (result["choices"][0]["message"]["content"] or "").strip()
        if text:
            logger.debug("consolidate: %d chars", len(text))
            return text
    except Exception as exc:
        logger.warning("consolidate failed: %s", exc)

    # Fallback: stitch summaries together
    if summaries:
        return summaries[-1][:500]
    return "(no answer)"
