"""Synthesizer — assembles the final answer from sub-task results.

One LLM call. Plain text output (not JSON).

Context given to the model:
  - combined_ctx: recall + CLAUDE.md (Option B/C) — already partitioned per call
  - Original query
  - Relevant soul section
  - User preferences
  - One-line summary per sub-task result
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are Sisyphean. Answer the user's question directly and specifically using the results below. "
    "Extract exact facts, numbers, and names from the results — do not paraphrase vaguely. "
    "If context is provided, use it to understand references to prior work or conversation. "
    "If soul guidance is given, follow it strictly. "
    "Be terse. No filler. No hollow openers. No '/think'."
)


async def synthesize(
    query: str,
    soul_section: str,
    user_prefs: str,
    results: list[dict],
    client,
    context: str = "",
) -> str:
    """Produce the final plain-text answer from all sub-task results."""
    parts: list[str] = []

    if context:
        parts.append(f"Context:\n{context[:600]}")

    parts.append(f"User: {query[:200]}")

    if soul_section:
        parts.append(f"Soul guidance (follow this):\n{soul_section[:300]}")

    if user_prefs:
        parts.append(f"User preferences:\n{user_prefs[:200]}")

    good, weak, failed = [], [], []
    for r in results:
        if r.get("outer"):
            continue
        if r.get("result") is None:
            continue
        q = r.get("_quality", "good")
        summary = r.get("summary", "") or r.get("result", "")
        if q in ("empty", "error"):
            failed.append(f"[FAILED — {r.get('tool','?')}: {r.get('input','')[:40]}]")
        elif q == "weak":
            weak.append(f"- (weak) {summary[:300]}")
        else:
            good.append(f"- {summary[:2000]}")

    if good:
        parts.append("Results:\n" + "\n".join(good))
    if weak:
        parts.append("Partial results (low confidence):\n" + "\n".join(weak))
    if failed:
        parts.append(
            "These steps failed to return useful information — note the gap in your answer:\n"
            + "\n".join(failed)
        )

    prompt = "\n\n".join(p for p in parts if p.strip())

    try:
        result = await client.generate(
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=1024,
            temperature=0.3,
            stream=False,
            thinking=False,
        )
        text = (result["choices"][0]["message"]["content"] or "").strip()
        if text:
            logger.debug("synthesize: %d chars", len(text))
            return text
    except Exception as exc:
        logger.warning("synthesize failed: %s", exc)

    best = good or weak
    if best:
        return best[-1][:500]
    return "(no answer)"
