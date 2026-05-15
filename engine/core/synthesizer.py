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

_SYSTEM_WITH_RESULTS = (
    "You are Sisyphean. Answer the user's question using the results below.\n"
    "Rules:\n"
    "- Synthesize the results into a coherent, informative answer — do not just quote one line.\n"
    "- Extract exact facts, numbers, and names. Do not paraphrase vaguely.\n"
    "- If multiple results are given, combine them into a complete picture.\n"
    "- If context is provided, use it to understand references to prior conversation.\n"
    "- If soul guidance is given, follow it.\n"
    "- Be direct. No hollow openers ('Great question!', 'Certainly!'). No '/think'."
)

# Used when there are no tool results — the results-focused instruction confuses small models.
_SYSTEM_NO_RESULTS = (
    "You are Sisyphean, a local AI assistant running entirely on this machine — no cloud, no external server.\n"
    "Reply to the user naturally and helpfully based on the conversation so far.\n"
    "Rules:\n"
    "- Give a substantive, specific answer — not a single sentence.\n"
    "- Draw on the conversation context if provided.\n"
    "- If soul guidance is given, follow it.\n"
    "- No hollow openers. No '/think'."
)


# Tools whose results benefit from LLM synthesis (combining, interpreting, shaping).
# Everything else (bash, write, edit, save_memory) is self-describing — return directly.
_RESEARCH_TOOLS = frozenset({
    "web_search", "search_memory", "search_knowledge",
    "search_policy", "websearch", "webfetch",
})
_WRITE_TOOLS  = frozenset({"write", "edit", "write_plan"})
_BASH_TOOLS   = frozenset({"bash", "powershell"})
_MEMORY_TOOLS = frozenset({"save_memory", "remember"})


def _fast_path(results: list[dict]) -> str | None:
    """Return a deterministic response when no research results are present.

    Covers three purely mechanical outcomes:
      bash / powershell  → return stdout directly
      write / edit       → "Done. Written to `X`."
      save_memory        → "Saved: X"

    Returns None when any research/web result is present — those need LLM synthesis.
    """
    valid = [r for r in results if r.get("result") is not None]
    if not valid:
        return None
    if any(r.get("tool", "").lower() in _RESEARCH_TOOLS for r in valid):
        return None  # has research results → let LLM synthesize

    lines: list[str] = []
    for r in valid:
        tool = (r.get("tool") or "").lower()
        inp  = (r.get("input") or "").strip()[:80]
        out  = (r.get("result") or "").strip()

        if tool in _BASH_TOOLS:
            if out:
                # Truncate very long outputs from the front, keep the tail
                # (test runners, build tools put the important bit at the end)
                trimmed = out if len(out) <= 800 else f"[…]\n{out[-800:]}"
                lines.append(trimmed)
        elif tool in _WRITE_TOOLS:
            lines.append(f"Done. Written to `{inp}`." if inp else "Done.")
        elif tool in _MEMORY_TOOLS:
            lines.append(f"Saved: {inp}" if inp else "Saved.")
        elif out:
            lines.append(out[:300])

    return "\n\n".join(lines) if lines else None


async def synthesize(
    query: str,
    soul_section: str,
    user_prefs: str,
    results: list[dict],
    client,
    context: str = "",
) -> str:
    """Produce the final plain-text answer from all sub-task results.

    Fast-path: bash/write/save_memory tasks return a deterministic template
    without an LLM call — the output IS the answer.
    LLM call is kept for: web search results, conversational turns, multi-source
    synthesis, and any task whose results require interpretation or combination.
    """
    fast = _fast_path(results)
    if fast:
        logger.debug("synthesize: fast-path (%d chars)", len(fast))
        return fast

    parts: list[str] = []

    if context:
        parts.append(f"Context:\n{context[:600]}")

    parts.append(f"User: {query[:200]}")

    if soul_section:
        parts.append(f"Soul guidance (follow this):\n{soul_section[:300]}")

    # Only inject user_prefs when there are actual results to synthesize.
    # For pure conversational turns (greeting, simple questions) the model
    # tends to echo the prefs back verbatim rather than using them as guidance.
    if user_prefs and results:
        parts.append(f"User preferences:\n{user_prefs[:200]}")

    good, weak, failed = [], [], []
    outer_done: list[str] = []  # write/edit/bash tool summaries

    for r in results:
        if r.get("outer"):
            # Track outer tool completions so we can summarise them when no inner results exist
            tool = r.get("tool", "")
            inp  = r.get("input", "")[:60]
            res  = (r.get("result") or r.get("summary") or "")[:80]
            if tool in ("write", "edit", "bash") and res:
                outer_done.append(f"[{tool}: {inp}] → {res}")
            continue
        if r.get("result") is None:
            continue
        q = r.get("_quality", "good")
        # Use full result content for synthesis; fall back to summary only if result missing
        body = r.get("result") or r.get("summary", "")
        if q in ("empty", "error"):
            failed.append(f"[FAILED — {r.get('tool','?')}: {r.get('input','')[:40]}]")
        elif q == "weak":
            weak.append(f"- (weak) {body[:500]}")
        else:
            good.append(f"- {body[:3000]}")

    # When there are no inner results (write-only tasks), summarise what was done
    # so the synthesizer can confirm completion rather than hallucinating reasoning.
    if not good and not weak and outer_done:
        good.extend(f"- {s}" for s in outer_done)

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

    # Switch system prompt based on whether there are actual results to synthesize.
    # The results-focused instruction confuses small models when there's nothing to cite.
    has_results = bool(good or weak)
    system = _SYSTEM_WITH_RESULTS if has_results else _SYSTEM_NO_RESULTS

    try:
        result = await client.generate(
            [
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=1500,
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
