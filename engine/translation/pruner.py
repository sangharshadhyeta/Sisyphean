"""Content pruner — remove low-information content before LLM injection.

Ported from BirdClaw birdclaw/llm/pruner.py.

Two tiers:

  keyword_prune(text, goal, max_chars)
      Zero LLM cost. Scores every sentence/line by keyword overlap with the
      goal, keeps the highest-scoring ones up to max_chars. Used everywhere:
      web search snippets, GraphRAG context, planning context, file reads.

  semantic_prune(text, goal, client, max_chars)
      One cheap LLM call. Asks Gemma to extract only the sentences relevant to
      the goal. Used only for large unstructured content where keyword overlap
      is noisy (raw web page text).

Both functions are safe to call with empty input — they return "" or the
original text unchanged when content is already short.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_MIN_PRUNE_CHARS = 200

_STOP = frozenset({
    "the", "and", "for", "are", "was", "this", "that", "with", "have",
    "from", "they", "will", "been", "had", "has", "its", "not", "but",
    "can", "all", "one", "you", "your", "our", "their", "also", "more",
})


def _tokenise(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9]+", text.lower())
        if len(w) > 2 and w not in _STOP
    }


def _split_chunks(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) > 3:
        return sentences
    return [ln for ln in text.splitlines() if ln.strip()]


def keyword_prune(text: str, goal: str, max_chars: int = 800) -> str:
    """Score chunks by keyword overlap with goal; return top chunks up to max_chars.

    Preserves reading order within the budget.
    """
    if not text or len(text) <= _MIN_PRUNE_CHARS:
        return text
    if not goal:
        return text[:max_chars]

    goal_tokens = _tokenise(goal)
    if not goal_tokens:
        return text[:max_chars]

    chunks = _split_chunks(text)
    scored: list[tuple[int, int, str]] = []
    for i, chunk in enumerate(chunks):
        score = len(goal_tokens & _tokenise(chunk))
        scored.append((score, i, chunk))

    scored.sort(key=lambda t: (-t[0], t[1]))

    selected: set[int] = set()
    budget = max_chars
    for score, idx, chunk in scored:
        if budget <= 0:
            break
        selected.add(idx)
        budget -= len(chunk) + 1

    result = "\n".join(chunk for i, chunk in enumerate(chunks) if i in selected)
    return result.strip() or text[:max_chars]


async def semantic_prune(text: str, goal: str, client, max_chars: int = 800) -> str:
    """Use Gemma to extract sentences relevant to goal from large unstructured text.

    Only used for content that keyword pruning handles poorly (raw web pages
    where the goal vocabulary doesn't appear verbatim in the relevant section).
    Falls back to keyword_prune on any error.
    """
    if not text or len(text) <= _MIN_PRUNE_CHARS:
        return text
    if not goal:
        return text[:max_chars]

    kw_result = keyword_prune(text, goal, max_chars)
    if len(kw_result) <= max_chars * 1.1:
        return kw_result

    try:
        trimmed = text[:2000]
        result = await client.generate(
            [
                {
                    "role": "system",
                    "content": (
                        "Extract only the sentences from the text that are relevant to the goal. "
                        "Return them as plain text. Omit navigation, ads, headers, footers, "
                        "unrelated paragraphs. Keep all relevant technical details verbatim."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Goal: {goal[:200]}\n\nText:\n{trimmed}",
                },
            ],
            max_tokens=512,
            temperature=0.1,
            stream=False,
            thinking=False,
        )
        extracted = result["choices"][0]["message"]["content"].strip()
        if extracted and len(extracted) > 50:
            logger.debug("semantic_prune: %d→%d chars", len(text), len(extracted))
            return extracted[:max_chars]
    except Exception as exc:
        logger.debug("semantic_prune: LLM failed (%s) — falling back to keyword_prune", exc)

    return kw_result
