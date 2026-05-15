"""Bigram-based context extraction — selects relevant history per pipeline stage.

Each sub-task receives a focused slice of the conversation history via Jaccard
bigram scoring — the same approach used by engine.core.recall.Recall.  No LLM
call; runs synchronously and saves 1-3 round-trips per pipeline invocation.

For tool selection a keyword scorer is used (no extra LLM call).
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_MIN_SIM     = 0.04   # minimum Jaccard similarity to include a block
_MAX_WORDS   = 300    # hard cap on returned excerpt
_SHORT_LIMIT = 100    # histories ≤ this many words are returned as-is

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "to", "and", "or", "for", "with", "it",
    "this", "that", "be", "by", "at", "from", "as", "on", "do", "if",
    "can", "will", "what", "how", "i", "my", "me", "we", "you",
})


def _bigrams(text: str) -> set[str]:
    words = re.findall(r'\w+', text.lower())
    return {f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


async def extract_for_task(task: str, history: str, client) -> str:  # noqa: ARG001
    """Return history excerpts relevant to the given sub-task.

    Uses bigram Jaccard scoring to select relevant blocks — no LLM call.
    The `client` parameter is retained for API compatibility but unused.
    Returns empty string if nothing in history is relevant.
    """
    if not history or not task:
        return history or ""

    words = history.split()
    if len(words) <= _SHORT_LIMIT:
        return history  # Already short — return as-is

    # Split into exchange blocks (role-prefixed lines or blank-line separated)
    blocks = re.split(r'\n(?=(?:User|Assistant|Human|AI)\s*:)', history)
    if len(blocks) <= 1:
        # No role markers — split on blank lines
        blocks = [b.strip() for b in re.split(r'\n{2,}', history) if b.strip()]

    task_bg = _bigrams(task)
    scored: list[tuple[float, str]] = []
    for block in blocks:
        if not block.strip():
            continue
        block_bg = _bigrams(block)
        sim = _jaccard(task_bg, block_bg)
        if sim >= _MIN_SIM:
            scored.append((sim, block))

    if not scored:
        logger.debug("context_extractor: nothing relevant for %r", task[:50])
        return ""

    scored.sort(key=lambda x: x[0], reverse=True)
    relevant = [b for _, b in scored[:5]]
    result = "\n".join(relevant)

    result_words = result.split()
    if len(result_words) <= 5:
        return ""

    logger.debug(
        "context_extractor: %d → %d words (bigram, no LLM)",
        len(words), min(len(result_words), _MAX_WORDS),
    )
    return " ".join(result_words[:_MAX_WORDS])


def filter_tools_for_task(task: str, available_tools: list[dict]) -> list[dict]:
    """Return the subset of available tools relevant to this task.

    Scores each tool by word-overlap between the task text and the tool's name +
    description.  Returns top matches (min 2, max 5) so the planner always has
    options without being overwhelmed by an unrelated tool list.

    No LLM call — this runs synchronously before the planning stage.
    """
    if not available_tools:
        return []
    if not task:
        return available_tools[:5]

    task_words = set(task.lower().split()) - _STOPWORDS

    scored: list[tuple[int, dict]] = []
    for t in available_tools:
        name = t.get("name", "").lower()
        desc = (t.get("description", "") or "").lower()
        # Weight name twice — exact name match in task text is a strong signal
        combined_words = set(f"{name} {name} {desc}".split())
        score = len(task_words & combined_words)
        if name in task.lower():
            score += 5
        scored.append((score, t))

    scored.sort(key=lambda x: x[0], reverse=True)

    relevant = [t for s, t in scored if s > 0]
    if len(relevant) < 2:
        # Nothing matched — fall back to first two tools so the planner isn't empty
        relevant = [t for _, t in scored[:2]]
    return relevant[:5]
