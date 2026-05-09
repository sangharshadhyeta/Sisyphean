"""LLM-based context extraction — selects relevant history per pipeline stage.

Each sub-task gets a focused slice of the conversation history via a quick LLM call
rather than every stage receiving the full dump.  Gemma4 e4b has a 131k context
window so we pass the full history and let *it* decide what's relevant.

For tool selection a keyword scorer is used (no extra LLM call).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are a context filter for an AI pipeline stage.
Given a task and the full conversation history between the user and an AI assistant,
copy ONLY the portions of the history that are directly relevant to completing the task.
Preserve exact wording — do not paraphrase.
If no part of the history is relevant, output a single dash: -
Stay under 300 words. No commentary, no headings."""

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "to", "and", "or", "for", "with", "it",
    "this", "that", "be", "by", "at", "from", "as", "on", "do", "if",
    "can", "will", "what", "how", "i", "my", "me", "we", "you",
})


async def extract_for_task(task: str, history: str, client) -> str:
    """Return history excerpts relevant to the given sub-task.

    Passes the full history to the LLM (Gemma4 has 131k context window).
    Falls back to first 300 words on any failure.
    Returns empty string if the LLM signals nothing is relevant.
    """
    if not history or not task:
        return history or ""

    if len(history.split()) <= 100:
        return history  # Already short — no extraction needed

    logger.debug("context_extractor: task=%r history=%d words", task[:50], len(history.split()))
    try:
        result = await client.generate(
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"Task: {task}\n\nConversation history:\n{history}"},
            ],
            max_tokens=400,
            temperature=0.1,
            stream=False,
            thinking=False,
        )
        extracted = result["choices"][0]["message"]["content"].strip()
        if extracted and extracted != "-" and len(extracted.split()) > 5:
            logger.debug(
                "context_extractor: %d → %d words",
                len(history.split()), len(extracted.split()),
            )
            return extracted
        return ""  # LLM says nothing in history is relevant
    except Exception as exc:
        logger.warning("context_extractor: failed: %s", exc)

    # Fallback — first 300 words of raw history
    return " ".join(history.split()[:300])


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
