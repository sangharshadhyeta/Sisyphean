"""Context compressor — summarizes large combined_ctx to a query-focused summary.

One LLM call converts tens-of-thousands of tokens (CLAUDE.md + recall) into
a focused ~200-word summary each decision step can actually use.

Falls back to word truncation on any LLM failure so the pipeline never blocks.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_SYSTEM = """\
Summarize ONLY the information relevant to the given task.
Output under 200 words. Include: key file paths, commands, relevant architecture notes, recent context.
Skip sections unrelated to the task. Be concise and factual."""


async def compress_context(query: str, context: str, client) -> str:
    """Compress large context to a query-focused summary.

    Returns the original unchanged if it's already short.
    Falls back to hard truncation on failure.
    """
    if not context:
        return ""

    word_count = len(context.split())
    if word_count <= 150:
        return context

    logger.debug("compressor: %d words → compressing for %r", word_count, query[:50])

    try:
        result = await client.generate(
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"Task: {query[:300]}\n\nContext:\n{context[:8000]}"},
            ],
            max_tokens=350,
            temperature=0.1,
            stream=False,
            thinking=False,
        )
        compressed = result["choices"][0]["message"]["content"].strip()
        if compressed and len(compressed.split()) > 10:
            logger.debug("compressor: %d → %d words", word_count, len(compressed.split()))
            return compressed
    except Exception as exc:
        logger.warning("compressor failed: %s", exc)

    words = context.split()
    return " ".join(words[:200])
