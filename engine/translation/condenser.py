"""Web content condenser — two-phase processing of raw web/tool content.

Ported and adapted from BirdClaw birdclaw/tools/condenser.py.

Key differences from BirdClaw:
- Async (asyncio) instead of threading — fits FastAPI/uvicorn
- No page_store dependency — results returned directly
- No pending_notes queue — distillation is called inline
- Uses LlamaClient instead of BirdClaw's llm_client singleton

Phase 1 — fast_clean(text) [sync, no LLM]:
    Strip noise from already-cleaned text.
    Returns first ~1200 chars of cleaned text immediately.
    Used to get something usable into context right now.

Phase 2 — distill(text, goal, client) [async, one Gemma call]:
    Full LLM pass: read clean text + current task goal.
    Produces {"cleaned": "...", "notes": "..."}
      cleaned  — condensed markdown of general content (up to 1500 chars)
      notes    — task-focused extract, what's relevant RIGHT NOW (up to 500 chars)
    Returns the notes string — inject this into decide_next_action's tool context.

Usage:
    from engine.translation.condenser import fast_clean, distill

    # Immediate context (no LLM)
    snippet = fast_clean(raw_html_stripped_text)

    # Task-focused extract (one Gemma call, ~0.5s)
    notes = await distill(raw_text, goal="current stage goal", client=llm_client)
"""
from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

# Fast-path: max chars returned without LLM
_FAST_PATH_CHARS = 1200

# LLM input cap — protect Gemma 4B context window
_LLM_INPUT_CHARS = 3000

# Minimum content size worth processing
_MIN_CHARS = 100


# ---------------------------------------------------------------------------
# Phase 1 — sync fast-path (no LLM)
# ---------------------------------------------------------------------------

def fast_clean(text: str) -> str:
    """Instant cleanup for immediate context injection. No LLM, no delay.

    - Collapses blank lines
    - Removes pure-noise lines (navigation, cookie banners, etc.)
    - Returns first _FAST_PATH_CHARS characters
    """
    if not text or len(text) < _MIN_CHARS:
        return text

    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [
        ln for ln in text.splitlines()
        if len(ln.strip()) > 3 and not re.match(r"^[\s\W]{0,3}$", ln)
    ]
    return "\n".join(lines)[:_FAST_PATH_CHARS]


# ---------------------------------------------------------------------------
# Phase 2 — async LLM distillation
# ---------------------------------------------------------------------------

_DISTILL_PROMPT = """\
You are a content distiller. Read the content below and produce a JSON object.

Rules:
- "cleaned": condensed markdown — remove boilerplate, keep facts, code, data. Max 1500 chars.
- "notes": extract only what is directly relevant to the goal. Max 500 chars.
- Output ONLY the JSON object. No explanation.

Goal: {goal}
Source: {source}

Content:
{content}

Output format:
{{"cleaned": "...", "notes": "..."}}"""


async def distill(
    text: str,
    goal: str,
    client,
    source: str = "tool_result",
) -> str:
    """Ask Gemma to extract the relevant parts from content given a goal.

    Returns the task-focused "notes" string (up to 500 chars).
    Falls back to fast_clean() on any error.

    Args:
        text:    Raw or HTML-stripped content to distill.
        goal:    Current stage goal — guides relevance extraction.
        client:  LlamaClient instance.
        source:  Label for logging (URL, tool name, etc.)
    """
    if not text or len(text) < _MIN_CHARS:
        return text

    clean = fast_clean(text)
    if len(text) <= _FAST_PATH_CHARS:
        return clean   # already short enough — no LLM needed

    truncated = text[:_LLM_INPUT_CHARS]
    prompt = _DISTILL_PROMPT.format(
        goal=goal[:200],
        source=source[:100],
        content=truncated,
    )

    try:
        result = await client.generate(
            [{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.1,
            response_format={"type": "json_object"},  # "{" prefix on llama.cpp
            stream=False,
            thinking=False,
        )
        raw = result["choices"][0]["message"]["content"].strip()
        parsed = _parse_json(raw)
        if parsed:
            notes = parsed.get("notes") or parsed.get("cleaned", "")
            if notes and len(notes) > 20:
                logger.debug("distill: %d→%d chars  source=%s", len(text), len(notes), source[:50])
                return notes[:500]
    except Exception as exc:
        logger.debug("distill: LLM failed (%s) — using fast_clean", exc)

    return clean


# ---------------------------------------------------------------------------
# Convenience: distill only if content exceeds threshold
# ---------------------------------------------------------------------------

async def maybe_distill(
    text: str,
    goal: str,
    client,
    source: str = "tool_result",
    threshold: int = 1500,
) -> str:
    """Distill only if text exceeds threshold; otherwise return as-is.

    This is the function called by decide_next_action() — cheap on small
    results, one Gemma call on large ones.
    """
    if not text or len(text) <= threshold:
        return text
    return await distill(text, goal, client, source)


# ---------------------------------------------------------------------------
# HTML stripping (sync, no extra deps if bs4 unavailable)
# ---------------------------------------------------------------------------

def strip_html(html: str) -> str:
    """Strip HTML tags and return plain text. Uses BeautifulSoup if available."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        body = soup.find("article") or soup.find("main") or soup.find("body")
        return body.get_text(separator="\n", strip=True) if body else soup.get_text(separator="\n", strip=True)
    except ImportError:
        # Fallback: regex-based tag stripping
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"&[a-z]+;", " ", text)
        return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None
