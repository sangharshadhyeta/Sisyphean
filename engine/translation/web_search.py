"""Web search stub for Sisyphean.

Sisyphean is a pure reasoning engine — web search is executed by the harness
(BirdClaw) via the WebSearch tool_use block, or by Claude Code's built-in
WebSearch tool.  This module exists only to satisfy internal loop references;
search() always returns [] so the model falls through to using the external
WebSearch tool instead.

fetch() still works directly (httpx only, no search-package dependency) so
the executor can fetch URLs when given one explicitly.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from engine.translation.pruner import keyword_prune
from engine.translation.condenser import fast_clean, distill, strip_html

logger = logging.getLogger(__name__)

_SEARCH_TOKEN_RE = re.compile(r"\[SEARCH:\s*([^\]]+)\]", re.IGNORECASE)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str

    def to_context(self) -> str:
        return f"**{self.title}**\n{self.snippet}\nSource: {self.url}"


# ---------------------------------------------------------------------------
# Search — stub (real search handled by BirdClaw via WebSearch tool_use)
# ---------------------------------------------------------------------------

async def search(query: str, max_results: int = 5) -> list[SearchResult]:
    """No-op stub — Sisyphean delegates search to the harness.

    Returns [] so the micro-loop skips internal search and the model uses
    the external WebSearch tool_use block dispatched by BirdClaw instead.
    """
    logger.debug("search stub called for %r — delegating to harness WebSearch tool", query[:50])
    return []


# ---------------------------------------------------------------------------
# Fetch — still works directly (httpx only, no extra deps)
# ---------------------------------------------------------------------------

async def fetch(url: str, goal: str = "", client=None) -> str:
    """Fetch a URL and return cleaned text."""
    try:
        import httpx
    except ImportError:
        return f"(httpx not installed — cannot fetch {url})"

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as http:
            resp = await http.get(url, headers={"User-Agent": "Sisyphean/1.0"})
            resp.raise_for_status()
    except Exception as exc:
        logger.warning("web fetch failed  url=%s  error=%s", url[:80], exc)
        return f"(fetch error: {exc})"

    content_type = resp.headers.get("content-type", "")
    if "html" not in content_type and "text" not in content_type:
        return f"(unsupported content type: {content_type})"

    raw_text = strip_html(resp.text)
    goal_hint = goal or url.split("/")[-1].replace("-", " ").replace("_", " ")

    if len(raw_text) > 3000:
        cleaned = keyword_prune(raw_text, goal=goal_hint, max_chars=2000)
    else:
        cleaned = fast_clean(raw_text)

    if goal and client and len(cleaned) > 1000:
        notes = await distill(cleaned, goal=goal, client=client, source=url)
        if notes and len(notes) > 50:
            return notes

    logger.debug("fetch ok  url=%s  chars=%d", url[:60], len(cleaned))
    return cleaned


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

def format_results(results: list[SearchResult]) -> str:
    """Format search results as a context block for injection into model prompts."""
    if not results:
        return "[No search results found]"
    lines = ["### Web Search Results\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.to_context()}\n")
    return "\n".join(lines)


def extract_search_queries(text: str) -> list[str]:
    """Extract [SEARCH: query] tokens from model output."""
    return [m.strip() for m in _SEARCH_TOKEN_RE.findall(text)]
