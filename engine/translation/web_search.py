"""Web search and fetch tools.

Ported and adapted from BirdClaw birdclaw/tools/web.py.

Changes from BirdClaw:
- Async throughout (no sync httpx calls)
- Uses DuckDuckGo search package (no SearXNG dependency)
- HTML stripping via condenser.strip_html() (BeautifulSoup if available, regex fallback)
- Two-phase content processing: fast_clean() immediately, distill() for large pages
- No tool registry integration (tools are forwarded via Claude Code)

Two public async functions:
    search(query, max_results) → list[SearchResult]
    fetch(url, goal, client)   → str (cleaned page content)

format_results() and extract_search_queries() are used by the executor to
inject search results into the model's context.
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
# Search
# ---------------------------------------------------------------------------

async def search(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search via DuckDuckGo. Returns [] on any failure.

    Snippets are keyword-pruned against the query before returning so the
    most relevant parts of each result are surfaced for Gemma 4B.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.warning("ddgs not installed — web search unavailable")
            return []

    try:
        results: list[SearchResult] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                raw_snippet = r.get("body", "")
                # Keyword-prune snippet against query so relevance is maximised
                snippet = keyword_prune(raw_snippet, goal=query, max_chars=400)
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=snippet or raw_snippet[:400],
                ))
        logger.debug("search '%s' → %d results", query[:50], len(results))
        return results
    except Exception as exc:
        logger.warning("web search failed (%s)", exc)
        return []


# ---------------------------------------------------------------------------
# Fetch + clean
# ---------------------------------------------------------------------------

async def fetch(url: str, goal: str = "", client=None) -> str:
    """Fetch a URL and return cleaned text.

    Phase 1 (always): HTML stripped → keyword-pruned fast-path snippet
    Phase 2 (if goal + client + page large): Gemma distillation for task focus

    Returns a ready-to-inject string.
    """
    try:
        import httpx
    except ImportError:
        return f"(httpx not installed — cannot fetch {url})"

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as http:
            resp = await http.get(
                url,
                headers={"User-Agent": "Sisyphean/1.0"},
            )
            resp.raise_for_status()
    except Exception as exc:
        logger.warning("web fetch failed  url=%s  error=%s", url[:80], exc)
        return f"(fetch error: {exc})"

    content_type = resp.headers.get("content-type", "")
    if "html" not in content_type and "text" not in content_type:
        return f"(unsupported content type: {content_type})"

    raw_text = strip_html(resp.text)

    # Keyword prune using URL slug as a goal hint if no explicit goal given
    goal_hint = goal or url.split("/")[-1].replace("-", " ").replace("_", " ")

    if len(raw_text) > 3000:
        cleaned = keyword_prune(raw_text, goal=goal_hint, max_chars=2000)
    else:
        cleaned = fast_clean(raw_text)

    # Phase 2: LLM distillation for large pages when we have a goal + client
    if goal and client and len(cleaned) > 1000:
        notes = await distill(cleaned, goal=goal, client=client, source=url)
        if notes and len(notes) > 50:
            logger.debug("fetch distilled: %d→%d chars  url=%s", len(cleaned), len(notes), url[:60])
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
