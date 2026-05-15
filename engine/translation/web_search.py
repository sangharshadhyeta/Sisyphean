"""Web search for Sisyphean.

Three-tier search with automatic fallback:
  1. SearXNG  — private, multi-engine. Requires local instance.
               Set `search.searxng_url` in config.yaml to enable.
  2. DuckDuckGo package (ddgs / duckduckgo_search) — full results, no API key.
               Install: pip install duckduckgo-search
  3. DuckDuckGo Instant Answers API — pure httpx fallback, no extra package.
               Limited to abstract + related topics.

When running under Claude Code, the outer WebSearch tool is preferred over
this module for real-time results. This module handles the internal search
path (standalone mode, or when graph-first check triggers before the outer tool).
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
# Config loader — read search config without importing the full app
# ---------------------------------------------------------------------------

def _get_search_config():
    """Return (searxng_url, max_results, timeout) from config.yaml if available."""
    try:
        from engine.config import load_config
        cfg = load_config()
        return cfg.search.searxng_url, cfg.search.max_results, cfg.search.timeout
    except Exception:
        return "", 5, 15.0


# ---------------------------------------------------------------------------
# Search — three-tier with automatic fallback
# ---------------------------------------------------------------------------

async def search(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search the web and return results as SearchResult objects.

    Falls through the three tiers automatically.  Returns [] only if all
    tiers fail (e.g. no network, all services down).
    """
    if not query or not query.strip():
        return []

    searxng_url, cfg_max, cfg_timeout = _get_search_config()
    n = max_results or cfg_max
    timeout = cfg_timeout

    # ── Tier 1: SearXNG ──────────────────────────────────────────────────────
    if searxng_url:
        results = await _search_searxng(query, searxng_url, n, timeout)
        if results:
            logger.info("web_search: SearXNG ok  query=%r  results=%d", query[:50], len(results))
            return results
        logger.debug("web_search: SearXNG miss  query=%r", query[:50])

    # ── Tier 2: DuckDuckGo package (ddgs / duckduckgo_search) ────────────────
    results = await _search_ddg_package(query, n)
    if results:
        logger.info("web_search: DDG package ok  query=%r  results=%d", query[:50], len(results))
        return results

    # ── Tier 3: DuckDuckGo Instant Answers API (pure httpx) ──────────────────
    results = await _search_ddg_instant(query, n, timeout)
    if results:
        logger.info("web_search: DDG instant ok  query=%r  results=%d", query[:50], len(results))
        return results

    logger.warning("web_search: all tiers failed  query=%r", query[:50])
    return []


async def _search_searxng(
    query: str, base_url: str, n: int, timeout: float
) -> list[SearchResult]:
    try:
        import httpx
        async with httpx.AsyncClient(timeout=timeout) as http:
            resp = await http.get(
                f"{base_url.rstrip('/')}/search",
                params={"q": query, "format": "json",
                        "engines": "google,bing,duckduckgo"},
            )
            resp.raise_for_status()
            data = resp.json()
        results = []
        for item in data.get("results", [])[:n]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", "")[:500],
            ))
        return results
    except Exception as exc:
        logger.debug("_search_searxng failed: %s", exc)
        return []


async def _search_ddg_package(query: str, n: int) -> list[SearchResult]:
    """Use ddgs or duckduckgo_search package for full web results."""
    import asyncio
    try:
        def _sync_search():
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=n):
                    results.append(SearchResult(
                        title=r.get("title", ""),
                        url=r.get("href", ""),
                        snippet=r.get("body", "")[:500],
                    ))
            return results
        # Run the blocking DDGS call in a thread pool to avoid blocking the event loop
        return await asyncio.get_event_loop().run_in_executor(None, _sync_search)
    except Exception as exc:
        logger.debug("_search_ddg_package failed: %s", exc)
        return []


async def _search_ddg_instant(
    query: str, n: int, timeout: float
) -> list[SearchResult]:
    """DuckDuckGo Instant Answers API — no package required."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=timeout) as http:
            resp = await http.get(
                "https://api.duckduckgo.com/",
                params={
                    "q": query,
                    "format": "json",
                    "no_html": "1",
                    "skip_disambig": "1",
                },
                headers={"User-Agent": "Sisyphean/1.0"},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.debug("_search_ddg_instant failed: %s", exc)
        return []

    results: list[SearchResult] = []

    abstract = data.get("AbstractText", "")
    if abstract:
        results.append(SearchResult(
            title=data.get("Heading", query),
            url=data.get("AbstractURL", ""),
            snippet=abstract[:500],
        ))

    for topic in data.get("RelatedTopics", []):
        if isinstance(topic, dict) and topic.get("Text"):
            results.append(SearchResult(
                title=topic.get("Text", "")[:80],
                url=topic.get("FirstURL", ""),
                snippet=topic.get("Text", "")[:500],
            ))
        if len(results) >= n:
            break

    return results


# ---------------------------------------------------------------------------
# Fetch — fetch a URL and return cleaned text
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
