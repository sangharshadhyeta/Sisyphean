"""Web search for Sisyphean — internal fallback path only.

Two-tier search for standalone / graph-first mode:
  1. SearXNG  — private, self-hosted, multi-engine. Requires local instance.
               Set `search.searxng_url` in config.yaml to enable.
  2. Jina AI  — free-tier AI-powered search (s.jina.ai). Returns clean,
               AI-processed content. No API key required for basic use.
               Results are marked is_ai_synthesized=True.

When running under Claude Code, the pipeline delegates web_search to Claude
Code's own WebSearch outer tool (see pipeline.py lines ~934-936).  This
module is only reached in standalone mode or when the outer tool is absent.
DuckDuckGo has been removed — it blocks automated requests and adds no value
over Jina for the standalone case.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from engine.translation.pruner import keyword_prune
from engine.translation.condenser import fast_clean, distill, strip_html

logger = logging.getLogger(__name__)

_SEARCH_TOKEN_RE = re.compile(r"\[SEARCH:\s*([^\]]+)\]", re.IGNORECASE)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str = field(default="")          # "jina", "searxng", "ddg", etc.
    is_ai_synthesized: bool = field(default=False)  # True when result is pre-processed by an AI tier

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

    Two-tier fallback for standalone / internal mode.  Returns [] if both
    tiers fail (no SearXNG configured and Jina unreachable).

    Jina AI (tier 2) returns AI-processed results marked with
    is_ai_synthesized=True — the synthesizer can skip its LLM call
    when all results carry this flag.

    Under Claude Code, this function is bypassed: the pipeline delegates
    web_search to Claude Code's own WebSearch outer tool instead.
    """
    if not query or not query.strip():
        return []

    searxng_url, cfg_max, cfg_timeout = _get_search_config()
    n = max_results or cfg_max
    timeout = cfg_timeout

    # ── Tier 1: SearXNG (self-hosted, private) ───────────────────────────────
    if searxng_url:
        results = await _search_searxng(query, searxng_url, n, timeout)
        if results:
            logger.info("web_search: SearXNG ok  query=%r  results=%d", query[:50], len(results))
            return results
        logger.debug("web_search: SearXNG miss  query=%r", query[:50])

    # ── Tier 2: Jina AI (free-tier, AI-processed results) ────────────────────
    results = await _search_jina(query, n, timeout)
    if results:
        logger.info("web_search: Jina ok  query=%r  results=%d", query[:50], len(results))
        return results
    logger.debug("web_search: Jina miss  query=%r", query[:50])

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
            url = item.get("url", "")
            # Normalise protocol-relative URLs ("//example.com") to https.
            if url.startswith("//"):
                url = f"https:{url}"
            results.append(SearchResult(
                title=item.get("title", ""),
                url=url,
                snippet=item.get("content", "")[:500],
            ))
        return results
    except Exception as exc:
        logger.debug("_search_searxng failed: %s", exc)
        return []


async def _search_jina(
    query: str, n: int, timeout: float
) -> list[SearchResult]:
    """Jina AI Reader search — free tier, no API key required.

    Endpoint: GET https://s.jina.ai/{encoded_query}
    Returns clean, AI-processed content from the top web results.
    Results are marked is_ai_synthesized=True so the synthesizer
    can skip its LLM call and return the content directly.
    """
    try:
        import httpx
        from urllib.parse import quote

        encoded = quote(query, safe="")
        url = f"https://s.jina.ai/{encoded}"

        headers = {
            "Accept": "application/json",
            "User-Agent": "Sisyphean/1.0",
            "X-Respond-With": "no-references",  # omit inline citations for cleaner text
        }

        async with httpx.AsyncClient(timeout=timeout) as http:
            resp = await http.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

    except Exception as exc:
        logger.debug("_search_jina failed: %s", exc)
        return []

    raw_items = data.get("data") or []
    if not raw_items:
        return []

    results: list[SearchResult] = []
    for item in raw_items[:n]:
        title   = (item.get("title") or "").strip()
        url_str = (item.get("url") or "").strip()
        # Prefer `content` (full cleaned text) over `description` (short snippet)
        body    = (item.get("content") or item.get("description") or "").strip()
        if not body:
            continue
        results.append(SearchResult(
            title=title,
            url=url_str,
            snippet=body[:800],
            source="jina",
            is_ai_synthesized=True,
        ))

    return results


# ---------------------------------------------------------------------------
# Fetch — fetch a URL and return cleaned text
# ---------------------------------------------------------------------------

async def fetch(url: str, goal: str = "", client=None) -> str:
    """Fetch a URL and return cleaned text.

    Tries Jina Reader first (https://r.jina.ai/{url}) which returns clean,
    article-extracted text without requiring HTML parsing.  Falls back to a
    direct HTTP fetch + HTML stripping when Jina is unavailable or returns
    an error.  Optionally distils the page against the goal using an LLM call
    when a client is provided and the cleaned text is long (>1000 chars).
    """
    try:
        import httpx
    except ImportError:
        return f"(httpx not installed — cannot fetch {url})"

    goal_hint = goal or url.split("/")[-1].replace("-", " ").replace("_", " ")

    # ── Tier 1: Jina Reader — clean article extraction, no HTML parsing ──────
    try:
        from urllib.parse import quote as _quote
        jina_url = f"https://r.jina.ai/{_quote(url, safe=':/?#[]@!$&\'()*+,;=%')}"
        async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as http:
            resp = await http.get(
                jina_url,
                headers={
                    "Accept": "text/plain",
                    "User-Agent": "Sisyphean/1.0",
                    "X-Remove-Selector": "nav,header,footer,aside",
                },
            )
            resp.raise_for_status()
            raw_text = resp.text.strip()
        if raw_text and len(raw_text) > 100:
            cleaned = keyword_prune(raw_text, goal=goal_hint, max_chars=3000) if len(raw_text) > 3000 else raw_text
            if goal and client and len(cleaned) > 1000:
                notes = await distill(cleaned, goal=goal, client=client, source=url)
                if notes and len(notes) > 50:
                    return notes
            logger.debug("fetch(jina) ok  url=%s  chars=%d", url[:60], len(cleaned))
            return cleaned
    except Exception as _jina_exc:
        logger.debug("fetch(jina) miss  url=%s  err=%s", url[:60], _jina_exc)

    # ── Tier 2: Direct HTTP fetch + HTML stripping ────────────────────────────
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

    if len(raw_text) > 3000:
        cleaned = keyword_prune(raw_text, goal=goal_hint, max_chars=2000)
    else:
        cleaned = fast_clean(raw_text)

    if goal and client and len(cleaned) > 1000:
        notes = await distill(cleaned, goal=goal, client=client, source=url)
        if notes and len(notes) > 50:
            return notes

    logger.debug("fetch(direct) ok  url=%s  chars=%d", url[:60], len(cleaned))
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
