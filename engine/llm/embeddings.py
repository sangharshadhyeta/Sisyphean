"""Embedding client — wraps Ollama's /api/embeddings endpoint.

Architecture
------------
EmbeddingCache
    In-memory dict keyed by node key.  Built lazily; never persisted to disk.
    Embeddings are derived from summary text, so the cache is always
    reconstructible.  Invalidated when a node's summary changes.

EmbeddingClient
    Thin async wrapper around Ollama's /api/embeddings.
    Returns None (never raises) so callers can fall back to keyword/Jaccard
    without branching on availability.

cosine_similarity
    Pure-Python, no NumPy.  Suitable for 768-dim nomic-embed-text vectors
    on CPU with 20-100 nodes in the graph.

Usage (extractor / graph)
--------------------------
    from engine.llm.embeddings import EmbeddingClient, EmbeddingCache, cosine_similarity

    cache  = EmbeddingCache()
    client = EmbeddingClient(ollama_url, model, cache)

    vec = await client.embed("some label")
    node_vec = await client.embed("name: summary", cache_key="node_key")
    sim = cosine_similarity(vec, node_vec)   # 0.0 – 1.0

    # When a node summary is updated, drop the stale vector:
    cache.invalidate("node_key")

    # At shutdown:
    await client.close()

Prerequisites
-------------
    ollama pull nomic-embed-text
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_TIMEOUT = 12.0       # seconds — generous for first-call model warm-up
_MAX_TEXT = 2048      # chars — hard cap to stay within nomic-embed-text context


# ── Pure-Python cosine similarity ─────────────────────────────────────────────

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [0, 1].  Returns 0.0 on empty / dimension mismatch."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (mag_a * mag_b)))


# ── In-memory embedding cache ─────────────────────────────────────────────────

class EmbeddingCache:
    """Thread/async-safe in-memory cache: node_key → embedding vector.

    Not persisted — rebuilt lazily each server run.  Invalidate when a
    node's summary changes so stale vectors don't bias similarity scores.
    """

    def __init__(self) -> None:
        self._data: dict[str, list[float]] = {}

    def get(self, key: str) -> list[float] | None:
        return self._data.get(key)

    def set(self, key: str, vec: list[float]) -> None:
        self._data[key] = vec

    def invalidate(self, key: str) -> None:
        self._data.pop(key, None)

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


# ── Embedding client ──────────────────────────────────────────────────────────

class EmbeddingClient:
    """Async Ollama embedding client with transparent in-process caching.

    Falls back silently (returns None) when:
      • Ollama is not running
      • The model has not been pulled (ollama pull nomic-embed-text)
      • Any HTTP / parse error occurs

    Once a failure is observed the client stays silent for the remainder of
    the process lifetime (_available = False) to avoid per-call overhead.
    Set _available back to None to retry.
    """

    def __init__(
        self,
        ollama_url: str = "http://127.0.0.1:11434",
        model: str = "nomic-embed-text",
        cache: EmbeddingCache | None = None,
    ) -> None:
        self._url   = ollama_url.rstrip("/") + "/api/embeddings"
        self._model = model
        self._cache = cache or EmbeddingCache()
        self._http  = httpx.AsyncClient(timeout=_TIMEOUT)
        # None = not yet tried; True = working; False = unavailable (skip)
        self._available: bool | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    async def embed(
        self,
        text: str,
        *,
        cache_key: str | None = None,
    ) -> list[float] | None:
        """Return embedding for *text*, or None if the service is unavailable.

        *cache_key* — when provided, the result is stored in / retrieved from
        the in-process cache.  Use the graph node key so stale entries can be
        invalidated when a node's summary changes.
        """
        if self._available is False:
            return None

        # Check cache first
        if cache_key:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        vec = await self._call(text)
        if vec is not None and cache_key:
            self._cache.set(cache_key, vec)
        return vec

    async def close(self) -> None:
        """Release the underlying httpx client."""
        await self._http.aclose()

    # ── Internal ───────────────────────────────────────────────────────────────

    async def _call(self, text: str) -> list[float] | None:
        try:
            r = await self._http.post(
                self._url,
                json={"model": self._model, "prompt": text[:_MAX_TEXT]},
            )
            r.raise_for_status()
            vec = r.json().get("embedding")
            if not isinstance(vec, list) or not vec:
                return None
            if self._available is not True:
                logger.info(
                    "EmbeddingClient: %s ready (%d dims)", self._model, len(vec)
                )
            self._available = True
            return [float(v) for v in vec]
        except Exception as exc:
            if self._available is None:
                # First failure — log once, then go silent
                logger.warning(
                    "EmbeddingClient: %s unavailable (%s) — "
                    "semantic dedup will use Jaccard fallback. "
                    "Run: ollama pull %s",
                    self._model, exc, self._model,
                )
            self._available = False
            return None
