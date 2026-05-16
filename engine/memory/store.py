"""Artifact store — JSONL + optional semantic search.

Stores discrete artifacts produced or referenced during conversations:
code snippets, file paths, decisions, outputs, and facts too small
for the knowledge graph.

Each entry carries an optional graph_node_id linking it back to a
KnowledgeGraph node so the two layers stay navigable.

Search strategy
---------------
- If sentence-transformers is installed and an embedding_model is
  configured: full semantic cosine-similarity search.
- Otherwise: keyword overlap fallback (always works, no GPU needed).

Embeddings are computed incrementally: new entries are encoded on save
and appended to the in-memory array. Full rebuild only happens on startup.
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _HAS_ST = True
except ImportError:
    _HAS_ST = False
    logger.info("sentence-transformers not available — using keyword search for artifact store")


class ArtifactStore:

    def __init__(
        self,
        path: Path,
        embedding_model: str | None = "all-MiniLM-L6-v2",
    ) -> None:
        self.path = Path(path)
        self._entries: list[dict] = []
        self._encoder: Any = None
        self._embeddings: Any = None  # np.ndarray shape (N, D) once built
        self._lock = threading.Lock()

        if _HAS_ST and embedding_model:
            try:
                self._encoder = SentenceTransformer(embedding_model)
                logger.info("Embedding model loaded: %s", embedding_model)
            except Exception as exc:
                logger.warning("Could not load embedding model (%s) — keyword search only", exc)

        self._load()

    # ── Write ────────────────────────────────────────────────────────────────

    def save(
        self,
        type: str,
        content: str,
        summary: str | None = None,
        tags: list[str] | None = None,
        graph_node_id: str | None = None,
    ) -> str:
        entry = {
            "id": str(uuid.uuid4()),
            "type": type,
            "content": content,
            "summary": summary or content[:120],
            "tags": tags or [],
            "graph_node_id": graph_node_id,
            "created_at": _now(),
        }
        with self._lock:
            self._entries.append(entry)
            self._append_disk(entry)
            self._encode_one(entry)
        return entry["id"]

    # ── Search ───────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_n: int = 5,
        type_filter: str | None = None,
    ) -> list[dict]:
        with self._lock:
            pool = [e for e in self._entries if type_filter is None or e.get("type") == type_filter]
            if not pool:
                return []
            if self._encoder is not None and self._embeddings is not None and _HAS_ST:
                return self._semantic(query, pool, top_n)
            return self._keyword(query, pool, top_n)

    # ── Load / append ────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path) as f:
                self._entries = [json.loads(ln) for ln in f if ln.strip()]
            logger.info("Artifact store: %d entries loaded", len(self._entries))
            self._rebuild_embeddings()
        except Exception as exc:
            logger.error("Failed to load artifact store (%s)", exc)

    def _append_disk(self, entry: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    # ── Embedding helpers ────────────────────────────────────────────────────

    def _text(self, entry: dict) -> str:
        return f"{entry.get('summary', '')} {entry.get('content', '')[:300]}"

    def _encode_one(self, entry: dict) -> None:
        """Encode a single new entry and append to the embeddings array."""
        if self._encoder is None:
            return
        import numpy as np
        try:
            vec = self._encoder.encode([self._text(entry)], show_progress_bar=False)[0]
            if self._embeddings is None:
                self._embeddings = vec.reshape(1, -1)
            else:
                self._embeddings = np.vstack([self._embeddings, vec])
        except Exception as exc:
            logger.warning("Embedding encode failed (%s)", exc)

    def _rebuild_embeddings(self) -> None:
        """Full rebuild — only called on startup when loading existing entries."""
        if self._encoder is None or not self._entries:
            return
        import numpy as np
        texts = [self._text(e) for e in self._entries]
        try:
            self._embeddings = self._encoder.encode(texts, show_progress_bar=False, batch_size=64)
        except Exception as exc:
            logger.warning("Embedding rebuild failed (%s)", exc)

    def _semantic(self, query: str, pool: list[dict], top_n: int) -> list[dict]:
        import numpy as np
        pool_ids = {e["id"] for e in pool}
        indices = [i for i, e in enumerate(self._entries) if e["id"] in pool_ids]
        if not indices or self._embeddings is None:
            return []
        q_emb = self._encoder.encode([query], show_progress_bar=False)[0]
        sub = self._embeddings[indices]
        norms = np.linalg.norm(sub, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        sims = (sub / norms) @ q_emb / (np.linalg.norm(q_emb) + 1e-9)
        top = np.argsort(sims)[::-1][:top_n]
        return [self._entries[indices[i]] for i in top]

    def _keyword(self, query: str, pool: list[dict], top_n: int) -> list[dict]:
        q = set(query.lower().split())
        scored = []
        for entry in pool:
            text = f"{entry.get('summary', '')} {entry.get('content', '')}".lower()
            score = sum(1 for t in q if t in text)
            if score:
                scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_n]]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
