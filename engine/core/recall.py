"""Recall — gathers relevant context before planning.

Three sources, searched in order:
  1. Current conversation history — recent turns with bigram overlap to query
  2. Graph memory             — stored facts and user preferences
  3. Files on disk            — any filename mentioned in the query that exists

All gathered text is compressed to ~100 words and returned as a plain
string injected into plan_task's prompt.  No LLM call — pure Python.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_WORDS     = 100   # hard cap on returned context
_HISTORY_TURNS = 8     # how many past messages to scan
_FILE_LINES    = 40    # max lines read from a mentioned file
_MEM_NODES     = 3     # top-k graph nodes to retrieve
_MIN_SIM       = 0.04  # minimum Jaccard similarity to include a history turn


# ── Text utilities ────────────────────────────────────────────────────────────

def _bigrams(text: str) -> set[str]:
    words = re.findall(r'\w+', text.lower())
    return {f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _word_count(text: str) -> int:
    return len(text.split())


def _truncate(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "…"


# ── Recall ────────────────────────────────────────────────────────────────────

class Recall:
    """Gathers and compresses relevant context for a query.

    Instantiated once in Pipeline and reused across requests.
    All methods are synchronous — no I/O blocking calls beyond small file reads.
    """

    def __init__(self, graph=None, workspace: str = ".") -> None:
        self.graph     = graph
        self.workspace = Path(workspace) if workspace else Path(".")

    def gather(self, query: str, raw_history: list[dict]) -> str:
        """Return a ~100-word context string relevant to *query*.

        Returns empty string if nothing relevant is found — callers should
        treat an empty return as "no context available" and proceed without it.
        """
        if not query:
            return ""

        q_bg   = _bigrams(query)
        budget = _MAX_WORDS
        parts: list[str] = []

        # ── 1. Conversation history ───────────────────────────────────────────
        hist = self._from_history(query, q_bg, raw_history, max_words=budget // 2)
        if hist:
            parts.append(f"Recent: {hist}")
            budget -= _word_count(hist)

        # ── 2. Graph memory ───────────────────────────────────────────────────
        if budget > 10:
            mem = self._from_memory(query, max_words=budget // 2)
            if mem:
                parts.append(f"Memory: {mem}")
                budget -= _word_count(mem)

        # ── 3. Files mentioned in query ───────────────────────────────────────
        if budget > 10:
            files = self._from_files(query, max_words=budget)
            if files:
                parts.append(files)

        if not parts:
            return ""

        result = "  |  ".join(parts)
        logger.debug("recall: %d words gathered for %r", _word_count(result), query[:50])
        return _truncate(result, _MAX_WORDS)

    # ── Sources ───────────────────────────────────────────────────────────────

    def _from_history(
        self,
        query: str,
        q_bg: set,
        raw_history: list[dict],
        max_words: int = 50,
    ) -> str:
        """Find recent conversation turns relevant to the query via Jaccard similarity."""
        if not raw_history:
            return ""

        scored: list[tuple[float, str]] = []
        recent = raw_history[-(_HISTORY_TURNS * 2):]

        for msg in recent:
            role = msg.get("role", "")
            if role not in ("user", "assistant"):
                continue

            content = msg.get("content", "")
            # Flatten content blocks to plain text
            if isinstance(content, list):
                text = " ".join(
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            else:
                text = str(content)

            text = text.strip()
            # Skip empty, pipeline state, thinking blocks
            if (not text or len(text) < 8
                    or "PIPELINE_STATE:" in text
                    or text.startswith("{")):
                continue

            sim = _jaccard(q_bg, _bigrams(text))
            if sim >= _MIN_SIM:
                prefix = "U:" if role == "user" else "A:"
                scored.append((sim, f"{prefix} {text[:100]}"))

        if not scored:
            return ""

        scored.sort(reverse=True)
        chosen = [t for _, t in scored[:3]]
        return _truncate("  ".join(chosen), max_words)

    def _from_memory(self, query: str, max_words: int = 40) -> str:
        """Search graph memory for nodes relevant to the query."""
        if self.graph is None:
            return ""
        try:
            nodes = self.graph.search(query, top_k=_MEM_NODES)
            if not nodes:
                return ""
            lines = []
            for n in nodes:
                text = (n.get("content") or n.get("summary") or "").strip()
                if text:
                    lines.append(text[:120])
            return _truncate("  ".join(lines), max_words)
        except Exception:
            return ""

    def _from_files(self, query: str, max_words: int = 40) -> str:
        """Read files named in the query if they exist on disk."""
        # Match filenames: word.ext (2-5 char extension)
        candidates = re.findall(r'\b(\w[\w/.-]*\.[a-zA-Z]{2,5})\b', query)
        if not candidates:
            return ""

        parts: list[str] = []
        seen: set[str] = set()

        for fname in candidates:
            if fname in seen:
                continue
            seen.add(fname)

            for base in (self.workspace, Path(".")):
                fpath = (base / fname).resolve()
                try:
                    if fpath.exists() and fpath.is_file() and fpath.stat().st_size < 100_000:
                        raw_lines = fpath.read_text(encoding="utf-8", errors="replace"
                                                    ).splitlines()[:_FILE_LINES]
                        content = "\n".join(raw_lines).strip()
                        if content:
                            parts.append(f"[{fname}]\n{content}")
                        break
                except Exception:
                    pass

        if not parts:
            return ""

        combined = "\n\n".join(parts)
        return _truncate(combined, max_words)
