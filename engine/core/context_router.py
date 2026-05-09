"""Context router — serves relevant slices of project context per stage.

Parses project context (CLAUDE.md extracted via Option B) into sections by
markdown headers, then uses Jaccard bigram similarity to select the most
relevant sections for each plan step or synthesis call.

Each plan_task() call gets a different slice rather than the full document —
keeps each LLM call focused and under token budget without losing anything.

Same algorithm as soul/router.py but tuned for markdown ## sections.
"""
from __future__ import annotations

import re

_HEADER_RE = re.compile(r'^#{1,3}\s+(.+)$', re.MULTILINE)
_WORD_RE   = re.compile(r'\b[a-z0-9\']+\b')
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "in", "of", "to", "and",
    "or", "for", "with", "it", "this", "that", "be", "by", "at", "from",
    "as", "on", "up", "do", "if", "not", "use", "can", "will", "how",
})


def _ngrams(text: str) -> frozenset[tuple[str, ...]]:
    words = [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS]
    uni = {(w,) for w in words}
    bi  = {(words[i], words[i + 1]) for i in range(len(words) - 1)}
    return frozenset(uni | bi)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _parse_sections(text: str) -> list[tuple[str, str]]:
    """Parse markdown into [(header, content)] preserving order."""
    sections: list[tuple[str, str]] = []
    matches = list(_HEADER_RE.finditer(text))
    for i, m in enumerate(matches):
        name    = m.group(1).strip()
        start   = m.end()
        end     = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            sections.append((name, content))
    return sections


class ContextRouter:
    """Builds once per request; serves query-relevant context slices per stage.

    Usage:
        router = ContextRouter(project_ctx, recall_ctx)
        plan_ctx     = router.for_task("write a Python file called foo.py")
        synthesis_ctx = router.for_synthesis("write a Python file called foo.py")
    """

    def __init__(self, project_ctx: str, recall_ctx: str) -> None:
        self.recall_ctx = recall_ctx
        self._sections  = _parse_sections(project_ctx) if project_ctx else []
        self._raw       = project_ctx

    def for_task(self, task: str, top_n: int = 2, max_chars: int = 1200) -> str:
        """Return context most relevant to a specific plan step."""
        return self._build(task, top_n=top_n, max_chars=max_chars)

    def for_synthesis(self, query: str, max_chars: int = 800) -> str:
        """Return context for the final synthesize call (slightly smaller budget)."""
        return self._build(query, top_n=1, max_chars=max_chars)

    def for_split(self, query: str, max_chars: int = 400) -> str:
        """Return minimal context for the split decision."""
        return self._build(query, top_n=1, max_chars=max_chars)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build(self, query: str, top_n: int, max_chars: int) -> str:
        parts: list[str] = []
        budget = max_chars

        if self.recall_ctx:
            snippet = self.recall_ctx[:budget]
            parts.append(snippet)
            budget -= len(snippet)

        if not self._sections:
            if self._raw:
                parts.append(self._raw[:budget])
            return "\n\n".join(p for p in parts if p)

        q_ng = _ngrams(query)
        scored: list[tuple[float, str, str]] = []
        for name, content in self._sections:
            s_ng  = _ngrams((name + " ") * 3 + content[:400])
            score = _jaccard(q_ng, s_ng)
            scored.append((score, name, content))
        scored.sort(reverse=True)

        for _, name, content in scored[:top_n]:
            if budget <= 80:
                break
            snippet = f"[{name}]\n{content}"[:budget]
            parts.append(snippet)
            budget -= len(snippet)

        return "\n\n".join(p for p in parts if p)
