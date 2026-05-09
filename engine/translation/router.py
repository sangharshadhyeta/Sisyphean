"""Intent classifier + tool router — ported from BirdClaw agent/router.py.

Two responsibilities:

1. Intent classification (new in Sisyphean):
   Decides whether a message needs the full translation loop or can be
   answered in one shot.  Runs before any LLM call — pure heuristics.

   Intents:
     direct  — simple Q&A, greetings, clarifications (1-shot answer)
     search  — needs current/recent information
     code    — code generation or editing task
     task    — multi-step work requiring planning + loop

2. Tool selection (ported from BirdClaw):
   Scores registered tools by keyword overlap with query + recent history.
   Keeps the model's tool list small so the 4B model stays focused.
"""
from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)

_PATH_RE = re.compile(r"[\w./~-]+\.(?:py|rs|ts|js|toml|json|yaml|yml|md|txt|sh|cfg|ini)")

# ── Intent signals ────────────────────────────────────────────────────────────

_DIRECT_STARTS = (
    "what is", "what are", "who is", "who are", "define", "explain",
    "how does", "tell me about", "hello", "hi", "hey", "thanks", "thank you",
    "yes", "no", "ok", "okay", "sure", "got it",
)
_SEARCH_WORDS = (
    "latest", "recent", "current", "today", "now", "2024", "2025", "2026",
    "news", "update", "release", "version", "announced", "just", "new",
)
_CODE_WORDS = (
    "write", "implement", "create", "build", "code", "function", "class",
    "script", "fix", "debug", "refactor", "edit", "modify", "add feature",
)
_TASK_WORDS = (
    "research", "find", "investigate", "analyse", "analyze", "report",
    "plan", "design", "set up", "configure", "install", "deploy",
    "compare", "evaluate", "summarise", "summarize",
)


def classify(message: str) -> str:
    """Return intent: 'direct' | 'search' | 'code' | 'task'."""
    msg = message.lower().strip()

    # Short messages are almost always direct
    if len(msg.split()) <= 4:
        return "direct"

    # Starts with direct patterns
    if any(msg.startswith(p) for p in _DIRECT_STARTS):
        if not any(w in msg for w in _SEARCH_WORDS + _CODE_WORDS + _TASK_WORDS):
            return "direct"

    # Check for search signals first — even code tasks may need a search step
    if any(w in msg for w in _SEARCH_WORDS):
        return "search"

    # Code signals
    if any(w in msg for w in _CODE_WORDS) or _PATH_RE.search(message):
        return "code"

    # Multi-step task signals
    if any(w in msg for w in _TASK_WORDS):
        return "task"

    # Default to task for anything complex-looking
    if len(msg.split()) > 20:
        return "task"

    return "direct"


# ── Tool selection (ported from BirdClaw) ────────────────────────────────────

def _tokenise(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _query_tokens(query: str, history: list[dict], lookback: int = 3) -> set[str]:
    combined = query
    for msg in history[-lookback:]:
        combined += " " + (msg.get("content") or "")
    return _tokenise(combined)


def select_tools(
    query: str,
    available: list[dict],
    history: list[dict] | None = None,
    max_n: int = 6,
) -> list[dict]:
    """Return up to max_n tools most relevant to this query.

    Each tool dict must have: {"name": str, "tags": list[str], "schema": dict}
    Control tools (think, answer) are NOT included — executor adds them always.
    """
    if not available:
        return []

    tokens = _query_tokens(query, history or [])
    is_question = _is_question(query)
    has_path = bool(_PATH_RE.search(query))

    scored: list[tuple[int, dict]] = []
    for tool in available:
        score = len(tokens & set(tool.get("tags", [])))
        scored.append((score, tool))
    scored.sort(key=lambda x: x[0], reverse=True)

    # Priority boosts — same logic as BirdClaw
    priority_names: list[str] = []
    if has_path:
        priority_names += ["write_file", "read_file"]
    if is_question:
        priority_names += ["web_search"]

    priority_set = set(priority_names)
    priority_tools = [t for _, t in scored if t["name"] in priority_set]
    rest = [t for _, t in scored if t["name"] not in priority_set]
    ordered = priority_tools + rest

    return ordered[:max_n]


def _is_question(text: str) -> bool:
    q = text.strip().lower()
    return (
        q.endswith("?") or
        q.startswith(("what", "who", "where", "when", "why", "how", "is ", "are ", "does ", "can "))
    )
