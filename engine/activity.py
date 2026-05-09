"""Sisyphean activity log — captures internal agent events for the dashboard.

The translation loop pushes one entry per step via `log_event()`.
The dashboard reads via `recent_events()`. No persistence — in-memory only.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Literal

_MAX_EVENTS = 200

EventKind = Literal[
    "tool",       # internal tool called (plan, search, web, etc.)
    "bash",       # bash command dispatched to Claude Code
    "answer",     # final answer returned
    "write",      # subtask write stage completed
    "error",      # something failed
    "stage",      # pipeline stage transition (router/planner/execute/synthesize)
    "llm",        # LLM call made — what was sent, what came back
    "plan",       # plan created — label=step count, detail=plan text
]

_events: deque[dict] = deque(maxlen=_MAX_EVENTS)


def log_event(
    kind: EventKind,
    label: str,
    detail: str = "",
    session_id: str = "",
    data: dict | None = None,
) -> None:
    _events.appendleft({
        "ts": round(time.time()),
        "kind": kind,
        "label": label,
        "detail": detail[:300],
        "session": session_id[-8:] if session_id else "",
        "data": data or {},
    })


def recent_events(n: int = 50) -> list[dict]:
    return list(_events)[:n]


def clear() -> None:
    _events.clear()
