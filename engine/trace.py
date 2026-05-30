"""Per-task pipeline trace — captures every decision for test inspection.

Usage (pipeline side):
    from engine.trace import trace as _trace
    _trace(task_id, "route",    "decision",  "search")
    _trace(task_id, "step",     "bash",      "python calc.py 2+2")

Usage (test side):
    GET /debug/trace            → last request's trace
    GET /debug/trace?task_id=X  → specific task's trace
"""
from __future__ import annotations

import time
from collections import deque

from engine import sse_bus

_MAX_TASKS = 20   # keep the last N completed request traces

_traces:  dict[str, list[dict]] = {}       # task_id → ordered event list
_order:   deque[str]            = deque(maxlen=_MAX_TASKS)


def trace(task_id: str, stage: str, key: str, value: str = "") -> None:
    """Append one event to the trace for task_id."""
    if not task_id:
        return
    if task_id not in _traces:
        _traces[task_id] = []
        if task_id in _order:
            _order.remove(task_id)
        _order.appendleft(task_id)
    event = {
        "ts":    round(time.time(), 3),
        "stage": stage,
        "key":   key,
        "value": str(value)[:300],
    }
    _traces[task_id].append(event)
    sse_bus.publish({"type": "trace", "task_id": task_id, **event})


def get_trace(task_id: str | None = None) -> list[dict]:
    """Return the event list for task_id, or the most-recent task if None."""
    if task_id:
        return list(_traces.get(task_id, []))
    if _order:
        return list(_traces.get(_order[0], []))
    return []


def latest_task_id() -> str | None:
    return _order[0] if _order else None


def clear(task_id: str | None = None) -> None:
    if task_id:
        _traces.pop(task_id, None)
    else:
        _traces.clear()
        _order.clear()
