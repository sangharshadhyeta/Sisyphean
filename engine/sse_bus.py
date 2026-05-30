"""In-process SSE broadcast bus.

Subscribers call subscribe() to get an asyncio.Queue.
publish() pushes events to every live subscriber.
put_nowait() is used throughout — no await, safe to call from sync code.
"""
from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

_subscribers: set[asyncio.Queue] = set()


def subscribe() -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=500)
    _subscribers.add(q)
    return q


def unsubscribe(q: asyncio.Queue) -> None:
    _subscribers.discard(q)


def publish(event: dict) -> None:
    """Push event to all subscribers; silently drop lagging clients."""
    if not _subscribers:
        return
    dead: set[asyncio.Queue] = set()
    for q in _subscribers:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            dead.add(q)
    for q in dead:
        _subscribers.discard(q)
        logger.debug("sse_bus: dropped lagging subscriber")
