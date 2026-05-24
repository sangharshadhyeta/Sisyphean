"""Session auto-merge — lightweight session end consolidation.

After each end_turn response, this upserts a 'session' graph node
summarising what was accomplished in the current session turn.  Runs
without an LLM call — composes the summary from stage_done events and
user messages already captured in the SessionLog.

This is the engine-side lightweight counterpart to BirdClaw's dream cycle.
BirdClaw's dream cycle does deeper consolidation (LLM calls, cross-session
pattern extraction, skill discovery).  This ensures the graph stays current
between dream cycle runs so that even a plain Claude Code client benefits
from recent session context.

Node written
------------
  type:    session
  name:    session:<YYYY-MM-DD>:<truncated_session_id>
  summary: "<date> | requests: <...> | completed: [<type>] <goal> → <outcome>"
  sources: ["session_log"]

The node is upserted (not created fresh) so repeated end_turn events in
the same session enrich the same node rather than creating duplicates.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # graph type imported lazily to avoid circular imports

logger = logging.getLogger(__name__)

_MAX_GOAL_LEN = 80    # chars per goal/outcome fragment
_MAX_GOALS = 6        # max completed stages to include
_MAX_REQ_LEN = 100    # chars per user request fragment
_MAX_REQUESTS = 3     # max user requests to include
_MAX_SUMMARY = 700    # hard cap on the full summary string


def _build_summary(session_id: str, slog) -> str:
    """Compose a plain-text summary without an LLM call.

    Pulls from:
      - slog.last_user_messages()  → what the user asked
      - slog.completed_stages()    → what the agent accomplished
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # User requests (last N)
    requests = slog.last_user_messages(_MAX_REQUESTS)
    req_text = "; ".join(r[:_MAX_REQ_LEN] for r in requests) if requests else ""

    # Completed stage outcomes
    stages = slog.completed_stages()
    outcomes: list[str] = []
    for s in stages[-_MAX_GOALS:]:
        stype   = s.get("stage_type", "")
        goal    = s.get("goal", "")[:_MAX_GOAL_LEN]
        outcome = s.get("summary", "")[:_MAX_GOAL_LEN]
        frag = f"[{stype}] {goal}"
        if outcome and outcome != goal:
            frag += f" → {outcome}"
        outcomes.append(frag)
    outcomes_text = "; ".join(outcomes)

    parts = [f"date: {today}"]
    if req_text:
        parts.append(f"requests: {req_text}")
    if outcomes_text:
        parts.append(f"completed: {outcomes_text}")

    return " | ".join(parts)[:_MAX_SUMMARY]


async def merge_session_to_graph(session_id: str, graph) -> None:
    """Upsert a session graph node summarising the current session state.

    Safe to call multiple times per session — each call refreshes the
    node's summary with the latest accumulated activity.  Silently
    no-ops if the session has no meaningful activity or graph is None.

    Parameters
    ----------
    session_id:
        The session key used by get_session_log() (typically derived from
        the first user message in anthropic.py).
    graph:
        A GraphStore instance (engine.memory.graph.knowledge_graph).
        Passed explicitly so this module stays harness-agnostic.
    """
    if graph is None:
        return

    try:
        from engine.memory.session_log import get_session_log
        slog = get_session_log(session_id)

        # Skip sessions with no meaningful activity
        stages   = slog.completed_stages()
        requests = slog.last_user_messages(1)
        if not stages and not requests:
            logger.debug("session_merge: nothing to merge for %r", session_id[:20])
            return

        summary   = _build_summary(session_id, slog)
        # Node name is date-scoped so old sessions don't collide with new ones
        node_name = (
            f"session:{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
            f":{session_id[:20]}"
        )

        # upsert_node auto-saves; no explicit graph.save() needed
        graph.upsert_node(
            name=node_name,
            node_type="session",
            summary=summary,
            sources=["session_log"],
            session_id=session_id,
        )

        logger.debug(
            "session_merge: upserted %r (%d stages, %d requests)",
            node_name, len(stages), len(requests),
        )

    except Exception as exc:
        logger.warning("session_merge failed for %r: %s", session_id[:20], exc)
