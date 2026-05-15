"""Session log — append-only JSONL event stream for a single session.

Records every significant event during a session so the dreaming pipeline
can later mine it for new knowledge and so the planning context can inject
recent activity into the system prompt.

Event types
-----------
  user_message      — raw message received from the user
  assistant_message — final text answer delivered back to the user
  tool_call         — tool invoked by the agent (name + truncated args)
  tool_result       — tool result back to the agent (name + truncated result)
  stage_start       — a new stage began (type + goal)
  stage_done        — a stage completed (type + goal + summary)
  compaction        — history was compacted (removed count + token estimate)
  plan              — the planner produced a step list

Storage
-------
  ~/.sisyphean/sessions/<session_id>.jsonl   — one JSON event per line
  Rotates at 256 KB; keeps last 3 archives.

Usage
-----
  log = SessionLog.new(session_id="abc123")
  log.user_message("do the thing")
  log.stage_start("research", "find out X")
  log.tool_call("bash", {"command": "ls /"})
  log.stage_done("research", "find out X", "found that X is Y")
  log.assistant_message("Here is what I found…")

  # Get a snippet to inject into context
  ctx = log.recent_context(n=5)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# ── Types ─────────────────────────────────────────────────────────────────────

EventType = Literal[
    "user_message",
    "assistant_message",
    "tool_call",
    "tool_result",
    "stage_start",
    "stage_done",
    "plan",
    "compaction",
]

_SESSIONS_DIR = Path.home() / ".sisyphean" / "sessions"

# Rotate log file after this many bytes
_ROTATE_BYTES: int = 256 * 1024
# Keep this many rotated archives (.1.jsonl, .2.jsonl, .3.jsonl)
_KEEP_ROTATED: int = 3
# Truncate tool result/arg content beyond this length in log entries
_MAX_CONTENT_CHARS: int = 500


# ── Event dataclass ───────────────────────────────────────────────────────────

@dataclass
class Event:
    type: EventType
    data: dict[str, Any]
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {"type": self.type, "ts": self.ts, "data": self.data}

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return cls(type=d["type"], data=d.get("data", {}), ts=d.get("ts", ""))


# ── SessionLog ────────────────────────────────────────────────────────────────

class SessionLog:
    """Append-only JSONL log for one Sisyphean session.

    Thread-safe for single-process use (asyncio-safe via GIL for the append).
    """

    def __init__(self, session_id: str, path: Path) -> None:
        self.session_id = session_id
        self._path = path
        self._events: list[Event] = []

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def new(cls, session_id: str) -> "SessionLog":
        """Create a fresh session log (or resume if the file already exists)."""
        _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        path = _SESSIONS_DIR / f"{session_id}.jsonl"
        log = cls(session_id=session_id, path=path)
        # Load existing events if any (idempotent on resume)
        if path.exists():
            log._load_events()
        logger.debug("SessionLog ready: %s", path)
        return log

    @classmethod
    def load(cls, session_id: str) -> "SessionLog":
        """Load an existing session log from disk."""
        path = _SESSIONS_DIR / f"{session_id}.jsonl"
        log = cls(session_id=session_id, path=path)
        if path.exists():
            log._load_events()
        return log

    def _load_events(self) -> None:
        try:
            for line in self._path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        self._events.append(Event.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        logger.warning("skipping malformed event in %s", self._path)
        except OSError as e:
            logger.warning("could not read session log %s: %s", self._path, e)

    # ── Rotation ──────────────────────────────────────────────────────────────

    def _rotate_if_needed(self) -> None:
        try:
            if not self._path.exists() or self._path.stat().st_size < _ROTATE_BYTES:
                return
        except OSError:
            return

        for i in range(_KEEP_ROTATED, 0, -1):
            src = self._path.parent / f"{self._path.stem}.{i}.jsonl"
            dst = self._path.parent / f"{self._path.stem}.{i + 1}.jsonl"
            if src.exists():
                if i >= _KEEP_ROTATED:
                    src.unlink(missing_ok=True)
                else:
                    src.rename(dst)

        archive = self._path.parent / f"{self._path.stem}.1.jsonl"
        try:
            self._path.rename(archive)
            logger.info("session log rotated → %s", archive.name)
        except OSError as e:
            logger.warning("session log rotation failed: %s", e)

    # ── Append ────────────────────────────────────────────────────────────────

    def _append(self, event: Event) -> None:
        self._events.append(event)
        try:
            self._rotate_if_needed()
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning("could not write session log: %s", e)

    def _trunc(self, text: str) -> str:
        if len(text) <= _MAX_CONTENT_CHARS:
            return text
        return text[:_MAX_CONTENT_CHARS] + "…"

    # ── Event writers ─────────────────────────────────────────────────────────

    def user_message(self, content: str) -> None:
        self._append(Event("user_message", {"content": self._trunc(content)}))

    def assistant_message(self, content: str) -> None:
        self._append(Event("assistant_message", {"content": self._trunc(content)}))

    def tool_call(self, name: str, arguments: dict) -> None:
        # Truncate large argument values (e.g. file contents) so the log stays readable
        truncated_args: dict[str, Any] = {}
        for k, v in arguments.items():
            if isinstance(v, str) and len(v) > _MAX_CONTENT_CHARS:
                truncated_args[k] = v[:_MAX_CONTENT_CHARS] + "…"
            else:
                truncated_args[k] = v
        self._append(Event("tool_call", {
            "name": name,
            "arguments": truncated_args,
            "called_at": time.time(),
        }))

    def tool_result(self, name: str, result: str, duration_ms: int = 0) -> None:
        self._append(Event("tool_result", {
            "name": name,
            "result": self._trunc(result),
            "duration_ms": duration_ms,
        }))

    def stage_start(self, stage_type: str, goal: str) -> None:
        self._append(Event("stage_start", {
            "stage_type": stage_type,
            "goal": self._trunc(goal),
            "started_at": time.time(),
        }))

    def stage_done(
        self,
        stage_type: str,
        goal: str,
        summary: str = "",
        duration_ms: int = 0,
        ok: bool = True,
    ) -> None:
        self._append(Event("stage_done", {
            "stage_type": stage_type,
            "goal": self._trunc(goal),
            "summary": self._trunc(summary),
            "duration_ms": duration_ms,
            "ok": ok,
        }))

    def plan(self, steps: list[str]) -> None:
        self._append(Event("plan", {
            "steps": steps,
            "planned_at": time.time(),
        }))

    def compaction(self, turns_removed: int, tokens_saved: int) -> None:
        self._append(Event("compaction", {
            "turns_removed": turns_removed,
            "tokens_saved": tokens_saved,
        }))

    # ── Query helpers ─────────────────────────────────────────────────────────

    def all_events(self) -> list[Event]:
        return list(self._events)

    def events_of_type(self, *types: EventType) -> list[Event]:
        return [e for e in self._events if e.type in types]

    def last_user_messages(self, n: int = 3) -> list[str]:
        msgs = [e.data["content"] for e in self._events if e.type == "user_message"]
        return msgs[-n:]

    def completed_stages(self) -> list[dict]:
        return [e.data for e in self._events if e.type == "stage_done" and e.data.get("ok")]

    # ── Context injection ─────────────────────────────────────────────────────

    def recent_context(self, n: int = 5) -> str:
        """Return a short human-readable snippet of the last N events.

        Injected into the planning system prompt so the model knows what it
        already did in the current session.
        """
        recent = self._events[-n:]
        if not recent:
            return ""
        lines: list[str] = ["Recent session activity:"]
        for e in recent:
            if e.type == "user_message":
                lines.append(f"  User: {e.data['content'][:120]}")
            elif e.type == "stage_start":
                lines.append(f"  Stage start [{e.data['stage_type']}]: {e.data['goal'][:100]}")
            elif e.type == "stage_done":
                status = "done" if e.data.get("ok") else "failed"
                lines.append(f"  Stage {status} [{e.data['stage_type']}]: {e.data.get('summary', '')[:100]}")
            elif e.type == "tool_call":
                lines.append(f"  Tool: {e.data['name']}")
            elif e.type == "assistant_message":
                lines.append(f"  Answer: {e.data['content'][:120]}")
        return "\n".join(lines)

    def planning_context(self) -> str:
        """Context block for the planning stage: last 2 user requests + completed stages."""
        parts: list[str] = []

        recent_reqs = self.last_user_messages(2)
        if recent_reqs:
            parts.append("Recent requests:\n" + "\n".join(f"  - {r}" for r in recent_reqs))

        done = self.completed_stages()
        if done:
            lines = [f"  - [{s['stage_type']}] {s['goal']}: {s.get('summary', '')}" for s in done[-4:]]
            parts.append("Completed stages:\n" + "\n".join(lines))

        return "\n\n".join(parts)


# ── Module-level session registry ─────────────────────────────────────────────
# Keyed by session_id so multiple concurrent sessions can coexist.

_active_logs: dict[str, SessionLog] = {}


def get_session_log(session_id: str) -> SessionLog:
    """Return (or create) the SessionLog for the given session_id."""
    if session_id not in _active_logs:
        _active_logs[session_id] = SessionLog.new(session_id)
    return _active_logs[session_id]
