"""Memory cleanup — retention policy for sessions and knowledge graph.

The cleanup pass runs after the memorise pass (inside dream).  It enforces
size and age limits so the memory store does not grow unbounded.

What gets cleaned
-----------------
  Session logs
    Files older than SESSION_RETENTION_DAYS are deleted.
    Archived rotation files (.1.jsonl, .2.jsonl, .3.jsonl) follow the same rule.

  Knowledge graph nodes
    Nodes with last_seen older than NODE_RETENTION_DAYS are pruned — UNLESS
    they are type 'soul', 'user', or 'project' (those are permanent).
    Nodes with no last_seen timestamp are kept (seeded / hand-crafted nodes).

  Budget tracker stage history
    Rows older than BUDGET_RETENTION_DAYS are trimmed from the JSONL files
    inside the mem_path budget directory (stage_history/*.jsonl).

Dry-run
-------
  Pass dry_run=True to get a CleanupResult without making any changes.
  Useful for reporting what *would* be deleted.

Defaults (all overridable via keyword args to CleanupPolicy)
-------------------------------------------------------------
  SESSION_RETENTION_DAYS = 30
  NODE_RETENTION_DAYS    = 90
  BUDGET_RETENTION_DAYS  = 60
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_SESSIONS_DIR = Path.home() / ".sisyphean" / "sessions"
_WATERMARK_FILE = _SESSIONS_DIR / ".memorised"

_PERMANENT_NODE_TYPES = {"policy", "soul", "user", "project"}  # "soul" kept for backward compat


# ── Policy dataclass ──────────────────────────────────────────────────────────

@dataclass
class CleanupPolicy:
    session_retention_days: int = 30
    node_retention_days: int = 90
    budget_retention_days: int = 60
    mem_path: Path | None = None       # set to config.memory.path for budget cleanup


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class CleanupResult:
    sessions_deleted: int = 0
    session_bytes_freed: int = 0
    nodes_pruned: int = 0
    budget_rows_trimmed: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = []
        if self.sessions_deleted:
            kb = self.session_bytes_freed // 1024
            parts.append(f"{self.sessions_deleted} session files ({kb} KB)")
        if self.nodes_pruned:
            parts.append(f"{self.nodes_pruned} graph nodes")
        if self.budget_rows_trimmed:
            parts.append(f"{self.budget_rows_trimmed} budget rows")
        if not parts:
            return "nothing to clean"
        return "cleaned: " + ", ".join(parts)


# ── Session cleanup ───────────────────────────────────────────────────────────

def _cleanup_sessions(
    policy: CleanupPolicy,
    dry_run: bool,
    result: CleanupResult,
) -> None:
    if not _SESSIONS_DIR.exists():
        return

    cutoff = datetime.now(timezone.utc) - timedelta(days=policy.session_retention_days)

    # Load watermark to clean entries for deleted sessions
    watermark: dict[str, str] = {}
    wm_dirty = False
    if _WATERMARK_FILE.exists():
        try:
            watermark = json.loads(_WATERMARK_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Find all .jsonl files (including rotation archives like .1.jsonl)
    for p in list(_SESSIONS_DIR.glob("*.jsonl")):
        if p.name.startswith("."):
            continue
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
        except OSError:
            continue
        if mtime < cutoff:
            size = p.stat().st_size
            if not dry_run:
                try:
                    p.unlink()
                    result.sessions_deleted += 1
                    result.session_bytes_freed += size
                    # Remove from watermark
                    stem = p.stem.split(".")[0]  # handle "foo.1" archives
                    if stem in watermark:
                        del watermark[stem]
                        wm_dirty = True
                except OSError as exc:
                    result.errors.append(f"delete {p.name}: {exc}")
            else:
                result.sessions_deleted += 1
                result.session_bytes_freed += size

    if wm_dirty and not dry_run:
        try:
            tmp = _WATERMARK_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(watermark, indent=2), encoding="utf-8")
            tmp.replace(_WATERMARK_FILE)
        except Exception as exc:
            result.errors.append(f"watermark update: {exc}")


# ── Graph node pruning ────────────────────────────────────────────────────────

def _cleanup_graph(
    policy: CleanupPolicy,
    dry_run: bool,
    result: CleanupResult,
) -> None:
    try:
        from engine.memory.graph import knowledge_graph as kg
    except ImportError:
        return

    cutoff = datetime.now(timezone.utc) - timedelta(days=policy.node_retention_days)
    to_remove: list[str] = []

    for key, data in list(kg._graph.nodes(data=True)):
        ntype = data.get("type", "")
        if ntype in _PERMANENT_NODE_TYPES:
            continue
        last_seen_str = data.get("last_seen") or data.get("updated_at") or ""
        if not last_seen_str:
            continue  # no timestamp — keep (hand-crafted / seeded)
        try:
            last_seen = datetime.fromisoformat(last_seen_str)
            if last_seen.tzinfo is None:
                last_seen = last_seen.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if last_seen < cutoff:
            to_remove.append(key)

    for key in to_remove:
        if not dry_run:
            try:
                kg._graph.remove_node(key)
                result.nodes_pruned += 1
            except Exception as exc:
                result.errors.append(f"remove node {key}: {exc}")
        else:
            result.nodes_pruned += 1

    if to_remove and not dry_run:
        try:
            kg.save()
        except Exception as exc:
            result.errors.append(f"graph save after prune: {exc}")


# ── Budget history trimming ───────────────────────────────────────────────────

def _cleanup_budget(
    policy: CleanupPolicy,
    dry_run: bool,
    result: CleanupResult,
) -> None:
    if not policy.mem_path:
        return
    budget_dir = policy.mem_path / "stage_history"
    if not budget_dir.exists():
        return

    cutoff = datetime.now(timezone.utc) - timedelta(days=policy.budget_retention_days)

    for jsonl_path in budget_dir.glob("*.jsonl"):
        try:
            lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue

        kept: list[str] = []
        trimmed = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                ts_str = row.get("ts") or row.get("timestamp") or ""
                if ts_str:
                    ts = datetime.fromisoformat(ts_str)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if ts < cutoff:
                        trimmed += 1
                        continue
            except (json.JSONDecodeError, ValueError):
                pass  # keep malformed rows
            kept.append(line)

        if trimmed and not dry_run:
            try:
                jsonl_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
                result.budget_rows_trimmed += trimmed
            except OSError as exc:
                result.errors.append(f"trim {jsonl_path.name}: {exc}")
        elif trimmed:
            result.budget_rows_trimmed += trimmed


# ── Public entry point ────────────────────────────────────────────────────────

def run_cleanup(
    policy: CleanupPolicy | None = None,
    dry_run: bool = False,
) -> CleanupResult:
    """Run all cleanup passes. Returns a CleanupResult with counts."""
    if policy is None:
        policy = CleanupPolicy()

    result = CleanupResult()

    _cleanup_sessions(policy, dry_run, result)
    _cleanup_graph(policy, dry_run, result)
    _cleanup_budget(policy, dry_run, result)

    if result.errors:
        for err in result.errors:
            logger.warning("cleanup error: %s", err)

    return result
