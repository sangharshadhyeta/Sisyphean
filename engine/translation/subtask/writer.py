"""Subtask writer — writes one item at a time with verify + retry.

Entry point: run_write_step(client, instruction, workspace, file_path="") -> StageResult

For each item in the manifest:
  1. Build a focused write prompt (goal + done items + file tail + this item)
  2. Call LLM → raw text output
  3. Snapshot file → append output → verify
  4. If previously-complete items regressed, rollback to snapshot and retry
  5. On completion, move to next item

Returns StageResult with progress summary.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

from .manifest import SubtaskItem, SubtaskManifest
from engine.translation.subtask import planner as _planner
from engine.translation.subtask import verifier as _verifier

logger = logging.getLogger(__name__)

MAX_ITEM_RETRIES = 2
MAX_WRITE_ITERATIONS = 20   # hard cap on total outer-loop iterations
_FILE_TAIL_LINES = 25


# ── Result ────────────────────────────────────────────────────────────────────

class StageResult:
    def __init__(self, manifest: SubtaskManifest, written_path: str):
        self.manifest = manifest
        self.written_path = written_path

    @property
    def summary(self) -> str:
        complete = [it.title for it in self.manifest.items if it.status == "complete"]
        partial  = [it.title for it in self.manifest.items if it.status == "partial"]
        missing  = [it.title for it in self.manifest.items if it.status in ("missing", "regressed")]
        parts = [f"{len(complete)}/{self.manifest.total} items complete"]
        if partial:
            parts.append(f"partial: {', '.join(partial)}")
        if missing:
            parts.append(f"missing: {', '.join(missing)}")
        return "; ".join(parts)


# ── File helpers ─────────────────────────────────────────────────────────────

def _read_file(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _write_file(path: str, content: str, append: bool) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with p.open(mode, encoding="utf-8") as f:
        f.write(content)


def _file_tail(path: str, n: int = _FILE_TAIL_LINES) -> str:
    content = _read_file(path)
    if not content:
        return ""
    lines = content.splitlines()
    return "\n".join(lines[-n:] if len(lines) > n else lines)


def _infer_output_path(stage_goal: str, file_type: Literal["doc", "code"], workspace: str) -> str:
    """Derive a filename from the stage goal when none is specified."""
    slug = stage_goal[:40].lower()
    slug = "".join(c if c.isalnum() or c == " " else "" for c in slug).strip()
    slug = "_".join(slug.split())[:30] or "output"
    ext = ".py" if file_type == "code" else ".md"
    return str(Path(workspace) / (slug + ext))


# ── Write prompt builder ─────────────────────────────────────────────────────

_WRITE_SYSTEM = (
    "You are a code/document writer. Write ONLY the requested item — no explanations, "
    "no preamble, no markdown fences unless writing a doc. Output raw text only."
)


def _build_write_prompt(
    item: SubtaskItem,
    manifest: SubtaskManifest,
    attempt: int,
    error_hint: str = "",
    done_summary: str = "",
) -> str:
    if attempt > 0:
        # Retry: show verifier gap analysis
        fc = _read_file(manifest.file_path)
        diff = _verifier.run(manifest, fc)
        prompt = diff.resume_context
        if error_hint:
            prompt += f"\n\nNote from last attempt: {error_hint}"
        return prompt

    # First attempt: show goal, accumulated done-summary, file tail, what to write next
    done = [it for it in manifest.items if it.status == "complete"]
    done_str = ", ".join(it.title for it in done) or "none yet"

    tail = _file_tail(manifest.file_path)
    file_state = (
        f"Current file tail ({manifest.file_path}):\n{tail}"
        if tail
        else f"[{manifest.file_path} — empty, start fresh]"
    )

    if manifest.file_type == "code":
        marker_hint = f"Start with exactly: def {item.anchor}( (or class {item.anchor}:)\n"
    else:
        marker_hint = f"Start with exactly: ## {item.anchor}\n"

    prompt = (
        f"Goal: {manifest.stage_goal}\n"
        f"Done items: {done_str}\n"
    )
    if done_summary:
        prompt += f"Already produced:\n{done_summary}\n"
    prompt += (
        f"\n{file_state}\n\n"
        f"Write next: {item.title} (min {item.expected_min_chars} chars)\n"
        f"{marker_hint}"
        f"Append after the content above. Do not repeat earlier content.\n"
    )
    if error_hint:
        prompt += f"\nNote: {error_hint}"
    return prompt


# ── Single item write + verify ──────────────────────────────────────────────

async def _write_item(
    item: SubtaskItem,
    manifest: SubtaskManifest,
    client: Any,
    done_summary: str = "",
) -> bool:
    """Write one item with up to MAX_ITEM_RETRIES retries. Returns True if complete.

    Snapshot+rollback: before each write we save the current file content.
    If the write causes any previously-complete item to regress, we restore
    the snapshot and retry with an error hint telling the model to append only.
    """
    item.status = "in_progress"
    is_first = item.index == 0
    error_hint = ""

    for attempt in range(MAX_ITEM_RETRIES + 1):
        logger.info(
            "[subtask] attempt=%d/%d  item=%r  file=%s",
            attempt, MAX_ITEM_RETRIES, item.title[:40], manifest.file_path,
        )

        prompt = _build_write_prompt(item, manifest, attempt, error_hint, done_summary)
        error_hint = ""

        messages = [
            {"role": "system", "content": _WRITE_SYSTEM},
            {"role": "user", "content": prompt},
        ]

        try:
            result = await client.generate(
                messages,
                max_tokens=1024,
                temperature=0.3,
                stream=False,
                thinking=False,
            )
            raw = result["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            error_hint = f"LLM error: {exc}"
            logger.warning("[subtask] LLM error attempt=%d: %s", attempt, exc)
            continue

        if not raw:
            error_hint = "Empty output — write the full item."
            continue

        if len(raw) < max(80, item.expected_min_chars // 3):
            logger.debug(
                "[subtask] output too short (%d chars, expected >=%d) — retrying",
                len(raw), item.expected_min_chars,
            )
            error_hint = (
                f"Output was only {len(raw)} chars — write the complete item "
                f"(minimum {item.expected_min_chars} chars). Do not stop early."
            )
            continue

        # Snapshot before writing so we can rollback on regression
        snapshot = _read_file(manifest.file_path)
        prev_complete = {it.anchor for it in manifest.items if it.status == "complete"}

        existing_before = snapshot  # already read above
        if is_first and attempt > 0 and item.status == "missing":
            _write_file(manifest.file_path, raw, append=False)
        elif is_first and attempt == 0 and not existing_before:
            _write_file(manifest.file_path, raw, append=False)
        else:
            _write_file(manifest.file_path, "\n\n" + raw, append=True)

        # Verify
        fc = _read_file(manifest.file_path)
        diff = _verifier.run(manifest, fc)

        # Rollback if previously-complete items regressed
        regressed_anchors = {it.anchor for it in diff.regressed} & prev_complete
        if regressed_anchors:
            logger.warning(
                "[subtask] regression on write (item=%r attempt=%d) — rolling back. regressed=%s",
                item.title[:40], attempt, sorted(regressed_anchors)[:3],
            )
            try:
                Path(manifest.file_path).write_text(snapshot, encoding="utf-8")
                # Restore item statuses to match the snapshot
                _verifier.run(manifest, snapshot)
            except OSError as exc:
                logger.error("[subtask] rollback failed: %s", exc)
            error_hint = (
                f"Your output caused {len(regressed_anchors)} previously-complete section(s) to disappear. "
                f"Append only — do not overwrite earlier content."
            )
            continue

        if item.status == "complete":
            logger.info("[subtask] item=%r complete  chars=%d", item.title[:40], item.actual_chars)
            return True

        if item.status == "partial":
            error_hint = f"{item.title} is partial ({item.actual_chars}c, need {item.expected_min_chars}c). Add more content."
        elif item.status == "missing":
            error_hint = f"{item.title} not found in file. Make sure to start with the exact anchor line."
        elif item.status == "regressed":
            error_hint = f"{item.title} regressed — earlier content may have been overwritten."

    logger.warning("[subtask] item=%r exhausted retries — marked partial", item.title[:40])
    return False


# ── Main entry point ────────────────────────────────────────────────────────

async def run_write_step(
    client: Any,
    stage_goal: str,
    file_type: Literal["doc", "code"],
    workspace: str,
    file_path: str = "",
) -> StageResult:
    """Execute a write_code or write_doc stage via the subtask pipeline.

    1. Plan: one LLM call → list of named items (functions / sections)
    2. Execute: write one item at a time, verify after each, retry if needed
    3. Rollback if a write causes previously-complete items to regress
    4. Return StageResult with progress summary

    file_path: explicit output path. Pass this when the pipeline or planner
               already knows the target filename; omit to infer from stage_goal.
    """
    if not file_path:
        file_path = _infer_output_path(stage_goal, file_type, workspace)

    # Ensure parent directory exists before planning
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    existing = _read_file(file_path)
    manifest = await _planner.plan(
        client=client,
        stage_goal=stage_goal,
        file_path=file_path,
        file_type=file_type,
        existing_content=existing,
    )

    logger.info(
        "[subtask] plan done  goal=%r  items=%d  file=%s",
        stage_goal[:60], manifest.total, file_path,
    )

    done_summary: str = ""
    _iteration = 0

    while manifest.current_item is not None:
        if _iteration >= MAX_WRITE_ITERATIONS:
            logger.warning(
                "[subtask] MAX_WRITE_ITERATIONS (%d) hit — forcing remaining items to partial",
                MAX_WRITE_ITERATIONS,
            )
            for it in manifest.items:
                if it.status not in ("complete",):
                    it.status = "partial"
            break
        _iteration += 1

        item = manifest.current_item
        success = await _write_item(item, manifest, client, done_summary=done_summary)

        status_word = "written" if success else "partial"
        content_hint = f" — {item.summary[:60]}" if item.summary else ""
        done_summary = (
            done_summary + f"\n- {item.title}: {status_word} ({item.actual_chars}c){content_hint}"
        ).strip()

        # Force advance: if current_item hasn't changed (stuck after retries), move on
        next_item = manifest.current_item
        if next_item is not None and next_item.index == item.index:
            item.status = "partial"
            break

    result = StageResult(manifest=manifest, written_path=file_path)
    logger.info("[subtask] stage done  %s  file=%s", result.summary, file_path)
    return result
