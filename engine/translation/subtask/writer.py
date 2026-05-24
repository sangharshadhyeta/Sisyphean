"""Subtask writer — writes one item at a time with verify + retry.

Entry point: run_write_step(client, instruction, workspace, file_path="") -> StageResult

For each item in the manifest:
  1. Build a focused write prompt (goal + done items + progressive context + this item)
     Context hierarchy: CLAUDE.md → exact section → relevant lines → continuation point
  2. Call LLM → raw text output
  3. Snapshot file → append output → verify
  4. Rollback on char-ratio regression (file shrank >20%) or anchor regression
  5. On completion, move to next item

Returns StageResult with progress summary.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from .manifest import SubtaskItem, SubtaskManifest
from engine.translation.subtask import planner as _planner
from engine.translation.subtask import verifier as _verifier
from engine.translation.subtask.line_search import (
    find_section,
    find_continuation_point,
    search_relevant,
)

logger = logging.getLogger(__name__)

MAX_ITEM_RETRIES = 2
MAX_WRITE_ITERATIONS = 20   # hard cap on total outer-loop iterations
_MAX_CTX_CHARS = 6_000      # max chars injected as file context (~1500 tokens)


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


def _read_for_context(path: str, item_title: str, file_type: str, workspace: str = "") -> str:
    """4-step progressive disclosure — most specific context first.

    1. CLAUDE.md in workspace root — project-level notes relevant to this item
    2. find_section   — exact section/function in the target file matching item_title
    3. search_relevant — goal-relevant lines scattered through the file
    4. find_continuation_point — last header → EOF (natural append point)

    Returns a pre-labelled context string ready for LLM injection.
    Caps at _MAX_CTX_CHARS to avoid context overflow.
    """
    parts: list[str] = []

    # Step 1: project notes (CLAUDE.md in workspace root)
    if workspace:
        claude_md = Path(workspace) / "CLAUDE.md"
        if claude_md.is_file():
            ws_ctx = search_relevant(item_title, [claude_md], context_lines=1, max_results=3)
            if ws_ctx:
                logger.debug("[ctx] CLAUDE.md hit  item=%r  chars=%d", item_title[:40], len(ws_ctx))
                parts.append(f"[CLAUDE.md]\n{ws_ctx}")
            else:
                logger.debug("[ctx] CLAUDE.md miss  item=%r", item_title[:40])

    # Step 2: exact section/function in the target file
    section = find_section(path, item_title, file_type)
    if section:
        logger.debug("[ctx] find_section hit  item=%r  chars=%d", item_title[:40], len(section))
        parts.append(f"[{Path(path).name} — {item_title}]\n{section}")
        ctx = "\n\n".join(parts)
        return ctx[:_MAX_CTX_CHARS] + "\n...[truncated]" if len(ctx) > _MAX_CTX_CHARS else ctx

    # Step 3: goal-relevant lines scattered through the file
    rel = search_relevant(item_title, [path], context_lines=2)
    if rel:
        logger.debug("[ctx] search_relevant hit  item=%r  chars=%d", item_title[:40], len(rel))
        parts.append(f"[{Path(path).name}]\n{rel}")
        ctx = "\n\n".join(parts)
        return ctx[:_MAX_CTX_CHARS] + "\n...[truncated]" if len(ctx) > _MAX_CTX_CHARS else ctx

    # Step 4: last section → EOF (natural continuation point)
    cont = find_continuation_point(path, file_type)
    if cont:
        logger.debug("[ctx] continuation fallback  item=%r  chars=%d", item_title[:40], len(cont))
        parts.append(f"[{Path(path).name}]\n{cont}")

    ctx = "\n\n".join(parts)
    return ctx[:_MAX_CTX_CHARS] + "\n...[truncated]" if len(ctx) > _MAX_CTX_CHARS else ctx


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
    context: str = "",
    workspace: str = "",
) -> str:
    if attempt > 0:
        # Retry: show verifier gap analysis
        fc = _read_file(manifest.file_path)
        diff = _verifier.run(manifest, fc)
        prompt = diff.resume_context
        if error_hint:
            prompt += f"\n\nNote from last attempt: {error_hint}"
        return prompt

    # First attempt: 4-step progressive context + goal + what to write next
    done = [it for it in manifest.items if it.status == "complete"]
    done_str = ", ".join(it.title for it in done) or "none yet"

    # Progressive disclosure: CLAUDE.md → exact section → relevant lines → continuation
    ctx_str = _read_for_context(manifest.file_path, item.title, manifest.file_type, workspace)
    file_state = ctx_str if ctx_str else f"[{manifest.file_path} — empty, start fresh]"

    if manifest.file_type == "code":
        marker_hint = f"Start with exactly: def {item.anchor}( (or class {item.anchor}:)\n"
    else:
        marker_hint = f"Start with exactly: ## {item.anchor}\n"

    prompt = ""
    if context:
        prompt += f"{context}\n\n"
    prompt += (
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
    context: str = "",
    workspace: str = "",
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

        prompt = _build_write_prompt(
            item, manifest, attempt, error_hint, done_summary,
            context=context if attempt == 0 else "",
            workspace=workspace if attempt == 0 else "",
        )
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

        # Rollback if file shrank by more than 20% (char-ratio regression guard)
        if snapshot and len(fc) < len(snapshot) * 0.8:
            logger.warning(
                "[subtask] char-ratio regression  item=%r attempt=%d  before=%d  after=%d — rolling back",
                item.title[:40], attempt, len(snapshot), len(fc),
            )
            try:
                Path(manifest.file_path).write_text(snapshot, encoding="utf-8")
                _verifier.run(manifest, snapshot)
            except OSError as exc:
                logger.error("[subtask] rollback failed: %s", exc)
            error_hint = (
                f"Your output caused the file to shrink from {len(snapshot)} to {len(fc)} chars. "
                f"Append only — do not overwrite or delete earlier content."
            )
            continue

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
    context: str = "",
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
        success = await _write_item(item, manifest, client, done_summary=done_summary, context=context, workspace=workspace)

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
