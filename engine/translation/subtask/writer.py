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

import re

from .manifest import SubtaskItem, SubtaskManifest
from engine.translation.subtask import planner as _planner
from engine.translation.subtask import verifier as _verifier
from engine.translation.subtask.verifier import parse_code_items, parse_doc_sections
from engine.translation.subtask.line_search import (
    find_section,
    find_continuation_point,
    search_relevant,
)

logger = logging.getLogger(__name__)

MAX_ITEM_RETRIES    = 3       # 3 attempts per item: first write + 2 continuations
MAX_WRITE_ITERATIONS = 20   # hard cap on total outer-loop iterations
_MAX_CTX_CHARS = 6_000      # max chars injected as file context (~1500 tokens)

# Writer token budget — 1024 is far too small for a real function/section.
# 2048 gives ~1600 words, enough for most complex single items.
MAX_TOKENS_WRITE = 2048

# How many lines of the file tail to show as the "seam" on continuation.
# 25 lines gives the model enough context (signature + partial body) without
# flooding the prompt with the entire file.
_SEAM_LINES = 25


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


# ── Edit-in-place helpers ─────────────────────────────────────────────────────
# Used by the verify-pass to patch partial/stub items without appending a second
# copy of the function/section (which causes duplicates and regressions).

def _apply_edit(path: str, old_text: str, new_text: str) -> bool:
    """Replace the first exact occurrence of old_text with new_text.

    Mirrors Claude Code's Edit tool: targeted old→new string swap rather than
    rewriting the whole file or blindly appending.  Returns True if the swap
    was applied, False if old_text was not found.
    """
    try:
        content = Path(path).read_text(encoding="utf-8")
    except OSError:
        return False
    if old_text in content:
        Path(path).write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        return True
    # Tolerant fallback: try stripped boundaries (handles trailing-newline mismatches)
    for candidate in (old_text.rstrip(), old_text.strip()):
        if candidate and candidate in content:
            Path(path).write_text(content.replace(candidate, new_text, 1), encoding="utf-8")
            return True
    logger.debug("[edit] old_text not found  path=%s  len=%d", Path(path).name, len(old_text))
    return False


def _key_match(anchor: str, parsed: dict) -> str | None:
    """Fuzzy key lookup: exact → substring → None."""
    clean = anchor.strip().lower()
    for k in parsed:
        if k.strip().lower() == clean:
            return k
    for k in parsed:
        if clean in k.strip().lower() or k.strip().lower() in clean:
            return k
    return None


def _get_item_text(item: SubtaskItem, manifest: SubtaskManifest, file_content: str) -> str | None:
    """Return the exact text of an item as it stands in the file.

    For code: the `def`/`class` line + current body (as parsed by parse_code_items).
    For docs: the `## Heading` line + current body.

    Returns None when the item's anchor is not yet present in the file.
    """
    if manifest.file_type == "code":
        parsed = parse_code_items(file_content)
        key = _key_match(item.anchor, parsed)
        return parsed.get(key) if key else None
    else:
        parsed = parse_doc_sections(file_content)
        key = _key_match(item.anchor, parsed)
        if key is None:
            return None
        body = parsed[key]
        # Reconstruct heading + body (heading was stripped by parse_doc_sections)
        for prefix in ("## ", "# "):
            heading = f"{prefix}{key}"
            if heading in file_content:
                return heading + ("\n" + body if body else "")
        return None


def _rebuild_item(existing_text: str, new_body: str) -> str:
    """Keep the header line (def/## line) from existing_text, swap in new_body.

    If the model accidentally echoed the header as the first line of new_body,
    that duplicate is stripped so the file stays clean.
    """
    lines = existing_text.splitlines()
    header = lines[0] if lines else ""
    body_lines = new_body.splitlines()
    # Drop any accidental header repetition from the LLM output
    if body_lines and body_lines[0].strip() == header.strip():
        body_lines = body_lines[1:]
    new_body_clean = "\n".join(body_lines).lstrip("\n")
    return (header + "\n" + new_body_clean) if header else new_body_clean


def _build_edit_prompt(item: SubtaskItem, manifest: SubtaskManifest, existing_text: str) -> str:
    """Prompt for the edit pass: show the partial item, ask for the complete body only.

    The LLM outputs ONLY the body (no def/class line for code, no ## heading for
    docs).  We then splice it back in via _rebuild_item + _apply_edit.
    """
    short_path = Path(manifest.file_path).name
    excerpt = existing_text[:700]
    if manifest.file_type == "code":
        header = existing_text.splitlines()[0] if existing_text.splitlines() else ""
        prompt = (
            f"The function in `{short_path}` is incomplete or stub:\n"
            f"```python\n{excerpt}\n```\n\n"
            f"Output the COMPLETE function body — the indented lines that come after "
            f"`{header}`.\n"
            f"Rules:\n"
            f"- Do NOT output the `def`/`class` line — body only.\n"
            f"- Properly indented (4 spaces).\n"
            f"- Min {item.expected_min_chars} chars of real logic.\n"
            f"- Raw Python only — no markdown fences.\n"
        )
    else:
        prompt = (
            f"The section `{item.anchor}` in `{short_path}` is incomplete:\n"
            f"```\n{excerpt}\n```\n\n"
            f"Output the COMPLETE section body — do NOT include the `## {item.anchor}` heading.\n"
            f"Min {item.expected_min_chars} chars. Raw text only, no markdown code fences.\n"
        )
    return prompt


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
    """Write one item with up to MAX_ITEM_RETRIES attempts.  Returns True if complete.

    Attempt 0 — fresh write
    ───────────────────────
    Build a focused prompt (goal + done items + 4-step progressive context) and
    call the LLM.  The output is appended (or written fresh for the very first item).

    Attempt ≥ 1 — EDIT pass  (if body already exists)
    ──────────────────────────────────────────────────
    When the item's anchor is already in the file (partial/stub body), use a
    targeted edit instead of appending a second copy of the function:

        _apply_edit(path, existing_text, _rebuild_item(existing_text, new_body))

    This is identical in principle to Claude Code's Edit tool — old_string →
    new_string swap.  No duplicates, no regression risk for surrounding items.

    Attempt ≥ 1 — APPEND pass  (if anchor is missing)
    ──────────────────────────────────────────────────
    Anchor not found → fall through to the standard append path with the full
    resume_context prompt so the model writes the item from scratch.

    Rollback guard
    ──────────────
    A snapshot is taken before every write/edit.  If previously-complete items
    regress after the change, the snapshot is restored.
    """
    item.status = "in_progress"
    is_first = item.index == 0
    error_hint = ""

    for attempt in range(MAX_ITEM_RETRIES + 1):
        logger.info(
            "[subtask] attempt=%d/%d  item=%r  file=%s",
            attempt, MAX_ITEM_RETRIES, item.title[:40], manifest.file_path,
        )

        # ── Read current file state (used for both snapshot and edit detection) ──
        snapshot = _read_file(manifest.file_path)
        prev_complete = {it.anchor for it in manifest.items if it.status == "complete"}

        # ── Decide: EDIT mode or APPEND mode? ────────────────────────────────
        # Edit mode when (a) we are on a retry AND (b) the item already has some
        # content in the file (partial/stub).  "already has content" means
        # actual_chars > 20 — avoids triggering on completely missing anchors.
        use_edit = attempt > 0 and item.actual_chars > 20
        existing_text: str | None = None
        if use_edit:
            existing_text = _get_item_text(item, manifest, snapshot)
            if not existing_text:
                use_edit = False   # anchor not in file — use append path instead

        # ── Build prompt ──────────────────────────────────────────────────────
        if use_edit and existing_text:
            prompt = _build_edit_prompt(item, manifest, existing_text)
        else:
            prompt = _build_write_prompt(
                item, manifest, attempt, error_hint, done_summary,
                context=context if attempt == 0 else "",
                workspace=workspace if attempt == 0 else "",
            )
        error_hint = ""

        # ── LLM call ──────────────────────────────────────────────────────────
        messages = [
            {"role": "system", "content": _WRITE_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        try:
            result = await client.generate(
                messages,
                max_tokens=MAX_TOKENS_WRITE,
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

        min_out = max(60, item.expected_min_chars // 4)
        if len(raw) < min_out:
            error_hint = (
                f"Output was only {len(raw)} chars — write the complete item "
                f"(minimum ~{item.expected_min_chars} chars). Do not stop early."
            )
            continue

        # ── Apply: EDIT path ──────────────────────────────────────────────────
        if use_edit and existing_text:
            new_text = _rebuild_item(existing_text, raw)
            applied = _apply_edit(manifest.file_path, existing_text, new_text)
            if applied:
                logger.info(
                    "[subtask] edit applied  item=%r  old=%dc  new=%dc",
                    item.title[:40], len(existing_text), len(new_text),
                )
            else:
                # Edit target drifted (e.g. file was concurrently modified).
                # Fall back to append so the attempt isn't wasted.
                logger.debug("[subtask] edit target not found — falling back to append")
                _write_file(manifest.file_path, "\n\n" + raw, append=True)

        # ── Apply: APPEND / FRESH-WRITE path ─────────────────────────────────
        else:
            if is_first and attempt == 0 and not snapshot:
                _write_file(manifest.file_path, raw, append=False)
            elif is_first and attempt > 0 and item.status == "missing":
                # Anchor was truly missing on a retry — write fresh
                _write_file(manifest.file_path, raw, append=False)
            else:
                _write_file(manifest.file_path, "\n\n" + raw, append=True)

        # ── Verify ────────────────────────────────────────────────────────────
        fc = _read_file(manifest.file_path)
        diff = _verifier.run(manifest, fc)

        # Char-ratio regression guard (only meaningful for append path)
        if not use_edit and snapshot and len(fc) < len(snapshot) * 0.8:
            logger.warning(
                "[subtask] char-ratio regression  before=%d  after=%d — rolling back",
                len(snapshot), len(fc),
            )
            try:
                Path(manifest.file_path).write_text(snapshot, encoding="utf-8")
                _verifier.run(manifest, snapshot)
            except OSError as exc:
                logger.error("[subtask] rollback failed: %s", exc)
            error_hint = (
                f"File shrank from {len(snapshot)} to {len(fc)} chars — "
                f"append only, do not overwrite earlier content."
            )
            continue

        # Anchor regression guard
        regressed_anchors = {it.anchor for it in diff.regressed} & prev_complete
        if regressed_anchors:
            logger.warning(
                "[subtask] regression  item=%r attempt=%d  regressed=%s — rolling back",
                item.title[:40], attempt, sorted(regressed_anchors)[:3],
            )
            try:
                Path(manifest.file_path).write_text(snapshot, encoding="utf-8")
                _verifier.run(manifest, snapshot)
            except OSError as exc:
                logger.error("[subtask] rollback failed: %s", exc)
            error_hint = (
                f"Caused {len(regressed_anchors)} previously-complete section(s) to disappear. "
                f"Output only the body of '{item.title}', nothing else."
            )
            continue

        if item.status == "complete":
            logger.info("[subtask] item=%r complete  chars=%d", item.title[:40], item.actual_chars)
            return True

        if item.status == "partial":
            error_hint = (
                f"'{item.title}' is partial ({item.actual_chars}c, "
                f"need {item.expected_min_chars}c). Continue the content."
            )
        elif item.status == "missing":
            error_hint = (
                f"'{item.title}' not found in file. "
                f"Start with the exact anchor line."
            )
        elif item.status == "regressed":
            error_hint = f"'{item.title}' regressed — earlier content may have been overwritten."

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
