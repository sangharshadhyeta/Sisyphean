"""Subtask verifier — parses written files and scores each manifest item.

Pure regex, no LLM calls. Works for both Python code (def/class) and
markdown documents (## headings).
"""
from __future__ import annotations

import logging
import re
import time
from typing import Optional

from .manifest import SubtaskItem, SubtaskManifest, SubtaskDiff

logger = logging.getLogger(__name__)

_STUB_BODIES = {"pass", "...", "..", "raise NotImplementedError", "raise NotImplementedError()"}


# ── Doc parser ────────────────────────────────────────────────────────────────

def parse_doc_sections(content: str) -> dict[str, str]:
    """Return {heading_text: body_text} splitting on ## or # headings."""
    sections: dict[str, str] = {}
    current: Optional[str] = None
    buf: list[str] = []

    for line in content.splitlines(keepends=True):
        m = re.match(r"^#{1,2}\s+(.+)", line)
        if m:
            if current is not None:
                sections[current] = "".join(buf).strip()
            current = m.group(1).strip()
            buf = []
        else:
            buf.append(line)

    if current is not None:
        sections[current] = "".join(buf).strip()

    return sections


# ── Code parser ───────────────────────────────────────────────────────────────

def parse_code_items(content: str) -> dict[str, str]:
    """Return {name: body} for top-level def/class in Python source."""
    items: dict[str, str] = {}
    lines = content.splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"^(def |class )(\w+)", lines[i])
        if m:
            name = m.group(2)
            body_lines: list[str] = [lines[i]]
            i += 1
            while i < len(lines):
                if lines[i] and not lines[i][0].isspace() and re.match(r"^(def |class |\S)", lines[i]):
                    break
                body_lines.append(lines[i])
                i += 1
            items[name] = "\n".join(body_lines).rstrip()
        else:
            i += 1
    return items


# ── Stub detection ────────────────────────────────────────────────────────────

def is_stub_body(body: str) -> bool:
    stripped = body.strip()
    if not stripped:
        return True
    meaningful = [
        line.strip() for line in stripped.splitlines()
        if line.strip()
        and not line.strip().startswith("#")
        and not line.strip().startswith('"""')
        and not line.strip().startswith("'''")
        and not re.match(r"^(def |class )", line)
    ]
    if not meaningful:
        return True
    return all(line in _STUB_BODIES for line in meaningful)


# ── Key matching ──────────────────────────────────────────────────────────────

def _match_key(anchor: str, parsed: dict[str, str]) -> Optional[str]:
    anchor_clean = anchor.strip().lower()
    for key in parsed:
        if key.strip().lower() == anchor_clean:
            return key
        if anchor_clean in key.strip().lower() or key.strip().lower() in anchor_clean:
            return key
    return None


# ── Main verifier ─────────────────────────────────────────────────────────────

def run(manifest: SubtaskManifest, file_content: str) -> SubtaskDiff:
    """Compare manifest items against actual file content; update item statuses."""
    if manifest.file_type == "doc":
        parsed = parse_doc_sections(file_content)
    else:
        parsed = parse_code_items(file_content)

    logger.debug(
        "[verifier] file=%s type=%s items=%d parsed_keys=%d",
        manifest.file_path, manifest.file_type, manifest.total, len(parsed),
    )
    diff = SubtaskDiff()

    for item in manifest.items:
        key = _match_key(item.anchor, parsed)

        if key is None:
            item.status = "missing"
            diff.missing.append(item)
            continue

        body = parsed[key]
        current_chars = len(body)
        stub = manifest.file_type == "code" and is_stub_body(body)

        if stub or current_chars < item.expected_min_chars:
            prev_complete = item.status == "complete"
            item.is_stub = stub
            item.mark_partial(body)
            if prev_complete:
                # Was complete before, now drops below threshold → genuine regression
                item.status = "regressed"
                diff.regressed.append(item)
            else:
                diff.partial.append(item)
            continue

        item.mark_complete(body)
        diff.complete.append(item)

    diff.seam_index = next(
        (it.index for it in manifest.items if it.status != "complete"),
        len(manifest.items),
    )
    manifest.update_file_hash(file_content)
    diff.resume_context = _build_resume_context(manifest, diff, file_content)

    logger.info(
        "[verifier] complete=%d partial=%d missing=%d regressed=%d",
        len(diff.complete), len(diff.partial), len(diff.missing), len(diff.regressed),
    )
    return diff


# ── Resume context ────────────────────────────────────────────────────────────

def _build_resume_context(manifest: SubtaskManifest, diff: SubtaskDiff, file_content: str) -> str:
    """Build a resume prompt for the retry path.

    NOTE: When the item's anchor already exists in the file (partial/stub body),
    the writer uses _build_edit_prompt + _apply_edit instead of this resume
    context.  This context is used only when the anchor is genuinely missing
    (first write failed, item still = "missing").
    """
    lines = [f"File: {manifest.file_path}"]
    for it in manifest.items:
        if it.status == "complete":
            lines.append(f"  [done] {it.title}  ({it.actual_chars}c)")
        elif it.status == "partial":
            lines.append(
                f"  [partial] {it.title}  — {it.actual_chars}c written "
                f"(needs {it.expected_min_chars}c)  <-- resume here"
            )
        elif it.status in ("missing", "regressed"):
            marker = "REGRESSED" if it.status == "regressed" else "MISSING"
            lines.append(f"  [{marker}] {it.title}")
        else:
            lines.append(f"  [pending] {it.title}")

    if diff.seam_index < len(manifest.items):
        seam = manifest.items[diff.seam_index]
        in_file = seam.actual_chars > 0
        lines += [
            "",
            f"Resume item [{diff.seam_index}]: \"{seam.anchor}\"",
            f"  Anchor {'IS already in the file (partial body — write body only, NO def/## line)' if in_file else 'is NOT yet in the file — write from the anchor header'}.",
            f"  Min {seam.expected_min_chars} chars of substantive content.",
        ]

    if file_content:
        # Show last 30 LINES (not arbitrary chars) — gives proper code/doc context
        tail_lines = file_content.splitlines()[-30:]
        tail = "\n".join(tail_lines)
        lines += [
            "",
            "--- current file tail (seam — continue AFTER the last line below) ---",
            tail,
            "--- end seam ---",
        ]

    return "\n".join(lines)
