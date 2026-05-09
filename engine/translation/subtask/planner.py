"""Subtask planner — one LLM call that breaks a write goal into named items.

For code: items are function/class names.
For docs: items are section headings.

Returns a SubtaskManifest ready for the executor to iterate.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

from .manifest import SubtaskItem, SubtaskManifest

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^#{1,3} .+", re.MULTILINE)
_DEF_RE     = re.compile(r"^(?:def |class )\w+", re.MULTILINE)


def _summarise_existing(content: str) -> str:
    """Extract headings/def lines + last 200 chars — much cheaper than full content."""
    lines: list[str] = []
    for m in _HEADING_RE.finditer(content):
        lines.append(m.group().strip())
    for m in _DEF_RE.finditer(content):
        lines.append(m.group().strip())
    summary = "\n".join(lines)
    tail = content.strip()[-200:] if len(content) > 200 else ""
    if tail and tail not in summary:
        summary = (summary + "\n...\n" + tail).strip()
    return summary[:600] or content[:300]


_PLAN_SYSTEM = (
    "You are a task planner. Given a writing goal, output a JSON subtask list.\n\n"
    "For documents: each subtask is one section with a heading.\n"
    "For code: each subtask is one function or class.\n\n"
    "Rules:\n"
    "- Output ONLY valid JSON, no explanation.\n"
    "- For docs: 'anchor' is the heading text (without ##).\n"
    "- For code: 'anchor' is just the function or class name (no def/class prefix).\n"
    "- 'min_chars' is the minimum expected body length in characters.\n"
    "- 'kind' is one of: section, function, class, test.\n"
    "- Order items in logical writing order.\n"
    "- Do NOT include items already marked complete in existing content.\n\n"
    'Output format:\n{"subtasks": [\n'
    '  {"title": "reverse_string", "anchor": "reverse_string", "kind": "function", "min_chars": 100},\n'
    "  ...\n]}"
)


def _parse_response(raw: str) -> list[dict]:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    # Find first {...}
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(raw[start:end])
        items = data.get("subtasks", [])
        if not isinstance(items, list):
            return []
        return items
    except json.JSONDecodeError as exc:
        logger.warning("SubtaskPlanner JSON error: %s — raw: %.200s", exc, raw)
        return []


def _items_from_raw(raw_items: list[dict], start_index: int = 0) -> list[SubtaskItem]:
    items = []
    for i, r in enumerate(raw_items):
        title = str(r.get("title", f"Item {start_index + i}"))
        anchor = str(r.get("anchor", title))
        kind = r.get("kind", "section")
        if kind not in ("section", "function", "class", "test"):
            kind = "section"
        min_chars = int(r.get("min_chars", 200))
        items.append(SubtaskItem(
            index=start_index + i,
            title=title,
            anchor=anchor,
            kind=kind,
            expected_min_chars=max(80, min_chars),
        ))
    return items


# Vague item titles that indicate the planner didn't decompose properly
_VAGUE_TITLES = {
    "main", "logic", "code", "function", "method", "class", "section",
    "content", "body", "text", "implementation", "module", "script",
    "write", "output", "result", "item", "part", "stuff", "things",
}

_STRICT_PLAN_SYSTEM = (
    _PLAN_SYSTEM
    + "\n\nIMPORTANT: Use SPECIFIC names only. "
    "For code: exact function/class names like 'reverse_string', 'BankAccount', 'test_reverse'. "
    "For docs: exact heading text like 'Installation', 'API Reference'. "
    "NEVER use vague titles like 'main logic', 'helper function', 'section 1'."
)


def _is_vague(items: list[dict]) -> bool:
    """True if any item title is too generic to be a useful anchor."""
    for item in items:
        title = item.get("title", "").strip().lower()
        # Too short
        if len(title) < 3:
            return True
        # Entirely generic word
        if title in _VAGUE_TITLES:
            return True
        # Looks like "item 1", "function 2", "step 3"
        if re.match(r"^(item|step|part|function|section|chunk)\s*\d+$", title):
            return True
    return False


async def _call_planner(client: Any, prompt: str, system: str) -> list[dict]:
    """One planner LLM call. Returns parsed items or []."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    try:
        result = await client.generate(
            messages,
            max_tokens=400,
            temperature=0.2,
            response_format={"type": "json_object"},
            stream=False,
            thinking=False,
        )
        raw = result["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.warning("SubtaskPlanner LLM error: %s", exc)
        return []
    return _parse_response(raw)


async def plan(
    client: Any,
    stage_goal: str,
    file_path: str,
    file_type: Literal["doc", "code"],
    existing_content: str = "",
) -> SubtaskManifest:
    """One LLM call → SubtaskManifest of named items to write.

    If the first plan looks vague (generic titles), retries once with a
    stricter prompt that demands specific names.
    """
    existing_note = ""
    if existing_content.strip():
        existing_note = (
            "\n\nExisting content (do not re-plan completed sections):\n"
            + _summarise_existing(existing_content)
        )

    prompt = (
        f"Stage goal: {stage_goal}\n"
        f"File: {file_path}\n"
        f"File type: {file_type}{existing_note}"
    )

    raw_items = await _call_planner(client, prompt, _PLAN_SYSTEM)

    # Retry with stricter prompt if plan looks vague
    if not raw_items or _is_vague(raw_items):
        if raw_items:
            logger.warning(
                "SubtaskPlanner: vague items detected (%s) — retrying with strict prompt",
                [i.get("title") for i in raw_items],
            )
        raw_items = await _call_planner(client, prompt, _STRICT_PLAN_SYSTEM)

    if not raw_items:
        logger.warning("SubtaskPlanner returned no items after retry — single-item fallback")
        default_kind = "function" if file_type == "code" else "section"
        raw_items = [{"title": stage_goal[:60], "anchor": stage_goal[:60], "kind": default_kind, "min_chars": 400}]

    return SubtaskManifest(
        stage_goal=stage_goal,
        file_path=file_path,
        file_type=file_type,
        items=_items_from_raw(raw_items),
    )
