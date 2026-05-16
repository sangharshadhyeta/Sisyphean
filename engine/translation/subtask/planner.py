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
    "For documents: each subtask is one CONTENT SECTION with a meaningful heading.\n"
    "For code: each subtask is one function or class.\n\n"
    "Rules:\n"
    "- Output ONLY valid JSON, no explanation.\n"
    "- For docs: 'anchor' MUST be a short content heading like 'Introduction', 'Why I Am Alive', "
    "'Philosophical Dimensions', 'Conclusion'. NEVER use filenames, operations, or task descriptions "
    "as anchors (e.g. NEVER 'edit_markdown', 'save_as_doc', 'query_on_self.md', 'write_essay').\n"
    "- For code: 'anchor' is just the function or class name (no def/class prefix).\n"
    "- 'min_chars' is the minimum expected body length in characters.\n"
    "- 'kind' is one of: section, function, class, test.\n"
    "- Order items in logical writing order.\n"
    "- Do NOT include items already marked complete in existing content.\n\n"
    'Output format:\n{"subtasks": [\n'
    '  {"title": "Introduction", "anchor": "Introduction", "kind": "section", "min_chars": 300},\n'
    '  {"title": "Why I Am Alive", "anchor": "Why I Am Alive", "kind": "section", "min_chars": 400},\n'
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
            expected_min_chars=max(200, min_chars),
        ))
    return items


# Vague item titles that indicate the planner didn't decompose properly
_VAGUE_TITLES = {
    "main", "logic", "code", "function", "method", "class", "section",
    "content", "body", "text", "implementation", "module", "script",
    "write", "output", "result", "item", "part", "stuff", "things",
    # Operation/filename anchors — common mistake on edit tasks
    "edit", "edit_markdown", "save", "save_as_doc", "save_doc", "convert",
    "update", "essay", "document", "write_essay", "write_doc",
    # Common essay headings that are terrible code function names
    "introduction", "conclusion", "overview", "summary", "background",
    "discussion", "analysis", "abstract", "preface", "appendix",
}

# Operation verbs that start a title — indicate planner describing a task, not a content heading
_OP_VERB_STARTS = frozenset({
    "editing", "saving", "converting", "writing", "creating", "updating",
    "adding", "removing", "fixing", "running", "generating", "processing",
})

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
        # Entirely generic word or operation name
        if title in _VAGUE_TITLES:
            return True
        # Looks like "item 1", "function 2", "step 3"
        if re.match(r"^(item|step|part|function|section|chunk)\s*\d+$", title):
            return True
        # Looks like a filename — contains a dot extension (e.g. "query_on_self.md")
        if re.search(r'\.[a-z]{2,5}$', title):
            return True
        # Looks like a snake_case operation (edit_markdown, save_as_doc, write_essay)
        if re.match(r'^(edit|save|write|update|convert|create|add|fix|get)_\w+$', title):
            return True
        # Starts with an operation verb in natural language ("Editing the essay content")
        first_word = title.split()[0] if title.split() else ""
        if first_word in _OP_VERB_STARTS:
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
            response_format={"type": "json_object"},  # "{" prefix on llama.cpp
            stream=False,
            thinking=False,
        )
        raw = result["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.warning("SubtaskPlanner LLM error: %s", exc)
        return []
    return _parse_response(raw)


# Regex patterns that can extract a single item name without an LLM call.
# Matches: "Write the reverse_string function", "Implement class BankAccount",
#          "Add section Installation", "Write a ## Getting Started section"
_SINGLE_FUNC_RE = re.compile(
    r'\b(?:write|implement|add|create|define)\s+(?:the\s+|a\s+)?'
    r'(?:def\s+|class\s+|function\s+|method\s+)?([A-Za-z_]\w+)',
    re.IGNORECASE,
)
_SINGLE_HEADING_RE = re.compile(
    r'\b(?:write|add|create)\s+(?:the\s+|a\s+)?(?:#+\s+)?'
    r'["\']?([A-Z][A-Za-z0-9 _-]{2,40})["\']?\s*(?:section|heading)?',
    re.IGNORECASE,
)


_MULTI_SECTION_KW = frozenset({
    "essay", "document", "report", "article", "guide", "readme",
    "tutorial", "proposal", "specification", "overview", "summary",
    "analysis", "review", "based on", "reasons", "discuss",
})


def _try_single_item(stage_goal: str, file_type: Literal["doc", "code"]) -> list[dict] | None:
    """Return a single-item list if the goal names exactly one function/section.

    Avoids an LLM call for the common case of "Write the X function" / "Add X section".
    Returns None if the goal describes multiple or ambiguous items.
    """
    sg_lower = stage_goal.lower()

    # Bail out if the goal clearly describes multiple items
    if any(kw in sg_lower for kw in (" and ", ", ", "multiple", "all ", "each ", "several")):
        return None

    # Bail out for multi-section doc goals — these need the LLM planner to generate sections
    if file_type == "doc" and any(kw in sg_lower for kw in _MULTI_SECTION_KW):
        return None

    if file_type == "code":
        m = _SINGLE_FUNC_RE.search(stage_goal)
        if m:
            name = m.group(1)
            if name.lower() not in {"the", "a", "an", "function", "class", "method"}:
                kind = "class" if re.search(r'\bclass\b', stage_goal, re.I) else "function"
                return [{"title": name, "anchor": name, "kind": kind, "min_chars": 120}]
    else:
        m = _SINGLE_HEADING_RE.search(stage_goal)
        if m:
            heading = m.group(1).strip()
            # Reject multi-word generic phrases — a valid heading is 1-3 clean words
            words = heading.split()
            if (len(heading) >= 3 and len(words) <= 5
                    and heading.lower() not in _VAGUE_TITLES
                    and words[0].lower() not in {"an", "the", "a"}):
                return [{"title": heading, "anchor": heading, "kind": "section", "min_chars": 200}]

    return None


async def plan(
    client: Any,
    stage_goal: str,
    file_path: str,
    file_type: Literal["doc", "code"],
    existing_content: str = "",
) -> SubtaskManifest:
    """SubtaskManifest of named items to write.

    For simple single-function/section goals, extracts the name with regex
    and skips the LLM call entirely.  Falls back to one (or two, if vague)
    LLM calls for multi-item or ambiguous goals.
    """
    existing_note = ""
    if existing_content.strip():
        existing_note = (
            "\n\nExisting content (do not re-plan completed sections):\n"
            + _summarise_existing(existing_content)
        )

    # ── Fast path: regex extraction for single-name goals ────────────────────
    if not existing_note:  # only skip LLM when there's no existing content to check
        fast_items = _try_single_item(stage_goal, file_type)
        if fast_items:
            logger.debug("SubtaskPlanner: single-item regex shortcut for %r", stage_goal[:60])
            return SubtaskManifest(
                stage_goal=stage_goal,
                file_path=file_path,
                file_type=file_type,
                items=_items_from_raw(fast_items),
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

    # ── Item count cap for code files ─────────────────────────────────────────
    # Code files are written whole-file in a single pass; over-segmenting into
    # many small functions causes excessive Write round-trips (10 per file).
    # Cap at 3 items for code so simple scripts stay within max_rounds budgets.
    # Doc files keep their original count (they can legitimately be long).
    if file_type == "code" and len(raw_items) > 3:
        logger.info(
            "SubtaskPlanner: capping code items from %d → 3 for %r",
            len(raw_items), file_path,
        )
        raw_items = raw_items[:3]

    return SubtaskManifest(
        stage_goal=stage_goal,
        file_path=file_path,
        file_type=file_type,
        items=_items_from_raw(raw_items),
    )
