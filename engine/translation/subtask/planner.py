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


_PLAN_SYSTEM_CODE = (
    "You are a code planner. List the functions and classes to write for the given goal.\n\n"
    "Rules:\n"
    "- Output ONLY valid JSON, no explanation.\n"
    "- Each item 'anchor' is the exact function or class name (no 'def'/'class' prefix, no spaces).\n"
    "- 'kind' is one of: function, class, test.\n"
    "- 'min_chars' is minimum expected body length in characters.\n"
    "- Order items in logical coding order.\n"
    "- Do NOT include items already in existing content.\n\n"
    'Output format:\n{"subtasks": [\n'
    '  {"title": "add", "anchor": "add", "kind": "function", "min_chars": 150},\n'
    '  {"title": "subtract", "anchor": "subtract", "kind": "function", "min_chars": 150},\n'
    '  {"title": "multiply", "anchor": "multiply", "kind": "function", "min_chars": 150},\n'
    "  ...\n]}"
)

_PLAN_SYSTEM_DOC = (
    "You are a document planner. List the sections to write for the given goal.\n\n"
    "Rules:\n"
    "- Output ONLY valid JSON, no explanation.\n"
    "- Each item 'anchor' MUST be a short, specific section heading that matches the document topic.\n"
    "  Use headings that fit the ACTUAL document being written, not generic placeholders.\n"
    "- NEVER use filenames, operations, or task descriptions as anchors.\n"
    "- 'kind' is always 'section'.\n"
    "- 'min_chars' is minimum expected body length in characters.\n"
    "- Order sections in logical document order.\n"
    "- Do NOT include sections already in existing content.\n\n"
    'Output format:\n{"subtasks": [\n'
    '  {"title": "Introduction", "anchor": "Introduction", "kind": "section", "min_chars": 300},\n'
    '  {"title": "Main Argument", "anchor": "Main Argument", "kind": "section", "min_chars": 400},\n'
    '  {"title": "Counterarguments", "anchor": "Counterarguments", "kind": "section", "min_chars": 300},\n'
    '  {"title": "Conclusion", "anchor": "Conclusion", "kind": "section", "min_chars": 200},\n'
    "  ...\n]}"
)

# Legacy alias — kept for _STRICT_PLAN_SYSTEM which appends to it
_PLAN_SYSTEM = _PLAN_SYSTEM_CODE


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

_STRICT_PLAN_SYSTEM_CODE = (
    _PLAN_SYSTEM_CODE
    + "\n\nIMPORTANT: Use SPECIFIC names only. "
    "Exact function/class names like 'reverse_string', 'BankAccount', 'test_reverse'. "
    "NEVER use vague titles like 'main', 'logic', 'helper', 'function1', 'section 1'."
)

_STRICT_PLAN_SYSTEM_DOC = (
    _PLAN_SYSTEM_DOC
    + "\n\nIMPORTANT: Use SPECIFIC heading text only. "
    "Exact headings like 'Installation', 'API Reference', 'Getting Started'. "
    "NEVER use generic placeholders like 'section 1', 'content', 'body'."
)

# Legacy aliases for any external references
_STRICT_PLAN_SYSTEM = _STRICT_PLAN_SYSTEM_CODE


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

    plan_sys  = _PLAN_SYSTEM_CODE  if file_type == "code" else _PLAN_SYSTEM_DOC
    strict_sys = _STRICT_PLAN_SYSTEM_CODE if file_type == "code" else _STRICT_PLAN_SYSTEM_DOC

    raw_items = await _call_planner(client, prompt, plan_sys)

    # Retry with stricter prompt if plan looks vague
    # Also retry if code items look like essay sections (model confuses file types)
    def _has_doc_leak(items: list[dict]) -> bool:
        """True when code items contain essay/doc section names."""
        if file_type != "code":
            return False
        doc_names = {"introduction", "conclusion", "overview", "why i am alive",
                     "philosophical dimensions", "counterarguments", "preface",
                     "background", "abstract", "appendix", "discussion"}
        for item in items:
            if item.get("anchor", "").strip().lower() in doc_names:
                return True
        return False

    def _has_invalid_code_anchors(items: list[dict]) -> bool:
        """True when any code anchor is not a valid Python identifier.

        Python function/class names cannot contain spaces or special chars.
        When the model echoes the full task description as the anchor
        (e.g. "Write a small program to give me daily updates on AI"),
        the writer would emit `def Write a small program...` which is a
        SyntaxError.  Catch this structurally — no content hardcoding.
        """
        if file_type != "code":
            return False
        for item in items:
            anchor = item.get("anchor", "").strip()
            if not re.match(r'^[A-Za-z_]\w*$', anchor):
                return True
        return False

    if not raw_items or _is_vague(raw_items) or _has_doc_leak(raw_items) or _has_invalid_code_anchors(raw_items):
        if raw_items:
            logger.warning(
                "SubtaskPlanner: vague/wrong items detected (%s) — retrying with strict prompt",
                [i.get("title") for i in raw_items],
            )
        raw_items = await _call_planner(client, prompt, strict_sys)

    # For code: drop any items whose anchor is not a valid Python identifier
    # even after the retry.  Invalid anchors (containing spaces) cause the
    # writer to emit `def write a small program...` which is a SyntaxError.
    # Dropping them leaves items=[] in _start_write_plan, which falls through
    # to the single-shot _generate_code path that writes correct Python.
    if file_type == "code" and raw_items:
        valid = [it for it in raw_items if re.match(r'^[A-Za-z_]\w*$', it.get("anchor", "").strip())]
        if not valid:
            logger.warning(
                "SubtaskPlanner: all code anchors still invalid after retry (%s) — "
                "dropping to trigger single-shot fallback",
                [i.get("anchor") for i in raw_items],
            )
            raw_items = []

    if not raw_items:
        logger.warning("SubtaskPlanner returned no items after retry — single-item fallback")
        default_kind = "function" if file_type == "code" else "section"
        # For deepen tasks, extract the specific section/function being deepened
        # rather than using the full "Deepen: ..." string as the anchor.
        _anchor = stage_goal[:60]
        if stage_goal.lower().startswith("deepen:"):
            _deepen_rest = stage_goal[7:].strip()
            # Prefer a quoted name: Deepen: 'Introduction' section → Introduction
            _quoted = re.search(r"['\"]([^'\"]{3,60})['\"]", _deepen_rest)
            if _quoted:
                _anchor = _quoted.group(1).strip()
            else:
                # Take text up to first "needs"/"must"/"should"/"is" verb
                _anchor = re.split(r'\b(?:needs|must|should|is |has |lacks)\b',
                                   _deepen_rest, maxsplit=1)[0].strip()[:60]
        # For code: if the fallback anchor isn't a valid Python identifier
        # (e.g. the full task description slipped through), convert it to
        # snake_case so the writer doesn't emit `def write a program...`.
        if file_type == "code" and not re.match(r'^[A-Za-z_]\w*$', _anchor):
            _parts = re.findall(r'[A-Za-z]+', _anchor)[:5]
            _anchor = "_".join(p.lower() for p in _parts)[:40] or "main_program"
            logger.info("SubtaskPlanner: sanitised fallback anchor → %r", _anchor)
        raw_items = [{"title": _anchor, "anchor": _anchor, "kind": default_kind, "min_chars": 400}]

    return SubtaskManifest(
        stage_goal=stage_goal,
        file_path=file_path,
        file_type=file_type,
        items=_items_from_raw(raw_items),
    )
