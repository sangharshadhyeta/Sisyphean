"""Decomposer — converts a user task into a self-contained instruction manifest.

This is the "conversion step": one LLM call that does the heavy thinking
upfront so every subsequent execution call is minimal and concrete.

Why a separate step
-------------------
A 4B model asked to "do a complex task" while also managing history,
tool results, and step tracking will lose context.  But asked to "follow
this specific instruction and call this one tool" it is reliable.

The decomposer bridges these two modes:
  - It runs with MORE context (full task + memory) to produce a plan
  - Each instruction it produces is written to be independently followable:
      BAD:  "Continue the previous work"
      GOOD: "Run `python scraper.py --url https://bbc.com/news` and
             check stdout for at least 5 article titles"
  - needs_prev=True when the instruction genuinely requires the previous
    step's output (e.g. "Write code using the library found in prev step")

Once the manifest is produced, the executor never needs to think about
the overall task — it just follows one instruction at a time with the
single most recent result as context.

JSON contract
-------------
The model is asked to produce:
{
  "goal": "one-sentence success description",
  "steps": [
    {
      "type": "research|write_code|write_doc|verify|reflect|direct",
      "text": "fully self-contained instruction",
      "needs_prev": true|false
    },
    ...
  ]
}

Valid types map to stage budgets in prompts.STAGE_BUDGETS.
"""
from __future__ import annotations

import json
import logging
import re

from engine.translation.manifest import Instruction, TaskManifest
from engine.translation.prompts import SYSTEM, STAGE_BUDGETS

logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

_DECOMPOSE_PROMPT = """\
You are a task planner. Convert the user's request into a concrete step-by-step plan.

USER REQUEST: {task}

{memory_block}

Rules for each step:
1. Write the instruction so it can be followed WITHOUT reading any previous steps.
   - Include specific filenames, URLs, function names, search queries.
   - Do NOT say "continue from before" or "use the previous result" in the text.
2. Set needs_prev=true ONLY when this step must use the actual output of the previous step.
3. Keep steps atomic: one action per step (one search, one file write, one bash run).
4. Scale to complexity: 1 step for simple/direct tasks; 2-3 for research tasks; 3-5 for
   complex multi-part tasks (reports, audits, code pipelines).
5. Research, analysis, and explanation tasks MUST have at least 2 steps.

Step text keyword rules:
- If the task says "search memory" or "check memory" or "user history" → step text starts with "Search memory for ..."
- If the task says "search the web" or "look up online" or "find current" → step text starts with "Search the web for ..."
- If the task says "greet" or is a greeting → ONE step only, text: "Search memory for user context and give a direct greeting as Sisyphean"
- If the task says "edit" or "update" or "modify" an existing file → step text starts with "Edit <filename> to ..." (never "write a new ...")
- If the task says "convert" a file to another format → step text starts with "Convert <filename> to <format> by running ..."
- CRITICAL: "search memory" and "search the web" are DIFFERENT actions. Never confuse them.
- CRITICAL: "edit an existing file" uses Edit tool — do NOT plan a "write new file" step for it.

Return ONLY a JSON object (no type field needed — it is determined automatically):
{{
  "goal": "<one sentence: what the user wants to achieve>",
  "steps": [
    {{"text": "<self-contained instruction>", "needs_prev": false}},
    ...
  ]
}}"""

# Fallback when LLM returns unparseable output — treat whole task as one research step
def _fallback_manifest(task: str) -> TaskManifest:
    return TaskManifest(
        goal=task[:120],
        steps=[
            Instruction(
                idx=0,
                type="research",
                text=task,
                needs_prev=False,
            )
        ],
    )


# ── Public API ────────────────────────────────────────────────────────────────

async def decompose(
    task: str,
    memory_context: str,
    client,
) -> TaskManifest:
    """Convert a user task into a TaskManifest of self-contained instructions.

    Falls back to a single-step manifest on any parse failure so the loop
    always has something to execute.
    """
    memory_block = ""
    if memory_context:
        # Keep memory context brief for the decomposition call
        memory_block = f"Context from memory:\n{memory_context[:600]}"

    prompt = _DECOMPOSE_PROMPT.format(
        task=task[:800],
        memory_block=memory_block,
    )

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": prompt},
    ]

    try:
        result = await client.generate(
            messages,
            max_tokens=800,
            temperature=0.2,          # low temperature for consistent structure
            response_format={"type": "json_object"},
            stream=False,
            thinking=False,           # JSON output: thinking breaks response_format
        )
        raw = result["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.warning("decompose: LLM call failed (%s) — using fallback", exc)
        return _fallback_manifest(task)

    data = _parse_json(raw)
    if not data:
        logger.warning("decompose: unparseable response — using fallback. Raw: %s", raw[:200])
        return _fallback_manifest(task)
    logger.debug("decompose: parsed data keys=%s raw=%s", list(data.keys()), raw[:100])

    return _build_manifest(task, data)


# ── Helpers ───────────────────────────────────────────────────────────────────

# Keywords used to derive step type from step text without asking the LLM.
# Checked in priority order: verify > edit(direct) > write_code > write_doc > reflect > research.
_VERIFY_KW    = ("run ", "bash", "execute", "mkdir", "create file", "create folder",
                 "touch ", "test ", "confirm ", "check if", "verify ", "chmod",
                 "install ", "pip ", "npm ", "git ", "convert ")
_CODE_KW      = (".py", ".js", ".ts", ".sh", ".rb", ".go", ".java", ".rs",
                 "function ", "def ", "class ", "implement", "script",
                 "module", "write code", "write a function", "write a class")
_DOC_KW       = (".md", ".txt", ".rst", ".html", "document", "report",
                 "readme", "write a ", "draft ", "write the ")
_REFLECT_KW   = ("analys", "summaris", "summariz", "review", "compare",
                 "synthesiz", "synthesise", "reflect", "evaluate")
# Edit/update steps should use the executor (read + Edit tools), NOT the subtask writer.
# Priority: checked BEFORE write_code / write_doc to avoid misclassifying "edit X.md" as write_doc.
_EDIT_KW      = ("edit ", "update ", "modify ", "revise ", "patch ",
                 "append to ", "add to ", "insert into ", "change ")


def _derive_type(text: str) -> str:
    """Derive the step type from its text using keyword rules — no LLM needed."""
    t = text.lower()
    if any(kw in t for kw in _VERIFY_KW):
        return "verify"
    # Edit/update existing content → direct (executor uses Read + Edit tools)
    if any(kw in t for kw in _EDIT_KW):
        return "direct"
    if any(kw in t for kw in _CODE_KW):
        return "write_code"
    if any(kw in t for kw in _DOC_KW):
        return "write_doc"
    if any(kw in t for kw in _REFLECT_KW):
        return "reflect"
    return "research"


def _build_manifest(task: str, data: dict) -> TaskManifest:
    """Build a TaskManifest from parsed LLM output.

    Validates and normalises each step; falls back to a single-step
    manifest if the steps list is empty or malformed.

    Step type is now derived locally via _derive_type() — the LLM no longer
    needs to output a type field, reducing JSON complexity and hallucination risk.
    """
    goal = (data.get("goal") or task[:120]).strip()
    raw_steps = data.get("steps") or []

    if not isinstance(raw_steps, list) or not raw_steps:
        logger.warning("decompose: empty steps — using fallback")
        return _fallback_manifest(task)

    steps: list[Instruction] = []

    for i, s in enumerate(raw_steps[:5]):  # cap at 5 steps
        if not isinstance(s, dict):
            continue
        text = (s.get("text") or "").strip()
        if not text:
            continue

        # Accept legacy `type` field from old prompts/cached responses, but
        # always re-derive locally to fix hallucinated or stale values.
        stype = _derive_type(text)

        needs_prev = bool(s.get("needs_prev", False))

        steps.append(Instruction(
            idx=i,
            type=stype,
            text=text,
            needs_prev=needs_prev,
        ))

    if not steps:
        logger.warning("decompose: no valid steps parsed — using fallback")
        return _fallback_manifest(task)

    # Re-index in case some steps were filtered
    for i, s in enumerate(steps):
        s.idx = i

    logger.info(
        "decompose: %d steps for %r",
        len(steps), goal[:60],
    )
    return TaskManifest(goal=goal, steps=steps)


def _parse_json(text: str) -> dict | None:
    """Find and parse the first complete balanced JSON object in text."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                fragment = text[start : i + 1]
                try:
                    return json.loads(fragment)
                except json.JSONDecodeError:
                    cleaned = re.sub(r",\s*([}\]])", r"\1", fragment)
                    try:
                        return json.loads(cleaned)
                    except json.JSONDecodeError:
                        return None
    return None
