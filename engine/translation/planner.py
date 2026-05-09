"""Task planner — stage type inference, plan generation, reflect gate.

Ported and merged from BirdClaw birdclaw/agent/planner.py.

Key upgrades over the previous version:
- parse_format_response(): 4-attempt JSON repair (handles 4B model output quirks)
- infer_stage_type(): richer keyword sets matching BirdClaw's production tuning
- reflect_on_stage(): post-stage reflection gate — evaluates quality, inserts follow-ups
- generate_plan(): 2 retry attempts with error feedback
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from engine.translation.prompts import (
    PLAN_SCHEMA_PROMPT,
    STAGE_BUDGETS,
    dynamic_context,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage type constants
# ---------------------------------------------------------------------------

# Stage types that use format-mode (json_object) instead of tool calls.
FORMAT_STAGE_TYPES = frozenset({"write_code", "write_doc"})

# Stage types where think() signals stage completion.
THINK_ADVANCES_TYPES = frozenset({"research", "reflect"})

# Stage types that get a post-completion reflection gate.
REFLECT_GATE_TYPES = frozenset({"research", "reflect", "write_code", "write_doc"})


# ---------------------------------------------------------------------------
# Stage keyword sets — ported from BirdClaw with production tuning
# ---------------------------------------------------------------------------

_DOC_KW = {"document", "proposal", "report", "markdown", "readme",
            "spec", "article", "essay", "write up", "write a"}
_CODE_KW = {"write", "create", "implement", "code", "function", "class",
             "script", "generate", "build", "develop", "program", "module"}
_VERIFY_KW = {"test", "check", "verify", "validate", "confirm",
               "assert", "ensure", "pass", "fail", "lint",
               "run", "execute", "install", "complete"}
_RESEARCH_KW = {"search", "research", "find", "look up", "fetch", "browse",
                 "investigate", "online", "web", "http"}


# ---------------------------------------------------------------------------
# Stage dataclass
# ---------------------------------------------------------------------------

@dataclass
class Stage:
    type: str
    goal: str
    budget: int = 0

    def __post_init__(self):
        if not self.budget:
            self.budget = STAGE_BUDGETS.get(self.type, 10)


@dataclass
class Plan:
    outcome: str
    stages: list[Stage] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.stages


# ---------------------------------------------------------------------------
# JSON repair + parsing (ported from BirdClaw — handles 4B output quirks)
# ---------------------------------------------------------------------------

def _repair_json(text: str) -> str:
    """Best-effort JSON repair for common small-model output errors."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)
    text = text.strip()
    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = re.sub(r"(?<![\\])'([^']*)'", r'"\1"', text)
    return text


def parse_format_response(content: str) -> dict | None:
    """Extract a JSON object from a format-mode response.

    4-attempt strategy: direct → code fence → bare {} → repair entire content.
    Handles trailing commas, single quotes, stray text, and markdown fences.
    """
    if not content:
        return None

    # Attempt 1: direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Attempt 2: code fence extraction
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            try:
                return json.loads(_repair_json(m.group(1)))
            except json.JSONDecodeError:
                pass

    # Attempt 3: bare {} extraction
    m2 = re.search(r"\{.*\}", content, re.DOTALL)
    if m2:
        raw = m2.group()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            try:
                return json.loads(_repair_json(raw))
            except json.JSONDecodeError:
                pass

    # Attempt 4: repair entire content
    try:
        return json.loads(_repair_json(content))
    except json.JSONDecodeError:
        pass

    return None


# ---------------------------------------------------------------------------
# Stage type inference
# ---------------------------------------------------------------------------

def infer_stage_type(step_text: str) -> str:
    """Classify a plain-English step into a stage type.

    Priority: research > verify > write_doc > write_code
    Research wins when a step mentions both search and write — the intent
    is to gather information, not produce a file.
    """
    s = step_text.lower()
    if any(k in s for k in _RESEARCH_KW):
        return "research"
    if any(k in s for k in _VERIFY_KW):
        return "verify"
    if any(k in s for k in _DOC_KW):
        return "write_doc"
    if any(k in s for k in _CODE_KW):
        return "write_code"
    return "research"


# ---------------------------------------------------------------------------
# Plan generation
# ---------------------------------------------------------------------------

_PLAN_SYSTEM = """\
Output a JSON plan. Format exactly:
{"outcome": "one sentence success criteria", "steps": "step1 | step2 | step3"}
Steps are pipe-separated plain English actions.
Use plain verbs: Run, Search, Write, Read, Summarise.
Scale steps to complexity: 1 for simple tasks, 3-5 for complex multi-part tasks.
RULES:
- NEVER plan a step that asks the user for input. If info is needed, answer immediately instead.
- Use 'Write' steps ONLY when the user explicitly asks for a file/document/report to be saved.
- For web research: use 'Search' or 'Fetch' steps.
OPTIONAL: Add "budgets": "12 | 60 | 8" (pipe-count must match steps) ONLY when a step
needs more than the default (research=12, write_doc=10, write_code=12, verify=8, reflect=5).
"""

async def make_plan(goal: str, client, workspace: str = "") -> Plan:
    """Generate a Plan for *goal*. Falls back gracefully on any failure.

    Two attempts with error feedback if the first parse fails.
    """
    logger.debug("[plan] goal=%r", goal[:80])
    ctx = dynamic_context(workspace=workspace, task_goal=goal)
    last_error = ""

    for attempt in range(2):
        error_hint = f"\n\nPrevious attempt failed: {last_error}" if last_error else ""
        try:
            result = await client.generate(
                [
                    {"role": "system", "content": _PLAN_SYSTEM},
                    {"role": "user", "content": f"{ctx}\n\nTask: {goal}{error_hint}"},
                ],
                max_tokens=256,
                temperature=0.2,
                response_format={"type": "json_object"},
                stream=False,
                thinking=False,
            )
            raw = result["choices"][0]["message"]["content"].strip()
            plan = _parse_plan(raw, goal)
            if not plan.empty:
                return plan
            last_error = f"empty steps in: {raw[:80]!r}"
        except Exception as exc:
            logger.warning("planner failed attempt %d: %s", attempt + 1, exc)
            last_error = str(exc)

    logger.warning("plan generation failed — single-stage fallback")
    return _fallback(goal)


# ---------------------------------------------------------------------------
# Reflection gate
# ---------------------------------------------------------------------------

_REFLECT_GATE_PROMPT = """\
Outcome target: {outcome}
Stage just completed ({stage_type}): {summary}

Evaluate: does the output sufficiently advance the outcome?
Output exactly one JSON choice:
  Sufficient — proceed: {{"decision": "continue"}}
  Outcome already fully met: {{"decision": "done"}}
  Needs more work (specific gap): {{"decision": "deepen", "goal": "what is missing"}}
"""

async def reflect_on_stage(
    outcome: str,
    stage_type: str,
    stage_summary: str,
    client,
    steps_remaining: int = 99,
) -> dict:
    """One cheap format-mode call after a stage completes.

    Evaluates whether the output is sufficient or needs follow-up.
    Returns one of:
      {"decision": "continue"}
      {"decision": "done"}
      {"decision": "deepen", "goal": "<what's missing>"}

    Falls back to {"decision": "continue"} on any failure — gate must never block.
    """
    budget_hint = (
        "\nBudget tight — prefer continue or done if close enough."
        if steps_remaining <= 5 else ""
    )
    prompt = _REFLECT_GATE_PROMPT.format(
        outcome=outcome,
        stage_type=stage_type,
        summary=stage_summary,
    ) + budget_hint

    _VALID = ("continue", "done", "deepen")
    for attempt in range(2):
        try:
            result = await client.generate(
                [{"role": "user", "content": prompt}],
                max_tokens=128,
                temperature=0.1,
                response_format={"type": "json_object"},
                stream=False,
                thinking=False,
            )
            raw = result["choices"][0]["message"]["content"].strip()
            parsed = parse_format_response(raw)
            if parsed:
                decision = parsed.get("decision", "")
                if decision in _VALID:
                    if decision == "deepen" and not parsed.get("goal"):
                        return {"decision": "continue"}
                    logger.debug("[reflect-gate] %s  stage=%s", decision, stage_type)
                    return parsed
        except Exception as exc:
            logger.debug("reflect gate attempt %d failed: %s", attempt + 1, exc)

    return {"decision": "continue"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_plan(raw: str, goal: str) -> Plan:
    data = parse_format_response(raw)
    if not data:
        logger.debug("plan parse failed — raw: %s", raw[:200])
        return _fallback(goal)

    outcome = (data.get("outcome") or "").strip()
    steps_raw = (data.get("steps") or "").strip()

    if not outcome or not steps_raw:
        return _fallback(goal)

    budgets_raw = (data.get("budgets") or "").strip()
    budget_list = [b.strip() for b in budgets_raw.split("|")] if budgets_raw else []

    if isinstance(steps_raw, list):
        plain_steps = [s.strip() for s in steps_raw if str(s).strip()]
    else:
        plain_steps = [s.strip() for s in str(steps_raw).split("|") if s.strip()]

    stages: list[Stage] = []
    for i, step_text in enumerate(plain_steps):
        stage_type = infer_stage_type(step_text)
        budget = 0
        if i < len(budget_list):
            try:
                budget = int(budget_list[i])
            except ValueError:
                pass
        stages.append(Stage(
            type=stage_type,
            goal=step_text,
            budget=budget or STAGE_BUDGETS.get(stage_type, 10),
        ))

    if not stages:
        return _fallback(goal)

    logger.debug("plan: outcome=%r stages=%s", outcome[:60], [s.type for s in stages])
    return Plan(outcome=outcome, stages=stages)


def _fallback(goal: str) -> Plan:
    return Plan(
        outcome=f"Complete: {goal[:80]}",
        stages=[Stage(type="research", goal=goal, budget=STAGE_BUDGETS["research"])],
    )


# ---------------------------------------------------------------------------
# New pipeline planner — split + plan_task
# Uses pipe-separated strings (more reliable on 0.6b than JSON arrays)
# ---------------------------------------------------------------------------

_SPLIT_SYSTEM = """\
Does message B use a pronoun (its/their/his/her) that refers to the answer of message A, \
AND would knowing A's SPECIFIC ANSWER (e.g. exact name, version, or identifier) allow a \
more precise search query for B than a combined search would?

If YES — the answer to A is needed to build an effective query for B:
  {"tasks": "part A | part B rephrased without pronouns, using specific placeholders"}
If NO — a single search covers both, or both parts are independent:
  {"tasks": "original message verbatim"}
Output only the JSON — no explanation."""

# Sisyphean's own internal tools — always available regardless of harness
_INTERNAL_TOOLS = [
    ("web_search",    "search the web for any factual or current information"),
    ("save_memory",   "save a user preference or important fact for later recall"),
    ("search_memory", "look up previously saved facts or research"),
]
_INTERNAL_TOOL_NAMES = frozenset(t[0] for t in _INTERNAL_TOOLS)


def _build_plan_system(outer_tools: list[dict]) -> str:
    """Build the Stage 2 system prompt from harness-provided tools + internal tools."""
    lines = ["Pick the right tool and write its exact input. Use pipe | to chain steps.\n"]
    lines.append("Available tools:")
    for t in outer_tools:
        name = t.get("name", "").lower()
        desc = (t.get("description") or "")[:80]
        if name:
            lines.append(f"  {name} — {desc}")
    for name, desc in _INTERNAL_TOOLS:
        lines.append(f"  {name} — {desc}")
    lines += [
        "",
        "NEVER answer factual questions from memory — always web_search.",
        "Each web_search query must be a short keyword phrase — strip question words.",
        "If the task asks for two dependent things (identify X, then details of X), use two steps.",
        "For write/create file tasks: bash input must be a SHORT description only — NEVER embed actual code.",
        'For greetings, thanks, or social messages output: {"steps": ""}',
        'Reply: {"steps": "toolname:input"}',
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Signal extraction — pure regex, annotates tasks for the LLM
# ---------------------------------------------------------------------------

_MATH_RE   = re.compile(r'^[\d\s\+\-\*\/\%\(\)\.\^]+$')
_REMEMBER_RE = re.compile(
    r'^\s*(?:remember|note that|keep in mind|don\'?t forget|save that|store that)\s+(.+)',
    re.IGNORECASE,
)
_SOUL_KW = frozenset(["alive", "sentient", "conscious", "exist", "real", "feel",
                       "inner self", "who are you", "what are you", "are you a", "do you have"])
_WEB_KW  = frozenset(["latest", "current", "today", "tonight", "now", "recent", "news",
                       "price", "cost", "version", "release", "update", "weather",
                       "stock", "rate", "live", "2024", "2025", "2026"])
_RESEARCH_KW = frozenset(["search online", "search for", "search the web", "look up",
                           "look it up", "find online", "browse", "fetch online", "google"])
_EXT_LANG = {   # language/type word → glob pattern
    "python": "*.py", "py": "*.py", "javascript": "*.js", "js": "*.js",
    "typescript": "*.ts", "ts": "*.ts", "markdown": "*.md", "md": "*.md",
    "text": "*.txt", "txt": "*.txt", "json": "*.json",
    "yaml": "*.yaml", "yml": "*.yml", "html": "*.html", "css": "*.css",
}
_DIR_SKIP = frozenset(["folder", "directory", "dir", "this", "the", "current", "here", "that"])
_SPLIT_JUNK = frozenset(["task1", "task2", "task_a", "task_b",
                          "first question", "second question",
                          "step 1", "step 2", "step1", "step2"])


def _spoken_math_to_python(text: str) -> str:
    """Convert spoken/symbolic math to a Python expression string."""
    t = text.strip()
    m = re.search(r'\bsqrt\s*\(?\s*(\d+(?:\.\d+)?)\s*\)?', t, re.IGNORECASE)
    if m:
        return f"__import__('math').sqrt({m.group(1)})"
    m = re.search(r'square\s+root\s+of\s+(\d+(?:\.\d+)?)', t, re.IGNORECASE)
    if m:
        return f"__import__('math').sqrt({m.group(1)})"
    t = re.sub(r'\b(what\s+is|what\'s|calculate|compute|evaluate|how\s+much\s+is|find|the)\b',
               '', t, flags=re.IGNORECASE)
    t = re.sub(r'[?!.,]', '', t).strip()
    t = re.sub(r'\btimes\b|\bmultiplied\s+by\b',   '*', t, flags=re.IGNORECASE)
    t = re.sub(r'\bplus\b|\badded\s+to\b',         '+', t, flags=re.IGNORECASE)
    t = re.sub(r'\bminus\b|\bsubtracted\s+from\b', '-', t, flags=re.IGNORECASE)
    t = re.sub(r'\bdivided\s+by\b|\bover\b',       '/', t, flags=re.IGNORECASE)
    t = re.sub(r'\bto\s+the\s+power\s+of\b|\braised\s+to\b', '**', t, flags=re.IGNORECASE)
    t = re.sub(r'\s*([\+\-\*\/])\s*', r'\1', t).strip()
    if re.search(r'\d[\+\-\*\/\*\*\^%]\d', t):
        return t
    orig_clean = re.sub(r'[?!.,\s]', '', text.strip())
    if re.search(r'\d[\+\-\*\/\^%]\d', orig_clean):
        return orig_clean
    return t or text.strip()


def _looks_like_math(text: str) -> bool:
    t = text.strip().rstrip('?').strip()
    return bool(_MATH_RE.match(t)) and any(c in t for c in '+-*/')


def _extract_signals(task: str) -> dict:
    """Extract structured signals from a task. Returns a hints dict.

    These signals are injected into the LLM prompt — regex annotates,
    LLM decides. Short-circuit keys (tool_hint == 'math'/'remember'/'soul')
    bypass the LLM entirely.
    """
    t  = task.strip()
    tl = t.lower()

    # ── Math ──────────────────────────────────────────────────────────────────
    spoken = _spoken_math_to_python(t)
    if _looks_like_math(spoken) or spoken.startswith("__import__"):
        return {"tool_hint": "math", "expr": spoken}
    if _looks_like_math(t):
        return {"tool_hint": "math", "expr": t.rstrip('?').strip()}

    # ── Remember ──────────────────────────────────────────────────────────────
    m = _REMEMBER_RE.match(t)
    if m:
        return {"tool_hint": "remember", "content": m.group(1).strip().rstrip('?.,')}

    # ── Policy / identity ─────────────────────────────────────────────────────
    if any(kw in tl for kw in _SOUL_KW):
        return {"tool_hint": "policy"}

    hints: dict = {}

    # ── Research intent (blocks bash short-circuit for multi-step tasks) ──────
    if any(kw in tl for kw in _RESEARCH_KW):
        hints["has_research"] = True

    # ── File / shell signals ──────────────────────────────────────────────────
    # Extract filename + extension
    fname_m = (re.search(r'(?:called|named)\s+(\w[\w.-]*\.[a-zA-Z]{2,5})', t, re.IGNORECASE) or
               re.search(r'(\w[\w.-]*\.[a-zA-Z]{2,5})\b', t, re.IGNORECASE))
    if fname_m:
        hints["filename"] = fname_m.group(1).rstrip('?.,')
        hints["filetype"] = hints["filename"].rsplit('.', 1)[-1].lower()

    write_m = re.search(r'\b(write|create|generate|make|produce|build)\b', tl)
    edit_m  = re.search(r'\b(edit|modify|change|update|replace|fix|set|patch|rename)\b', tl)
    run_m   = re.search(r'\b(run|execute|then\s+run|and\s+run|then\s+test|and\s+test\s+it)\b', tl)

    if write_m:
        ftype = hints.get("filetype", "")
        if ftype in ("py", "js", "ts", "sh", "rb", "go", "rs", "c", "cpp", "java"):
            hints["tool_hint"] = "bash"
            hints["action"]    = "write_code"
        elif hints.get("filename"):
            hints["tool_hint"] = "bash"
            hints["action"]    = "write_text"
            # Extract literal content if specified
            cm = re.search(
                r"(?:containing|with\s+(?:the\s+)?(?:content|text|word|line)s?)\s+"
                r"(?:the\s+(?:word|text|line)\s+)?[\"']?(.+?)[\"']?(?:\.|$)",
                t, re.IGNORECASE,
            )
            if cm:
                hints["content"] = cm.group(1).strip().strip("'\"").rstrip('?.,')  # strip surrounding quotes
        elif re.search(r'\b(folder|directory|dir)\b', tl):
            hints["tool_hint"] = "bash"
            hints["action"]    = "mkdir"
        if run_m:
            hints["run_after"] = True

    if not hints.get("tool_hint") and edit_m:
        # Broaden: "the python file", "the file", "the script" count even without an explicit filename
        has_file_ref = (hints.get("filename") or
                        re.search(r'\b(port|config|value|setting)\b', tl) or
                        re.search(r'\bthe\s+(python\s+)?(?:file|script|code|program)\b', tl))
        if has_file_ref:
            hints["tool_hint"] = "bash"
            hints["action"]    = "edit_file"
            # If only a generic file reference, mark as semantic rewrite
            if not hints.get("filename"):
                # Guess extension from context
                if re.search(r'\bpython\b|\bpy\b', tl):
                    hints["filetype"]  = "py"
                    hints["rewrite"]   = True  # signals write_code-style regen
            # Extract "from X to Y" for deterministic sed construction
            chg_m = re.search(r'\bfrom\s+["\']?(\S+?)["\']?\s+to\s+["\']?(\S+?)["\']?(?:\s|$|\.)',
                               t, re.IGNORECASE)
            if chg_m:
                hints["from_val"] = chg_m.group(1).rstrip('?.,')
                hints["to_val"]   = chg_m.group(2).rstrip('?.,')
            # "change X to Y" without from keyword
            elif not chg_m:
                chg_m2 = re.search(r'\b(?:change|set|update|replace)\b\s+\S+\s+to\s+["\']?(\S+?)["\']?(?:\s|$|\.)',
                                    t, re.IGNORECASE)
                if chg_m2:
                    hints["to_val"] = chg_m2.group(1).rstrip('?.,')

    # Shell ops: ls, list files, mkdir, rm, touch
    if not hints.get("tool_hint"):
        if re.search(r'\b(list|ls|show)\b.{0,30}\bfiles?\b', tl):
            hints["tool_hint"] = "bash"
            hints["action"]    = "list_files"
            # Language/extension hint
            for word in re.findall(r'\w+', tl):
                if word in _EXT_LANG:
                    hints["pattern"] = _EXT_LANG[word]
                    break
            if not hints.get("pattern"):
                ext_m2 = re.search(r'\*?\.([\w]+)\b', t)
                if ext_m2:
                    hints["pattern"] = f"*.{ext_m2.group(1)}"
            # Directory hint
            dir_m = re.search(r'\bin\s+(?:this\s+)?(?:folder|directory|dir|the\s+)?(\S+)?',
                               t, re.IGNORECASE)
            if dir_m and dir_m.group(1) and dir_m.group(1).lower() not in _DIR_SKIP:
                hints["directory"] = dir_m.group(1).rstrip('?.,')
        elif re.search(r'\b(mkdir|rmdir|touch|rm\b|del\b|mv\b|cp\b)\b', tl):
            hints["tool_hint"] = "bash"
            hints["action"]    = "shell_op"
        elif run_m and re.search(r'\b(script|file|program|\.py|\.sh)\b', tl):
            hints["tool_hint"] = "bash"
            hints["action"]    = "run_script"

    # ── Grep / search-in-code signals ─────────────────────────────────────────
    if not hints.get("tool_hint"):
        grep_m = re.search(
            r'\b(grep|search\s+(for|in|code|files?)|find\s+(in|inside|within|all|where)|'
            r'look\s+for\s+.{1,30}\bin\b)\b', tl)
        if grep_m and not re.search(r'\b(web|online|internet|google)\b', tl):
            hints["tool_hint"] = "grep"
            hints["action"]    = "grep"
            # Extract the thing to search for
            kw_m = re.search(r'(?:grep|search\s+for|find)\s+["\']?([^"\']+?)["\']?\s*(?:in|$)', t,
                              re.IGNORECASE)
            if kw_m:
                hints["pattern"] = kw_m.group(1).strip().rstrip('?.,')

    # ── WebFetch: explicit URL in the task ────────────────────────────────────
    if not hints.get("tool_hint"):
        url_m = re.search(r'https?://\S+', t)
        if url_m and any(kw in tl for kw in ("fetch", "get", "scrape", "read", "open", "visit")):
            hints["tool_hint"] = "webfetch"
            hints["action"]    = "web_fetch"
            hints["url"]       = url_m.group(0).rstrip('.,)')

    # ── Web search signals ────────────────────────────────────────────────────
    if not hints.get("tool_hint"):
        if any(kw in tl for kw in _WEB_KW):
            hints["tool_hint"] = "web"
            q = re.sub(r'^(what is|what\'s|who is|where is|when is|how much is)\s+',
                       '', tl, flags=re.IGNORECASE).rstrip('?').strip()
            hints["search_q"] = q

    return hints


def _hints_to_str(sig: dict) -> str:
    """Convert signal dict to a short hint string for the LLM prompt."""
    parts = []
    if sig.get("tool_hint"):
        parts.append(f"likely tool: {sig['tool_hint']}")
    if sig.get("action"):
        parts.append(f"action: {sig['action']}")
    if sig.get("filename"):
        parts.append(f"filename: {sig['filename']}")
    if sig.get("filetype"):
        parts.append(f"filetype: {sig['filetype']}")
    if sig.get("run_after"):
        parts.append("run after writing")
    if sig.get("content"):
        parts.append(f"content: {sig['content'][:60]}")
    if sig.get("pattern"):
        parts.append(f"pattern: {sig['pattern']}")
    if sig.get("directory"):
        parts.append(f"directory: {sig['directory']}")
    if sig.get("search_q"):
        parts.append(f"search query: {sig['search_q'][:60]}")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# split + plan_task
# ---------------------------------------------------------------------------

async def split_deep(query: str, client, max_depth: int = 3) -> list[str]:
    """Iteratively split until every sub-task is atomic or max_depth reached.

    "Atomic" means split() returns the task unchanged — the LLM judged it
    needs no further decomposition.  Each pass re-offers changed tasks to
    split(); unchanged ones are left alone.  Stops as soon as a full pass
    produces no new splits.
    """
    tasks = [query]
    for depth in range(max_depth):
        expanded: list[str] = []
        changed = False
        for t in tasks:
            subs = await split(t, client)
            if len(subs) > 1:
                expanded.extend(subs)
                changed = True
                logger.debug("split_deep depth=%d: %r → %d tasks", depth + 1, t[:60], len(subs))
            else:
                expanded.append(t)
        tasks = expanded
        if not changed:
            logger.debug("split_deep: stable at depth=%d, %d task(s)", depth, len(tasks))
            break
    return tasks


async def split(query: str, client) -> list[str]:
    """Split a query into independent sub-tasks.

    Short-circuits to [query] for short/math inputs.
    Falls back to [query] on any failure.
    """
    words = query.strip().split()
    if len(words) <= 4 or _looks_like_math(query):
        return [query]

    try:
        result = await client.generate(
            [
                {"role": "system", "content": _SPLIT_SYSTEM},
                {"role": "user",   "content": query[:300]},
            ],
            max_tokens=800,
            temperature=0.1,
            response_format={"type": "json_object"},
            stream=False,
            thinking=False,
        )
        raw  = result["choices"][0]["message"]["content"].strip()
        data = parse_format_response(raw)
        if data:
            tasks_raw = str(data.get("tasks") or "").strip()
            if tasks_raw.startswith("[") or tasks_raw.startswith("("):
                return [query]
            tasks = [t.strip() for t in tasks_raw.split("|") if t.strip()]
            if any(t.lower() in _SPLIT_JUNK for t in tasks):
                return [query]
            if tasks:
                logger.debug("split: %d tasks from %r", len(tasks), query[:60])
                return tasks
    except Exception as exc:
        logger.warning("split failed: %s", exc)
    return [query]


_CODE_GEN_SYSTEM = """\
Write only the code — no explanation, no markdown fences, no comments.
Keep it concise: minimal imports, no docstrings, no unittest boilerplate.
Output raw code only. Must be syntactically complete and runnable."""

_BASH_CMD_SYSTEM = """\
Output ONE bash command as JSON: {"cmd": "bash command here"}
Examples:
  Change value:  sed -i 's/old/new/g' file.txt
  Run script:    python script.py
  Create file:   echo 'content' > file.txt
  Rename:        mv old.py new.py
No explanation. Only the JSON."""


_CODE_START_RE = re.compile(
    r'^(?:def |class |import |from |for |while |if |print\(|#!|# |return |'
    r'[a-zA-Z_]\w*\s*[=(])',
)


async def _generate_code(task: str, client) -> str | None:
    """Ask LLM to generate code for a file-write task. Returns raw code string.

    Post-processes the response to strip any reasoning/preamble text that the
    model emits before the actual code (common with small thinking models).
    """
    try:
        result = await client.generate(
            [
                {"role": "system", "content": _CODE_GEN_SYSTEM},
                {"role": "user",   "content": f"Task: {task[:300]}"},
            ],
            max_tokens=1600,
            temperature=0.1,
            stream=False,
            thinking=False,
        )
        raw = result["choices"][0]["message"]["content"].strip()
        # Strip markdown fences
        raw = re.sub(r'^```\w*\n?', '', raw, flags=re.MULTILINE).strip()
        raw = re.sub(r'\n?```\s*$', '', raw, flags=re.MULTILINE).strip()
        # Strip leading reasoning lines — find first line that looks like code
        lines = raw.splitlines()
        for i, line in enumerate(lines):
            if _CODE_START_RE.match(line.strip()):
                raw = '\n'.join(lines[i:]).strip()
                break
        return raw if len(raw) >= 4 else None
    except Exception as exc:
        logger.warning("_generate_code failed: %s", exc)
    return None


async def _generate_bash_cmd(task: str, client) -> str | None:
    """Ask LLM to generate a single bash command (edit/run/misc tasks).

    Accepts JSON {"cmd": "..."} or a plain bash command line — 0.6b models
    often drop the JSON wrapper but emit the command directly.
    """
    _BAD = frozenset(["bash command here", "cmd", "command", ""])
    try:
        result = await client.generate(
            [
                {"role": "system", "content": _BASH_CMD_SYSTEM},
                {"role": "user",   "content": f"Task: {task[:300]}"},
            ],
            max_tokens=100,
            temperature=0.1,
            response_format={"type": "json_object"},
            stream=False,
            thinking=False,
        )
        raw = result["choices"][0]["message"]["content"].strip()

        # Try JSON parse first
        data = parse_format_response(raw)
        if data:
            cmd = (data.get("cmd") or "").strip()
            if cmd and len(cmd) > 4 and cmd.lower() not in _BAD:
                return cmd

        # Fall back: raw output might be a bare command (strip code fences)
        bare = re.sub(r'^```\w*\n?', '', raw, flags=re.MULTILINE).strip()
        bare = re.sub(r'\n?```$', '', bare).strip()
        # Accept only if it looks like a shell command (starts with a known cmd word)
        if bare and len(bare) > 4 and bare.lower() not in _BAD:
            first_word = bare.split()[0].lower().rstrip(';')
            if first_word in ("sed", "mv", "cp", "rm", "echo", "python", "python3",
                              "cat", "touch", "mkdir", "find", "grep", "awk"):
                return bare

    except Exception as exc:
        logger.warning("_generate_bash_cmd failed: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Flipped 3-stage planner
#
# Pre-filter  (0ms, regex)  : hard short-circuits — math / remember / soul
# Stage 1     (LLM normalize): raw natural language → structured intent JSON
# Stage 2     (Python dispatch): intent JSON → concrete {tool, input} steps
#             (may call _generate_code or _generate_bash_cmd as sub-calls)
# ---------------------------------------------------------------------------

_NORMALIZE_SYSTEM = """\
Classify the task. Output ONE short JSON object only — no prose, no markdown.

"intent" must be one of:
  write_code  write_text  edit_file  list_files  run_script
  web_search  web_fetch  grep  search_code
  direct  save_memory  math  mkdir  shell_op

Use "direct" for: greetings, thanks, acknowledgements, simple social exchanges,
or any task that needs no tool calls to answer.

Include only relevant extra fields:
  file, run, from_val, to_val, pattern, query, expr, description, cmd

CRITICAL: NEVER include actual code, file contents, or generated text in this JSON.
For write_code: output ONLY {"intent": "write_code", "file": "filename.ext"} plus
"run": true if the user wants to test/execute it after. Nothing else.

"run": true — set when the user wants to execute/test the file after writing it

For web_search: "query" must be a short keyword phrase (2-5 words) targeting the FIRST
unknown fact needed. Strip question words.

Output only the JSON object — no examples, no prose."""


async def _normalize_intent(task: str, context: str, client) -> dict:
    """Stage 1: Ask the LLM to convert natural language → structured intent JSON.

    Returns a dict with at least an 'intent' key.
    Falls back to {} on any failure — caller must handle empty dict gracefully.
    """
    prompt = f"Task: {task[:300]}"
    if context:
        prompt += f"\nContext:\n{context[:200]}"
    try:
        result = await client.generate(
            [
                {"role": "system", "content": _NORMALIZE_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=200,
            temperature=0.1,
            response_format={"type": "json_object"},
            stream=False,
            thinking=False,
        )
        raw  = result["choices"][0]["message"]["content"].strip()
        data = parse_format_response(raw) or {}
        if data:
            logger.debug("normalize: %r → intent=%s", task[:60], data.get("intent", "?"))
        else:
            logger.warning("normalize: parse failed for %r — raw=%r", task[:40], raw[:120])
        return data
    except Exception as exc:
        logger.warning("_normalize_intent failed: %s", exc)
    return {}


async def _dispatch_intent(intent: dict, task: str, client) -> list[dict]:
    """Stage 2: Pure Python dispatch on a structured intent dict.

    Deterministic for most intents; calls _generate_code / _generate_bash_cmd
    only when content must be synthesised (write_code with no given code, etc.).
    """
    kind  = intent.get("intent", "").lower()
    fname = (intent.get("file") or "").strip()

    # ── write_code ────────────────────────────────────────────────────────────
    if kind == "write_code":
        if not fname:
            fname = "script.py"
        desc = intent.get("description") or intent.get("task_description") or task
        code = await _generate_code(desc, client)
        if code:
            final_code = code if code.endswith('\n') else code + '\n'
            # Use the Write tool — clean diff display in Claude Code UI,
            # no shell quoting issues. Pipeline resolves file_path to absolute.
            steps = [{"tool": "write", "input": final_code, "file_path": fname}]
            if intent.get("run"):
                steps.append({"tool": "bash", "input": f"python {fname}"})
            return steps

    # ── write_text ────────────────────────────────────────────────────────────
    if kind == "write_text":
        content = (intent.get("content") or "").replace("'", "\\'")
        if content and fname:
            return [{"tool": "bash", "input": f"echo '{content}' > {fname}"}]
        # No literal content specified — generate it
        cmd = await _generate_bash_cmd(task, client)
        if cmd:
            return [{"tool": "bash", "input": cmd}]

    # ── edit_file ─────────────────────────────────────────────────────────────
    if kind == "edit_file":
        from_val = (intent.get("from_val") or "").strip()
        to_val   = (intent.get("to_val")   or "").strip()
        if from_val and to_val and fname:
            # Exact string replacement — use bash sed
            return [{"tool": "bash", "input": f"sed -i 's/{from_val}/{to_val}/g' {fname}"}]
        if fname:
            # Semantic edit with known filename: read it first, then replan
            return [{"tool": "read", "input": fname, "file_path": fname}]
        # No filename known: search for files matching the filetype, then replan
        ftype   = (intent.get("filetype") or "py").strip()
        pattern = f"*.{ftype}"
        return [{"tool": "glob", "input": pattern, "pattern": pattern}]

    # ── list_files ────────────────────────────────────────────────────────────
    if kind == "list_files":
        pattern   = (intent.get("pattern")   or "*").strip()
        directory = (intent.get("directory") or "." ).strip()
        cmd = f"ls {directory}/{pattern}" if directory != "." else f"ls {pattern}"
        return [{"tool": "bash", "input": cmd}]

    # ── mkdir ─────────────────────────────────────────────────────────────────
    if kind == "mkdir":
        name = fname or "output"
        return [{"tool": "bash", "input": f"mkdir -p {name}"}]

    # ── math ──────────────────────────────────────────────────────────────────
    if kind == "math":
        expr = (intent.get("expr") or intent.get("math_expr") or "").strip()
        if expr:
            return [{"tool": "bash", "input": f'python -c "print({expr})"'}]

    # ── run_script ────────────────────────────────────────────────────────────
    if kind == "run_script":
        target = fname or intent.get("cmd") or ""
        if target:
            return [{"tool": "bash", "input": f"python {target}"}]
        cmd = await _generate_bash_cmd(task, client)
        if cmd:
            return [{"tool": "bash", "input": cmd}]

    # ── shell_op ──────────────────────────────────────────────────────────────
    if kind == "shell_op":
        cmd = (intent.get("cmd") or "").strip()
        if not cmd:
            cmd = await _generate_bash_cmd(task, client)
        if cmd:
            return [{"tool": "bash", "input": cmd}]

    # ── web_search ────────────────────────────────────────────────────────────
    # Pipeline will upgrade to Claude Code's WebSearch outer tool at runtime if available.
    if kind == "web_search":
        q = (intent.get("query") or intent.get("search_query") or task).strip()
        return [{"tool": "web_search", "input": q}]

    # ── web_fetch ─────────────────────────────────────────────────────────────
    if kind == "web_fetch":
        url    = (intent.get("url") or intent.get("query") or "").strip()
        prompt = (intent.get("prompt") or "Extract all relevant information.").strip()
        if url:
            return [{"tool": "webfetch", "input": url, "url": url, "prompt": prompt}]

    # ── grep / search_code ────────────────────────────────────────────────────
    if kind in ("grep", "search_code"):
        pattern  = (intent.get("pattern") or intent.get("query") or task).strip()
        filetype = (intent.get("filetype") or "").strip()
        step: dict = {"tool": "grep", "input": pattern, "pattern": pattern}
        if filetype:
            step["filetype"] = filetype
        return [step]

    # ── save_memory ───────────────────────────────────────────────────────────
    if kind == "save_memory":
        return [{"tool": "save_memory", "input": task}]

    # ── unknown / fallback ────────────────────────────────────────────────────
    logger.warning("_dispatch_intent: unknown intent %r for task %r — letting synthesizer answer", kind, task[:60])
    return []


async def plan_task(
    task: str,
    outer_tools: list[dict],
    client,
    context: str = "",
    soul_section: str = "",
    user_prefs: str = "",
) -> list[dict]:
    """Plan a single task → list of {tool, input} steps.

    Two-stage pipeline:
      Pre-filter — regex signals for unambiguous tasks (no LLM)
      Stage 1    — LLM normalize → structured intent JSON → Python dispatch
      Stage 2    — LLM fallback; soul + user prefs shape the approach

    context may include [Memory from previous research] blocks — the planner
    uses these to avoid redundant web searches.
    """
    # ── Regex pre-filter — bypass LLM entirely for unambiguous tasks ─────────
    # The LLM normalize is unreliable for write/edit tasks on small models
    # (they output prose or embed full code inside the JSON). Regex is faster
    # and more reliable for these patterns.
    signals = _extract_signals(task)
    sig_hint   = signals.get("tool_hint", "")
    sig_action = signals.get("action", "")

    if sig_hint == "math":
        expr = signals.get("expr", "")
        if expr:
            return [{"tool": "bash", "input": f'python -c "print({expr})"'}]

    if sig_hint == "remember":
        return [{"tool": "save_memory", "input": signals.get("content", task)}]

    if sig_hint == "policy":
        return [{"tool": "search_policy", "input": task}]

    if sig_action == "write_code":
        intent_direct = {
            "intent": "write_code",
            "file": signals.get("filename", "script.py"),
            "description": task,
            "run": bool(signals.get("run_after")),
        }
        steps = await _dispatch_intent(intent_direct, task, client)
        if steps:
            logger.debug("plan_task: regex→dispatch write_code → %d step(s)", len(steps))
            return steps

    if sig_action in ("write_text", "edit_file", "list_files", "mkdir",
                      "run_script", "shell_op"):
        intent_direct = {
            "intent": sig_action,
            "file":    signals.get("filename", ""),
            "content": signals.get("content", ""),
            "from_val": signals.get("from_val", ""),
            "to_val":   signals.get("to_val", ""),
            "pattern":  signals.get("pattern", ""),
            "directory": signals.get("directory", ""),
            "filetype": signals.get("filetype", ""),
            "rewrite":  signals.get("rewrite", False),
        }
        steps = await _dispatch_intent(intent_direct, task, client)
        if steps:
            logger.debug("plan_task: regex→dispatch %s → %d step(s)", sig_action, len(steps))
            return steps

    # ── Stage 1: LLM normalize → Python dispatch ─────────────────────────────
    intent = await _normalize_intent(task, context, client)

    if intent:
        kind = intent.get("intent", "")
        # Only treat as "direct" (no tools needed) for pure social exchanges —
        # not for any task that contains an action verb or file/tool reference.
        _ACTION_HINTS = frozenset({
            "list", "write", "create", "run", "find", "show", "search", "check",
            "edit", "make", "build", "install", "execute", "read", "get", "open",
            "delete", "move", "copy", "fetch", "look", "print", "generate",
        })
        task_words = set(task.lower().split())
        _looks_like_task = bool(task_words & _ACTION_HINTS)

        if kind in ("direct", "answer") and not _looks_like_task:
            logger.debug("plan_task: intent=%s, no action hints → synthesizer answers", kind)
            return []
        if kind in ("direct", "answer") and _looks_like_task:
            logger.debug("plan_task: intent=%s but task has action hints → falling to Stage 2", kind)
            # Fall through to Stage 2 which has the tool list
        elif kind:
            steps = await _dispatch_intent(intent, task, client)
            if steps:
                logger.debug("plan_task: normalize→dispatch → %d step(s) intent=%s",
                             len(steps), kind)
                return steps

    # ── Stage 2: LLM fallback for complex/multi-step tasks ───────────────────
    logger.debug("plan_task: normalize missed %r — falling back to LLM", task[:60])
    prompt = f"Task: {task[:200]}"
    if context:
        prompt += f"\nContext:\n{context[:400]}"
    if soul_section:
        prompt += f"\nPersonality guidance:\n{soul_section[:200]}"
    if user_prefs:
        prompt += f"\nUser preferences:\n{user_prefs[:150]}"

    try:
        result = await client.generate(
            [
                {"role": "system", "content": _build_plan_system(outer_tools)},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=400,
            temperature=0.1,
            response_format={"type": "json_object"},
            stream=False,
            thinking=False,
        )
        raw  = result["choices"][0]["message"]["content"].strip()
        data = parse_format_response(raw)
        if data:
            steps_raw = str(data.get("steps") or "").strip()
            if steps_raw.startswith("["):
                steps_raw = ""
            steps = []
            for part in steps_raw.split("|"):
                part = part.strip()
                if ":" in part:
                    tool, _, inp = part.partition(":")
                    tool = tool.strip().lower()
                    inp  = inp.strip()
                    if tool == "tool" and inp == "input":
                        continue
                    if tool:
                        steps.append({"tool": tool, "input": inp})
                elif part and part.lower() not in ("tool", "input"):
                    steps.append({"tool": part.lower(), "input": task})
            if steps:
                logger.debug("plan_task: LLM fallback → %d step(s)", len(steps))
                return steps
    except Exception as exc:
        logger.warning("plan_task fallback LLM failed: %s", exc)

    # No steps produced — synthesizer will answer directly from query + context
    return []
