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

    Priority: save_memory > research > verify > write_doc > write_code
    Research wins when a step mentions both search and write — the intent
    is to gather information, not produce a file.
    """
    s = step_text.lower()
    if s.startswith("save:") or s.startswith("save_memory:") or s.startswith("save_memory "):
        return "save_memory"
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
Use plain verbs: Run, Search, Write, Read, Summarise, Verify.
Scale steps to complexity:
- 1 step: simple direct queries (time, math, single bash command, greeting)
- 2-3 steps: research tasks (Search topic | Summarise findings), single-file edits
- 3-5 steps: complex tasks (reports, multi-part research, code pipelines, audits)
RULES:
- Research, analysis, and explanation tasks MUST have at least 2 steps: gather then synthesise.
- Complex tasks (reports, code pipelines, audits, multi-part research) MUST have at least 3 steps.
- NEVER plan a step that asks the user for input. If info is needed, answer immediately instead.
- Use 'Write' steps ONLY when the user explicitly asks for a file/document/report to be saved.
- For web research: use 'Search' or 'Fetch' steps.
- Use 'Run' steps (bash) for queries about the CURRENT state of this machine: running \
processes, GPU/CPU/memory usage, hardware info, disk space, network status, uptime. \
These require a shell command — never use 'Search' for live local machine data.
- In 'Run' step text write the exact command to execute. Never prefix a command with \
'python' unless it is a .py script (e.g. 'python myscript.py'). Standalone tools like \
nvidia-smi, systeminfo, ipconfig run directly without any prefix.
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
                max_tokens=512,
                temperature=0.2,
                # No response_format: llama.cpp + thinking model → content empty
                # with json_object; free-form output is parsed by parse_format_response
                stream=False,
                thinking=True,
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
                temperature=0.1,
                response_format={"type": "json_object"},  # "{" prefix on llama.cpp
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
# save_memory is intentionally excluded: it is a routing decision made by
# think_decompose() when the user explicitly asks to save something, NOT a
# step the planner should suggest — small models otherwise emit save_memory
# for tasks that should run bash.
_INTERNAL_TOOLS = [
    ("direct",        "answer directly without any tool call — use for greetings, thanks, simple chat, capability questions"),
    ("web_search",    "search the web for any factual or current information"),
    ("search_memory", "look up previously saved facts or research from past sessions"),
]
_INTERNAL_TOOL_NAMES = frozenset(t[0] for t in _INTERNAL_TOOLS)


def _build_plan_system(outer_tools: list[dict]) -> str:
    """Build the plan system prompt from harness-provided tools + internal tools."""
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
        "Use 'direct' for: hi, hello, thanks, what can you do, are you alive, simple social exchanges.",
        "Use web_search for any factual question worth knowing — current or timeless.",
        "Each web_search query must be a short keyword phrase — strip question words.",
        "If the task asks for two dependent things (identify X, then details of X), use two steps.",
        "Research, analysis, or explanation tasks MUST use at least 2 steps (e.g. web_search then direct).",
        "Complex multi-part tasks MUST use 3+ steps — use pipe-separated steps for each action.",
        'Single step: {"steps": "toolname:input"}',
        'Multiple steps: {"steps": "toolname:input | toolname:input | toolname:input"}',
    ]
    return "\n".join(lines)


_SPLIT_JUNK = frozenset(["task1", "task2", "task_a", "task_b",
                          "first question", "second question",
                          "step 1", "step 2", "step1", "step2"])




# ---------------------------------------------------------------------------
# split + plan_task
# ---------------------------------------------------------------------------

async def split_deep(query: str, client, max_depth: int = 1) -> list[str]:
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


_SPLIT_CONJUNCTIONS = re.compile(
    r'\b(and\s+(?:also\s+)?|then\s+(?:also\s+)?|also\s+|additionally\s+|furthermore\s+|plus\s+)'
    r'(?:search|find|look\s+up|write|create|build|run|check|download|fetch|get|list)',
    re.IGNORECASE,
)


async def split(query: str, client) -> list[str]:
    """Split a query into independent sub-tasks.

    Short-circuits to [query] for short/math inputs or when no conjunction
    keywords indicate multiple independent actions.  Falls back to [query]
    on any failure.
    """
    words = query.strip().split()
    if len(words) <= 4:
        return [query]

    # Regex pre-check: if no conjunction pattern found, skip the LLM call
    if not _SPLIT_CONJUNCTIONS.search(query):
        return [query]

    try:
        result = await client.generate(
            [
                {"role": "system", "content": _SPLIT_SYSTEM},
                {"role": "user",   "content": query[:300]},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},  # "{" prefix on llama.cpp
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


_RESOLVE_BASH_SYSTEM = """\
Output ONLY a JSON object: {"command": "<shell command to run>"}
The tool is already decided: bash. Your only job is to write the command.
Do not explain. Do not pick a different tool. Output the command string only.
"""


_CONCRETE_CMD_RE = re.compile(
    r"^(python\d*|pip\d*|node|npm|npx|go|cargo|make|cmake|gcc|g\+\+|"
    r"powershell|pwsh|cmd|bash|sh|zsh|Get-|Set-|New-|Remove-|"
    r"git|docker|kubectl|curl|wget|ls|dir|cd|mkdir|rm|cp|mv|cat|echo|"
    r"systeminfo|ipconfig|ifconfig|ps|top|htop|nvidia-smi|wmic|tasklist|"
    r"pytest|unittest|coverage|mypy|ruff|black|flake8)\b",
    re.IGNORECASE,
)


async def resolve_bash_command(goal: str, client, context: str = "") -> str:
    """Ask the model what bash command to run for a verify-stage goal.

    Returns the command string, or empty string on failure.
    The tool selection has already been made (bash) — this only resolves WHAT to run.
    Short-circuits without an LLM call when the goal is already a concrete command.
    """
    # Goal is already a runnable command — no LLM call needed.
    goal_stripped = goal.strip()
    if _CONCRETE_CMD_RE.match(goal_stripped):
        logger.debug("resolve_bash_command: goal IS the command — skipping LLM: %s", goal_stripped[:80])
        return goal_stripped

    prompt = f"Goal: {goal}"
    if context:
        prompt += f"\nContext: {context[:300]}"
    try:
        result = await client.generate(
            [
                {"role": "system", "content": _RESOLVE_BASH_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
            stream=False,
            thinking=False,
        )
        raw = result["choices"][0]["message"]["content"].strip()
        data = parse_format_response(raw)
        if data and isinstance(data.get("command"), str):
            return data["command"].strip()
    except Exception as exc:
        logger.warning("resolve_bash_command failed: %s", exc)
    return ""


_THINK_DECOMPOSE_SYSTEM = """\
You are a task router. Output ONLY valid JSON: {"outcome": "...", "steps": "..."}

steps="" means answer directly from knowledge — no tool needed.
steps="Run CMD" means execute that exact shell command.
steps="Search QUERY" means web search.
steps="Write FILE" means create a file.
Pipe-separate multi-step tasks: "step1 | step2"

EXAMPLES — study these carefully:

User: hi
{"outcome": "greet user", "steps": ""}

User: thanks
{"outcome": "acknowledge", "steps": ""}

User: thank you so much
{"outcome": "acknowledge", "steps": ""}

User: sounds good
{"outcome": "acknowledge", "steps": ""}

User: what can you do?
{"outcome": "describe capabilities", "steps": ""}

User: what is the capital of France?
{"outcome": "answer geography question", "steps": ""}

User: 2+2
{"outcome": "compute arithmetic", "steps": "Run python -c 'print(2+2)'"}

User: what is 15 * 7?
{"outcome": "compute arithmetic", "steps": "Run python -c 'print(15*7)'"}

User: what's the weather today?
{"outcome": "find current weather", "steps": "Search weather today"}

User: latest Python release?
{"outcome": "find latest Python version", "steps": "Search latest Python release 2024"}

User: remember I prefer dark mode
{"outcome": "save user preference", "steps": "Save: user prefers dark mode"}

User: what processes are running?
{"outcome": "list running processes", "steps": "Run Get-Process | Select-Object -First 20"}

User: check my GPU
{"outcome": "show GPU info", "steps": "Run nvidia-smi"}

RULES:
- Greetings, thanks, acknowledgements, social replies → steps="" ALWAYS. Never Search for these.
- Arithmetic (2+2, 15*7, etc.) → Run python -c 'print(expr)' ALWAYS. Never answer from memory.
- Save: ONLY when user explicitly says remember/save/note/keep-in-mind.
- Search: ONLY for live/current data. Not for math. Not for greetings.
"""


async def think_decompose(
    query: str,
    client,
    context: str = "",
    soul_section: str = "",
) -> tuple[str, list[dict]]:
    """LLM planning call that decomposes the task into stages.

    This is the FIRST call in the pipeline — it happens before any execution.
    Returns (outcome, stages) where stages is:
      [{"type": str, "goal": str}, ...]

    Stage types map to how they are executed:
      direct     → synthesizer answers directly (no tool call)
      research   → web_search
      write_code → write_plan (subtask pipeline)
      write_doc  → write_plan (doc variant)
      verify     → bash
      edit       → plan_task (complex, keeps LLM routing)

    The LLM decides routing — greetings, social replies, capability questions
    all return steps="" (direct) based on the prompt instructions.
    No regex pre-filters: the model owns the routing decision.
    """
    prompt = f"Task: {query[:300]}"
    if context:
        prompt += f"\nContext:\n{context[:300]}"
    if soul_section:
        prompt += f"\nPersonality guidance:\n{soul_section[:150]}"

    try:
        result = await client.generate(
            [
                {"role": "system", "content": _THINK_DECOMPOSE_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},  # "{" prefix on llama.cpp; json_object elsewhere
            stream=False,
            thinking=False,
        )
        raw  = result["choices"][0]["message"]["content"].strip()
        data = parse_format_response(raw)
        if not data:
            return query[:80], []

        outcome  = (data.get("outcome") or query[:80]).strip()
        raw_steps = data.get("steps") or ""
        if isinstance(raw_steps, list):
            plain_steps = [str(s).strip() for s in raw_steps if str(s).strip()]
        else:
            plain_steps = [s.strip() for s in str(raw_steps).strip().split("|") if s.strip()]

        # Model explicitly returned steps="" — direct answer, no tool calls needed.
        # Return a "direct" stage so the pipeline skips planning entirely.
        if not plain_steps:
            logger.info("think_decompose: steps='' → direct answer for %r", query[:60])
            return outcome, [{"type": "direct", "goal": query}]

        stages: list[dict] = []
        for step in plain_steps:
            stype = infer_stage_type(step)
            stages.append({"type": stype, "goal": step})

        logger.info("think_decompose: outcome=%r stages=%s",
                    outcome[:60], [(s["type"], s["goal"][:40]) for s in stages])
        return outcome, stages

    except Exception as exc:
        logger.warning("think_decompose failed: %s", exc)
        return query[:80], []




async def plan_task(
    task: str,
    outer_tools: list[dict],
    client,
    context: str = "",
    soul_section: str = "",
    user_prefs: str = "",
) -> list[dict]:
    """Plan a single task → list of {tool, input} steps.

    One LLM call with the available tools listed. The model picks the right
    tool and provides the input — no regex pre-filters, no conditional routing.
    """
    logger.debug("plan_task: LLM planning call for %r", task[:60])
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
            temperature=0.1,
            response_format={"type": "json_object"},  # "{" prefix on llama.cpp; json_object elsewhere
            stream=False,
            thinking=False,
        )
        raw  = result["choices"][0]["message"]["content"].strip()
        data = parse_format_response(raw)
        if data:
            steps_raw_val = data.get("steps")

            # Model sometimes returns steps as a dict or list of dicts instead
            # of the pipe-separated string format. Handle all cases cleanly.
            if isinstance(steps_raw_val, dict):
                # Single step as dict: {"tool": "websearch", "input": "..."}
                t = steps_raw_val.get("tool", "").strip().lower()
                i = steps_raw_val.get("input", "").strip()
                if t and i:
                    logger.debug("plan_task: LLM returned step dict → %s", t)
                    return [{"tool": t, "input": i}]
            elif isinstance(steps_raw_val, list):
                # List of step dicts: [{"tool": ..., "input": ...}, ...]
                steps = [
                    {"tool": s.get("tool", "").strip().lower(),
                     "input": s.get("input", "").strip()}
                    for s in steps_raw_val
                    if isinstance(s, dict) and s.get("tool") and s.get("input")
                ]
                if steps:
                    logger.debug("plan_task: LLM returned step list → %d step(s)", len(steps))
                    return steps

            # Standard pipe-separated string format
            steps_raw = str(steps_raw_val or "").strip()
            if steps_raw.startswith("[") or steps_raw.startswith("{"):
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
                    # Drop fragments with no meaningful input — these are
                    # trailing garbage the model appended (e.g. "tab: ",
                    # "space: ") after a valid step.
                    if tool and inp:
                        steps.append({"tool": tool, "input": inp})
                    elif tool:
                        logger.debug("plan_task: dropping empty-input fragment %r", tool)
                elif part and part.lower() not in ("tool", "input"):
                    logger.debug("plan_task: skipping fragment (no colon): %r", part[:60])
            if steps:
                logger.debug("plan_task: LLM → %d step(s)", len(steps))
                return steps
    except Exception as exc:
        logger.warning("plan_task fallback LLM failed: %s", exc)

    # No steps produced — synthesizer will answer directly from query + context
    return []
