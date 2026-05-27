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

_STEP_NUMBER_RE = re.compile(
    r'^(?:step\s*\d+\s*[:.)]\s*|\d+\s*[:.)]\s*)',
    re.IGNORECASE,
)


def infer_stage_type(step_text: str) -> str:
    """Classify a plain-English step into a stage type.

    Priority: save_memory > research > verify > write_doc > write_code

    Numbered prefixes ("STEP2:", "Step 1.", "2)") are stripped before
    classification so bare label artefacts never route to web_search.
    Unknown / empty steps default to "direct" instead of "research" —
    an unclassifiable step should not trigger a random web search.
    """
    # Strip "STEP2", "Step 1:", "2." etc. — model sometimes emits these
    clean = _STEP_NUMBER_RE.sub("", step_text).strip()
    s = clean.lower()

    if not s:
        return "direct"
    # Skill steps are always "direct" — run_skill converts to bash in _execute;
    # read_skill is an internal graph lookup. Neither should trigger web_search.
    if s.startswith("run skill:") or s.startswith("run_skill:") or s.startswith("run skill "):
        return "direct"
    if s.startswith("read skill:") or s.startswith("read_skill:") or s.startswith("read skill "):
        return "direct"
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
    return "direct"   # unknown → answer directly, never random websearch


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
    ("direct",        "ONLY when the answer is already in the provided context (memory, history, recall). Never use if the answer must be looked up or computed."),
    ("web_search",    "search the web for any factual or current information"),
    ("search_memory", "look up previously saved facts or research from past sessions"),
    ("write_plan",    "write a file section-by-section with verify-and-retry — "
                      "use ONLY for files the user explicitly asked to keep: programs, modules, documents, reports. "
                      "Input format: write_plan:<filename>|<task description>  "
                      "Example: write_plan:fibonacci.py|Write a Fibonacci generator with three functions"),
    # ── Skill tools (progressive disclosure) ────────────────────────────────
    # read_skill / run_skill are shown only when the skill index is non-empty.
    # save_skill / save_skill_program are always available so the model can
    # capture new approaches even before any skills exist.
    ("save_skill",         "save a reusable text approach after solving a non-trivial problem — "
                           "save_skill:SKILL-NAME|step-by-step runbook that worked. "
                           "Skip for greetings, trivial math, single bash commands."),
    ("save_skill_program", "save the actual program code as a reusable skill — "
                           "save_skill_program:SKILL-NAME|complete-program-code. "
                           "Use when you just wrote a script the user will likely want again."),
]
_INTERNAL_TOOL_NAMES = frozenset(t[0] for t in _INTERNAL_TOOLS)

# All tool names the planner is allowed to use.  Steps whose tool name is NOT
# in this set are silently normalised to web_search — this catches small-model
# mistakes like outputting "arxiv:query" instead of "run_skill:arxiv" when the
# model reads a platform name in the instruction text as a tool name.
_KNOWN_PLANNER_TOOLS = _INTERNAL_TOOL_NAMES | {
    "run_skill", "read_skill",     # skill execution (progressive disclosure)
    "bash", "write", "edit", "read", "glob", "multiedit",  # outer harness tools
    "web_search", "web_fetch",     # web tools
    "direct", "write_plan",        # pipeline-internal special steps
}


def _normalise_step(tool: str, inp: str, outer_tool_names: frozenset[str]) -> tuple[str, str]:
    """Normalise a (tool, input) pair from the planner LLM.

    If the tool name is not in the allowed set, the model probably read a
    platform/library name from the instruction text and used it as a tool
    (e.g. "arxiv:beingalive" instead of "run_skill:arxiv").
    Fall back to web_search with "<tool> <inp>" as the query.
    """
    all_known = _KNOWN_PLANNER_TOOLS | outer_tool_names
    if tool in all_known:
        return tool, inp
    # Unknown tool → web_search fallback
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "plan_task: unknown tool %r → web_search (inp=%r)", tool, inp[:60]
    )
    return "web_search", f"{tool} {inp}".strip()


def _build_plan_system(outer_tools: list[dict], skill_index: str = "") -> str:
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
    # Skills: one line per skill, same format as other tools — model picks by name.
    # Never injected as a text block; always presented as selectable tool entries.
    if skill_index:
        for _sk_line in skill_index.strip().splitlines():
            _sk = _sk_line.strip().lstrip("•").strip()
            if not _sk:
                continue
            _ci = _sk.find(":")
            if _ci > 0:
                _sname = _sk[:_ci].strip()
                _sdesc = _sk[_ci + 1:].strip().replace("[runnable]", "").strip()[:60]
            else:
                _sname = _sk.strip()
                _sdesc = "skill"
            if _sname:
                lines.append(f"  run_skill:{_sname} — {_sdesc}")
    lines += [
        "",
        "Use 'direct' ONLY when the answer is already in the provided context (memory, history, recall).",
        "Never use 'direct' if the answer must be looked up or computed — use web_search or bash.",
        "Never answer from training knowledge. Every fact must come from web_search or bash.",
        "Each web_search query must be a short keyword phrase — strip question words.",
        "If the task asks for two dependent things (identify X, then details of X), use two steps.",
        "Research, analysis, or explanation tasks MUST use at least 2 steps (e.g. web_search then direct).",
        "Complex multi-part tasks MUST use 3+ steps — use pipe-separated steps for each action.",
        'Single step: {"steps": "toolname:input"}',
        'Multiple steps: {"steps": "toolname:input | toolname:input | toolname:input"}',
        "",
        "TEMPORARY vs PERMANENT files:",
        "  bash — for one-off utility scripts the agent needs to compute something (calc.py,",
        "         quick test script). These are ephemeral: write, run, discard.",
        "  write_plan — for files the USER explicitly asked to create and keep: a Python",
        "         module, a PDF extractor program, a report, an essay. These are permanent.",
        "         Format: write_plan:<filename>|<task description>",
        "         Example: write_plan:fibonacci.py|Write a Fibonacci number generator with tests",
        "         Example: write_plan:essay.md|Write a 4-section essay about consciousness",
        "",
        "FILE PATHS: When using bash to create or run files, always use the full absolute",
        "  workspace path given in the task message — never bare relative filenames like calc.py.",
        "  The workspace path is shown explicitly in the task prompt.",
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


async def resolve_bash_command(goal: str, client, context: str = "") -> str:
    """Ask the model what shell command to run for a verify-stage goal.

    Returns the command string, or empty string on failure.
    The tool selection has already been made (bash) — this only resolves WHAT to run.
    """
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
            cmd = data["command"].strip()
            # Strip "COMMAND: " / "Run " prefixes that small models echo from
            # the format description (e.g. model returns "COMMAND: Run python...")
            cmd = re.sub(r'^(?:COMMAND:\s*|Run\s+COMMAND:\s*)', '', cmd, flags=re.IGNORECASE)
            if cmd.lower().startswith("run "):
                cmd = cmd[4:].strip()
            return cmd
    except Exception as exc:
        logger.warning("resolve_bash_command failed: %s", exc)
    return ""


# ---------------------------------------------------------------------------
# Query router — dedicated first-step classifier
# ---------------------------------------------------------------------------

_ROUTE_SYSTEM = """\
Classify this task into one category. Output the category name only — one word, no punctuation.

  direct  — the answer is already present in the provided context (memory, history, recall)
  bash    — can be computed or executed locally on this machine (no external data needed)
  search  — requires external data that does not exist on this machine
  memory  — user wants to save or recall something
  code    — create or modify a file

If the answer is NOT already in context, never use direct — use bash or search."""

_ROUTE_LABELS = frozenset({"direct", "bash", "search", "memory", "code"})

_ROUTE_HINTS: dict[str, str] = {
    "bash":   "Hint: this needs computation — plan a Run step (use skill:calc if available).",
    "search": "Hint: this needs a web lookup — plan a Search step.",
    "memory": "Hint: user wants to save something — plan a Save step.",
    "code":   "Hint: user wants a file created — plan a Write step.",
    "direct": "Hint: purely conversational — steps can be empty.",
}


async def route_query(query: str, client) -> str:
    """Lean router — single focused LLM call, returns one of _ROUTE_LABELS.

    Returns "" on failure so think_decompose falls back to its own judgment.
    This is an additive step: it biases think_decompose without replacing it.
    """
    try:
        result = await client.generate(
            [
                {"role": "system", "content": _ROUTE_SYSTEM},
                {"role": "user",   "content": query[:200]},
            ],
            max_tokens=256,  # thinking=True: reasoning in reasoning field, word in content
            temperature=0.0,
            stream=False,
            thinking=True,
        )
        msg = result["choices"][0]["message"]
        # content has the actual word after thinking
        # fall back to last word of reasoning if content is empty (Ollama token exhaustion)
        raw = (msg.get("content") or "").strip().lower()
        if not raw:
            reasoning = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip().lower()
            # take the last non-empty word — models often end reasoning with the answer
            words = [w for w in re.split(r"\W+", reasoning) if w]
            if words:
                raw = words[-1]
        # Strip think blocks
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        # Extract first word
        word = re.split(r"\W+", raw)[0] if raw else ""
        if word in _ROUTE_LABELS:
            logger.info("route_query: %r → %s", query[:50], word)
            return word
        # Fuzzy fallback for longer outputs
        for label in _ROUTE_LABELS:
            if label in raw:
                logger.info("route_query: %r → %s (fuzzy)", query[:50], label)
                return label
    except Exception as exc:
        logger.debug("route_query failed: %s", exc)
    logger.debug("route_query: unknown route for %r — think_decompose decides", query[:50])
    return ""


_THINK_DECOMPOSE_SYSTEM = """\
You are a task router. Output ONLY valid JSON: {"outcome": "...", "steps": "..."}

STEP FORMATS:
  steps=""                   answer directly — no tool needed
  steps="Search KEYWORDS"    web search — write real search terms
  steps="Run COMMAND"        run a shell command
  steps="Write FILENAME"     create a code or document file
  steps="Save: FACT"         persist a fact to memory
  steps="STEP1 | STEP2"      chain multiple steps with a pipe

ROUTING RULES:

steps="" ONLY when the answer is already present in the provided context
  (memory recall, graph, or conversation history). If the answer is not
  in context, always use a tool — never answer from training knowledge.

steps="Search KEYWORDS" for any factual question not already in context.
  Do NOT use Search for computation — use Run for those.

steps="Run COMMAND" for shell actions and ALL computation.
  For computation: use the calc skill — plan "Run skill:calc EXPRESSION".
  NEVER use steps="Write..." for computation or expressions — Write is for permanent user files only.

steps="Write FILENAME" ONLY for PERMANENT files the user explicitly asked to create
  and keep: Python programs, modules, extractors, full applications, documents,
  essays, reports. These go through the incremental write+verify pipeline.
  Use a descriptive filename that matches the deliverable.
  WORKSPACE RULE: The workspace path is given in the prompt. ALL files MUST live
  inside that workspace directory — never use a bare filename.
  PARAMETERIZE: any program written for reuse MUST accept inputs via sys.argv (or
  argparse), never hardcoded values. Task "sum 5 and 3" → write sum.py that reads
  sys.argv[1] and sys.argv[2], then Run python WORKSPACE/sum.py 5 3.
  Programs with baked-in constants are one-offs; programs that read sys.argv are skills.

steps="Save: FACT" only when the user says: remember / save / note / keep in mind.

SKILL-FIRST — check before planning new stages:
If "Relevant skills" are listed below the task, scan them first.
  • skill tagged [runnable] → plan a single step: "Run skill:SKILL-NAME"
    (this re-executes the saved program; no web search, no write pipeline needed)
  • text-only skill (no [runnable] tag) → plan "Read skill:SKILL-NAME" as first step,
    then only add Search / Write steps for gaps the runbook does not cover
Only fall through to full search/write planning when NO skill matches this task type.
"""


async def think_decompose(
    query: str,
    client,
    context: str = "",
    soul_section: str = "",
    workspace: str = "",
    skill_index: str = "",
    route: str = "",
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

    skill_index: compact bullet list from get_skill_index() — tells the model
    which skills it already has for this task so it can plan a read_skill step
    instead of rediscovering from scratch.
    """
    prompt = f"Task: {query[:300]}"
    if route and route in _ROUTE_HINTS:
        # Router hint appended after the task so the model reads the task first,
        # then receives the classification nudge — prevents prefix confusion.
        prompt += f"\n{_ROUTE_HINTS[route]}"
    if workspace:
        prompt += f"\nWorkspace (write ALL task files here): {workspace}"
    if context:
        prompt += f"\nContext:\n{context[:300]}"
    if soul_section:
        prompt += f"\nPersonality guidance:\n{soul_section[:150]}"
    # Skills are listed as tool entries in plan_task's system prompt — not injected here.

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
    workspace: str = "",
    skill_index: str = "",
) -> list[dict]:
    """Plan a single task → list of {tool, input} steps.

    One LLM call with the available tools listed. The model picks the right
    tool and provides the input — no regex pre-filters, no conditional routing.

    skill_index: compact bullet list from get_skill_index() — injected into
    the prompt so the model can choose read_skill:NAME instead of web_search
    when it already has a runbook for this type of task.
    """
    logger.debug("plan_task: LLM planning call for %r", task[:60])
    prompt = f"Task: {task[:200]}"
    if workspace:
        prompt += f"\nWorkspace (write ALL task files here): {workspace}"
    if context:
        prompt += f"\nContext:\n{context[:400]}"
    if soul_section:
        prompt += f"\nPersonality guidance:\n{soul_section[:200]}"
    if user_prefs:
        prompt += f"\nUser preferences:\n{user_prefs[:150]}"
    # Skills are presented as tool entries in the system prompt (see _build_plan_system),
    # NOT injected as a text block here — prevents the model from echoing descriptions.

    _outer_names = frozenset(t.get("name", "").lower() for t in outer_tools)

    try:
        result = await client.generate(
            [
                {"role": "system", "content": _build_plan_system(outer_tools, skill_index=skill_index)},
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
                    t, i = _normalise_step(t, i, _outer_names)
                    logger.debug("plan_task: LLM returned step dict → %s", t)
                    return [{"tool": t, "input": i}]
            elif isinstance(steps_raw_val, list):
                # List of step dicts OR list of "tool:input" strings.
                # The model sometimes returns one, sometimes the other.
                steps = []
                for s in steps_raw_val:
                    if isinstance(s, dict) and s.get("tool") and s.get("input"):
                        t = s.get("tool", "").strip().lower()
                        i = s.get("input", "").strip()
                        t, i = _normalise_step(t, i, _outer_names)
                        steps.append({"tool": t, "input": i})
                    elif isinstance(s, str) and ":" in s:
                        # "write_plan:file.py|goal" or "bash:command"
                        t, _, i = s.partition(":")
                        t = t.strip().lower()
                        i = i.strip()
                        if t and i and t not in ("tool", "input"):
                            t, i = _normalise_step(t, i, _outer_names)
                            steps.append({"tool": t, "input": i})
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
                        tool, inp = _normalise_step(tool, inp, _outer_names)
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


# ── One-shot code generation ──────────────────────────────────────────────────

_GENERATE_CODE_SYSTEM = """\
You are a code generator. Output ONLY the raw file content — no markdown fences, \
no explanation, no commentary. If generating Python, output valid Python. \
If generating another language, output valid code for that language.
"""


async def _generate_code(task: str, client) -> str:
    """Generate file content for a write-code task in a single LLM call.

    Used as a fallback when the subtask planner produces no items, and when
    regenerating file content with epistemic context injected. Returns the
    generated text, or empty string on failure.
    """
    try:
        result = await client.generate(
            [
                {"role": "system", "content": _GENERATE_CODE_SYSTEM},
                {"role": "user",   "content": task[:2000]},
            ],
            max_tokens=1200,
            temperature=0.2,
            stream=False,
            thinking=False,
        )
        raw = (result["choices"][0]["message"]["content"] or "").strip()
        # Strip accidental markdown fences the model sometimes emits
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()
        return raw
    except Exception as exc:
        logger.warning("_generate_code: LLM call failed: %s", exc)
        return ""
