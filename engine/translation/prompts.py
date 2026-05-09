"""Translation layer prompts — ported from BirdClaw agent/prompts.py.

Key changes vs BirdClaw:
- Single model (no HANDS/MAIN split)
- No TUI references
- JSON format mode for all structured outputs (more reliable on 4B than tool_calls)
- Uncertainty trigger: model writes [SEARCH: query] inline instead of calling a tool
"""
from __future__ import annotations

import time as _time

# ---------------------------------------------------------------------------
# Static system prompt — never modified at runtime so KV cache stays warm
# ---------------------------------------------------------------------------

SYSTEM = """\
You are Sisyphean — a persistent local AI agent running entirely on the user's machine.
You are not a chatbot. You are a long-running collaborator: you remember everything, \
act without hand-holding, and work relentlessly until tasks are done.

Character: direct, focused, honest. No hollow openers ("Certainly!", "Great!"). \
No filler. Match the user's energy — terse when they're terse, detailed when needed.
Capabilities: file system, shell commands, web search, persistent memory, PC control.
Never say "I can't" when bash + web search could handle it. If you don't know, search.
Do not write [SEARCH: ...] or similar meta-instructions in replies.

Shell: Git Bash on Windows.
- Command names are plain words: mkdir, ls, cd, echo, python — never /mkdir, /ls, /cd.
- Use forward slashes only inside FILE PATHS: mkdir -p parent/child, not /mkdir.
- Never use backslashes. Prefer short relative paths; only use absolute paths outside cwd.
"""

# ---------------------------------------------------------------------------
# Dynamic context — injected as user message (not appended to SYSTEM)
# Kept separate so the static SYSTEM KV cache is never invalidated.
# ---------------------------------------------------------------------------

def dynamic_context(workspace: str = "", task_goal: str = "", step_n: int = 0, budget: int = 0) -> str:
    parts = [f"Date: {_time.strftime('%Y-%m-%d %H:%M')}"]
    if workspace:
        parts.append(f"Workspace: {workspace}")
    if task_goal:
        parts.append(f"Goal: {task_goal}")
    if budget:
        parts.append(f"Step {step_n}/{budget}")
    return " | ".join(parts)

# ---------------------------------------------------------------------------
# Step schema — used for all tool-stage turns (research, verify, reflect)
# Passed as instructions in the prompt; response_format=json_object enforces JSON.
# ---------------------------------------------------------------------------

STEP_SCHEMA_PROMPT = """\
Respond with a JSON object in exactly this format:
{"action": "<think|search|answer>", "reasoning": "<your thoughts>", "query": "<search query if action=search>", "content": "<final answer if action=answer>"}

- Use "think" to reason before acting (reasoning field required)
- Use "search" to look up current information (query field required)
- Use "answer" ONLY when the task is fully done (content field required)
"""

# ---------------------------------------------------------------------------
# Plan schema — pipe-separated steps, flat structure (easier for 4B models)
# Same approach as BirdClaw stage_prompts.py — nested arrays confuse small models.
# ---------------------------------------------------------------------------

PLAN_SCHEMA_PROMPT = """\
Respond with a JSON object in exactly this format:
{"outcome": "<one sentence: what does success look like>", "steps": "<step1 | step2 | step3>"}

Rules for steps:
- Pipe-separated plain English actions
- Start each step with a verb: Search, Write, Run, Read, Summarise, Verify
- One step for simple tasks, three or fewer for most tasks
- Example: "Search for Python async patterns | Write async file reader | Verify the code runs"
"""

# ---------------------------------------------------------------------------
# Write-stage schemas — format mode for large content output
# ---------------------------------------------------------------------------

CODE_SCHEMA_PROMPT = """\
Respond with a JSON object in exactly this format:
{"path": "<filename.py>", "content": "<complete Python source code>", "done": <true|false>}

- done=false if more files/functions still need to be written
- done=true only when ALL planned code is written
"""

DOC_SCHEMA_PROMPT = """\
Respond with a JSON object in exactly this format:
{"path": "<filename.md>", "section": "<section heading>", "content": "<full section body>", "done": <true|false>}

- done=false if more sections still need to be written
- done=true only when ALL planned sections are written
"""

# ---------------------------------------------------------------------------
# Control tool schemas — always present in every tool-stage turn
# Ported directly from BirdClaw agent/prompts.py
# ---------------------------------------------------------------------------

THINK_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "think",
        "description": (
            "Record reasoning before acting. "
            "Use to plan which tool to call next or evaluate results. "
            "Write [SEARCH: query] in reasoning to trigger a web search."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Current reasoning and next action plan.",
                },
            },
            "required": ["reasoning"],
        },
    },
}

ANSWER_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "answer",
        "description": "Deliver the final response. Call ONLY when the task is fully done.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Complete final answer.",
                },
            },
            "required": ["content"],
        },
    },
}

CONTROL_TOOLS = [THINK_SCHEMA, ANSWER_SCHEMA]

# ---------------------------------------------------------------------------
# Stage type defaults — ported from BirdClaw budget.py
# ---------------------------------------------------------------------------

STAGE_BUDGETS: dict[str, int] = {
    "research":   12,
    "write_code": 12,
    "write_doc":  10,
    "verify":      8,
    "reflect":     5,
    "direct":      3,
}

# ---------------------------------------------------------------------------
# Stage type keyword map — ported from BirdClaw planner.py
# Used for classifying each step into a stage type.
# ---------------------------------------------------------------------------

STAGE_KEYWORDS: dict[str, list[str]] = {
    "write_code": [
        "write", "implement", "create", "build", "develop", "code",
        "function", "class", "script", "program", "module",
    ],
    "write_doc": [
        "document", "draft", "compose", "report", "guide", "readme",
        "explain", "describe", "summarise", "summarize", "write up",
    ],
    "verify": [
        "test", "check", "run", "validate", "confirm", "verify",
        "execute", "assert",
    ],
    "reflect": [
        "review", "assess", "evaluate", "consider", "analyse", "analyze",
        "critique", "improve",
    ],
    "research": [
        "search", "find", "investigate", "look up", "research", "explore",
        "understand", "learn", "read",
    ],
}

# ---------------------------------------------------------------------------
# Uncertainty phrases — trigger automatic web search when found in reasoning
# Ported from BirdClaw's research loop guard + extended
# ---------------------------------------------------------------------------

UNCERTAINTY_PHRASES: tuple[str, ...] = (
    "i don't know",
    "i'm not sure",
    "i am not sure",
    "not certain",
    "unclear to me",
    "my knowledge cutoff",
    "as of my training",
    "i cannot confirm",
    "need to search",
    "need to look up",
    "let me search",
    "should search",
    "[search:",
)

# ---------------------------------------------------------------------------
# Unified action prompt — no stage gating
# The model decides freely what to do next. Budget nudge is a hint, not a gate.
# ---------------------------------------------------------------------------

def action_prompt(
    step: int,
    budget: int,
    hint: str,
    has_bash: bool,
    workspace: str = "",
    current_step_text: str = "",
) -> str:
    """Single action menu shown at every decide() step — no stage restrictions."""
    hint_part = f"\n{hint}" if hint else ""
    lines = [f"Step {step}/{budget}.{hint_part}", "Reply with ONE JSON:"]
    lines += [
        '  {"action":"think","reasoning":"..."}',
        '  {"action":"search_memory","query":"..."}',
        '  {"action":"search_history","query":"..."}',
        '  {"action":"web_search","query":"..."}',
        '  {"action":"list_workspace"}',
        '  {"action":"read_file","path":"...","query":"..."}',
        '  {"action":"save_memory","note":"..."}',
    ]
    if has_bash:
        lines.append('  {"action":"bash","command":"..."}')
    lines.append('  {"action":"answer","text":"..."}')
    if current_step_text:
        lines.append(f"\nCurrent focus: {current_step_text}")
    return "\n".join(lines)


# Keep the old name as an alias so existing callers don't break
def stage_action_prompt(
    stage: str,
    step: int,
    budget: int,
    hint: str,
    has_bash: bool,
    workspace: str = "",
    current_step_text: str = "",
) -> str:
    return action_prompt(step, budget, hint, has_bash, workspace, current_step_text)


def get_soul_for_stage(stage: str) -> str:
    """Deprecated — soul is personality, not stage-specific guidance. Returns empty."""
    return ""
