"""Stage executor — decides next action for Claude Code to execute.

New design: decide_next_action() uses two-phase tool selection:
  1. _filter_tools()  — token-overlap search over tool names + descriptions,
                        returns the 4 most relevant tools for this instruction.
  2. Fill-in-the-blank prompt — each option is shown as a concrete JSON template
     with the exact field names from the tool's input_schema.  Gemma fills in
     values, not structure.  Much more reliable for 4B than free-form JSON.

Response parsing is trivial: read response["tool"] to know which option was
chosen, strip that key, use the rest as tool_input.  No nested wrapper,
no action/tool_name/tool_input triple to invent.

Legacy execute_stage() is kept for fallback when tools are unavailable.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field

from engine.translation.planner import Stage
from engine.translation.manifest import TaskManifest
from engine.translation.prompts import (
    SYSTEM,
    STEP_SCHEMA_PROMPT,
    CODE_SCHEMA_PROMPT,
    DOC_SCHEMA_PROMPT,
    UNCERTAINTY_PHRASES,
    dynamic_context,
    action_prompt,
    stage_action_prompt,  # kept as alias for any remaining callers
)
from engine.translation.pruner import keyword_prune
from engine.translation.condenser import maybe_distill
from engine.translation.web_search import (
    search as web_search,
    format_results,
    extract_search_queries,
)

logger = logging.getLogger(__name__)

_WRITE_STAGE_TYPES = {"write_code", "write_doc"}


# ── New micro-loop executor ───────────────────────────────────────────────────

def _detect_stage(
    internal_messages: list[dict],
    summary: str,
    step: int,
    budget: int,
) -> str:
    """Determine the current stage from loop state.

    plan    — first step, no plan created yet
    answer  — budget nearly exhausted or forced synthesis
    execute — default: working through steps
    """
    if step >= budget - 2:
        return "answer"
    has_plan = any(m.get("tool") == "plan_task" for m in internal_messages)
    if not has_plan and summary:
        has_plan = "Planned" in summary or "planned '" in summary.lower()
    if step == 0 and not has_plan:
        return "plan"
    return "execute"


async def decide(
    user_message: str,
    internal_messages: list[dict],
    available_tools: list[dict],
    static_context: str,
    client,
    step: int = 0,
    budget: int = 12,
    semantic_history: str = "",  # LLM-summarized past sessions relevant to this task
    summary: str = "",           # running progress log for this specific task
    hint: str = "",
    workspace: str = "",
    current_step_text: str = "",
    plan_steps: list[dict] | None = None,   # full plan steps list
    plan_step_idx: int = 0,                 # which step we're on right now
) -> "Action":
    """Single-step decision: given full context, what is the next action?

    Context is a proper multi-turn conversation:
      [system]       SYSTEM + soul personality (full, no stage filtering)
      [user]         [Past sessions] + [Task progress] + [Current Step] + [Task]
      [assistant]    {"action": "search_memory", ...}   ← this turn's prior choice
      [user]         [Memory] ...                        ← its result
      ... one pair per internal tool called this turn
      [user]         unified action prompt (all tools always visible)

    semantic_history: cross-session context (what was done in earlier sessions).
    summary: in-task progress (what this task has done so far).
    No stage gating — the model decides freely what to do next.
    """
    # ── System: personality always present, no stage-specific filtering ───────
    system_parts = [SYSTEM]
    if static_context:
        system_parts.append(static_context)
    system = "\n\n".join(p.strip() for p in system_parts if p.strip())

    # ── Hint ─────────────────────────────────────────────────────────────────
    # External hint (e.g. stall guard) takes priority; fall through to auto-hints.
    if not hint:
        if step >= budget - 2:
            hint = "Budget nearly exhausted — use answer now."
        elif internal_messages:
            last = internal_messages[-1]
            if last.get("tool") == "bash_result" and _looks_like_error(last.get("result", "")):
                last_result = last.get("result", "")
                # web_search is an internal tool, not a shell command — give a precise redirect
                if "web_search" in last_result and "command not found" in last_result.lower():
                    hint = 'web_search is not a shell command. Use {"action": "web_search", "query": "..."} or answer with what you already know.'
                else:
                    hint = "The last command failed — try a corrected command or a different approach."

    has_bash = any(t.get("name", "").lower() == "bash" for t in available_tools)
    _action_prompt = action_prompt(step, budget, hint, has_bash, workspace, current_step_text)

    # ── Build multi-turn messages ─────────────────────────────────────────────
    messages: list[dict] = [{"role": "system", "content": system}]

    # User turn: structural blocks — past sessions, task progress, current step, task.
    # Each block is clearly labelled so the model treats them as context, not requests.
    # semantic_history = what happened in prior sessions relevant to this task.
    # summary = what this specific task has accomplished so far (built incrementally).
    # Prompt order: original task → current step → already done → past sessions.
    # The model must always see WHAT the user asked first, then WHAT to do right now,
    # then WHAT has already been completed.  Putting task last caused the model to lose
    # track of the original objective when the plan + progress text was long.
    user_turn_parts: list[str] = []

    # 1. Original task — always first so the model never loses sight of the goal
    user_turn_parts.append(f"[Original task]\n{user_message}")

    # 2. Full plan with current position marked — model sees all steps and knows
    #    where it is. Each step decides whether the previous output is useful.
    if plan_steps:
        plan_lines = ["This is a suggested plan — skip any step you judge unnecessary."]
        for i, s in enumerate(plan_steps):
            if i < plan_step_idx:
                marker = "  [done]"
            elif i == plan_step_idx:
                marker = "  ← do this now"
            else:
                marker = ""
            plan_lines.append(f"  Step {i+1}: [{s['type']}] {s['text']}{marker}")
        user_turn_parts.append("[Plan]\n" + "\n".join(plan_lines))
    elif current_step_text:
        user_turn_parts.append(f"[Current step]\n{current_step_text}")

    # 3. What has already been accomplished this task
    if summary:
        user_turn_parts.append(f"[Already done]\n{summary}")

    # 4. Bash outputs from previous outer turns — don't repeat these
    bash_done = [m for m in internal_messages if m.get("tool") == "bash_result"]
    if bash_done:
        bash_lines = []
        for br in bash_done:
            res = str(br.get("result", "")).strip()[:200].replace("\n", " ")
            bash_lines.append(res)
        user_turn_parts.append(
            "[Commands already run — DO NOT repeat]\n" + "\n".join(bash_lines)
        )

    # 5. Cross-session context (past sessions)
    if semantic_history:
        user_turn_parts.append(f"[Past sessions]\n{semantic_history}")
    messages.append({"role": "user", "content": "\n\n".join(user_turn_parts)})

    # This turn's internal tool calls — each as [assistant action, user result].
    for imsg in internal_messages:
        tool = imsg["tool"]
        if tool == "bash_result":
            continue   # already shown in [Commands already run] above
        inp = imsg.get("input", {})
        result = str(imsg.get("result", "")).strip()

        messages.append({"role": "assistant", "content": _tool_to_action_json(tool, inp)})
        label = {
            "plan_task":        "[Plan]",
            "search_knowledge": "[Memory]",
            "search_history":   "[History]",
            "save_memory":      "[Saved]",
            "web_search":       "[Web]",
            "think":            "[Thinking]",
        }.get(tool, "[Result]")
        messages.append({"role": "user", "content": f"{label}\n{result}"})

    # Append next-action prompt to the last user message (keeps valid alternation)
    if messages and messages[-1]["role"] == "user":
        messages[-1]["content"] += f"\n\n{_action_prompt}"
    else:
        messages.append({"role": "user", "content": _action_prompt})
    try:
        result = await client.generate(
            messages,
            max_tokens=256,
            temperature=0.2,
            response_format={"type": "json_object"},
            stream=False,
            thinking=False,   # thinking=True adds 100-200s per call on CPU; JSON is clear enough
        )
        raw = result["choices"][0]["message"]["content"].strip()

        # Strip <think>...</think> blocks that qwen3 leaks when thinking=True
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # Rescue cases where qwen3 outputs prose instead of JSON:
        # 1. Empty (thinking-only output — already stripped)
        # 2. Long prose with JSON embedded somewhere — trim leading prose
        # 3. Long prose with no JSON at all — retry clean
        # 4. Truncated JSON (hit max_tokens mid-object) — retry clean
        if raw and not raw.startswith("{"):
            json_start = raw.find("{")
            if json_start > 0:
                raw = raw[json_start:]   # strip leading prose, keep JSON
            else:
                raw = ""  # no JSON at all — fall through to retry below

        # Truncated JSON: opened { but never closed } — treat as no JSON
        if raw and raw.startswith("{") and "}" not in raw:
            logger.debug("decide: truncated JSON (no closing }) — treating as empty")
            raw = ""

        if not raw:
            logger.debug("decide: no JSON in response — retrying without thinking")
            # Inject continuation hint so the model picks up the next step
            # rather than defaulting to an answer after a truncation event.
            retry_messages = messages[:]
            retry_messages.append({
                "role": "user",
                "content": (
                    "Your previous output was unusable. "
                    "Continue with the planned task — do NOT answer yet unless all steps are done. "
                    + action_prompt
                ),
            })
            result2 = await client.generate(
                retry_messages, max_tokens=256, temperature=0.1, stream=False, thinking=False,
                response_format={"type": "json_object"},
            )
            raw2 = (result2["choices"][0]["message"]["content"] or "").strip()
            if raw2:
                parsed2 = _parse_json(raw2)
                if parsed2 and "action" in parsed2:
                    raw = raw2
                else:
                    raw = json.dumps({"action": "answer", "text": raw2[:300]})
            else:
                return Action(type="answer", content="(no response)")
    except Exception as exc:
        logger.warning("decide failed: %s", exc)
        return Action(type="answer", content=f"(Error: {exc})")

    action = _map_action_to_tool(raw, available_tools)
    logger.info("decide: raw=%r tools=%s → type=%s tool=%s", raw[:80], [t.get('name') for t in available_tools], action.type, action.tool_name)

    # Empty answer guard: model chose answer but left content blank — retry once
    if action.type == "answer" and not action.content.strip():
        logger.debug("decide: empty answer — retrying to extract content")
        try:
            fill_messages = messages + [{
                "role": "user",
                "content": (
                    "Your previous answer was empty. Provide a real response now.\n"
                    + action_prompt
                ),
            }]
            r_fill = await client.generate(
                fill_messages, max_tokens=256, temperature=0.1,
                stream=False, thinking=False,
                response_format={"type": "json_object"},
            )
            raw_fill = (r_fill["choices"][0]["message"]["content"] or "").strip()
            if raw_fill:
                a_fill = _map_action_to_tool(raw_fill, available_tools)
                if a_fill.content.strip():
                    action = a_fill
        except Exception as exc:
            logger.debug("empty answer retry failed: %s", exc)

    if action.type == "answer":
        logger.info("decide → Answer: %s", action.content[:80])
    elif action.tool_name == "think":
        logger.info("decide → Think: %s", str(action.tool_input.get("reasoning", ""))[:80])
    else:
        logger.info("decide → tool=%s input=%s", action.tool_name, str(action.tool_input)[:120])
    return action

def _tool_to_action_json(tool: str, inp: dict) -> str:
    """Reconstruct the action-JSON the model 'would have' output for an internal tool.

    Used to populate assistant turns in the multi-turn context so the model
    can see its own previous decisions in a format it recognises.
    """
    mapping = {
        "plan_task":        ("plan",           "task"),
        "search_knowledge": ("search_memory",  "query"),
        "search_history":   ("search_history", "query"),
        "save_memory":      ("save_memory",    "note"),
        "web_search":       ("web_search",     "query"),
        "think":            ("think",          "reasoning"),
        "list_workspace":   ("list_workspace", "input"),
        "read_file":        ("read_file",      "path"),
    }
    action_word, key = mapping.get(tool, (tool, "input"))
    value = inp.get(key) or inp.get(list(inp.keys())[0], "") if inp else ""
    return json.dumps({"action": action_word, key: value})


_ERROR_INDICATORS = (
    "invalid query", "error:", "command not found",
    "access denied", "not recognized", "is not recognized",
    "cannot find", "traceback", "exception", "syntax error",
    "failed", "invalid syntax", "no such file", "permission denied",
)


def _history_to_messages(raw_history: list[dict], max_turns: int = 6) -> list[dict]:
    """Convert Anthropic-format conversation history to flat Ollama messages.

    - Thinking blocks are skipped (includes SISYPHEAN_STATE).
    - tool_use → "[bash: ls -la]" style annotation.
    - tool_result → "[Output]\\n..." text.
    - system-reminder injections are stripped from user messages.

    Limiting to max_turns keeps the token count manageable for small models.
    """
    import re as _re

    messages: list[dict] = []
    tail = raw_history[-max_turns:] if len(raw_history) > max_turns else raw_history

    for msg in tail:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if isinstance(content, str):
            text = _re.sub(
                r"<system-reminder>.*?</system-reminder>", "", content, flags=_re.DOTALL
            ).strip()
            if text:
                messages.append({"role": role, "content": text})
            continue

        if not isinstance(content, list):
            continue

        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")

            if btype == "thinking":
                continue  # strip SISYPHEAN_STATE and all thinking blocks

            if btype == "text":
                text = block.get("text", "").strip()
                text = _re.sub(
                    r"<system-reminder>.*?</system-reminder>", "", text, flags=_re.DOTALL
                ).strip()
                if text:
                    parts.append(text)

            elif btype == "tool_use":
                # Emit action-JSON consistent with what the model outputs itself
                name = block.get("name", "")
                inp = block.get("input", {})
                cmd = (
                    inp.get("command")
                    or inp.get("query")
                    or inp.get("task")
                    or str(inp)[:80]
                )
                parts.append(json.dumps({"action": "bash", "command": cmd})
                             if name.lower() == "bash"
                             else json.dumps({"action": name, **{k: v for k, v in inp.items()}}))

            elif btype == "tool_result":
                rc = block.get("content", "")
                if isinstance(rc, list):
                    rc = "\n".join(
                        b.get("text", "") for b in rc
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                rc_text = str(rc).strip()[:500]
                if rc_text:
                    parts.append(f"[Output]\n{rc_text}")

        combined = "\n\n".join(parts).strip()
        if combined:
            messages.append({"role": role, "content": combined})

    return messages

def _looks_like_error(text: str) -> bool:
    """Return True if a bash result looks like a failure/error."""
    lower = text.lower().strip()
    return any(kw in lower for kw in _ERROR_INDICATORS)


def _map_action_to_tool(raw: str, available_tools: list[dict]) -> "Action":
    """Map simple action words from the model to actual tool Actions.

    The model outputs {"action": "bash", "command": "..."} — we translate
    "bash" to the real Bash tool name from available_tools, ensuring the
    model never needs to know exact tool names.
    """
    parsed = _parse_json(raw)
    if not parsed:
        logger.debug("decide: unparseable response — %s", raw[:120])
        return Action(type="answer", content=raw[:2000])

    action_word = str(parsed.get("action", "answer")).strip().lower()

    # ── Think — reasoning step, no external tool ──────────────────────────────
    if action_word in ("think", "reason", "reasoning"):
        reasoning = (
            parsed.get("reasoning")
            or parsed.get("thought")
            or parsed.get("text")
            or ""
        )
        return Action(type="tool", tool_name="think", tool_input={"reasoning": reasoning})

    # ── Direct answer ──────────────────────────────────────────────────────────
    if action_word in ("answer", "done", "complete", "finished", "reply"):
        text = (
            parsed.get("text")
            or parsed.get("summary")
            or parsed.get("content")
            or parsed.get("result")
            or raw
        )
        return Action(type="answer", content=str(text).strip())

    # ── Internal tools (handled inside micro-loop) ─────────────────────────────
    if action_word in ("plan", "plan_task"):
        task = parsed.get("task") or parsed.get("text") or ""
        return Action(type="tool", tool_name="plan_task", tool_input={"task": task})

    if action_word in ("search_memory", "search_knowledge", "search"):
        query = parsed.get("query") or parsed.get("text") or ""
        return Action(type="tool", tool_name="search_knowledge", tool_input={"query": query})

    if action_word in ("search_history",):
        query = parsed.get("query") or parsed.get("text") or ""
        return Action(type="tool", tool_name="search_history", tool_input={"query": query})

    if action_word in ("save_memory", "remember"):
        note = parsed.get("note") or parsed.get("text") or ""
        return Action(type="tool", tool_name="save_memory", tool_input={"note": note})

    if action_word in ("web_search", "web", "search_web"):
        query = parsed.get("query") or parsed.get("text") or ""
        return Action(type="tool", tool_name="web_search", tool_input={"query": query})

    if action_word in ("list_workspace", "ls", "list_files"):
        return Action(type="tool", tool_name="list_workspace", tool_input={})

    if action_word in ("read_file", "read"):
        path = parsed.get("path") or parsed.get("file") or ""
        query = parsed.get("query") or ""
        return Action(type="tool", tool_name="read_file", tool_input={"path": path, "query": query})

    # ── Outer tools (Claude Code executes these) ───────────────────────────────
    # Map action word → tool name lookup in available_tools
    _ACTION_TO_TOOL_HINT: dict[str, list[str]] = {
        "bash":       ["bash", "Bash"],
        "run":        ["bash", "Bash"],
        "shell":      ["bash", "Bash"],
        "read_file":  ["read", "Read"],
        "read":       ["read", "Read"],
        "write_file": ["write", "Write"],
        "write":      ["write", "Write"],
        "edit_file":  ["edit", "Edit"],
        "edit":       ["edit", "Edit"],
        "find_files": ["glob", "Glob"],
        "glob":       ["glob", "Glob"],
        "grep":       ["grep", "Grep"],
        "search_code":["grep", "Grep"],
    }
    hints = _ACTION_TO_TOOL_HINT.get(action_word, [action_word])

    # Find the actual tool in available_tools (case-insensitive)
    tool_map = {t.get("name", "").lower(): t.get("name", "") for t in available_tools}
    canonical = None
    for hint in hints:
        canonical = tool_map.get(hint.lower())
        if canonical:
            break

    if not canonical:
        # Unknown action — fall back to Bash if available, else Answer
        canonical = tool_map.get("bash")
        if not canonical:
            logger.warning("decide: unknown action %r, no Bash available — answering", action_word)
            return Action(type="answer", content=raw[:2000])
        logger.warning("decide: unknown action %r — defaulting to Bash", action_word)

    # Build tool_input: everything except "action"
    tool_input = {k: v for k, v in parsed.items() if k != "action"}
    # Strip unfilled placeholder values
    tool_input = {
        k: v for k, v in tool_input.items()
        if not (isinstance(v, str) and v.startswith("<") and v.endswith(">"))
    }

    # Guard: bash with no command or placeholder → skip (prevents "Invalid tool parameters")
    if canonical.lower() == "bash" and not tool_input.get("command", "").strip():
        logger.warning("decide: bash action with missing/empty command — falling back to answer")
        return Action(type="answer", content="(bash called without a command — please be more specific)")

    # Sanitize bash commands
    if canonical.lower() == "bash" and "command" in tool_input:
        cmd: str = tool_input["command"].strip()

        # Convert pure echo commands to answer actions — the model sometimes
        # wraps a text response in `echo "..."` instead of using the answer action.
        import re as _re
        _echo_re = _re.compile(r'^echo\s+["\'](.+)["\']$', _re.DOTALL)
        echo_match = _echo_re.match(cmd)
        if echo_match and "&&" not in cmd and "|" not in cmd and ";" not in cmd:
            answer_text = echo_match.group(1)
            logger.debug("converted echo command to answer action")
            return Action(type="answer", content=answer_text)

        # web_search / google are not shell commands — redirect to internal web_search tool
        _web_cmd_re = _re.compile(r'^(web_search|websearch|google|ddgs?)\b', _re.IGNORECASE)
        if _web_cmd_re.match(cmd):
            query = _re.sub(r'^[^\s]+\s*(-q\s*)?', '', cmd).strip().strip('"\'')
            logger.debug("redirected bash web_search → internal tool: %r", query)
            return Action(type="tool", tool_name="web_search", tool_input={"query": query or cmd})

        # Strip a spurious leading "/" from plain command names.
        # e.g. "/mkdir -p dir" → "mkdir -p dir"  (qwen3 sometimes adds this)
        # Legitimate absolute paths like /usr/bin/python are left untouched.
        if cmd.startswith("/"):
            first_token = cmd.split()[0]
            rest = first_token[1:]
            if rest and "/" not in rest and not rest.startswith("usr") and not rest.startswith("bin"):
                cmd = cmd[1:]
                logger.debug("sanitized bash command: stripped leading '/'")
        tool_input["command"] = cmd

    return Action(type="tool", tool_name=canonical, tool_input=tool_input)


# Maximum tools shown per decision call (keeps context small for 4B)
_MAX_TOOLS_PER_CALL: int = 4


# ── Action — what Gemma decided to do next ────────────────────────────────────

@dataclass
class Action:
    type: str           # "tool" | "answer"
    reasoning: str = ""
    tool_name: str = ""
    tool_id: str = field(default_factory=lambda: f"toolu_{uuid.uuid4().hex[:16]}")
    tool_input: dict = field(default_factory=dict)
    content: str = ""   # final answer text when type="answer"


# ── Phase 1: Tool filtering ───────────────────────────────────────────────────

def _filter_tools(
    available_tools: list[dict],
    instruction: str,
    top_n: int = _MAX_TOOLS_PER_CALL,
) -> list[dict]:
    """Return the top_n most relevant tools for this instruction.

    Uses token-overlap search over tool name + description.  Simple and fast —
    no embedding needed since tool descriptions are short and keyword-rich.

    Bash is always included as a fallback (it can do almost anything via shell).
    """
    if not available_tools:
        return []
    if len(available_tools) <= top_n:
        return list(available_tools)

    tokens = set(re.findall(r"[a-z0-9]+", instruction.lower()))

    scored: list[tuple[int, int, dict]] = []
    for i, t in enumerate(available_tools):
        text = f"{t.get('name', '')} {t.get('description', '')}".lower()
        score = len(tokens & set(re.findall(r"[a-z0-9]+", text)))
        scored.append((score, i, t))  # i is stable tiebreaker

    scored.sort(key=lambda x: (-x[0], x[1]))
    result: list[dict] = [t for _, _, t in scored[:top_n]]

    # Ensure Bash is always available — it's a universal fallback
    bash = next(
        (t for t in available_tools if t.get("name", "").lower() == "bash"),
        None,
    )
    if bash and bash not in result:
        result[-1] = bash  # replace last slot with Bash

    return result


# ── Phase 2: Fill-in-the-blank prompt ────────────────────────────────────────

def _build_tool_prompt(tools: list[dict]) -> str:
    """Build a fill-in-the-blank options block from the filtered tools.

    Each option shows the exact JSON template the model must fill in.
    The model replies with ONLY the chosen JSON — no wrapper, no explanation.

    Template field values are written as <HINT> placeholders so Gemma
    knows exactly what type of value to supply.
    """
    labels = "ABCDEFGHIJ"
    # Build the allowed names list upfront so the model knows the exact valid values
    allowed_names = [t.get("name", "") for t in tools] + ["Answer"]
    allowed_str = ", ".join(f'"{n}"' for n in allowed_names)
    lines: list[str] = [
        f'The "tool" field MUST be exactly one of: {allowed_str}.',
        "Choose ONE action. Reply with ONLY that JSON — no explanation.\n",
    ]

    for i, tool in enumerate(tools):
        name = tool.get("name", "")
        desc = (tool.get("description") or "")[:80]
        schema = tool.get("input_schema") or {}
        props: dict = schema.get("properties", {}) if schema else {}
        required: list[str] = schema.get("required", []) if schema else []

        # Build template: {"tool": "Name", "field1": "<HINT>", ...}
        template: dict = {"tool": name}
        fields = required if required else list(props.keys())[:3]
        for key in fields:
            info = props.get(key, {})
            hint_raw = info.get("description", key).split(".")[0]
            hint = re.sub(r"[^A-Za-z0-9 _/-]", "", hint_raw)[:40].strip().upper()
            template[key] = f"<{hint or key.upper()}>"

        label = labels[i] if i < len(labels) else str(i + 1)
        lines.append(f"Option {label} – {name}  ({desc})")
        lines.append(json.dumps(template))
        lines.append("")

    # Answer option is always the last choice
    answer_label = labels[len(tools)] if len(tools) < len(labels) else str(len(tools) + 1)
    lines.append(f"Option {answer_label} – Answer  (answer directly if simple, or when all steps are done)")
    lines.append('{"tool": "Answer", "text": "<YOUR FULL ANSWER OR COMPLETION SUMMARY>"}')

    return "\n".join(lines)


# ── Decision prompt ────────────────────────────────────────────────────────────

_DECIDE_PROMPT = """\
{execution_ctx}

Step: {step}/{budget}
{hint}
{tool_options}"""


# ── Main decision function ────────────────────────────────────────────────────

async def decide_next_action(
    manifest: TaskManifest,
    step: int,
    budget: int,
    available_tools: list[dict],
    memory_context: str,
    client,
    workspace: str = "",
    extra_hint: str = "",
) -> Action:
    """Ask Gemma what to do next for the current manifest instruction.

    Two-phase approach:
      1. Filter available_tools to the 4 most relevant for this instruction.
      2. Build a fill-in-the-blank prompt — Gemma fills field values, not
         JSON structure.  Much more reliable for 4B models.

    Context given to the model:
      - manifest.execution_prompt() — goal + done summaries + current
        self-contained instruction + last result if needs_prev
      - Filtered tool options with their exact input_schema templates
      - Step / budget counter
      - Optional extra_hint (stall guard, budget warning)
    """
    cur = manifest.current
    instruction_text = cur.text if cur else manifest.goal

    # Phase 1: filter to relevant tools
    relevant_tools = _filter_tools(available_tools, instruction_text, top_n=_MAX_TOOLS_PER_CALL)

    # Phase 2: build fill-in-the-blank prompt
    tool_options = _build_tool_prompt(relevant_tools)

    execution_ctx = manifest.execution_prompt()
    system = SYSTEM if not memory_context else f"{SYSTEM}\n\n{memory_context}"
    ctx_line = dynamic_context(
        workspace=workspace,
        task_goal=instruction_text,
        step_n=step,
        budget=budget,
    )

    hint_lines: list[str] = []
    if step >= budget - 1:
        hint_lines.append("Budget nearly exhausted — summarise findings and use Answer.")
    if extra_hint:
        hint_lines.append(extra_hint)
    hint_block = "\n".join(hint_lines)

    prompt = _DECIDE_PROMPT.format(
        execution_ctx=execution_ctx,
        step=step,
        budget=budget,
        hint=hint_block,
        tool_options=tool_options,
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": f"{ctx_line}\n\n{prompt}"},
    ]

    try:
        result = await client.generate(
            messages,
            max_tokens=256,          # flat JSON is short — no need for 512
            temperature=0.2,         # lower than before: fill-in-blank needs precision
            response_format={"type": "json_object"},
            stream=False,
            thinking=False,          # JSON output: thinking breaks response_format
        )
        raw = result["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.warning("decide_next_action failed: %s", exc)
        return Action(
            type="answer",
            reasoning=str(exc),
            content=f"(Error completing task: {exc})",
        )

    action = _parse_action(raw, relevant_tools)
    if action.type == "answer":
        logger.info("decide_next_action → Answer: %s", action.content[:80])
    else:
        logger.info("decide_next_action → tool=%s input=%s", action.tool_name, str(action.tool_input)[:120])
    return action


# ── Response parser ───────────────────────────────────────────────────────────

def _parse_action(raw: str, relevant_tools: list[dict]) -> Action:
    """Parse the flat fill-in-the-blank response into an Action.

    Expected: {"tool": "Bash", "command": "ls -la"}
    or:       {"tool": "Answer", "summary": "Found 3 files."}

    Falls back to Answer with raw content on any parse failure.
    """
    parsed = _parse_json(raw)
    if not parsed:
        logger.debug("decide_next_action: unparseable — %s", raw[:120])
        return Action(type="answer", reasoning="unparseable", content=raw[:2000])

    tool_name = str(parsed.get("tool", "Answer")).strip()

    # Answer / step complete
    if tool_name.lower() in ("answer", "done", "complete", "finished"):
        summary = (
            parsed.get("text")
            or parsed.get("summary")
            or parsed.get("content")
            or parsed.get("result")
            or raw
        )
        return Action(type="answer", reasoning="", content=str(summary).strip())

    # Validate tool name against the tools we offered
    offered_names = {t.get("name", "") for t in relevant_tools}
    # Case-insensitive match
    canonical = next(
        (n for n in offered_names if n.lower() == tool_name.lower()),
        None,
    )
    if canonical is None:
        # Model used an unexpected name — try prefix match
        canonical = next(
            (n for n in offered_names if n.lower().startswith(tool_name.lower()[:4])),
            None,
        )
    if canonical is None:
        # Model hallucinated a tool name — try to salvage.
        # If Bash was offered and the response looks like a shell/file operation,
        # construct a Bash action from whatever fields the model filled in.
        bash_name = next((n for n in offered_names if n.lower() == "bash"), None)
        if bash_name:
            # Extract any command-like field from the parsed dict
            cmd = (
                parsed.get("command")
                or parsed.get("cmd")
                or parsed.get("script")
                or parsed.get("action")
                or parsed.get("description")
                or parsed.get("subject")
            )
            if cmd and isinstance(cmd, str):
                logger.warning(
                    "decide: unknown tool %r — salvaging as Bash command: %s",
                    tool_name, cmd[:80],
                )
                return Action(type="tool", tool_name=bash_name, tool_input={"command": cmd})
        logger.warning(
            "decide: unknown tool %r (offered: %s) — defaulting to Answer",
            tool_name,
            ", ".join(offered_names),
        )
        return Action(
            type="answer",
            reasoning=f"unknown tool: {tool_name}",
            content=raw[:2000],
        )

    # Build tool_input: everything except the "tool" key
    tool_input = {k: v for k, v in parsed.items() if k != "tool"}

    # Strip placeholder values the model forgot to fill in
    tool_input = {
        k: v for k, v in tool_input.items()
        if not (isinstance(v, str) and v.startswith("<") and v.endswith(">"))
    }

    return Action(
        type="tool",
        reasoning="",
        tool_name=canonical,
        tool_input=tool_input,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict | None:
    """Find and parse the first complete balanced JSON object in text.

    Uses brace-depth tracking instead of rfind('}') so that a model
    outputting two JSON blocks in sequence (e.g. bash action + answer)
    doesn't produce an invalid combined string that fails to parse.
    """
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


# ── Legacy: execute_stage (used when tools=None, no Claude Code harness) ──────

@dataclass
class StageResult:
    stage: Stage
    output: str
    steps_taken: int
    searches_done: list[str] = field(default_factory=list)


async def execute_stage(
    stage: Stage,
    memory_context: str,
    client,
    workspace: str = "",
) -> StageResult:
    """Legacy executor — runs stage internally when no external tools available."""
    is_write = stage.type in _WRITE_STAGE_TYPES
    schema_prompt = CODE_SCHEMA_PROMPT if stage.type == "write_code" else (
        DOC_SCHEMA_PROMPT if stage.type == "write_doc" else STEP_SCHEMA_PROMPT
    )

    system_content = SYSTEM if not memory_context else f"{SYSTEM}\n\n{memory_context}"
    ctx_line = dynamic_context(workspace=workspace, task_goal=stage.goal, step_n=0, budget=stage.budget)
    messages: list[dict] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"{ctx_line}\n\n{schema_prompt}\n\nTask: {stage.goal}"},
    ]

    output_parts: list[str] = []
    searches_done: list[str] = []
    step = 0

    while step < stage.budget:
        step += 1
        messages[1]["content"] = re.sub(
            r"Step \d+/\d+",
            f"Step {step}/{stage.budget}",
            messages[1]["content"],
        )

        try:
            result = await client.generate(
                messages,
                max_tokens=1024 if is_write else 512,
                temperature=0.3 if is_write else 0.5,
                response_format={"type": "json_object"},
                stream=False,
                thinking=False,
            )
            raw = result["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.warning("Stage %s step %d failed: %s", stage.type, step, exc)
            break

        parsed = _parse_json(raw)
        if not parsed:
            parsed = {"action": "think", "reasoning": raw[:500]}

        action = parsed.get("action", "think")

        if action == "answer":
            content = parsed.get("content", "").strip()
            if content:
                output_parts.append(content)
            break

        if is_write and ("path" in parsed or "content" in parsed):
            content = parsed.get("content", "")
            path = parsed.get("path", "")
            if content:
                label = f"**File: {path}**\n" if path else ""
                output_parts.append(label + content)
            if parsed.get("done", False):
                break
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": "Continue. Write the next section or file."})
            continue

        if action in ("think", "search"):
            reasoning = parsed.get("reasoning") or parsed.get("query") or raw
            queries = extract_search_queries(reasoning)
            if not queries and _is_uncertain(reasoning):
                queries = [_make_search_query(stage.goal, reasoning)]
            if action == "search":
                queries = [parsed.get("query", "").strip() or stage.goal]

            if queries:
                for query in queries:
                    results = await web_search(query, max_results=4)
                    search_block = format_results(results)
                    searches_done.append(query)
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "user", "content": f"Search results:\n\n{search_block}\n\nContinue."})
            else:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "Continue."})

    if not output_parts:
        output_parts.append(await _force_answer(stage.goal, messages, client))

    return StageResult(
        stage=stage,
        output="\n\n".join(p for p in output_parts if p.strip()),
        steps_taken=step,
        searches_done=searches_done,
    )


def _is_uncertain(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in UNCERTAINTY_PHRASES)


def _make_search_query(goal: str, reasoning: str) -> str:
    words = [w for w in reasoning.lower().split() if len(w) > 4][:5]
    return f"{goal[:60]} {' '.join(words)}".strip()


async def _force_answer(goal: str, messages: list[dict], client) -> str:
    force_messages = messages + [{
        "role": "user",
        "content": (
            "Budget exhausted. Provide the best answer you can now. "
            'Respond: {"action": "answer", "content": "<your best answer>"}'
        ),
    }]
    try:
        result = await client.generate(
            force_messages, max_tokens=512, temperature=0.3,
            response_format={"type": "json_object"}, stream=False,
            thinking=False,
        )
        raw = result["choices"][0]["message"]["content"].strip()
        parsed = _parse_json(raw)
        if parsed and parsed.get("content"):
            return parsed["content"]
        return raw
    except Exception as exc:
        logger.warning("Force answer failed: %s", exc)
        return f"(Could not complete: {goal})"
