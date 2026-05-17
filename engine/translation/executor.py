"""Stage executor — decides next action for Claude Code to execute.

Decomposed into focused sub-modules:
  executor_action.py   — Action / StageResult dataclasses
  executor_context.py  — history conversion, stage detection, action formatting
  executor_parsing.py  — JSON parsing, _parse_action, _map_action_to_tool
  executor_tools.py    — tool filtering, fill-in-the-blank prompt builder
  executor_legacy.py   — execute_stage() for when no external tools available

This file is the public interface: decide() and decide_next_action() live here;
all sub-module symbols are re-exported for backward compatibility.
"""
from __future__ import annotations

import json
import logging
import re

from engine.translation.manifest import TaskManifest
from engine.translation.prompts import (
    SYSTEM,
    UNCERTAINTY_PHRASES,
    dynamic_context,
    action_prompt,
    stage_action_prompt,  # kept as alias for any remaining callers
)
from engine.translation.executor_action import Action, StageResult
from engine.translation.executor_context import (
    _ERROR_INDICATORS,
    _looks_like_error,
    _detect_stage,
    _tool_to_action_json,
    _history_to_messages,
)
from engine.translation.executor_parsing import (
    _parse_json,
    _parse_action,
    _map_action_to_tool,
)
from engine.translation.executor_tools import (
    _MAX_TOOLS_PER_CALL,
    _filter_tools,
    _build_tool_prompt,
)
from engine.translation.executor_legacy import execute_stage

logger = logging.getLogger(__name__)

_WRITE_STAGE_TYPES = {"write_code", "write_doc"}

# ── Decide prompt template ────────────────────────────────────────────────────

_DECIDE_PROMPT = """\
{execution_ctx}

Step: {step}/{budget}
{hint}
{tool_options}"""


# ── Main micro-loop decision function ────────────────────────────────────────

async def decide(
    user_message: str,
    internal_messages: list[dict],
    available_tools: list[dict],
    static_context: str,
    client,
    step: int = 0,
    budget: int = 12,
    semantic_history: str = "",
    summary: str = "",
    hint: str = "",
    workspace: str = "",
    current_step_text: str = "",
    plan_steps: list[dict] | None = None,
    plan_step_idx: int = 0,
) -> Action:
    """Single-step decision: given full context, what is the next action?"""
    system_parts = [SYSTEM]
    if static_context:
        system_parts.append(static_context)
    system = "\n\n".join(p.strip() for p in system_parts if p.strip())

    if not hint:
        if step >= budget - 2:
            hint = "Budget nearly exhausted — use answer now."
        elif internal_messages:
            last = internal_messages[-1]
            if last.get("tool") == "bash_result" and _looks_like_error(last.get("result", "")):
                last_result = last.get("result", "")
                if "web_search" in last_result and "command not found" in last_result.lower():
                    hint = 'web_search is not a shell command. Use {"action": "web_search", "query": "..."} or answer with what you already know.'
                else:
                    hint = "The last command failed — try a corrected command or a different approach."

    has_bash = any(t.get("name", "").lower() == "bash" for t in available_tools)
    _action_prompt = action_prompt(step, budget, hint, has_bash, workspace, current_step_text)

    messages: list[dict] = [{"role": "system", "content": system}]

    user_turn_parts: list[str] = []
    user_turn_parts.append(f"[Original task]\n{user_message}")

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

    if summary:
        user_turn_parts.append(f"[Already done]\n{summary}")

    bash_done = [m for m in internal_messages if m.get("tool") == "bash_result"]
    if bash_done:
        bash_lines = []
        for br in bash_done:
            res = str(br.get("result", "")).strip()[:200].replace("\n", " ")
            bash_lines.append(res)
        user_turn_parts.append(
            "[Commands already run — DO NOT repeat]\n" + "\n".join(bash_lines)
        )

    if semantic_history:
        user_turn_parts.append(f"[Past sessions]\n{semantic_history}")
    messages.append({"role": "user", "content": "\n\n".join(user_turn_parts)})

    for imsg in internal_messages:
        tool = imsg["tool"]
        if tool == "bash_result":
            continue
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
            thinking=False,
        )
        raw = result["choices"][0]["message"]["content"].strip()

        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        if raw and not raw.startswith("{"):
            json_start = raw.find("{")
            if json_start > 0:
                raw = raw[json_start:]
            else:
                raw = ""

        if raw and raw.startswith("{") and "}" not in raw:
            logger.debug("decide: truncated JSON (no closing }) — treating as empty")
            raw = ""

        if not raw:
            logger.debug("decide: no JSON — retrying with explicit JSON instruction")
            retry_messages = messages[:]
            retry_messages.append({
                "role": "user",
                "content": (
                    "Output ONLY a single valid JSON object — no prose, no markdown fences. "
                    "Do NOT answer yet unless all steps are done. "
                    + _action_prompt
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

    if action.type == "answer" and not action.content.strip():
        logger.debug("decide: empty answer — retrying to extract content")
        try:
            fill_messages = messages + [{
                "role": "user",
                "content": (
                    "Your previous answer was empty. Provide a real response now.\n"
                    + _action_prompt
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


# ── decide_next_action (fill-in-the-blank, manifest-based) ───────────────────

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
    """Ask the model what to do next for the current manifest instruction."""
    cur = manifest.current
    instruction_text = cur.text if cur else manifest.goal

    relevant_tools = _filter_tools(available_tools, instruction_text, top_n=_MAX_TOOLS_PER_CALL)
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
            max_tokens=256,
            temperature=0.2,
            response_format={"type": "json_object"},
            stream=False,
            thinking=False,
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
