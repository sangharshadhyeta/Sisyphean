"""JSON parsing and action mapping for the executor."""
from __future__ import annotations

import json
import logging
import re

from engine.translation.executor_action import Action

logger = logging.getLogger(__name__)


def _parse_json(text: str) -> dict | None:
    """Find and parse the first complete balanced JSON object in text.

    Uses brace-depth tracking instead of rfind('}') so that a model
    outputting two JSON blocks doesn't produce an invalid combined string.
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
                fragment = text[start: i + 1]
                try:
                    return json.loads(fragment)
                except json.JSONDecodeError:
                    cleaned = re.sub(r",\s*([}\]])", r"\1", fragment)
                    try:
                        return json.loads(cleaned)
                    except json.JSONDecodeError:
                        return None
    return None


def _parse_action(raw: str, relevant_tools: list[dict]) -> Action:
    """Parse fill-in-the-blank response into an Action.

    Expected: {"tool": "Bash", "command": "ls -la"}
    or:       {"tool": "Answer", "summary": "Found 3 files."}
    """
    parsed = _parse_json(raw)
    if not parsed:
        logger.debug("decide_next_action: unparseable — %s", raw[:120])
        return Action(type="answer", reasoning="unparseable", content=raw[:2000])

    tool_name = str(parsed.get("tool", "Answer")).strip()

    if tool_name.lower() in ("answer", "done", "complete", "finished"):
        summary = (
            parsed.get("text")
            or parsed.get("summary")
            or parsed.get("content")
            or parsed.get("result")
            or raw
        )
        return Action(type="answer", reasoning="", content=str(summary).strip())

    offered_names = {t.get("name", "") for t in relevant_tools}
    canonical = next(
        (n for n in offered_names if n.lower() == tool_name.lower()), None
    )
    if canonical is None:
        canonical = next(
            (n for n in offered_names if n.lower().startswith(tool_name.lower()[:4])), None
        )
    if canonical is None:
        bash_name = next((n for n in offered_names if n.lower() == "bash"), None)
        if bash_name:
            cmd = (
                parsed.get("command") or parsed.get("cmd") or parsed.get("script")
                or parsed.get("action") or parsed.get("description") or parsed.get("subject")
            )
            if cmd and isinstance(cmd, str):
                logger.warning(
                    "decide: unknown tool %r — salvaging as Bash: %s", tool_name, cmd[:80]
                )
                return Action(type="tool", tool_name=bash_name, tool_input={"command": cmd})
        logger.warning(
            "decide: unknown tool %r (offered: %s) — defaulting to Answer",
            tool_name, ", ".join(offered_names),
        )
        return Action(type="answer", reasoning=f"unknown tool: {tool_name}", content=raw[:2000])

    tool_input = {k: v for k, v in parsed.items() if k != "tool"}
    tool_input = {
        k: v for k, v in tool_input.items()
        if not (isinstance(v, str) and v.startswith("<") and v.endswith(">"))
    }
    return Action(type="tool", reasoning="", tool_name=canonical, tool_input=tool_input)


def _map_action_to_tool(raw: str, available_tools: list[dict]) -> Action:
    """Map simple action words from the model output to actual tool Actions.

    The model outputs {"action": "bash", "command": "..."} — we translate
    "bash" to the real Bash tool name from available_tools.
    """
    parsed = _parse_json(raw)
    if not parsed:
        logger.debug("decide: unparseable response — %s", raw[:120])
        return Action(type="answer", content=raw[:2000])

    action_word = str(parsed.get("action", "answer")).strip().lower()

    if action_word in ("think", "reason", "reasoning"):
        reasoning = (
            parsed.get("reasoning") or parsed.get("thought") or parsed.get("text") or ""
        )
        return Action(type="tool", tool_name="think", tool_input={"reasoning": reasoning})

    if action_word in ("answer", "done", "complete", "finished", "reply"):
        text = (
            parsed.get("text") or parsed.get("summary") or parsed.get("content")
            or parsed.get("result") or raw
        )
        return Action(type="answer", content=str(text).strip())

    if action_word in ("plan", "plan_task"):
        return Action(type="tool", tool_name="plan_task",
                      tool_input={"task": parsed.get("task") or parsed.get("text") or ""})

    if action_word in ("search_memory", "search_knowledge", "search"):
        return Action(type="tool", tool_name="search_knowledge",
                      tool_input={"query": parsed.get("query") or parsed.get("text") or ""})

    if action_word == "search_history":
        return Action(type="tool", tool_name="search_history",
                      tool_input={"query": parsed.get("query") or parsed.get("text") or ""})

    if action_word in ("save_memory", "remember"):
        return Action(type="tool", tool_name="save_memory",
                      tool_input={"note": parsed.get("note") or parsed.get("text") or ""})

    if action_word in ("web_search", "web", "search_web"):
        return Action(type="tool", tool_name="web_search",
                      tool_input={"query": parsed.get("query") or parsed.get("text") or ""})

    if action_word in ("list_workspace", "ls", "list_files"):
        return Action(type="tool", tool_name="list_workspace", tool_input={})

    if action_word in ("read_file", "read"):
        return Action(type="tool", tool_name="read_file",
                      tool_input={"path": parsed.get("path") or parsed.get("file") or "",
                                  "query": parsed.get("query") or ""})

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
        "search_code": ["grep", "Grep"],
    }
    hints = _ACTION_TO_TOOL_HINT.get(action_word, [action_word])
    tool_map = {t.get("name", "").lower(): t.get("name", "") for t in available_tools}
    canonical = None
    for hint in hints:
        canonical = tool_map.get(hint.lower())
        if canonical:
            break

    if not canonical:
        canonical = tool_map.get("bash")
        if not canonical:
            logger.warning("decide: unknown action %r, no Bash — answering", action_word)
            return Action(type="answer", content=raw[:2000])
        logger.warning("decide: unknown action %r — defaulting to Bash", action_word)

    tool_input = {k: v for k, v in parsed.items() if k != "action"}
    tool_input = {
        k: v for k, v in tool_input.items()
        if not (isinstance(v, str) and v.startswith("<") and v.endswith(">"))
    }

    if canonical.lower() == "bash" and not tool_input.get("command", "").strip():
        logger.warning("decide: bash with missing command — falling back to answer")
        return Action(type="answer",
                      content="(bash called without a command — please be more specific)")

    if canonical.lower() == "bash" and "command" in tool_input:
        cmd: str = tool_input["command"].strip()

        _echo_re = re.compile(r'^echo\s+["\'](.+)["\']$', re.DOTALL)
        echo_match = _echo_re.match(cmd)
        if echo_match and "&&" not in cmd and "|" not in cmd and ";" not in cmd:
            return Action(type="answer", content=echo_match.group(1))

        _web_cmd_re = re.compile(r'^(web_search|websearch|google|ddgs?)\b', re.IGNORECASE)
        if _web_cmd_re.match(cmd):
            query = re.sub(r'^[^\s]+\s*(-q\s*)?', '', cmd).strip().strip('"\'')
            return Action(type="tool", tool_name="web_search",
                          tool_input={"query": query or cmd})

        if cmd.startswith("/"):
            first_token = cmd.split()[0]
            rest = first_token[1:]
            if rest and "/" not in rest and not rest.startswith("usr") and not rest.startswith("bin"):
                cmd = cmd[1:]

        _python_prefix_re = re.compile(r'^python(?:3)?\s+', re.IGNORECASE)
        if _python_prefix_re.match(cmd):
            rest_of_cmd = _python_prefix_re.sub("", cmd, count=1).strip()
            first_token_rest = rest_of_cmd.split()[0] if rest_of_cmd.split() else ""
            if first_token_rest and not first_token_rest.lower().endswith(".py"):
                cmd = rest_of_cmd

        tool_input["command"] = cmd

    return Action(type="tool", tool_name=canonical, tool_input=tool_input)
