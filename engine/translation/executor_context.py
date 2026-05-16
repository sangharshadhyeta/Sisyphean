"""Context helpers for the executor: history conversion, stage detection, action formatting."""
from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

_ERROR_INDICATORS = (
    "invalid query", "error:", "command not found",
    "access denied", "not recognized", "is not recognized",
    "cannot find", "traceback", "exception", "syntax error",
    "failed", "invalid syntax", "no such file", "permission denied",
)


def _looks_like_error(text: str) -> bool:
    lower = text.lower().strip()
    return any(kw in lower for kw in _ERROR_INDICATORS)


def _detect_stage(
    internal_messages: list[dict],
    summary: str,
    step: int,
    budget: int,
) -> str:
    """Determine the current stage from loop state: plan | answer | execute."""
    if step >= budget - 2:
        return "answer"
    has_plan = any(m.get("tool") == "plan_task" for m in internal_messages)
    if not has_plan and summary:
        has_plan = "Planned" in summary or "planned '" in summary.lower()
    if step == 0 and not has_plan:
        return "plan"
    return "execute"


def _tool_to_action_json(tool: str, inp: dict) -> str:
    """Reconstruct the action-JSON the model 'would have' output for an internal tool.

    Used to populate assistant turns in multi-turn context so the model can
    see its own previous decisions in a format it recognises.
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


def _history_to_messages(raw_history: list[dict], max_turns: int = 6) -> list[dict]:
    """Convert Anthropic-format conversation history to flat Ollama messages.

    - Thinking blocks are skipped (includes SISYPHEAN_STATE).
    - tool_use → action-JSON annotation.
    - tool_result → "[Output]\\n..." text.
    - system-reminder injections are stripped.
    """
    messages: list[dict] = []
    tail = raw_history[-max_turns:] if len(raw_history) > max_turns else raw_history

    for msg in tail:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if isinstance(content, str):
            text = re.sub(
                r"<system-reminder>.*?</system-reminder>", "", content, flags=re.DOTALL
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
                continue

            if btype == "text":
                text = block.get("text", "").strip()
                text = re.sub(
                    r"<system-reminder>.*?</system-reminder>", "", text, flags=re.DOTALL
                ).strip()
                if text:
                    parts.append(text)

            elif btype == "tool_use":
                name = block.get("name", "")
                inp = block.get("input", {})
                cmd = (
                    inp.get("command")
                    or inp.get("query")
                    or inp.get("task")
                    or str(inp)[:80]
                )
                parts.append(
                    json.dumps({"action": "bash", "command": cmd})
                    if name.lower() == "bash"
                    else json.dumps({"action": name, **{k: v for k, v in inp.items()}})
                )

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
