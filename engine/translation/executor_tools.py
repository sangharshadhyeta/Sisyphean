"""Tool filtering and fill-in-the-blank prompt builder for the executor."""
from __future__ import annotations

import json
import re

_MAX_TOOLS_PER_CALL: int = 4


def _filter_tools(
    available_tools: list[dict],
    instruction: str,
    top_n: int = _MAX_TOOLS_PER_CALL,
) -> list[dict]:
    """Return the top_n most relevant tools for this instruction.

    Uses token-overlap search over tool name + description.
    Bash is always included as a universal fallback.
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
        scored.append((score, i, t))

    scored.sort(key=lambda x: (-x[0], x[1]))
    result: list[dict] = [t for _, _, t in scored[:top_n]]

    # Bash is always included regardless of relevance score: it's a universal
    # fallback for any shell operation the model might need, and small models
    # often emit "bash" for tasks that don't obviously keyword-match any tool.
    bash = next(
        (t for t in available_tools if t.get("name", "").lower() == "bash"),
        None,
    )
    if bash and bash not in result:
        result[-1] = bash  # replace lowest-scored slot

    return result


def _build_tool_prompt(tools: list[dict]) -> str:
    """Build a fill-in-the-blank options block from the filtered tools.

    Each option shows the exact JSON template the model must fill in.
    Template field values are written as <HINT> placeholders.
    """
    labels = "ABCDEFGHIJ"
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

    answer_label = labels[len(tools)] if len(tools) < len(labels) else str(len(tools) + 1)
    lines.append(f"Option {answer_label} – Answer  (answer directly if simple, or when all steps are done)")
    lines.append('{"tool": "Answer", "text": "<YOUR FULL ANSWER OR COMPLETION SUMMARY>"}')

    return "\n".join(lines)
