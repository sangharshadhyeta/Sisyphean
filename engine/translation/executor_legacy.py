"""Legacy execute_stage() for when no external tools are available."""
from __future__ import annotations

import logging
import re

from engine.translation.planner import Stage
from engine.translation.prompts import (
    SYSTEM,
    STEP_SCHEMA_PROMPT,
    CODE_SCHEMA_PROMPT,
    DOC_SCHEMA_PROMPT,
    UNCERTAINTY_PHRASES,
    dynamic_context,
)
from engine.translation.web_search import (
    search as web_search,
    format_results,
    extract_search_queries,
)
from engine.translation.executor_action import StageResult
from engine.translation.executor_parsing import _parse_json

logger = logging.getLogger(__name__)

_WRITE_STAGE_TYPES = {"write_code", "write_doc"}


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
