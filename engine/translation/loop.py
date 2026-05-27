"""Translation loop — thin adapter between the Anthropic API handler and the core Pipeline.

Design overview
---------------
All agent logic lives in engine/core/pipeline.py.  This module is the
public entry-point used by the API handlers — it delegates every request
to Pipeline.process() (when tools are available) or _direct() (no tools).

No standalone loop exists here.  BirdClaw is the standalone agent.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from engine.translation.prompts import SYSTEM, dynamic_context

logger = logging.getLogger(__name__)


# ── Response container ────────────────────────────────────────────────────────

@dataclass
class LoopResponse:
    content: list[dict]   # Anthropic-format content blocks
    stop_reason: str      # "end_turn" | "tool_use"


# ── Main loop ─────────────────────────────────────────────────────────────────

class TranslationLoop:

    def __init__(
        self,
        client,
        ctx_manager,
        budget_tracker=None,
        workspace: str = "",
        permission_guard=None,
        injector=None,
        knowledge_graph=None,
    ) -> None:
        self.client = client
        self.ctx = ctx_manager
        self.budget = budget_tracker
        self.workspace = workspace
        self.permission_guard = permission_guard
        self.injector = injector
        self.knowledge_graph = knowledge_graph
        # ── Core pipeline ─────────────────────────────────────────────────────
        from pathlib import Path as _Path
        from engine.core.pipeline import Pipeline
        from engine.config import load_config as _load_config
        try:
            _cfg = _load_config()
            _soul_path  = _Path(_cfg.memory.engine_policy_file)
            _prefs_path = _Path(_cfg.memory.path) / "user_prefs.md"
        except Exception:
            _soul_path  = _Path("engine_policy.md")
            _prefs_path = _Path("memory/user_prefs.md")
        self._pipeline = Pipeline(
            client=client,
            policy_path=_soul_path,
            prefs_path=_prefs_path,
            knowledge_graph=knowledge_graph,
            workspace=workspace,
            budget_tracker=budget_tracker,
        )

    async def process(
        self,
        user_message: str,
        raw_history: list[dict],
        available_tools: list[dict],
        memory_context: str = "",
        system_context: str = "",
    ) -> LoopResponse:
        """Delegate to the core pipeline."""
        # cwd from system_context is the Claude Code project dir — used by the pipeline
        # for file path resolution (project_dir), NOT for Sisyphean's own workspace.
        # Workspace stays as configured so skill files and temp files go to the right place.

        # No tools → direct generation (bypass pipeline)
        if not available_tools:
            text = await self._direct(user_message, raw_history, "")
            return LoopResponse(content=[{"type": "text", "text": text}], stop_reason="end_turn")

        return await self._pipeline.process(
            user_message=user_message,
            raw_history=raw_history,
            available_tools=available_tools,
            system_context=system_context,
            memory_ctx=memory_context,
        )

    async def _direct(
        self,
        message: str,
        raw_history: list[dict],
        static_context: str,
    ) -> str:
        """Single LLM call for simple questions (no loop, no tool use)."""
        system = SYSTEM if not static_context else f"{SYSTEM}\n\n{static_context}"
        ctx_line = dynamic_context(workspace=self.workspace)

        flat_history = _flatten_history(raw_history)
        messages = await self.ctx.fit(
            flat_history + [{"role": "user", "content": f"{ctx_line}\n\n{message}"}],
            system=system,
        )
        try:
            result = await self.client.generate(
                messages, max_tokens=1024, temperature=0.7, stream=False, thinking=False,
            )
            text = result["choices"][0]["message"]["content"].strip()
            import re as _re
            text = _re.sub(r"\[SEARCH:[^\]]*\]", "", text).strip()
            return text
        except Exception as exc:
            logger.error("Direct generation failed: %s", exc)
            return f"(Error: {exc})"

    # ── Legacy compatibility ───────────────────────────────────────────────────

    async def run(
        self,
        message: str,
        history: list[dict],
        memory_context: str = "",
    ) -> str:
        """Legacy: run loop, return final text string."""
        resp = await self.process(
            message,
            raw_history=_ensure_block_format(history),
            available_tools=[],
            memory_context=memory_context,
        )
        for block in resp.content:
            if block.get("type") == "text":
                return block["text"]
        return ""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _flatten_history(raw_history: list[dict], max_turns: int = 6) -> list[dict]:
    """Convert the last max_turns messages to plain text for _direct() Q&A.

    Intentionally simpler than executor_context._history_to_messages():
    - Drops ALL non-text blocks (thinking, tool_use, tool_result).
    - No JSON annotation of tool actions — the direct path doesn't need them.
    executor_context._history_to_messages() is for the micro-loop where the
    model needs to see its own past tool actions in action-JSON form.
    """
    tail = raw_history[-max_turns:] if len(raw_history) > max_turns else raw_history
    result = []
    for msg in tail:
        content = msg.get("content", "")
        if isinstance(content, str):
            result.append({"role": msg["role"], "content": content})
        elif isinstance(content, list):
            text_parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            combined = "\n".join(text_parts).strip()
            if combined:
                result.append({"role": msg["role"], "content": combined})
    return result


def _ensure_block_format(history: list[dict]) -> list[dict]:
    """Convert flat string history to block format if needed."""
    result = []
    for msg in history:
        content = msg.get("content", "")
        if isinstance(content, str):
            result.append({
                "role": msg["role"],
                "content": [{"type": "text", "text": content}],
            })
        else:
            result.append(msg)
    return result
