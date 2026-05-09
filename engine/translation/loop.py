"""Translation loop — micro-loop where the LLM is the sole decision maker.

Design overview
---------------
The loop no longer pre-decomposes the task into a manifest.  Instead a
MicroState (step counter + user message + internal tool results) is passed
to the executor on every iteration.  The executor chooses:

  1. An **internal tool** (plan_task, search_knowledge, search_history,
     save_memory, web_search) — handled inside the loop, never visible to
     Claude Code.
  2. An **outer tool** (Bash, etc.) — the loop encodes its state in a
     thinking block and returns a tool_use block to Claude Code.
  3. An **answer** — the loop returns end_turn with final text.

State is encoded in thinking blocks as SISYPHEAN_STATE:<json> — same
pattern as before, but now keyed on MicroState rather than TaskManifest.
The server remains fully stateless; all state lives in the conversation
history Claude Code manages.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator

from engine.translation.decomposer import decompose
from engine.translation.executor import decide, Action
from engine.translation.prompts import SYSTEM, dynamic_context
from engine.translation.planner import _spoken_math_to_python  # shared math converter
from engine.translation.web_search import search as _web_search, format_results
from engine.translation.subtask.writer import run_write_step
from engine.activity import log_event
import engine.task_tracker as _tracker

logger = logging.getLogger(__name__)

_STATE_PREFIX = "SISYPHEAN_STATE:"

MAX_STEPS = 12


# ---------------------------------------------------------------------------
# Spoken-math → Python expression converter
# Handles natural language like "what is 19 times 294?" → "19*294"
# ---------------------------------------------------------------------------

def _spoken_math_to_python(text: str) -> str:
    """Convert a spoken or symbolic math query into a Python expression string.

    Examples:
      "what is 19 times 294?"          → "19*294"
      "what is 100 divided by 4?"      → "100/4"
      "what is the square root of 144?"→ "__import__('math').sqrt(144)"
      "2+2"                            → "2+2"
      "sqrt(144)"                      → "__import__('math').sqrt(144)"
    """
    t = text.strip()

    # sqrt(x) or sqrt x  (symbolic form already — just normalise)
    m = re.search(r'\bsqrt\s*\(?\s*(\d+(?:\.\d+)?)\s*\)?', t, re.IGNORECASE)
    if m:
        return f"__import__('math').sqrt({m.group(1)})"

    # "square root of N"
    m = re.search(r'square\s+root\s+of\s+(\d+(?:\.\d+)?)', t, re.IGNORECASE)
    if m:
        return f"__import__('math').sqrt({m.group(1)})"

    # Strip question / filler words so only the math part remains
    t = re.sub(
        r'\b(what\s+is|what\'s|calculate|compute|evaluate|how\s+much\s+is|find|the)\b',
        '', t, flags=re.IGNORECASE,
    )
    t = re.sub(r'[?!.,]', '', t).strip()

    # Spoken operator → symbol
    t = re.sub(r'\btimes\b|\bmultiplied\s+by\b',  '*', t, flags=re.IGNORECASE)
    t = re.sub(r'\bplus\b|\badded\s+to\b',        '+', t, flags=re.IGNORECASE)
    t = re.sub(r'\bminus\b|\bsubtracted\s+from\b', '-', t, flags=re.IGNORECASE)
    t = re.sub(r'\bdivided\s+by\b|\bover\b',       '/', t, flags=re.IGNORECASE)
    t = re.sub(r'\bto\s+the\s+power\s+of\b|\braised\s+to\b', '**', t, flags=re.IGNORECASE)

    # Collapse whitespace around operators
    t = re.sub(r'\s*([\+\-\*\/])\s*', r'\1', t).strip()

    # If it now looks like a valid expression return it; else fall back to original
    if re.search(r'\d[\+\-\*\/\*\*\^%]\d', t):
        return t
    # Original had a symbolic operator — return stripped original
    orig_clean = re.sub(r'[?!.,\s]', '', text.strip())
    if re.search(r'\d[\+\-\*\/\^%]\d', orig_clean):
        return orig_clean
    return t or text.strip()

# ── Internal tools ────────────────────────────────────────────────────────────
# These are prepended to available_tools before the executor sees them.
# Claude Code never executes these — the loop handles them internally.

INTERNAL_TOOLS: list[dict] = [
    {
        "name": "plan_task",
        "description": "Break a complex multi-step task into a plan before executing. Use when task has 3 or more distinct steps.",
        "input_schema": {
            "type": "object",
            "properties": {"task": {"type": "string", "description": "Full task to plan"}},
            "required": ["task"],
        },
    },
    {
        "name": "search_knowledge",
        "description": "Search stored facts, concepts and research from memory. Use for domain knowledge or past research results.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to search for"}},
            "required": ["query"],
        },
    },
    {
        "name": "search_history",
        "description": "Search past conversation history across sessions. Use when user asks about previous work or past sessions.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to search for in history"}},
            "required": ["query"],
        },
    },
    {
        "name": "save_memory",
        "description": "Save a fact, preference or note to long-term memory. Use when user says remember, note that, or shares something to retain.",
        "input_schema": {
            "type": "object",
            "properties": {"note": {"type": "string", "description": "Fact or preference to save"}},
            "required": ["note"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web for current information not in your knowledge.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    },
    {
        "name": "list_workspace",
        "description": "List files and folders currently in the workspace. Call this BEFORE running mkdir, creating files, or modifying existing files — so you know what already exists and use correct paths.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file in the workspace. Provide a query to get only the relevant portion — do not read the whole file unnecessarily.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path":  {"type": "string", "description": "File path relative to workspace or absolute"},
                "query": {"type": "string", "description": "What you are looking for (used to extract relevant lines)"},
            },
            "required": ["path"],
        },
    },
]

INTERNAL_TOOL_NAMES: frozenset[str] = frozenset(t["name"] for t in INTERNAL_TOOLS)


# ── State ─────────────────────────────────────────────────────────────────────

@dataclass
class MicroState:
    """All loop state for one user request.

    Encoded as JSON in a SISYPHEAN_STATE thinking block so the server stays
    stateless and Claude Code carries the state forward in the history.

    summary — a compact running narrative of what has been accomplished across
    outer turns (bash commands run and their outcomes, plans created, searches
    done).  Carried in SISYPHEAN_STATE and shown to qwen3 at the start of each
    decide() call so it understands the full progress without having to parse
    raw conversation history.

    pending_write_steps — write_code/write_doc steps extracted from the plan;
    consumed one at a time by the subtask pipeline before the normal loop
    resumes.  Not serialised into SISYPHEAN_STATE (write stages complete
    within a single outer turn).
    """
    step: int = 0
    user_message: str = ""
    internal_messages: list[dict] = field(default_factory=list)
    summary: str = ""
    pending_write_steps: list[dict] = field(default_factory=list, repr=False)
    # Full plan produced by plan_task — all steps in order.
    # Kept so every subsequent decide() can see the complete plan and know its position.
    plan_steps: list[dict] = field(default_factory=list)
    # Index into plan_steps pointing at the step currently being executed.
    # Advances each time a plan step's primary tool completes.
    plan_step_idx: int = 0
    # Derived from plan_steps[plan_step_idx] — convenience field for decide().
    current_step_text: str = ""
    task_id: str = ""           # tracker ID — survives outer turns via SISYPHEAN_STATE


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
        # ── New pipeline ──────────────────────────────────────────────────────
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
        )

    async def process(
        self,
        user_message: str,
        raw_history: list[dict],
        available_tools: list[dict],
        memory_context: str = "",     # kept for signature compat
        system_context: str = "",
    ) -> LoopResponse:
        """Delegate to the new core pipeline."""
        if system_context:
            import re as _re
            m = _re.search(r"cwd:\s*([^|]+)", system_context)
            if m:
                self.workspace = m.group(1).strip()
                self._pipeline.workspace = self.workspace

        # No tools → direct generation (bypass pipeline)
        if not available_tools:
            text = await self._direct(user_message, raw_history, "")
            return LoopResponse(content=[{"type": "text", "text": text}], stop_reason="end_turn")

        return await self._pipeline.process(
            user_message=user_message,
            raw_history=raw_history,
            available_tools=available_tools,
            system_context=system_context,
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

    async def stream(
        self,
        message: str,
        history: list[dict],
        memory_context: str = "",
    ) -> AsyncIterator[str]:
        """Legacy: stream final text word by word."""
        text = await self.run(message, history, memory_context)
        words = text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            if i % 10 == 0:
                await asyncio.sleep(0.01)


# ── Micro-loop ────────────────────────────────────────────────────────────────

async def _micro_loop(
    state: MicroState,
    available_tools: list[dict],
    static_context: str,
    client,
    injector,
    knowledge_graph,
    raw_history: list[dict] | None = None,
    workspace: str = ".",
) -> LoopResponse:
    """Single outer-turn decision loop.

    Design
    ------
    Each call to _micro_loop() represents ONE outer Claude Code turn.  The
    inner for-loop handles synchronous internal tools (plan_task,
    search_knowledge, etc.) and think steps, each consuming one inner slot.
    After at most MAX_INTERNAL inner steps the loop is forced to produce an
    outer action (Bash / answer) and return — there is no unbounded while loop.

    Semantic history is computed once per outer turn via _semantic_history_summary()
    — a Jaccard search over session logs followed by one LLM summarization call.
    Only exchanges above the similarity threshold are included (no fixed K).
    This replaces raw_history passthrough; decide() receives a compact summary string.
    """
    import base64 as _b64

    all_tools = INTERNAL_TOOLS + available_tools
    internal_messages: list[dict] = list(state.internal_messages)

    # Register task with the tracker on first outer turn (step==0, no task_id yet)
    if not getattr(state, "task_id", None):
        state.task_id = _tracker.start_task(
            session_id=getattr(state, "session_id", ""),
            user_message=state.user_message,
        )

    # Compute semantic history once per outer turn — one LLM summarization call.
    # Kept separate from state.summary: semantic_history = cross-session context,
    # state.summary = running progress log for this specific task.
    semantic_history = await _semantic_history_summary(state.user_message, client)

    # Stall guard: track (tool_name, canonical_input) seen this turn
    _seen_calls: set[tuple[str, str]] = set()
    _stall_hint: str = ""
    _consecutive_thinks: int = 0
    _saved_lessons: set[str] = set()   # dedup failure lessons within this session

    # Save failure lessons from any bash results already in state (from continuation)
    for _imsg in state.internal_messages:
        if _imsg.get("tool") == "bash_result":
            _lesson = _extract_failure_lesson("bash_result", {}, _imsg.get("result", ""))
            if _lesson:
                _save_lesson(_lesson, injector, knowledge_graph, _saved_lessons)

    while state.step < MAX_STEPS:
        # ── Write stage dispatch ───────────────────────────────────────────────
        # If the plan produced write_code/write_doc steps, consume them via the
        # subtask pipeline before returning to the normal decide() loop.
        if state.pending_write_steps:
            ws = state.pending_write_steps.pop(0)
            file_type = "code" if ws["type"] == "write_code" else "doc"
            logger.info("micro-loop: write stage  type=%s  goal=%r", file_type, ws["text"][:60])
            log_event("write", f"Write {file_type}", ws["text"][:120], session_id=state.task_id)
            try:
                stage_result = await run_write_step(
                    client=client,
                    stage_goal=ws["text"],
                    file_type=file_type,
                    workspace=workspace or ".",
                )
                summary_line = f"Wrote {file_type}: {stage_result.summary} → {stage_result.written_path}"
                log_event("write", f"Done {file_type}", stage_result.summary[:120], session_id=state.task_id)
            except Exception as exc:
                summary_line = f"Write stage failed: {exc}"
                log_event("error", "Write failed", str(exc)[:120], session_id=state.task_id)
                logger.warning("micro-loop: write stage error: %s", exc)
            internal_messages.append({
                "tool": "write_result",
                "input": ws,
                "result": summary_line,
            })
            state.summary = _extend_summary(state.summary, "write_result", ws, summary_line)
            state.step += 1
            # After the last write step, nudge the model to answer rather than loop
            if not state.pending_write_steps:
                _stall_hint = "All planned steps are complete. Give your final answer now."
            continue

        import time as _t
        _t0 = _t.time()
        action = await decide(
            user_message=state.user_message,
            internal_messages=internal_messages,
            available_tools=available_tools,  # outer tools only — INTERNAL_TOOLS handled by action_word checks
            static_context=static_context,
            client=client,
            step=state.step,
            budget=MAX_STEPS,
            semantic_history=semantic_history,
            summary=state.summary,
            hint=_stall_hint,
            workspace=workspace,
            current_step_text=state.current_step_text,
            plan_steps=state.plan_steps,
            plan_step_idx=state.plan_step_idx,
        )
        _elapsed_ms = int((_t.time() - _t0) * 1000)
        _stall_hint = ""  # consume hint
        _consecutive_thinks = 0  # reset on any concrete decision
        state.step += 1

        # ── Log the decision ─────────────────────────────────────────────────
        _sid = state.task_id
        if action.type == "answer":
            log_event("answer", "Answer", (action.content or "")[:120], session_id=_sid,
                      data={"action": "answer", "step": state.step, "budget": MAX_STEPS,
                            "content_preview": (action.content or "")[:300], "elapsed_ms": _elapsed_ms})
        elif action.tool_name in INTERNAL_TOOL_NAMES:
            _inp_preview = str(list(action.tool_input.values())[0])[:80] if action.tool_input else ""
            log_event("tool", action.tool_name, _inp_preview, session_id=_sid)
            log_event("llm", f"decide → {action.tool_name}", _inp_preview, session_id=_sid,
                      data={"action": action.tool_name, "input": _inp_preview,
                            "step": state.step, "budget": MAX_STEPS,
                            "elapsed_ms": _elapsed_ms})
        elif action.tool_name:
            _cmd = action.tool_input.get("command", str(action.tool_input)[:80])
            log_event("bash", action.tool_name, str(_cmd)[:120], session_id=_sid,
                      data={"action": "bash", "tool": action.tool_name,
                            "command": str(_cmd)[:300], "step": state.step,
                            "budget": MAX_STEPS, "elapsed_ms": _elapsed_ms})

        # ── Answer ────────────────────────────────────────────────────────────
        if action.type == "answer":
            text = action.content or ""
            if not text.strip() and internal_messages:
                # Fall back to the last bash output — but NEVER use error/warning
                # messages (empty command, permission denied, etc.) as the answer.
                _error_markers = (
                    "bash called without", "please be more specific",
                    "permission denied", "command not found", "error:",
                )
                bash_outputs = [
                    m["result"] for m in internal_messages
                    if m.get("tool") == "bash_result"
                    and m.get("result", "").strip()
                    and not any(
                        marker in m.get("result", "").lower()
                        for marker in _error_markers
                    )
                ]
                if bash_outputs:
                    text = bash_outputs[-1][:800]
            _tracker.add_inline_step(
                getattr(state, "task_id", ""), "answer", "Final answer",
                input_text="", output_text=text[:200], status="done",
            )
            _tracker.finish_task(getattr(state, "task_id", ""), "done")
            blocks: list[dict] = []
            visible = [m for m in internal_messages if m.get("tool") != "bash_result"]
            if visible:
                blocks.append(_reasoning_block(visible))
            blocks.append({"type": "text", "text": text})
            return LoopResponse(content=blocks, stop_reason="end_turn")

        # ── Think ─────────────────────────────────────────────────────────────
        if action.tool_name == "think" or action.type == "think":
            reasoning = action.tool_input.get("reasoning") or action.content or ""
            _consecutive_thinks += 1

            # Hard cap: 3 consecutive thinks without action → force next step
            if _consecutive_thinks >= 3:
                logger.warning("micro-loop: %d consecutive thinks — forcing action", _consecutive_thinks)
                _stall_hint = "Stop reasoning in circles. Take a concrete action or give your final answer now."
                continue

            # Dedup: skip if this reasoning overlaps heavily with a recent think
            recent_think_texts = [
                m["result"] for m in internal_messages if m.get("tool") == "think"
            ]
            if recent_think_texts and _think_is_duplicate(reasoning, recent_think_texts):
                logger.debug("micro-loop: duplicate think — skipping, nudging to act")
                _stall_hint = "You've already reasoned about this. Take action on what you know."
                continue

            internal_messages.append({"tool": "think", "input": {}, "result": reasoning[:300]})
            logger.info("micro-loop: think → %d chars", len(reasoning))
            continue

        # ── Internal tool ─────────────────────────────────────────────────────
        if action.tool_name in INTERNAL_TOOL_NAMES:
            # Cross-turn duplicate guard for web_search: check state.summary
            if action.tool_name == "web_search":
                _q = action.tool_input.get("query", "").strip().lower()
                if _q and f"web search '{_q[:35]}" in state.summary.lower():
                    logger.debug("micro-loop: cross-turn duplicate web_search '%s' — skipping", _q)
                    _stall_hint = f"Already searched for '{_q}' (see [Progress]). Use those results and answer."
                    continue

            # Stall guard: only one plan_task per turn
            if action.tool_name == "plan_task" and any(
                m.get("tool") == "plan_task" for m in internal_messages
            ):
                logger.warning("micro-loop: second plan_task call — skipping, forcing execute")
                _stall_hint = (
                    "A plan already exists (see [Plan] above). "
                    "Execute the current step now — call bash to run the command, or answer."
                )
                continue

            # Stall guard: skip duplicate calls within this turn
            call_key = (action.tool_name, json.dumps(action.tool_input, sort_keys=True))
            if call_key in _seen_calls:
                logger.warning("micro-loop: duplicate %s call — skipping, nudging to answer", action.tool_name)
                _stall_hint = "You already tried that. Use what you have and answer now."
                continue
            _seen_calls.add(call_key)

            _inp_str = str(list(action.tool_input.values())[0])[:120] if action.tool_input else ""
            _tracker.add_inline_step(
                getattr(state, "task_id", ""), action.tool_name,
                _inp_str, input_text=_inp_str, status="running",
            )
            result = await _handle_internal_tool(
                tool_name=action.tool_name,
                tool_input=action.tool_input,
                client=client,
                injector=injector,
                knowledge_graph=knowledge_graph,
                workspace=workspace,
                user_message=getattr(state, "user_message", ""),
            )
            # Update the last inline step to done with output
            tid = getattr(state, "task_id", "")
            task_steps = _tracker._tasks.get(tid, {}).get("steps", [])
            if task_steps:
                task_steps[-1]["status"] = "done"
                task_steps[-1]["output"] = result[:200]
                import time as _t; task_steps[-1]["finished_at"] = round(_t.time())
            # After plan_task: extract write stages into pending_write_steps so
            # the subtask pipeline handles them instead of the normal decide() loop.
            # Only queue write steps that have no preceding research/verify steps —
            # if research comes first, let decide() handle the whole plan so the
            # web search runs before the writer fires.
            if action.tool_name == "plan_task":
                plan_steps = _parse_plan_steps(result)
                _tracker.set_plan(getattr(state, "task_id", ""), plan_steps)
                log_event("plan", f"plan: {len(plan_steps)} steps", result[:200],
                          session_id=state.task_id,
                          data={"action": "plan", "steps": plan_steps,
                                "step_count": len(plan_steps)})
                has_prior_research = False
                write_steps = []
                for s in plan_steps:
                    if s.get("type") in ("research", "verify", "reflect"):
                        has_prior_research = True
                    elif s.get("type") in ("write_code", "write_doc") and not has_prior_research:
                        write_steps.append(s)
                if write_steps:
                    state.pending_write_steps = write_steps
                    logger.info(
                        "micro-loop: %d write step(s) queued from plan (no research prereqs)",
                        len(write_steps),
                    )
                # Store full plan on state so every decide() call can see all steps
                # and know exactly where in the plan we are.
                exec_steps = [s for s in plan_steps if s.get("type") not in ("write_code", "write_doc")]
                state.plan_steps = exec_steps
                state.plan_step_idx = 0
                if exec_steps:
                    state.current_step_text = f"[{exec_steps[0]['type']}] {exec_steps[0]['text']}"
                    logger.debug("micro-loop: plan loaded, step 1/%d: %r", len(exec_steps), state.current_step_text[:80])

            # After a research/verify/reflect tool completes, advance to next plan step.
            # This keeps current_step_text pointing at what needs doing NOW, not what was just done.
            elif action.tool_name in ("search_knowledge", "search_history", "web_search") \
                    and state.plan_steps:
                next_idx = state.plan_step_idx + 1
                if next_idx < len(state.plan_steps):
                    state.plan_step_idx = next_idx
                    s = state.plan_steps[next_idx]
                    state.current_step_text = f"[{s['type']}] {s['text']}"
                    logger.debug("micro-loop: advanced to step %d/%d: %r",
                                 next_idx + 1, len(state.plan_steps), state.current_step_text[:80])
                else:
                    # All steps done — clear so model isn't nudged to repeat
                    state.current_step_text = ""
                    logger.debug("micro-loop: all plan steps done, clearing current_step_text")

            internal_messages.append({
                "tool": action.tool_name,
                "input": action.tool_input,
                "result": result,
            })
            # Auto-save failure lessons to the knowledge graph
            lesson = _extract_failure_lesson(action.tool_name, action.tool_input, result)
            if lesson:
                _save_lesson(lesson, injector, knowledge_graph, _saved_lessons)
            # Extend the running summary so future turns remember what was done
            state.summary = _extend_summary(state.summary, action.tool_name, action.tool_input, result)
            logger.info("micro-loop: internal %s → %d chars", action.tool_name, len(result))
            # After a successful web_search, nudge toward answer — prevents the model
            # from confusing the internal tool name with a bash command on the next step.
            # Only set hint when actual results were returned (not on error strings).
            if action.tool_name == "web_search" and result.startswith("### Web Search Results"):
                _stall_hint = "Web search complete. Synthesize the results above and give your final answer now."
                state.current_step_text = ""  # step done, clear so model doesn't loop
            # After a memory search with no results, direct to web_search for factual questions
            if action.tool_name == "search_knowledge" and "(No knowledge found)" in result:
                _stall_hint = (
                    "Memory has nothing on this. "
                    "For factual or research questions, use web_search to find a reliable answer. "
                    "Do not guess or make up facts."
                )
            continue

        # ── Outer tool (Bash, etc.) — return to Claude Code ───────────────────
        _cmd_str = action.tool_input.get("command", str(action.tool_input)[:80])
        _tracker.add_inline_step(
            getattr(state, "task_id", ""), "bash", str(_cmd_str)[:100],
            input_text=str(_cmd_str)[:200], status="running",
        )
        # SISYPHEAN_STATE carries step + user_message + summary.
        # The full bash results live in Claude Code's conversation history.
        # The summary gives the model a compact cross-turn narrative without
        # requiring it to re-parse the raw history on every outer turn.
        state_payload = {
            "step": state.step,
            "user_message": state.user_message,
            "summary": state.summary,
            "current_step_text": state.current_step_text,
            "task_id": getattr(state, "task_id", ""),
        }
        state_b64 = _b64.b64encode(json.dumps(state_payload).encode()).decode()
        thinking_block = {"type": "thinking", "thinking": f"{_STATE_PREFIX}{state_b64}"}

        response_blocks: list[dict] = [thinking_block]
        visible = [m for m in internal_messages if m.get("tool") != "bash_result"]
        if visible:
            response_blocks.append(_reasoning_block(visible))

        tool_id = action.tool_id or f"toolu_{uuid.uuid4().hex[:16]}"
        response_blocks.append({
            "type": "tool_use",
            "id": tool_id,
            "name": action.tool_name,
            "input": action.tool_input,
        })
        return LoopResponse(response_blocks, stop_reason="tool_use")

    # MAX_STEPS exceeded
    logger.warning("micro-loop: MAX_STEPS (%d) exceeded — forcing fallback answer", MAX_STEPS)
    system = SYSTEM if not static_context else f"{SYSTEM}\n\n{static_context}"
    try:
        result = await client.generate(
            [{"role": "system", "content": system},
             {"role": "user", "content": state.user_message + "\nBriefly answer from what you know."}],
            max_tokens=512, temperature=0.5, stream=False, thinking=False,
        )
        fallback_text = result["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        fallback_text = f"(Could not complete after {MAX_STEPS} steps: {exc})"
    return LoopResponse(content=[{"type": "text", "text": fallback_text}], stop_reason="end_turn")


# ── Internal tool handlers ────────────────────────────────────────────────────

async def _handle_internal_tool(
    tool_name: str,
    tool_input: dict,
    client,
    injector,
    knowledge_graph,
    workspace: str = ".",
    user_message: str = "",
) -> str:
    """Execute an internal tool and return a result string."""
    try:
        if tool_name == "plan_task":
            task = tool_input.get("task", "")
            # Let the decomposer decide — no regex overrides, no keyword routing.
            # The plan stage prompt already guides the model on what to do for each
            # task type (math → bash, greetings → memory, etc.).
            manifest = await decompose(task, "", client)
            lines = [
                f"{i + 1}. [{step.type}] {step.text}"
                for i, step in enumerate(manifest.steps)
            ]
            return "\n".join(lines) if lines else "(No plan generated)"

        if tool_name == "search_knowledge":
            query = tool_input.get("query", "")
            parts: list[str] = []
            if injector and hasattr(injector, "graph"):
                hits = injector.graph.search(
                    query, top_n=5, node_types=["fact", "concept", "project"]
                )
                if hits:
                    parts.extend(
                        f"- [{h['type']}] {h['label']}: {h['content'][:200]}"
                        for h in hits
                    )
            try:
                from engine.memory.retrieval import retrieve as _retrieve
                kg_text = _retrieve(query, top_n=5)
                if kg_text:
                    parts.append(kg_text)
            except Exception:
                pass
            return "\n".join(parts) if parts else "(No knowledge found)"

        if tool_name == "search_history":
            query = tool_input.get("query", "")
            return _search_history(query)

        if tool_name == "save_memory":
            note = tool_input.get("note", "")
            # Use knowledge_graph first; fall back to injector.graph only if they
            # are different objects (injector.graph IS knowledge_graph in normal setup,
            # so writing both would create duplicate nodes).
            _target_graph = knowledge_graph or (
                injector.graph if injector and hasattr(injector, "graph") else None
            )
            if _target_graph:
                try:
                    _target_graph.upsert_node("fact", note[:60], summary=note)
                except Exception as exc:
                    logger.debug("save_memory: graph write failed: %s", exc)
            return f"Saved: {note[:120]}"

        if tool_name == "web_search":
            query = (tool_input.get("query") or "").strip()  # handles None/null from model
            if not query:
                return "(web_search: empty query — provide a specific search term)"
            try:
                results = await _web_search(query, max_results=4)
                formatted = format_results(results)
                # Auto-save top result snippet so future sessions can find it
                if results:
                    snippet = results[0].snippet[:150] if results[0].snippet else ""
                    note = f"[web:{query[:40]}] {snippet}"
                    for _graph in (
                        knowledge_graph,
                        injector.graph if injector and hasattr(injector, "graph") else None,
                    ):
                        if _graph is None:
                            continue
                        try:
                            _graph.add_node(type="fact", label=f"search:{query[:40]}", content=note)
                        except Exception:
                            pass
                        break
                return formatted
            except Exception as exc:
                return f"(Web search failed: {exc})"

        if tool_name == "list_workspace":
            return _workspace_snapshot(workspace)

        if tool_name == "read_file":
            from pathlib import Path as _Path
            raw_path = tool_input.get("path", "")
            query = tool_input.get("query", "").strip().lower()
            try:
                p = _Path(raw_path)
                if not p.is_absolute():
                    p = _Path(workspace) / raw_path
                p = p.resolve()
                if not p.exists():
                    return f"(File not found: {raw_path})"
                content = p.read_text(encoding="utf-8", errors="replace")
                # If query given: extract relevant lines (±5 context around matches)
                if query and len(content) > 800:
                    lines = content.splitlines()
                    q_tokens = set(query.split())
                    hits: list[int] = []
                    for i, line in enumerate(lines):
                        if any(tok in line.lower() for tok in q_tokens):
                            hits.append(i)
                    if hits:
                        keep: set[int] = set()
                        for h in hits:
                            keep.update(range(max(0, h - 5), min(len(lines), h + 6)))
                        extracted = "\n".join(lines[i] for i in sorted(keep))
                        return f"[{p.name} — relevant to '{query}']\n{extracted[:2000]}"
                    # No hits — return head + tail summary
                    head = "\n".join(lines[:20])
                    tail = "\n".join(lines[-10:]) if len(lines) > 30 else ""
                    return f"[{p.name} — no match for '{query}', showing head/tail]\n{head}\n...\n{tail}"
                if len(content) > 2000:
                    content = content[:2000] + "\n...(truncated)"
                return content
            except Exception as exc:
                return f"(read_file error: {exc})"

    except Exception as exc:
        logger.warning("_handle_internal_tool %s failed: %s", tool_name, exc)
        return f"(Tool error: {exc})"

    return "(Unknown internal tool)"


_COMMAND_NOT_FOUND_RE = re.compile(r"(\S+):\s*command not found", re.IGNORECASE)


def _extract_failure_lesson(tool_name: str, tool_input: dict, result: str) -> str | None:
    """Return a saveable lesson string if this result indicates a recoverable failure.

    Lessons are stored in the knowledge graph so search_knowledge surfaces them
    in future sessions before the model repeats the same mistake.
    """
    result_lower = result.lower()

    # Bash: command not found — tell future sessions what's unavailable
    if tool_name == "bash_result":
        m = _COMMAND_NOT_FOUND_RE.search(result)
        if m:
            cmd = m.group(1)
            return (
                f"Shell command '{cmd}' is not available on this system. "
                f"Do not run '{cmd}' in bash — use an available alternative."
            )

    # Internal web_search: empty results — nudge toward better queries
    if tool_name == "web_search" and "[no search results found]" in result_lower:
        query = tool_input.get("query", "").strip()
        if query:
            return (
                f"web_search returned no results for '{query[:60]}'. "
                f"Try a shorter or more specific query next time."
            )

    return None


def _save_lesson(lesson: str, injector, knowledge_graph, saved: set[str]) -> None:
    """Save a failure lesson to the knowledge graph, skipping duplicates."""
    key = lesson[:80]
    if key in saved:
        return
    saved.add(key)
    for graph in (
        knowledge_graph,
        injector.graph if injector and hasattr(injector, "graph") else None,
    ):
        if graph is None:
            continue
        try:
            graph.add_node(type="fact", label=lesson[:60], content=lesson)
            logger.info("auto-saved failure lesson: %s", lesson[:80])
            break
        except Exception as exc:
            logger.debug("_save_lesson failed: %s", exc)


def _extend_summary(summary: str, tool_name: str, tool_input: dict, result: str) -> str:
    """Append a one-line entry to the running progress summary.

    The summary is shown to qwen3 at the start of every decide() call so it
    knows what has been accomplished without having to re-read raw history.
    Each entry is intentionally short (~80 chars) to stay within token budget.
    """
    result_preview = result.strip()[:80].replace("\n", " ")
    if tool_name == "plan_task":
        task = tool_input.get("task", "")[:50]
        entry = f"Planned '{task}': {result_preview}"
    elif tool_name == "search_knowledge":
        query = tool_input.get("query", "")[:40]
        entry = f"Searched memory '{query}': {result_preview}"
    elif tool_name == "web_search":
        query = tool_input.get("query", "")[:40]
        entry = f"Web search '{query}': {result_preview}"
    elif tool_name == "save_memory":
        entry = f"Saved to memory: {result_preview}"
    elif tool_name == "search_history":
        entry = f"Searched history: {result_preview}"
    elif tool_name == "think":
        entry = f"Reasoned: {result_preview}"
    else:
        entry = f"{tool_name}: {result_preview}"
    return (summary + "\n" + entry).strip()


def _parse_plan_steps(plan_text: str) -> list[dict]:
    """Extract step dicts from plan_task result text.

    plan_task returns lines like:
      1. [write_code] Write the reverse_string function...
      2. [verify] Run the tests...
    """
    steps = []
    for line in plan_text.splitlines():
        m = re.match(r"^\d+\.\s*\[(\w+)\]\s*(.+)", line.strip())
        if m:
            steps.append({"type": m.group(1), "text": m.group(2).strip()})
    return steps


def _extend_summary_bash(summary: str, command: str, result: str, is_error: bool) -> str:
    """Append a bash execution entry to the running summary."""
    result_preview = result.strip()[:80].replace("\n", " ")
    prefix = "FAILED" if is_error else "OK"
    cmd_short = command[:60]
    return (summary + f"\nRan `{cmd_short}` [{prefix}]: {result_preview}").strip()


_TOOL_LABEL: dict[str, str] = {
    "think":            "Thinking",
    "plan_task":        "Plan",
    "search_knowledge": "Memory",
    "search_history":   "History",
    "save_memory":      "Saved",
    "list_workspace":   "Workspace",
    "read_file":        "File",
    "write_result":     "Wrote",
    "web_search":       "Web",
    "bash_result":      "Result",
}

_STEP_TYPE_RE = re.compile(r"^\d+\.\s*\[\w+\]\s*")   # strips "1. [verify] "


def _reasoning_block(internal_messages: list[dict]) -> dict:
    """Build a visible block showing each internal stage: what went in and what came back.

    Format per stage:
      >Plan  ← "find all Python files and count lines"
        1. Search for *.py | 2. Run wc -l | 3. Summarise

    This gives the user a clear trace of what Sisyphean decided at each step
    and what result it got back, so failures are easy to spot.
    """
    lines = ["*Sisyphean stages:*"]
    for msg in internal_messages:
        tool = msg["tool"]
        label = _TOOL_LABEL.get(tool, tool)
        inp = msg.get("input", {})
        result_raw = str(msg.get("result", "")).strip()
        result_clean = _STEP_TYPE_RE.sub("", result_raw).replace("\n", " ")

        # Build a short "input summary" showing what was asked
        if tool == "plan_task":
            inp_line = inp.get("task", "")[:70]
        elif tool in ("search_knowledge", "search_history", "web_search"):
            inp_line = f'"{inp.get("query", "")[:60]}"'
        elif tool == "save_memory":
            inp_line = inp.get("note", "")[:60]
        elif tool == "think":
            inp_line = ""
        else:
            inp_line = str(inp)[:60] if inp else ""

        # Truncate result to fit
        if len(result_clean) > 120:
            result_clean = result_clean[:117].rsplit(" ", 1)[0] + "…"

        header = f">**{label}**" + (f"  ← {inp_line}" if inp_line else "")
        lines.append(header)
        if result_clean:
            lines.append(f"  {result_clean}")

    return {"type": "text", "text": "\n".join(lines)}


def _workspace_snapshot(workspace: str, max_entries: int = 40) -> str:
    """Return a compact tree of the workspace so the model knows what exists.

    Only shows files and directories — not sizes or dates. Capped at
    max_entries to stay within the context budget. The model should
    read this before running mkdir, touch, or write commands so it
    doesn't recreate things that already exist or write to wrong paths.
    """
    import os
    from pathlib import Path

    root = Path(workspace)
    if not root.exists():
        return f"Workspace {workspace} (empty — does not exist yet)"

    lines: list[str] = [f"Workspace: {workspace}"]
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden dirs
        dirnames[:] = [d for d in sorted(dirnames) if not d.startswith(".")]
        rel = Path(dirpath).relative_to(root)
        depth = len(rel.parts)
        if depth > 3:
            dirnames.clear()
            continue
        prefix = "  " * depth
        if depth > 0:
            lines.append(f"{prefix}{rel.parts[-1]}/")
        for fname in sorted(filenames):
            if fname.startswith("."):
                continue
            lines.append(f"{'  ' * (depth + 1)}{fname}")
            count += 1
            if count >= max_entries:
                lines.append("  ... (truncated)")
                return "\n".join(lines)

    if count == 0:
        lines.append("  (empty)")
    return "\n".join(lines)


def _search_history(query: str) -> str:
    """Search recent session logs for exchanges matching the query."""
    try:
        from engine.memory.session_log import _active_logs
    except ImportError:
        return "(History search unavailable)"

    query_lower = query.lower()
    query_tokens = set(query_lower.split())
    matches: list[str] = []

    for session_id, slog in list(_active_logs.items()):
        events = slog.all_events()
        for i, event in enumerate(events):
            content = event.data.get("content", "") or event.data.get("result", "")
            if not content:
                continue
            content_lower = content.lower()
            if any(tok in content_lower for tok in query_tokens if len(tok) > 3):
                snippet = content[:200].replace("\n", " ")
                matches.append(f"[{event.type}] {snippet}")
                if len(matches) >= 5:
                    break
        if len(matches) >= 5:
            break

    if not matches:
        return "(No matching history found)"
    return "\n".join(matches)


def _jaccard(a_tokens: set[str], b_tokens: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a_tokens or not b_tokens:
        return 0.0
    intersection = a_tokens & b_tokens
    union = a_tokens | b_tokens
    return len(intersection) / len(union)


# Common words that inflate Jaccard scores without carrying meaning.
# Filtering these prevents "meaning of life" matching "check system status"
# because both contain "check", "what", "is", etc.
_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "what", "how", "why", "when", "where",
    "who", "which", "that", "this", "it", "its", "i", "you", "me", "my",
    "your", "we", "our", "they", "their", "he", "she", "his", "her",
    "check", "run", "get", "set", "use", "make", "take", "give", "go",
    "see", "know", "think", "want", "need", "look", "come", "just", "also",
    "not", "no", "so", "if", "as", "up", "out", "about", "into", "then",
    "than", "more", "some", "any", "all", "one", "two", "new", "good",
    "please", "now", "here", "there", "like", "very", "well", "only",
})


def _meaningful_tokens(text: str) -> set[str]:
    """Extract tokens longer than 2 chars with stop words removed."""
    return {
        t for t in re.findall(r"[a-z0-9]+", text.lower())
        if len(t) > 2 and t not in _STOP_WORDS
    }


async def _semantic_history_summary(
    user_message: str,
    client,
    min_similarity: float = 0.25,
) -> str:
    """Retrieve session history relevant to the current task and summarize with LLM.

    Scores each past user/assistant exchange by Jaccard similarity against
    user_message. Only exchanges above min_similarity are included — no fixed K.
    Stop words and short tokens are excluded before scoring so generic words
    like "check", "what", "is" don't create false matches.

    Returns empty string when nothing relevant is found (saves the LLM call).
    The current turn's user_message is excluded from results (it's always there).
    """
    try:
        from engine.memory.session_log import _active_logs
    except ImportError:
        return ""

    query_tokens = _meaningful_tokens(user_message)
    if len(query_tokens) < 2:
        return ""

    # Current message text for exact-match exclusion
    current_stripped = user_message.strip()[:120]

    relevant: list[str] = []

    for slog in list(_active_logs.values()):
        events = slog.all_events()
        i = 0
        while i < len(events):
            ev = events[i]

            if ev.type == "user_message":
                content = ev.data.get("content", "")
                # Skip the current turn's message
                if content.strip()[:120] == current_stripped:
                    i += 1
                    continue
                content_tokens = _meaningful_tokens(content)
                if _jaccard(query_tokens, content_tokens) >= min_similarity:
                    entry = f"User: {content[:150]}"
                    # Attach the following assistant reply if present
                    for j in range(i + 1, min(i + 3, len(events))):
                        if events[j].type == "assistant_message":
                            entry += f"\nAssistant: {events[j].data.get('content', '')[:150]}"
                            break
                    relevant.append(entry)

            elif ev.type == "tool_call":
                args = ev.data.get("arguments", {})
                cmd = str(args.get("command") or args.get("query") or "")
                if cmd:
                    cmd_tokens = _meaningful_tokens(cmd)
                    if _jaccard(query_tokens, cmd_tokens) >= min_similarity:
                        # Include adjacent tool_result if any
                        result_snippet = ""
                        if i + 1 < len(events) and events[i + 1].type == "tool_result":
                            result_snippet = events[i + 1].data.get("result", "")[:100]
                        entry = f"Ran: {cmd[:100]}"
                        if result_snippet:
                            entry += f" → {result_snippet}"
                        relevant.append(entry)

            i += 1

    if not relevant:
        return ""

    relevant_text = "\n\n".join(relevant[:8])

    # One LLM call to distill relevant exchanges into a compact summary
    try:
        msgs = [
            {"role": "system", "content": "Summarize past work concisely. Be specific about commands and results."},
            {"role": "user", "content": (
                f"Current task: {user_message[:200]}\n\n"
                f"Relevant past exchanges:\n{relevant_text}\n\n"
                "Summarize in 2-3 short sentences what was done that is relevant to this task. "
                "Name specific commands run and key results."
            )},
        ]
        result = await client.generate(
            msgs, max_tokens=120, temperature=0.1, stream=False, thinking=False,
        )
        summary = (result["choices"][0]["message"]["content"] or "").strip()
        return summary
    except Exception as exc:
        logger.debug("_semantic_history_summary LLM call failed: %s", exc)
        # Fallback: return raw relevant text trimmed
        return relevant_text[:300]


# ── Static context builder ────────────────────────────────────────────────────

def _build_static_context(injector) -> str:
    """Read soul + user nodes from graph directly — no LLM call, no vector search.

    Returns a short string capped at ~300 tokens worth of content.
    """
    if not injector or not hasattr(injector, "graph"):
        return ""
    parts: list[str] = []

    policy_nodes = injector.graph.all_nodes(node_type="policy") or injector.graph.all_nodes(node_type="soul")
    if policy_nodes:
        parts.append("### Engine Policy\n" + "\n".join(n["content"] for n in policy_nodes[:3]))

    user_nodes = [
        n for n in injector.graph.all_nodes(node_type="user")
        if n.get("content", "").strip() and "to be filled" not in n["content"]
    ]
    if user_nodes:
        parts.append("### User\n" + "\n".join(n["content"] for n in user_nodes[:5]))

    return "\n\n".join(parts)


# ── Continuation extraction ───────────────────────────────────────────────────

def _extract_continuation(history: list[dict]) -> MicroState | None:
    """Check if last user message has tool_result blocks and extract MicroState.

    Returns MicroState if this is a continuation, None if it's a new request.

    Rules:
    - Last user message must contain tool_result blocks (mid-task continuation).
    - If it ALSO contains new user text (e.g. user typed "hi" while a tool was
      running), that new message takes priority — return None so the caller
      starts a fresh loop for the new request instead of resuming the old one.
    - The tool_result content is appended to internal_messages so the model
      knows what the bash command produced before deciding what to do next.
    """
    if not history:
        return None

    last = history[-1]
    if last.get("role") != "user":
        return None

    content = last.get("content", "")
    if isinstance(content, str):
        return None

    tool_results = [
        b for b in content
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    if not tool_results:
        return None

    # If the user also typed new text alongside the tool_result, start fresh —
    # their new message takes priority over resuming the previous task.
    new_text = [
        b for b in content
        if isinstance(b, dict) and b.get("type") == "text"
        and b.get("text", "").strip()
    ]
    if new_text:
        logger.debug(
            "_extract_continuation: new user text alongside tool_result — starting fresh"
        )
        return None

    # Extract MicroState from the preceding assistant message's thinking block
    preceding = history[-2] if len(history) >= 2 else None
    if not preceding or preceding.get("role") != "assistant":
        return None

    msg_content = preceding.get("content", [])
    if not isinstance(msg_content, list):
        return None

    for block in msg_content:
        if not isinstance(block, dict) or block.get("type") != "thinking":
            continue
        thinking = block.get("thinking", "")
        idx = thinking.find(_STATE_PREFIX)
        if idx == -1:
            continue
        raw = thinking[idx + len(_STATE_PREFIX):].strip()
        try:
            import base64 as _b64
            # Try base64 decode first (new format); fall back to raw JSON (legacy)
            try:
                data = json.loads(_b64.b64decode(raw.split()[0]).decode())
            except Exception:
                end = raw.rfind("}") + 1
                data = json.loads(raw[:end] if end > 0 else raw)

            # Restore summary from SISYPHEAN_STATE.
            # internal_messages is no longer stored in state — the summary is
            # the compact cross-turn narrative, and bash results live in
            # semantic_history summary which decide() receives separately.
            restored_summary = data.get("summary", "")

            # Legacy: if old state has internal_messages but no summary, ignore them
            state = MicroState(
                step=data.get("step", 0),
                user_message=data.get("user_message", ""),
                internal_messages=[],
                summary=restored_summary,
                current_step_text=data.get("current_step_text", ""),
                task_id=data.get("task_id", ""),
            )

            # Extract the bash command that was run (from preceding tool_use block)
            bash_command = ""
            if msg_content:
                for blk in msg_content:
                    if isinstance(blk, dict) and blk.get("type") == "tool_use":
                        bash_command = blk.get("input", {}).get("command", "")
                        break

            # Append current tool_results as bash_result for error detection,
            # and extend the summary with what the command produced.
            from engine.translation.executor import _looks_like_error
            for tr in tool_results:
                tr_content = tr.get("content", "")
                if isinstance(tr_content, list):
                    tr_content = " ".join(
                        b.get("text", "") for b in tr_content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                result_text = str(tr_content).strip()[:300] or "(no output)"
                is_error = _looks_like_error(result_text)
                if is_error:
                    result_text = f"ERROR: {result_text}"
                state.internal_messages.append({
                    "tool": "bash_result",
                    "input": {},
                    "result": result_text,
                })
                state.summary = _extend_summary_bash(
                    state.summary, bash_command, result_text, is_error
                )
            return state
        except Exception as exc:
            logger.debug("_extract_continuation: parse failed: %s", exc)

    return None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _flatten_history(raw_history: list[dict], max_turns: int = 6) -> list[dict]:
    """Convert the last max_turns messages to flat string format.

    Only used by _direct() for simple Q&A.  Skips thinking, tool_use, and
    tool_result blocks — only keeps text blocks from the actual conversation.
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


def _recent_history_summary(raw_history: list[dict], max_pairs: int = 2) -> str:
    """Build a brief processed summary of the last max_pairs conversation turns.

    Purpose: give the model enough continuity context without flooding its
    limited context window with full raw history.  Tool-use/thinking blocks
    are stripped; only clean Q&A text survives.  Each entry is capped at
    100 chars so the total overhead is tiny (~400 chars for 2 pairs).

    The last message (current user question) is intentionally excluded —
    it arrives as user_message in the loop already.
    """
    if not raw_history or len(raw_history) < 2:
        return ""

    pairs: list[tuple[str, str]] = []   # (user_text, assistant_text)
    # Walk history in reverse (skip the last message = current request)
    pending_a = ""
    for msg in reversed(raw_history[:-1]):
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "assistant":
            if isinstance(content, list):
                text_parts = [
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                    and not b.get("text", "").startswith("*Sisyphean:")  # skip reasoning blocks
                ]
                pending_a = " ".join(text_parts).strip()[:100]
            else:
                pending_a = str(content).strip()[:100]

        elif role == "user" and pending_a:
            if isinstance(content, list):
                u_parts = [
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                u_text = " ".join(u_parts).strip()
            else:
                u_text = str(content).strip()
            # Skip slash commands and tool-result-only turns
            if u_text and not u_text.startswith("/"):
                # Strip any <system-reminder> injections
                u_text = re.sub(r"<system-reminder>.*?</system-reminder>", "", u_text, flags=re.DOTALL).strip()
                pairs.append((u_text[:100], pending_a))
                pending_a = ""
                if len(pairs) >= max_pairs:
                    break

    if not pairs:
        return ""

    # Only show prior questions, not answers — showing answers causes the model
    # to repeat stale responses instead of re-running tools for fresh data.
    lines = ["Prior questions:"]
    for u, _ in reversed(pairs):  # chronological order, answers intentionally dropped
        lines.append(f"  - {u}")
    return "\n".join(lines)


def _think_is_duplicate(reasoning: str, recent_thinks: list[str], threshold: float = 0.55) -> bool:
    """Return True if reasoning overlaps too heavily with a recent think result.

    Uses Jaccard similarity on word tokens. A threshold of 0.55 means the model
    is essentially restating the same thought — skip it and force action instead.
    Only checks the last 3 think results to avoid false positives from unrelated
    earlier reasoning.
    """
    tokens = set(re.findall(r"[a-z0-9]+", reasoning.lower()))
    if len(tokens) < 5:
        return False
    for prev in recent_thinks[-3:]:
        prev_tokens = set(re.findall(r"[a-z0-9]+", prev.lower()))
        if not prev_tokens:
            continue
        union = tokens | prev_tokens
        if union and len(tokens & prev_tokens) / len(union) >= threshold:
            return True
    return False


_CONVERSATIONAL_RE = re.compile(
    r"^(hi|hello|hey|howdy|greetings|good\s+morning|good\s+afternoon|good\s+evening"
    r"|thanks|thank\s+you|cheers|bye|goodbye|see\s+you|ok|okay|alright|sure|got\s+it"
    r"|cool|great|nice|sounds\s+good|perfect|yep|yes|no|nope|what['’s]*\s+up"
    r"|how\s+are\s+you|how\s+do\s+you\s+do)[?.!,]*$",
    re.IGNORECASE,
)


def _is_conversational(message: str) -> bool:
    """Return True for short social messages that don't need tool execution."""
    stripped = message.strip()
    # Short (≤8 words) AND matches a conversational pattern
    if len(stripped.split()) > 8:
        return False
    return bool(_CONVERSATIONAL_RE.match(stripped))


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
