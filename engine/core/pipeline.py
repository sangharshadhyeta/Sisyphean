"""Sisyphean core pipeline.

Flow:
  Router   → soul section search (no LLM)
  Planner  → split into sub-tasks, plan each (2 LLM calls total for simple tasks)
  Executor → run each planned step (internal tools handled here; outer tools
             returned as tool_use blocks to Claude Code)
  Consolidator → assemble final answer (1 LLM call, plain text)

State between outer tool calls is encoded in a thinking block as:
  PIPELINE_STATE:<json>
so the server stays stateless and Claude Code manages the conversation history.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time as _time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from engine.core.synthesizer import synthesize
from engine.core.recall import Recall
from engine.core.context_extractor import extract_for_task, filter_tools_for_task
from engine.translation.planner import split_deep, plan_task, parse_format_response
from engine.translation.web_search import search as _web_search, format_results
import engine.task_tracker as _tracker
from engine.activity import log_event as _log

logger = logging.getLogger(__name__)

_STATE_PREFIX = "PIPELINE_STATE:"

# Tools whose results should trigger dynamic replanning of subsequent steps.
# After these run, pre-guessed follow-up steps are replaced with steps derived
# from what was actually found — the result IS the reasoning.
_INFO_TOOLS = frozenset({"web_search", "search_memory", "search_knowledge"})

# Outer tools (returned to Claude Code harness) whose results require replanning.
# Read/Grep/Glob → find/read file, then generate edit steps.
# WebSearch/WebFetch → delegate to _replan_after_search for execution steps.
_OUTER_INFO_TOOLS = frozenset({"read", "glob", "grep", "websearch", "webfetch"})

_EDIT_SYSTEM = """\
Given the task and current file, decide the minimal change needed. Output ONE JSON object only.

For small targeted changes (URL, variable, import, small block):
{"mode": "edit", "old": "exact text to replace", "new": "replacement text"}

For major rewrites (different logic, new structure, significant feature):
{"mode": "write"}

Rules: "old" must be an EXACT substring of the file. Include 1-2 surrounding lines for uniqueness.
No prose, no markdown, no explanation — only the JSON."""

_REFRAME_SYSTEM = """\
A search query returned poor or empty results. Rewrite it as a better, more specific query.
Output only the new query — no explanation, no quotes, no punctuation at the end.
If the original query cannot be meaningfully improved, output a single dash: -"""

_PROCEDURE_EXTRACT_SYSTEM = """\
From web search results, extract ONLY reusable procedural knowledge: exact commands,
installation steps, configuration methods, or approaches that will remain valid in the future.

Output the extracted method as plain text — exact commands or steps only.
If the results contain purely time-sensitive data (current system status, live metrics,
today's news, real-time prices, current values of anything), output a single dash: -
Extract the METHOD of how to do something, never the current state of something."""

_REPLAN_SYSTEM = """\
A task is being completed. A search just returned relevant information.
Derive ONLY the execution steps from what was found — copy commands, package names,
and values VERBATIM from the search result. Do not guess or paraphrase.

Reply as JSON: {"steps": "toolname:exact input | toolname:exact input"}

Available tools:
  bash        — run a shell command exactly as described in the result
  web_search  — search for more specific information if still needed
  save_memory — save a key fact that was discovered

If the search result already fully answers the task and no execution is needed,
reply: {"steps": ""}"""


def _result_quality(result: dict) -> str:
    """Grade a step result: 'good' | 'weak' | 'empty' | 'error'."""
    content = str(result.get("result", "")).strip()
    if not content or len(content) < 8:
        return "empty"
    lower = content.lower()
    if any(k in lower for k in ("traceback", "error:", "exception", "search failed",
                                 "no results", "not found", "command not found")):
        return "error"
    if len(content.split()) < 15:
        return "weak"
    return "good"


# ── Pipeline state ────────────────────────────────────────────────────────────

@dataclass
class PipelineState:
    query: str = ""
    soul_section: str = ""
    user_prefs: str = ""
    # All planned sub-tasks: list of {task: str, steps: [{tool, input}]}
    sub_tasks: list[dict] = field(default_factory=list)
    current_task_idx: int = 0
    current_step_idx: int = 0
    # Accumulated results across all steps
    results: list[dict] = field(default_factory=list)
    # Query-relevant context for the final synthesize call (pre-computed by ContextRouter)
    synthesis_ctx: str = ""
    # Dashboard task id (empty = not tracked yet)
    task_id: str = ""
    # Absolute path of the Claude Code project directory (from cwd in system prompt)
    project_dir: str = ""

    def to_json(self) -> str:
        return json.dumps({
            "q":    self.query,
            "ss":   self.soul_section[:400],
            "up":   self.user_prefs[:200],
            "st":   self.sub_tasks,
            "ti":   self.current_task_idx,
            "si":   self.current_step_idx,
            "res":  self.results,
            "sctx": self.synthesis_ctx[:600],
            "tid":  self.task_id,
            "pd":   self.project_dir,
        })

    @classmethod
    def from_json(cls, s: str) -> "PipelineState":
        d = json.loads(s)
        obj = cls()
        obj.query             = d.get("q",   "")
        obj.soul_section      = d.get("ss",  "")
        obj.user_prefs        = d.get("up",  "")
        obj.sub_tasks         = d.get("st",  [])
        obj.current_task_idx  = d.get("ti",  0)
        obj.current_step_idx  = d.get("si",  0)
        obj.results           = d.get("res", [])
        obj.synthesis_ctx     = d.get("sctx", "")
        obj.task_id           = d.get("tid", "")
        obj.project_dir       = d.get("pd",  "")
        return obj


# ── LoopResponse (imported by translation/loop.py) ────────────────────────────

@dataclass
class LoopResponse:
    content: list[dict]
    stop_reason: str


# ── Pipeline ──────────────────────────────────────────────────────────────────

class Pipeline:

    def __init__(
        self,
        client,
        policy_path: Path,
        prefs_path: Path,
        knowledge_graph=None,
        workspace: str = "",
    ) -> None:
        self.client  = client
        self.soul_path  = policy_path  # kept as soul_path internally for compatibility
        self.prefs_path = prefs_path
        self.graph   = knowledge_graph
        self.workspace = workspace
        self.recall  = Recall(graph=knowledge_graph, workspace=workspace or ".")

    async def process(
        self,
        user_message: str,
        raw_history: list[dict],
        available_tools: list[dict],
        system_context: str = "",
    ) -> LoopResponse:
        """Entry point. Handles fresh requests and tool_result continuations."""

        # ── Continuation: tool_result came back ──────────────────────────────
        state = _extract_state(raw_history)
        if state:
            return await self._continue(state, raw_history, available_tools)

        # ── Fresh request ─────────────────────────────────────────────────────
        if not user_message:
            return LoopResponse(content=[{"type": "text", "text": ""}], stop_reason="end_turn")

        # Extract session_id from system_context ("session: <id> | ...")
        session_id = ""
        if system_context:
            m = re.search(r"session:\s*([a-f0-9\-]+)", system_context)
            if m:
                session_id = m.group(1)

        # Extract project context (CLAUDE.md + env facts) from the harness's system prompt.
        # Already processed by translate_system() — small and passed whole to every stage.
        project_ctx = ""
        if system_context:
            m = re.search(r"\[Project context\]\n(.*)", system_context, re.DOTALL)
            if m:
                project_ctx = m.group(1).strip()
        env_line = system_context.split("\n")[0] if system_context else ""
        if env_line and "[Project context]" not in env_line:
            project_ctx = "\n\n".join(p for p in (env_line, project_ctx) if p)

        # Extract working directory for the Write tool's file_path resolution.
        # translate_system() puts "cwd: /path/to/project" in the first line.
        project_dir = ""
        m_cwd = re.search(r"cwd:\s*([^|]+)", env_line)
        if m_cwd:
            project_dir = m_cwd.group(1).strip().replace("/", os.sep)

        # Format full conversation history — gemma4 has 131k context, let it decide
        # what's relevant rather than pre-filtering with bigrams.
        history_text = _format_history(raw_history)

        return await self._start(user_message, available_tools, session_id=session_id,
                                 history_text=history_text, project_ctx=project_ctx,
                                 project_dir=project_dir)

    # ── Fresh request ─────────────────────────────────────────────────────────

    async def _start(self, query: str, available_tools: list[dict], session_id: str = "",
                     history_text: str = "", project_ctx: str = "",
                     project_dir: str = "") -> LoopResponse:
        from engine.policy.router import load_user_prefs

        task_id = _tracker.start_task(session_id=session_id, user_message=query)

        soul_section = self.soul_path.read_text(encoding="utf-8").strip() if self.soul_path.exists() else ""
        user_prefs   = load_user_prefs(self.prefs_path)

        logger.info("pipeline.start: query=%r history=%d chars", query[:60], len(history_text))

        # ── Stage 1: Extract top-level context before splitting ───────────────
        # The extractor runs first so the splitter understands what the conversation
        # is about before deciding how to decompose the query.
        _tracker.tree_context_running(task_id, query)
        _log("stage", "extractor", query[:80], session_id=task_id, data={"stage": "extractor"})
        top_context = await extract_for_task(query, history_text, self.client)
        top_quality  = "relevant" if len(top_context.split()) > 10 else ("minimal" if top_context else "none")
        _log("llm", f"extractor → {top_quality}",
             top_context[:120] if top_context else "(nothing extracted)",
             session_id=task_id,
             data={"action": "extract", "task": query[:80],
                   "quality": top_quality, "extracted": top_context[:300]})
        _tracker.tree_context_done(task_id, f"top-level extract: {top_quality}")

        # ── Stage 2: Split — informed by extracted context ────────────────────
        _log("stage", "split", query[:80], session_id=task_id, data={"stage": "split"})
        tasks = await split_deep(query, self.client)
        logger.info("pipeline: %d task(s): %s", len(tasks), tasks)
        _log("llm", f"split → {len(tasks)} task(s)", " | ".join(t[:40] for t in tasks[:3]),
             session_id=task_id, data={"action": "split", "tasks": [t[:80] for t in tasks]})

        # ── Stage 3: Per-task context extraction + planning ───────────────────
        sub_tasks: list[dict] = []
        all_extracted: list[str] = []

        for task in tasks:
            # Re-extract focused on this specific sub-task (skip if only one task —
            # top-level extract already covers it)
            if len(tasks) > 1:
                _log("stage", f"extractor: {task[:50]}", "", session_id=task_id,
                     data={"stage": "extractor", "task": task[:80]})
                task_history = await extract_for_task(task, history_text, self.client)
                extract_quality = "relevant" if len(task_history.split()) > 10 else ("minimal" if task_history else "none")
                _log("llm", f"extractor → {extract_quality}",
                     task_history[:120] if task_history else "(nothing extracted)",
                     session_id=task_id,
                     data={"action": "extract", "task": task[:80],
                           "quality": extract_quality, "extracted": task_history[:300]})
            else:
                task_history = top_context
                extract_quality = top_quality

            # ── Graph memory recall — check what was already researched ───────
            # If the knowledge graph has relevant results from previous tasks,
            # include them so the planner can skip redundant web searches.
            graph_recall = _search_graph(task, self.graph)
            graph_block  = f"[Web research from previous tasks]\n{graph_recall}" if graph_recall else ""

            task_ctx = "\n\n".join(p for p in (task_history, graph_block, project_ctx) if p)
            all_extracted.append(task_history)

            if graph_recall:
                logger.info("pipeline: graph recall hit for task=%r (%d chars)", task[:50], len(graph_recall))

            # ── Tool filtering (keyword score, synchronous) ───────────────────
            relevant_tools = filter_tools_for_task(task, available_tools)
            filtered_names = [t.get("name", "") for t in relevant_tools]
            logger.info("pipeline: task=%r filtered_tools=%s", task[:60], filtered_names)

            # ── Planning stage ────────────────────────────────────────────────
            _log("stage", f"plan: {task[:50]}", "", session_id=task_id,
                 data={"stage": "plan", "task": task[:120]})
            steps = await plan_task(task, relevant_tools or available_tools, self.client,
                                    context=task_ctx,
                                    soul_section=soul_section,
                                    user_prefs=user_prefs)
            sub_tasks.append({"task": task, "steps": steps})
            steps_preview = " → ".join(f"[{s['tool']}] {s['input'][:30]}" for s in steps[:4])
            logger.info("pipeline: task=%r steps=%s", task[:60],
                        [(s["tool"], s["input"][:40]) for s in steps])
            _log("llm", f"plan: {task[:40]}", steps_preview, session_id=task_id,
                 data={"action": "plan", "task": task[:120],
                       "steps": [{"tool": s["tool"], "input": s["input"][:80]} for s in steps]})

        _tracker.tree_plan_done(task_id, sub_tasks)

        # Synthesis context — extract what's relevant to the original query.
        # If there's only one sub-task its extracted context is already good;
        # for multi-task queries re-run extraction against the full query.
        if len(tasks) == 1 and all_extracted:
            synthesis_history = all_extracted[0]
        else:
            synthesis_history = await extract_for_task(query, history_text, self.client)
        synthesis_ctx = "\n\n".join(p for p in (synthesis_history, project_ctx) if p)

        state = PipelineState(
            query=query,
            soul_section=soul_section,
            user_prefs=user_prefs,
            sub_tasks=sub_tasks,
            synthesis_ctx=synthesis_ctx,
            task_id=task_id,
            project_dir=project_dir,
        )
        return await self._execute(state, available_tools)

    # ── Continue after outer tool_result ─────────────────────────────────────

    async def _continue(self, state: PipelineState, raw_history: list[dict],
                        available_tools: list[dict]) -> LoopResponse:
        # Find the step that triggered this outer tool call so we can annotate the result
        outer_step = {}
        try:
            sub = state.sub_tasks[state.current_task_idx]
            outer_step = sub.get("steps", [])[state.current_step_idx]
        except (IndexError, KeyError):
            pass
        outer_tool  = outer_step.get("tool", "bash")
        outer_input = outer_step.get("input", "")

        # Extract tool results from the last user message
        tool_results = _extract_tool_results(raw_history)
        for tr in tool_results:
            # Empty stdout from bash/write means the command succeeded silently
            if not tr and outer_input:
                tr = f"Completed successfully: {outer_input[:80]}"
            state.results.append({
                "tool": outer_tool,
                "input": outer_input,
                "result": tr,
                "summary": tr[:120],
            })
            logger.info("pipeline.continue: tool_result=%s", tr[:80])
            # Mark outer tool as done in dashboard
            _tracker.tree_subtask_step(
                state.task_id, state.current_task_idx,
                outer_tool, outer_input, tr[:200], status="done",
            )

        # Advance step index
        state.current_step_idx += 1

        return await self._execute(state, available_tools)

    # ── Execute next step ─────────────────────────────────────────────────────

    async def _execute(self, state: PipelineState, available_tools: list[dict]) -> LoopResponse:
        outer_tool_names = {t.get("name", "").lower() for t in available_tools}

        # If the last outer tool was Read or Glob, replan the remaining steps
        # based on what was actually found before continuing.
        if state.results:
            last_result = state.results[-1]
            if last_result.get("tool") in _OUTER_INFO_TOOLS and not last_result.get("_replanned"):
                last_result["_replanned"] = True
                await self._replan_from_outer_result(state, available_tools)

        # Walk through sub_tasks and steps until we hit an outer tool or finish
        while state.current_task_idx < len(state.sub_tasks):
            sub = state.sub_tasks[state.current_task_idx]
            steps = sub.get("steps", [])

            while state.current_step_idx < len(steps):
                step = steps[state.current_step_idx]
                tool  = step.get("tool", "").strip().lower()
                inp   = step.get("input", "").strip()

                # ── Smart upgrade: use Claude Code's WebSearch/WebFetch if available ─
                if tool == "web_search" and "websearch" in outer_tool_names:
                    tool = "websearch"
                    step["tool"] = "websearch"
                elif tool in ("web_fetch", "fetch_url") and "webfetch" in outer_tool_names:
                    tool = "webfetch"
                    step["tool"] = "webfetch"

                # ── Outer tool → return tool_use to Claude Code ───────────────
                if tool in outer_tool_names:
                    logger.info("pipeline: outer tool=%s input=%s", tool, inp[:60])
                    _log("bash", tool, inp[:80], session_id=state.task_id,
                         data={"action": "bash", "tool": tool, "input": inp[:200],
                               "task_idx": state.current_task_idx,
                               "step_idx": state.current_step_idx})
                    # Mark as running in dashboard so outer tools are visible
                    _tracker.tree_subtask_step(
                        state.task_id, state.current_task_idx,
                        tool, inp, "", status="running",
                    )
                    # Find canonical name
                    canonical = next(
                        t.get("name", tool) for t in available_tools
                        if t.get("name", "").lower() == tool
                    )
                    tool_id = f"toolu_{uuid.uuid4().hex[:16]}"
                    cl = canonical.lower()
                    if cl == "bash":
                        tool_input = {"command": inp}
                    elif cl == "write":
                        file_path = step.get("file_path") or inp[:120]
                        if state.project_dir and not os.path.isabs(file_path):
                            file_path = os.path.join(state.project_dir, file_path)
                        tool_input = {"file_path": file_path, "content": inp}
                    elif cl == "read":
                        file_path = step.get("file_path") or inp
                        if state.project_dir and not os.path.isabs(file_path):
                            file_path = os.path.join(state.project_dir, file_path)
                        # Store resolved path so replan can reference it
                        step["_resolved_path"] = file_path
                        # Also update input so _continue stores the resolved path
                        step["input"] = file_path
                        tool_input = {"file_path": file_path}
                    elif cl == "glob":
                        pattern = step.get("pattern") or inp or "*.py"
                        tool_input = {"pattern": pattern}
                        if state.project_dir:
                            tool_input["path"] = state.project_dir
                    elif cl == "edit":
                        file_path = step.get("file_path", "")
                        if state.project_dir and not os.path.isabs(file_path):
                            file_path = os.path.join(state.project_dir, file_path)
                        tool_input = {
                            "file_path":  file_path,
                            "old_string": step.get("old_string", ""),
                            "new_string": step.get("new_string", ""),
                        }
                    elif cl == "powershell":
                        # Same input schema as Bash
                        tool_input = {"command": inp}
                    elif cl == "grep":
                        pattern = step.get("pattern") or inp
                        tool_input = {"pattern": pattern, "output_mode": "files_with_matches"}
                        if state.project_dir:
                            tool_input["path"] = state.project_dir
                        if step.get("glob"):
                            tool_input["glob"] = step["glob"]
                        elif step.get("filetype"):
                            tool_input["type"] = step["filetype"]
                        step["input"] = pattern  # so _continue stores the pattern
                    elif cl == "websearch":
                        tool_input = {"query": inp}
                    elif cl == "webfetch":
                        url  = step.get("url") or inp
                        prompt = step.get("prompt") or "Extract all relevant information from this page."
                        tool_input = {"url": url, "prompt": prompt}
                        step["input"] = url
                    else:
                        tool_input = {"input": inp}

                    state_block = {
                        "type": "thinking",
                        "thinking": f"{_STATE_PREFIX}{state.to_json()}",
                    }
                    tool_block = {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": canonical,
                        "input": tool_input,
                    }
                    return LoopResponse(
                        content=[state_block, tool_block],
                        stop_reason="tool_use",
                    )

                # ── Internal tool ─────────────────────────────────────────────
                _log("tool", tool, inp[:80], session_id=state.task_id,
                     data={"action": "tool", "tool": tool, "input": inp[:200],
                           "task_idx": state.current_task_idx,
                           "step_idx": state.current_step_idx})
                _tracker.tree_subtask_step(
                    state.task_id, state.current_task_idx,
                    tool, inp, "", status="running",
                )
                t0 = _time.time()
                result = await self._run_internal(tool, inp, state)
                elapsed_ms = int((_time.time() - t0) * 1000)

                # ── Quality gate + inline retry ───────────────────────────────
                quality = _result_quality(result)
                result["_quality"] = quality

                if quality in ("empty", "error", "weak") and tool in _INFO_TOOLS:
                    logger.info("pipeline: %s result is %s for %r — reframing query",
                                tool, quality, inp[:40])
                    _log("stage", "retry", f"{tool} was {quality} — reframing",
                         session_id=state.task_id,
                         data={"stage": "retry", "tool": tool, "quality": quality,
                               "original": inp[:80]})
                    better_query = await self._reframe_query(
                        inp, result.get("result", ""), sub["task"]
                    )
                    if better_query:
                        logger.info("pipeline: retrying %s with %r", tool, better_query[:60])
                        retry_result = await self._run_internal(tool, better_query, state)
                        retry_quality = _result_quality(retry_result)
                        retry_result["_quality"] = retry_quality
                        retry_result["_retry_of"] = inp
                        if retry_quality in ("good", "weak"):
                            result = retry_result
                            quality = retry_quality
                            inp = better_query  # update for downstream replan
                            logger.info("pipeline: retry %s — quality now %s", tool, quality)
                            _log("llm", f"retry {tool} → {quality}", better_query[:80],
                                 session_id=state.task_id,
                                 data={"action": "retry", "tool": tool,
                                       "new_query": better_query[:120],
                                       "quality": quality})
                    else:
                        logger.warning("pipeline: could not reframe %r — proceeding with %s result",
                                       inp[:40], quality)

                state.results.append(result)
                summary = result.get("summary", "")
                _tracker.tree_subtask_step(
                    state.task_id, state.current_task_idx,
                    tool, inp, summary, status="done",
                )
                logger.info("pipeline: internal tool=%s quality=%s → %s",
                            tool, quality, summary[:60])
                _log("llm", f"{tool} → {quality}", summary[:120], session_id=state.task_id,
                     data={"action": tool, "input": inp[:120], "result": summary[:200],
                           "quality": quality, "elapsed_ms": elapsed_ms})

                # ── Dynamic replan after info-gathering ───────────────────────
                # The search result IS the reasoning — replace any pre-guessed
                # follow-up steps with steps derived from what was actually found.
                if tool in _INFO_TOOLS:
                    found = result.get("result", "")
                    remaining = len(steps) - state.current_step_idx - 1
                    if found and remaining > 0:
                        new_steps = await self._replan_after_search(
                            sub["task"], found, available_tools,
                            soul_section=state.soul_section,
                            user_prefs=state.user_prefs,
                        )
                        old_count = remaining
                        sub["steps"] = steps[:state.current_step_idx + 1] + new_steps
                        steps = sub["steps"]
                        logger.info(
                            "pipeline: replan after %s — %d pre-guessed step(s) → %d result-driven step(s)",
                            tool, old_count, len(new_steps),
                        )
                        _log("stage", "replan", f"{old_count} → {len(new_steps)} steps",
                             session_id=state.task_id,
                             data={"stage": "replan", "tool": tool,
                                   "old_steps": old_count, "new_steps": len(new_steps),
                                   "derived": [s["input"][:80] for s in new_steps]})
                        if new_steps:
                            _tracker.tree_subtask_replanned(
                                state.task_id, state.current_task_idx, new_steps
                            )

                state.current_step_idx += 1

            # Task done — move to next
            _tracker.tree_subtask_done(state.task_id, state.current_task_idx)
            state.current_task_idx += 1
            state.current_step_idx = 0

        # ── All steps done → consolidate ──────────────────────────────────────
        logger.info("pipeline: consolidating %d results", len(state.results))
        result_summary = "; ".join(r.get("summary", "")[:60] for r in state.results[-3:])
        _log("stage", "synthesize", result_summary[:120], session_id=state.task_id,
             data={"stage": "synthesize", "results_count": len(state.results),
                   "input_preview": result_summary[:200]})
        _tracker.tree_synthesizer_running(state.task_id, result_summary)
        t0 = _time.time()
        answer = await synthesize(
            state.query,
            state.soul_section,
            state.user_prefs,
            state.results,
            self.client,
            context=state.synthesis_ctx,
        )
        elapsed_ms = int((_time.time() - t0) * 1000)
        _tracker.tree_synthesizer_done(state.task_id, answer[:200])
        _tracker.finish_task(state.task_id, "done")
        _log("llm", "synthesize → answer", answer[:120], session_id=state.task_id,
             data={"action": "synthesize", "answer_preview": answer[:300], "elapsed_ms": elapsed_ms})
        _log("answer", "answer", answer[:120], session_id=state.task_id,
             data={"answer": answer[:500]})
        return LoopResponse(
            content=[{"type": "text", "text": answer}],
            stop_reason="end_turn",
        )

    # ── Dynamic replanning after info-gathering ───────────────────────────────

    async def _replan_after_search(
        self, task: str, search_result: str, available_tools: list[dict],
        soul_section: str = "", user_prefs: str = "",
    ) -> list[dict]:
        """Derive follow-up execution steps from an actual search/memory result.

        Replaces pre-guessed steps with steps grounded in what was found.
        Soul guidance and user preferences shape which steps are chosen and how.
        Returns [] if the result already answers the task (no execution needed),
        or on any failure (caller keeps original steps in that case).
        """
        logger.debug("pipeline: replan from search result for task=%r", task[:60])
        soul_block = f"\nPersonality guidance (follow this):\n{soul_section[:300]}" if soul_section else ""
        prefs_block = f"\nUser preferences:\n{user_prefs[:200]}" if user_prefs else ""
        try:
            r = await self.client.generate(
                [
                    {"role": "system", "content": _REPLAN_SYSTEM + soul_block + prefs_block},
                    {"role": "user",   "content": f"Task: {task}\n\nFound:\n{search_result[:3000]}"},
                ],
                max_tokens=500,
                temperature=0.1,
                response_format={"type": "json_object"},
                stream=False,
                thinking=False,
            )
            raw  = r["choices"][0]["message"]["content"].strip()
            data = parse_format_response(raw) or {}
            steps_raw = str(data.get("steps") or "").strip()
            if not steps_raw:
                return []  # result fully answers the task
            steps: list[dict] = []
            for part in steps_raw.split("|"):
                part = part.strip()
                if ":" in part:
                    tool_name, _, inp = part.partition(":")
                    tool_name = tool_name.strip().lower()
                    inp = inp.strip()
                    if tool_name and inp:
                        steps.append({"tool": tool_name, "input": inp})
            logger.debug("pipeline: replan → %d steps", len(steps))
            return steps
        except Exception as exc:
            logger.warning("pipeline: _replan_after_search failed: %s", exc)
        return []

    # ── Replan after outer info tool (Read / Glob) ────────────────────────────

    async def _replan_from_outer_result(
        self, state: PipelineState, available_tools: list[dict]
    ) -> None:
        """Replace remaining planned steps based on the result of Read or Glob.

        Glob result  → inject a Read step for the matched file.
        Read result  → call _generate_edit_steps to produce Edit or Write steps.
        Modifies state.sub_tasks in-place; no return value.
        """
        last = state.results[-1]
        tool   = last.get("tool", "")
        result = last.get("result", "")
        inp    = last.get("input", "")   # resolved file path for read; pattern for glob

        sub   = state.sub_tasks[state.current_task_idx]
        # Truncate planned steps at current index — we replace what comes next
        sub["steps"] = sub["steps"][:state.current_step_idx]

        if tool in ("glob", "grep"):
            # Skip Read→Edit replan for informational queries — let synthesis answer directly
            if _is_informational_query(state.query):
                logger.info("pipeline: %s informational query — skipping read→edit replan", tool)
                return
            # Filter out summary lines like "Found 2 files" — keep only actual file paths
            # A real path has an extension, a path separator, or looks like a relative filename
            files = [
                f.strip() for f in result.splitlines()
                if f.strip() and (
                    re.search(r'\.[a-zA-Z0-9]{1,8}$', f.strip())
                    or os.sep in f.strip()
                    or '/' in f.strip()
                )
            ]
            if not files:
                logger.warning("pipeline: %s returned no parseable file paths — cannot replan", tool)
                return
            target = files[0]
            # Resolve against project_dir if relative
            if state.project_dir and not os.path.isabs(target):
                target = os.path.join(state.project_dir, target)
            logger.info("pipeline: %s→read replan: reading %s", tool, target)
            new_steps = [{"tool": "read", "input": target, "file_path": target}]
            sub["steps"].extend(new_steps)
            _tracker.tree_subtask_replanned(state.task_id, state.current_task_idx, new_steps)

        elif tool == "read":
            file_path = inp  # already resolved in _execute
            logger.info("pipeline: read→edit replan for %s", file_path)
            new_steps = await self._generate_edit_steps(
                state.query, result, file_path
            )
            if new_steps:
                sub["steps"].extend(new_steps)
                _tracker.tree_subtask_replanned(state.task_id, state.current_task_idx, new_steps)
            else:
                logger.warning("pipeline: _generate_edit_steps returned nothing — skipping edit")

        elif tool in ("websearch", "webfetch"):
            # Delegate to the existing search-result replan (derives bash/save steps)
            logger.info("pipeline: %s→replan from web result", tool)
            new_steps = await self._replan_after_search(
                state.query, result, available_tools,
                soul_section=state.soul_section,
                user_prefs=state.user_prefs,
            )
            if new_steps:
                sub["steps"].extend(new_steps)
                _tracker.tree_subtask_replanned(state.task_id, state.current_task_idx, new_steps)

    async def _generate_edit_steps(
        self, task: str, file_content: str, file_path: str
    ) -> list[dict]:
        """Given a task and current file content, produce Edit or Write steps.

        Two-phase: small decision call first (edit vs write + old/new strings),
        then _generate_code for the write path — avoids asking the LLM to output
        an entire file in a single constrained JSON call.
        """
        from engine.translation.planner import parse_format_response, _generate_code
        try:
            r = await self.client.generate(
                [
                    {"role": "system", "content": _EDIT_SYSTEM},
                    {"role": "user", "content":
                        f"Task: {task}\n\nFile:\n{file_content[:3000]}"},
                ],
                max_tokens=400,
                temperature=0.1,
                response_format={"type": "json_object"},
                stream=False,
                thinking=False,
            )
            raw  = r["choices"][0]["message"]["content"].strip()
            data = parse_format_response(raw) or {}
            mode = data.get("mode", "")

            if mode == "edit":
                old_s = (data.get("old") or "").strip()
                new_s = (data.get("new") or "").strip()
                if old_s and old_s in file_content:
                    logger.info("pipeline: edit mode — surgical replacement in %s", file_path)
                    return [{"tool": "edit", "input": "",
                             "file_path": file_path,
                             "old_string": old_s,
                             "new_string": new_s}]
                logger.warning("pipeline: edit old_string not found in file — falling back to write")

            # write mode (or edit fallback): regenerate with _generate_code
            logger.info("pipeline: write mode — regenerating %s", file_path)
            code = await _generate_code(task, self.client)
            if code:
                final = code if code.endswith("\n") else code + "\n"
                return [{"tool": "write", "input": final, "file_path": file_path}]

        except Exception as exc:
            logger.warning("pipeline: _generate_edit_steps failed: %s", exc)
        return []

    # ── Query reframing for failed steps ─────────────────────────────────────

    async def _reframe_query(self, original: str, failed_result: str, task: str) -> str | None:
        """Rewrite a failed search query into a better, more specific one.

        Called when a web_search or search_memory step returns empty/error/weak.
        Returns None if the query cannot be meaningfully improved.
        """
        try:
            r = await self.client.generate(
                [
                    {"role": "system", "content": _REFRAME_SYSTEM},
                    {"role": "user", "content": (
                        f"Task: {task[:200]}\n"
                        f"Original query: {original}\n"
                        f"Result was: {failed_result[:300] or '(empty)'}"
                    )},
                ],
                max_tokens=300,
                temperature=0.2,
                stream=False,
                thinking=False,
            )
            q = r["choices"][0]["message"]["content"].strip().strip('"').strip("'")
            if q and q != "-" and q.lower() != original.lower() and len(q) > 4:
                return q
        except Exception as exc:
            logger.warning("pipeline: _reframe_query failed: %s", exc)
        return None

    # ── Research graph persistence ────────────────────────────────────────────

    async def _save_research_to_graph(self, query: str, result: str) -> None:
        """Extract reusable procedural knowledge from a web result and save to graph.

        Runs a small LLM call to separate procedure from live data:
        - "how to install X" → saves the install command (reusable)
        - "what is system status" → live data, saves nothing

        The graph stores HOW to achieve things, not what the current state IS.
        """
        if not self.graph or not result:
            return
        try:
            r = await self.client.generate(
                [
                    {"role": "system", "content": _PROCEDURE_EXTRACT_SYSTEM},
                    {"role": "user",   "content": f"Query: {query}\n\nResults:\n{result[:2000]}"},
                ],
                max_tokens=400,
                temperature=0.1,
                stream=False,
                thinking=False,
            )
            procedure = r["choices"][0]["message"]["content"].strip()
            if not procedure or procedure == "-" or len(procedure.split()) < 5:
                logger.debug("pipeline: no reusable procedure in result for %r — not saving to graph", query[:40])
                return
            self.graph.upsert_node(
                name=query[:80],
                node_type="research",
                summary=procedure,
                sources=["web_search"],
            )
            self.graph.save()
            logger.debug("pipeline: saved procedure to graph for %r", query[:40])
        except Exception as exc:
            logger.warning("pipeline: _save_research_to_graph failed: %s", exc)

    # ── Internal tool execution ───────────────────────────────────────────────

    async def _run_internal(self, tool: str, inp: str, state: PipelineState) -> dict:

        if tool in ("search_soul", "search_policy"):
            from engine.policy.router import parse_policy_sections, match_policy_section, load_inner_self
            parse_soul_sections = parse_policy_sections  # alias for the code below
            match_soul_section  = match_policy_section
            # Check inner_self first for philosophical queries
            q = inp.lower()
            if any(kw in q for kw in ("alive", "think", "feel", "conscious", "exist",
                                       "sentient", "inner", "self", "real")):
                inner = load_inner_self(self.soul_path)
                if inner:
                    return {"tool": tool, "input": inp, "result": inner, "summary": inner[:120]}
            sections = parse_soul_sections(self.soul_path)
            _, content = match_soul_section(inp, sections)
            if not content and inp:
                content = sections.get(inp.lower(), "")
            return {"tool": tool, "input": inp, "result": content, "summary": content[:120]}

        if tool in ("search_memory", "search_knowledge"):
            content = _search_graph(inp, self.graph)
            return {"tool": tool, "input": inp, "result": content, "summary": f"memory: {content[:100]}"}

        if tool == "search_user_prefs":
            from engine.policy.router import load_user_prefs
            content = load_user_prefs(self.prefs_path)
            return {"tool": tool, "input": inp, "result": content, "summary": f"prefs: {content[:80]}"}

        if tool == "web_search":
            # Check graph first — previous tasks may have already fetched this research.
            # Graph stores web research results (node_type="research"), not answers.
            if self.graph:
                mem = _search_graph(inp, self.graph)
                if mem and len(mem.strip()) > 40:
                    logger.info("pipeline: web_search → graph hit for %r", inp[:40])
                    return {"tool": "search_memory", "input": inp, "result": mem,
                            "summary": f"recalled: {mem[:120]}"}
            # Nothing in graph — fetch from web
            try:
                raw = await _web_search(inp, max_results=4)
                content = format_results(raw) if raw else "No results."
            except Exception as exc:
                content = f"Search failed: {exc}"
            # Extract and save reusable procedural knowledge (not raw data) to graph.
            # Live data (status, metrics) is filtered out by the extraction LLM.
            if content and "No results" not in content and "failed" not in content:
                await self._save_research_to_graph(inp, content)
            return {"tool": tool, "input": inp, "result": content, "summary": content[:2000]}

        if tool in ("save_memory", "remember"):
            from engine.policy.router import save_user_pref
            save_user_pref(inp, self.prefs_path)
            if self.graph:
                try:
                    self.graph.upsert_node(name=inp[:80], node_type="user",
                                           summary=inp, sources=["pipeline"])
                    self.graph.save()
                except Exception:
                    pass
            return {"tool": tool, "input": inp, "result": "saved", "summary": f"saved: {inp[:80]}"}

        # Unknown tool — log and skip
        logger.warning("pipeline: unknown internal tool %r — skipping", tool)
        return {"tool": tool, "input": inp, "result": "", "summary": f"unknown: {tool}"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_state(raw_history: list[dict]) -> PipelineState | None:
    """Find and decode PIPELINE_STATE from thinking blocks in history."""
    for msg in reversed(raw_history):
        if msg.get("role") != "assistant":
            continue
        for block in msg.get("content", []):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "thinking":
                t = block.get("thinking", "")
                if _STATE_PREFIX in t:
                    idx = t.index(_STATE_PREFIX) + len(_STATE_PREFIX)
                    try:
                        return PipelineState.from_json(t[idx:])
                    except Exception as exc:
                        logger.warning("pipeline: state decode failed: %s", exc)
    return None


def _extract_tool_results(raw_history: list[dict]) -> list[str]:
    """Extract tool result content strings from the last user message."""
    results = []
    if not raw_history:
        return results
    last = raw_history[-1]
    if last.get("role") != "user":
        return results
    for block in last.get("content", []):
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_result":
            c = block.get("content", "")
            if isinstance(c, list):
                c = "\n".join(
                    b.get("text", "") for b in c
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            if c:
                results.append(str(c).strip()[:1000])
    return results



def _format_history(raw_history: list[dict]) -> str:
    """Format full conversation history as a clean transcript for LLM context.

    Includes tool_use and tool_result blocks so the model can see what tools
    were called and what they returned — critical for follow-up requests like
    "modify the file" where the filename only appears in a tool_result.

    Skips: thinking blocks and pipeline state blobs only.
    """
    turns: list[str] = []
    for msg in raw_history:
        role = msg.get("role", "")
        if role not in ("user", "assistant"):
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for b in content:
                if not isinstance(b, dict):
                    continue
                btype = b.get("type", "")
                if btype == "thinking":
                    continue
                if btype == "text":
                    t = b.get("text", "")
                    if _STATE_PREFIX not in t:
                        text_parts.append(t)
                elif btype == "tool_use":
                    name = b.get("name", "tool")
                    inp  = b.get("input") or {}
                    # Summarise the most meaningful input field
                    summary = (inp.get("command") or inp.get("file_path")
                               or inp.get("pattern") or str(inp)[:120])
                    text_parts.append(f"[called {name}: {summary}]")
                elif btype == "tool_result":
                    c = b.get("content", "")
                    if isinstance(c, list):
                        c = " ".join(
                            x.get("text", "") for x in c
                            if isinstance(x, dict) and x.get("type") == "text"
                        )
                    c = str(c).strip()
                    if c:
                        text_parts.append(f"[result: {c[:300]}]")
            text = " ".join(text_parts)
        else:
            text = str(content)
        text = text.strip()
        if not text or len(text) < 4:
            continue
        label = "User" if role == "user" else "Assistant"
        turns.append(f"{label}: {text}")
    return "\n".join(turns)


def _last_turns_text(raw_history: list[dict], n: int = 4) -> str:
    """Return the last n user+assistant turns as plain text (legacy helper)."""
    turns: list[str] = []
    for msg in raw_history[-(n * 2):]:
        role = msg.get("role", "")
        if role not in ("user", "assistant"):
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            text = " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            text = str(content)
        text = text.strip()
        if not text or len(text) < 4 or _STATE_PREFIX in text:
            continue
        turns.append(f"{'User' if role == 'user' else 'Assistant'}: {text[:300]}")
    return "\n".join(turns)


_INFORMATIONAL_PREFIXES = re.compile(
    r"^(which|what|who|how\s+many|how\s+much|does|do\s+any|is\s+there|are\s+there"
    r"|show\s+(me\s+)?|list|find\s+(which|what|all)|tell\s+me|can\s+you\s+tell"
    r"|check\s+(if|whether)|does\s+any|which\s+(of|files|ones))",
    re.IGNORECASE,
)


def _is_informational_query(query: str) -> bool:
    """Return True if the query is asking for information rather than requesting an action.

    Used to decide whether grep/glob results should trigger a Read→Edit replan
    or just flow straight to synthesis for answering.
    """
    q = query.strip()
    if _INFORMATIONAL_PREFIXES.match(q):
        return True
    # Questions ending with "?" are almost always informational
    if q.endswith("?"):
        return True
    return False


def _search_graph(query: str, graph) -> str:
    if graph is None:
        return ""
    try:
        nodes = graph.search(query, top_k=3)
        if not nodes:
            return ""
        return "\n".join(n.get("content", n.get("summary", ""))[:200] for n in nodes)
    except Exception:
        return ""
