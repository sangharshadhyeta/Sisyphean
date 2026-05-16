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
from engine.translation.planner import split_deep, plan_task, think_decompose, infer_stage_type, parse_format_response
from engine.translation.web_search import search as _web_search, fetch as _web_fetch_page, format_results
import engine.task_tracker as _tracker
from engine.activity import log_event as _log

logger = logging.getLogger(__name__)

_STATE_PREFIX = "PIPELINE_STATE:"

# Tools whose results should trigger dynamic replanning of subsequent steps.
# After these run, pre-guessed follow-up steps are replaced with steps derived
# from what was actually found — the result IS the reasoning.
_INFO_TOOLS = frozenset({"web_search", "web_fetch", "fetch_url", "search_memory", "search_knowledge"})

# Outer tools (returned to Claude Code harness) whose results require replanning.
# Read/Grep/Glob → find/read file, then generate edit steps.
# WebSearch/WebFetch → delegate to _replan_after_search for execution steps.
_OUTER_INFO_TOOLS = frozenset({"read", "glob", "grep", "websearch", "webfetch"})

_EDIT_SYSTEM = """\
Given the task and current file, decide the minimal change needed. Output ONE JSON object only.

For small targeted changes (URL, variable, import, small block, or adding content to a document):
{"mode": "edit", "old": "exact text to replace", "new": "replacement text"}

For major rewrites (different logic, new structure, significant restructuring):
{"mode": "write"}

Rules:
- "old" must be an EXACT substring of the file. Include 1-2 surrounding lines for uniqueness.
- For adding new content to a document, use "edit" mode: set "old" to the last line of the file,
  set "new" to that same last line plus the new content appended below it.
- Never output Python code for a document (.md, .txt) file.
- No prose, no markdown, no explanation — only the JSON."""

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

# ── Write item JSON schema — grammar-constrained output forces full content ────
# Mirrors BirdClaw's WRITE_ITEM_SCHEMA.  Ollama grammar-constrained decoding
# masks invalid tokens so the model CANNOT output preambles, stop early, or
# wrap output in markdown fences — it MUST fill "content" with actual text.
_WRITE_ITEM_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "write_item",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
            },
            "required": ["content"],
            "additionalProperties": False,
        },
    },
}

_WRITE_ITEM_SYSTEM = (
    "You are a focused writer producing prose for a document. "
    "Output ONLY a JSON object: "
    '{"content": "<section heading + full body text>"}\n\n'
    "Rules:\n"
    "- The content field MUST start with the exact section heading line given in the instruction.\n"
    "- After the heading, write multiple dense paragraphs (at least 3) with specific detail and depth.\n"
    "- Write PROSE PARAGRAPHS — sentences and paragraphs, not code.\n"
    "- Do NOT write Python, JavaScript, or any programming language code in the content.\n"
    "- Do NOT write a document title (# Title). Only the section heading (## Section) and body.\n"
    "- Do NOT repeat content that already exists in the file tail shown to you.\n"
    "- Do NOT add preamble, commentary, or markdown fences outside the JSON.\n"
    "- Fill the content field completely — do not stop after one sentence."
)

# ── Reflect gate — evaluates write depth after all items are done ─────────────
_REFLECT_SYSTEM = (
    "Evaluate whether the written content sufficiently achieves the stated goal. "
    "Output exactly one JSON object — no other text:\n"
    '  {"decision": "done"}\n'
    '  {"decision": "deepen", "goal": "specific aspect that is missing or too shallow"}\n'
    "Choose 'done' if the content covers the goal well enough. "
    "Choose 'deepen' only when a clearly important aspect is absent or underdeveloped."
)

_REPLAN_SYSTEM = """\
A task is being completed. A web search was just run.
Your job: decide what to do next based on what was found.

IF the search returned useful results:
  Derive execution steps from what was found — copy commands, package names,
  and values VERBATIM from the search result. Do not guess or paraphrase.
  If the result already answers the question → reply {"steps": ""}

IF the search returned nothing useful OR off-topic results:
  Do NOT invent commands from thin air.
  Read the result carefully — extract clues about what went wrong:
    - Result about wrong OS/platform (e.g. Android result for a PC task)?
      → reformulate with the correct OS context (e.g. add "Windows" or "Linux")
    - Result about a different meaning of the query?
      → narrow with the specific domain or context
    - Task is about THIS machine's current state (processes, CPU, memory, disk,
      system status, uptime, network) → plan a bash step, do NOT search the web
    - If a better query is genuinely needed → plan web_search with the improved query

RULES (enforce strictly):
  - NEVER output the exact same search query that was just run — always reformulate
  - A query that already ran once must not appear again in your steps
  - For machine/system state queries: always bash, never web_search

Reply as JSON: {"steps": "toolname:exact input | toolname:exact input"}

Available tools:
  bash        — run a shell command (write the exact command; never prefix with
                'python' unless running a .py file — other executables run directly)
  web_fetch   — fetch a specific URL to read its full page content; use the exact
                URL from the search results above (e.g. web_fetch:https://example.com/page)
  web_search  — search for more specific information if still needed
  save_memory — save a key fact that was discovered

WHEN TO USE web_fetch:
  - Search result snippets mention the answer but are too short to be useful
  - A result title/URL clearly contains what you need (docs, tutorial, spec)
  - The task requires exact commands, version numbers, or configuration details

If the result already fully answers the task, reply: {"steps": ""}"""


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
    # ── Epistemic state (persists across Claude Code turns) ───────────────────
    # Tracks what has been read/written/run so subsequent steps have full context.
    # Serialised compactly so the thinking-block payload stays small.
    files_read:    list[dict] = field(default_factory=list)  # [{"path":str,"head":str}]
    files_written: list[str]  = field(default_factory=list)  # paths created/modified
    commands_run:  list[dict] = field(default_factory=list)  # [{"cmd":str,"brief":str}]
    # ── Incremental write plan (one active write operation at a time) ─────────
    # Sisyphean decides what to write section-by-section; each section is emitted
    # as a separate Write outer-tool call.  After each Write result the file is
    # read from disk to provide context for the next section.
    wp_items:    list[dict] = field(default_factory=list)  # [{anchor, title, min_chars}]
    wp_goal:     str = ""   # full task description for the write operation
    wp_file:     str = ""   # absolute path of target file
    wp_ftype:    str = ""   # "code" or "doc"
    wp_idx:      int = 0    # index of the NEXT item to write
    wp_run_after: bool = False  # run the file with python after all items done
    wp_retry_count: int = 0  # retries for the current item (resets on advance)
    wp_resume_ctx:  str = ""  # verifier resume context for the current retry

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
            # Epistemic state — capped to keep JSON size manageable
            "fr":  [{"path": r["path"], "head": r.get("head", "")[:150]}
                    for r in self.files_read[-10:]],
            "fw":  self.files_written[-20:],
            "cr":  [{"cmd": r["cmd"][:80], "brief": r.get("brief", "")[:80]}
                    for r in self.commands_run[-10:]],
            # Write plan — compact
            "wp": {
                "items": [{"a": it["anchor"], "t": it["title"], "m": it["min_chars"]}
                          for it in self.wp_items],
                "goal":  self.wp_goal[:300],
                "file":  self.wp_file,
                "ftype": self.wp_ftype,
                "idx":   self.wp_idx,
                "run":   self.wp_run_after,
                "rc":    self.wp_retry_count,
                "rctx":  self.wp_resume_ctx[:800],
            } if self.wp_items else {},
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
        obj.files_read        = d.get("fr",  [])
        obj.files_written     = d.get("fw",  [])
        obj.commands_run      = d.get("cr",  [])
        wp = d.get("wp", {})
        obj.wp_items       = [{"anchor": i["a"], "title": i["t"], "min_chars": i["m"]}
                              for i in wp.get("items", [])]
        obj.wp_goal        = wp.get("goal",  "")
        obj.wp_file        = wp.get("file",  "")
        obj.wp_ftype       = wp.get("ftype", "")
        obj.wp_idx         = wp.get("idx",   0)
        obj.wp_run_after   = wp.get("run",   False)
        obj.wp_retry_count = wp.get("rc",    0)
        obj.wp_resume_ctx  = wp.get("rctx",  "")
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

        # ── BirdClaw pre-made plan ────────────────────────────────────────────
        # When BirdClaw routes via engine_client it calls generate_plan() first
        # (thinking=True) and injects the result as a [BirdClaw-Plan] message.
        # Detect it here so _start() can skip think_decompose() and go straight
        # to per-task execution — no duplicate planning call.
        bc_plan = _extract_birdclaw_plan(raw_history)

        return await self._start(user_message, available_tools, session_id=session_id,
                                 history_text=history_text, project_ctx=project_ctx,
                                 project_dir=project_dir, bc_plan=bc_plan)

    # ── Fresh request ─────────────────────────────────────────────────────────

    async def _start(self, query: str, available_tools: list[dict], session_id: str = "",
                     history_text: str = "", project_ctx: str = "",
                     project_dir: str = "",
                     bc_plan: "tuple[str, list[dict]] | None" = None) -> LoopResponse:
        from engine.policy.router import load_user_prefs

        task_id = _tracker.start_task(session_id=session_id, user_message=query)

        soul_section = self.soul_path.read_text(encoding="utf-8").strip() if self.soul_path.exists() else ""
        user_prefs   = load_user_prefs(self.prefs_path)

        logger.info("pipeline.start: query=%r history=%d chars", query[:60], len(history_text))

        # ── Stage 1: Extract top-level context before splitting ───────────────
        _tracker.tree_context_running(task_id, query)
        top_context = await extract_for_task(query, history_text, self.client)
        top_quality  = "relevant" if len(top_context.split()) > 10 else ("minimal" if top_context else "none")
        _tracker.tree_context_done(task_id, f"top-level extract: {top_quality}")

        # ── Stage 2: Decompose into stages ───────────────────────────────────
        # If BirdClaw already ran generate_plan() (thinking=True) before routing
        # here, use that plan directly — no duplicate LLM planning call.
        # Otherwise fall back to think_decompose() (standalone Sisyphean path).
        if bc_plan is not None:
            outcome, stages = bc_plan
            _tracker.tree_context_running(task_id, "birdclaw-plan")
            _tracker.tree_context_done(task_id, f"BirdClaw plan: {len(stages)} stage(s)")
            logger.info("pipeline: using BirdClaw plan — outcome=%r stages=%s",
                        outcome[:60], [(s["type"], s["goal"][:40]) for s in stages])
        else:
            # One thinking-enabled planning call that reasons about the task
            # before any execution begins. Returns (outcome, stages) where each
            # stage has a type (research/write_code/verify/direct/…) and a
            # plain-English goal. Sets PIPELINE_STATE accurately from the very
            # first API turn — the full plan is visible before execution starts.
            _tracker.tree_context_running(task_id, "think-decompose")
            outcome, stages = await think_decompose(
                query, self.client,
                context=top_context,
                soul_section=soul_section,
            )
            _tracker.tree_context_done(task_id, f"decomposed into {len(stages)} stage(s)")
            logger.info("pipeline: outcome=%r stages=%s", outcome[:60],
                        [(s["type"], s["goal"][:40]) for s in stages])

        # Fall back only if think_decompose raised an exception (it now explicitly
        # returns a "direct" stage when the model decides steps="").
        if not stages:
            stages = [{"type": infer_stage_type(query), "goal": query}]

        # Map stages back to the task list that plan_task expects.
        # Each stage goal becomes the sub-task text; infer_stage_type already
        # annotated the type which pipeline uses for tool selection below.
        tasks = [s["goal"] for s in stages]

        # ── Stage 3: Per-task context extraction + planning ───────────────────
        sub_tasks: list[dict] = []
        all_extracted: list[str] = []

        for task, stage in zip(tasks, stages):
            # Re-extract focused on this specific sub-task (bigram Jaccard, no LLM).
            # Gives each sub-task a focused slice of history rather than the
            # top-level extract which may include irrelevant exchanges.
            if len(tasks) > 1:
                task_history = await extract_for_task(task, history_text, self.client)
                extract_quality = "relevant" if len(task_history.split()) > 10 else ("minimal" if task_history else "none")
            else:
                task_history = top_context
                extract_quality = top_quality

            # ── Graph memory recall — check what was already researched ───────
            # If the knowledge graph has relevant results from previous tasks,
            # include them so the planner can skip redundant web searches.
            graph_recall = _search_graph(task, self.graph)
            graph_block  = f"[Web research from previous tasks]\n{graph_recall}" if graph_recall else ""

            # Inject a live workspace snapshot so the planner can see what files
            # already exist before generating a write/edit plan.
            ws_snap = _workspace_snapshot(self.workspace) if self.workspace else ""
            ws_block = f"[Workspace files]\n{ws_snap}" if ws_snap else ""

            task_ctx = "\n\n".join(p for p in (task_history, graph_block, ws_block, project_ctx) if p)
            all_extracted.append(task_history)

            if graph_recall:
                logger.info("pipeline: graph recall hit for task=%r (%d chars)", task[:50], len(graph_recall))

            # ── Tool filtering (keyword score, synchronous) ───────────────────
            relevant_tools = filter_tools_for_task(task, available_tools)
            filtered_names = [t.get("name", "") for t in relevant_tools]
            logger.info("pipeline: task=%r filtered_tools=%s", task[:60], filtered_names)

            # ── Planning stage ────────────────────────────────────────────────
            # Stage type from think_decompose guides tool selection so plan_task
            # can skip the LLM for well-understood stage types.
            stage_type = stage.get("type", "")

            # For "direct" stages (social messages, trivial answers) skip planning.
            if stage_type == "direct":
                steps = []
            else:
                steps = await plan_task(task, relevant_tools or available_tools, self.client,
                                        context=task_ctx,
                                        soul_section=soul_section,
                                        user_prefs=user_prefs)

            # ── Websearch fallback — if plan is empty and no graph recall ──────
            if not steps and stage_type == "research" and not graph_recall:
                ws_tool = "websearch" if any(
                    t.get("name", "").lower() == "websearch"
                    for t in (relevant_tools or available_tools)
                ) else "web_search"
                steps = [{"tool": ws_tool, "input": task}]
                logger.info("pipeline: empty plan → websearch fallback for %r", task[:60])
            elif not steps and stage_type not in ("direct",) and not graph_recall:
                # Non-research stage with no plan — let synthesizer handle it
                logger.info("pipeline: empty plan for %s stage %r", stage_type, task[:60])

            sub_tasks.append({"task": task, "type": stage_type, "steps": steps})
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

        # ── Write plan continuation ────────────────────────────────────────────
        # If a write plan is active the outer tool result was a Write for one item.
        # Verify the item against the actual file; retry if partial/missing.
        if state.wp_items:
            tool_results = _extract_tool_results(raw_history)
            # Item that was just written
            written_item = state.wp_items[state.wp_idx] if state.wp_idx < len(state.wp_items) else {}
            item_title  = written_item.get("title",  f"item {state.wp_idx + 1}")
            item_anchor = written_item.get("anchor", item_title)
            item_min    = written_item.get("min_chars", 200)

            for tr in tool_results:
                logger.info("pipeline.wp_continue: item=%d/%d result=%s",
                            state.wp_idx, len(state.wp_items), tr[:60])
                if state.wp_file not in state.files_written:
                    state.files_written.append(state.wp_file)
                _tracker.tree_subtask_step(
                    state.task_id, state.current_task_idx,
                    "write", item_title, tr[:120], status="done",
                )

            # ── Verify the item that was just written ─────────────────────────
            _MAX_ITEM_RETRIES = 2
            item_complete = True  # default: accept and advance

            if state.wp_file and os.path.isfile(state.wp_file):
                try:
                    from engine.translation.subtask.verifier import (
                        parse_doc_sections, parse_code_items,
                        _match_key, is_stub_body,
                    )
                    file_content = Path(state.wp_file).read_text(encoding="utf-8")
                    parsed = (parse_doc_sections(file_content)
                              if state.wp_ftype == "doc"
                              else parse_code_items(file_content))
                    key  = _match_key(item_anchor, parsed)
                    if key:
                        body  = parsed[key]
                        stub  = state.wp_ftype == "code" and is_stub_body(body)
                        if stub or len(body) < item_min:
                            item_complete = False
                            status_str = "stub" if stub else f"{len(body)}c < {item_min}c"
                            logger.info("pipeline.wp_verify: item %r %s", item_title, status_str)
                        else:
                            logger.info("pipeline.wp_verify: item %r complete (%dc)",
                                        item_title, len(body))
                    else:
                        item_complete = False
                        logger.info("pipeline.wp_verify: item %r missing from file", item_title)

                    if not item_complete:
                        state.wp_resume_ctx = _build_verify_resume_ctx(
                            state.wp_items, state.wp_idx, file_content, state.wp_ftype
                        )
                except Exception as _ve:
                    logger.warning("pipeline.wp_verify failed: %s — accepting", _ve)

            if item_complete or state.wp_retry_count >= _MAX_ITEM_RETRIES:
                if not item_complete:
                    logger.warning("pipeline.wp_verify: item %r exhausted retries — advancing",
                                   item_title)
                state.wp_idx        += 1
                state.wp_retry_count = 0
                state.wp_resume_ctx  = ""
            else:
                state.wp_retry_count += 1
                logger.info("pipeline.wp_verify: retrying item %r (%d/%d)",
                            item_title, state.wp_retry_count, _MAX_ITEM_RETRIES)

            return await self._execute(state, available_tools)

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

            # ── Update epistemic state ─────────────────────────────────────
            if outer_tool == "read":
                file_path = outer_step.get("_resolved_path") or outer_input
                if os.path.isfile(file_path):
                    head = _file_head(file_path)
                else:
                    # Use the first few lines of the returned content
                    head = "\n".join(
                        ln for ln in tr.split("\n")[:8] if ln.strip()
                    )[:300]
                if not any(fr.get("path") == file_path for fr in state.files_read):
                    state.files_read.append({"path": file_path, "head": head})
                    logger.debug("epistemic: read %s", file_path)
            elif outer_tool == "write":
                file_path = outer_step.get("file_path") or outer_input[:200]
                if state.project_dir and not os.path.isabs(file_path):
                    file_path = os.path.join(state.project_dir, file_path)
                if file_path not in state.files_written:
                    state.files_written.append(file_path)
                    logger.debug("epistemic: wrote %s", file_path)
            elif outer_tool in ("bash", "powershell"):
                brief = tr[:120].replace("\n", " ").strip()
                state.commands_run.append({"cmd": outer_input[:100], "brief": brief})
                logger.debug("epistemic: ran cmd len=%d", len(outer_input))

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

        # ── Resume active write plan ───────────────────────────────────────────
        # Sisyphean writes one section at a time — each section is a separate
        # Write outer-tool call.  After each result _continue advances wp_idx.
        if state.wp_items:
            if state.wp_idx < len(state.wp_items):
                return await self._write_plan_next_item(state)
            else:
                # All items done — reflect gate: check if content is deep enough.
                # One cheap LLM call. If "deepen", insert one more write_plan pass
                # targeting the specific gap (capped to 1 deepen to prevent loops).
                wp_goal_was  = state.wp_goal
                wp_file_was  = state.wp_file
                wp_ftype_was = state.wp_ftype
                _already_deepened = wp_goal_was.startswith("Deepen:")

                if not _already_deepened:
                    deepen_goal = await self._reflect_write_plan(
                        state.wp_goal, state.wp_file
                    )
                    if deepen_goal:
                        logger.info("pipeline: reflect gate → deepen: %s", deepen_goal[:80])
                        deepen_step = {
                            "tool":      "write_plan",
                            "input":     f"Deepen: {deepen_goal}",
                            "file_path": wp_file_was,
                            "file_type": wp_ftype_was,
                        }
                        try:
                            sub = state.sub_tasks[state.current_task_idx]
                            sub["steps"].insert(state.current_step_idx + 1, deepen_step)
                        except (IndexError, KeyError):
                            pass

                # Optionally run the file after writing
                if state.wp_run_after and state.wp_file:
                    run_step = {"tool": "bash", "input": f"python {state.wp_file}"}
                    try:
                        sub = state.sub_tasks[state.current_task_idx]
                        sub["steps"].insert(state.current_step_idx + 1, run_step)
                    except (IndexError, KeyError):
                        pass
                logger.info("pipeline: write plan done (%d items) → %s",
                            len(state.wp_items), state.wp_file)
                state.wp_items       = []
                state.wp_goal        = ""
                state.wp_file        = ""
                state.wp_ftype       = ""
                state.wp_idx         = 0
                state.wp_run_after   = False
                state.wp_retry_count = 0
                state.wp_resume_ctx  = ""
                state.current_step_idx += 1  # advance past the write_plan step

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

                # ── Graph-first: check knowledge graph before any web search ────
                # If the graph already has research relevant to this query,
                # use it directly and skip the web search entirely.
                # This applies to both the internal web_search and the outer
                # WebSearch upgrade — the graph check happens before either.
                if tool in ("web_search", "websearch") and self.graph:
                    mem = _search_graph(inp, self.graph)
                    if mem and len(mem.strip()) > 40:
                        logger.info(
                            "pipeline: graph-first hit for web_search %r — skipping web, using graph",
                            inp[:50],
                        )
                        result = {
                            "tool": "search_memory",
                            "input": inp,
                            "result": mem,
                            "summary": f"recalled from graph: {mem[:120]}",
                        }
                        state.results.append(result)
                        _tracker.tree_subtask_step(
                            state.task_id, state.current_task_idx,
                            "search_memory", inp, result["summary"], status="done",
                        )
                        state.current_step_idx += 1
                        continue

                # ── Smart upgrade: use Claude Code's WebSearch/WebFetch if available ─
                if tool == "web_search" and "websearch" in outer_tool_names:
                    tool = "websearch"
                    step["tool"] = "websearch"
                elif tool in ("web_fetch", "fetch_url") and "webfetch" in outer_tool_names:
                    tool = "webfetch"
                    step["tool"] = "webfetch"

                # ── direct — skip all steps, synthesizer answers from query ──
                # Used for greetings, thanks, simple questions where no tool
                # call is needed. Break out of both step and task loops.
                if tool == "direct":
                    logger.info("pipeline: direct step → skipping to synthesizer")
                    state.current_step_idx = len(steps)   # exhaust steps
                    break

                # ── write_plan — start the incremental write pipeline ─────────
                # Not an outer tool; handled entirely inside Sisyphean.
                # Runs the subtask planner to get items, then yields to _execute
                # which emits one Write outer-tool per item via _write_plan_next_item.
                if tool == "write_plan":
                    return await self._start_write_plan(step, state)

                # ── Outer tool → return tool_use to Claude Code ───────────────
                if tool in outer_tool_names:
                    # ── Deduplication guard ───────────────────────────────────
                    # Prevent the same (tool, input) from being dispatched twice.
                    # A replan loop can otherwise generate identical web searches
                    # indefinitely when the small model can't decide what to do.
                    if tool in ("websearch", "webfetch", "web_search"):
                        norm_inp = inp.strip().lower()
                        already_run = any(
                            r.get("tool", "").lower() in ("websearch", "web_search", "webfetch")
                            and r.get("input", "").strip().lower() == norm_inp
                            for r in state.results
                        )
                        if already_run:
                            logger.warning(
                                "pipeline: skipping duplicate outer %s(%r) — "
                                "already in results", tool, inp[:60]
                            )
                            state.current_step_idx += 1
                            continue
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
                        if not file_path or not file_path.strip():
                            # Malformed write step with no file path — skip it
                            logger.warning("pipeline: skipping write step with empty file_path")
                            state.current_step_idx += 1
                            continue
                        if state.project_dir and not os.path.isabs(file_path):
                            file_path = os.path.join(state.project_dir, file_path)
                        # Guard: if file_path resolves to a directory, skip
                        if os.path.isdir(file_path):
                            logger.warning("pipeline: skipping write step — path is a directory: %s", file_path)
                            state.current_step_idx += 1
                            continue
                        # Lazy content regeneration: if files have already been written
                        # in this session, regenerate the content with epistemic context
                        # so the new file is consistent with what already exists.
                        # Only for code/doc files when we have real context to add.
                        _code_exts = (".py", ".js", ".ts", ".sh", ".rb", ".go",
                                      ".java", ".rs", ".md", ".txt", ".html", ".css")
                        if (state.files_written
                                and file_path.lower().endswith(_code_exts)
                                and inp.strip()):  # skip empty write steps
                            ep = _epistemic_block(state)
                            if ep:
                                from engine.translation.planner import _generate_code
                                sub = state.sub_tasks[state.current_task_idx]
                                task_goal = sub.get("task", state.query)
                                enriched = (
                                    f"{task_goal}\n\n{ep}\n\n"
                                    f"Now write file: {os.path.basename(file_path)}"
                                )
                                try:
                                    new_content = await _generate_code(enriched, self.client)
                                    if new_content and len(new_content) > 20:
                                        inp = new_content
                                        step["input"] = new_content
                                        logger.info(
                                            "pipeline: regenerated %s with epistemic context "
                                            "(%d chars, %d prior files)",
                                            os.path.basename(file_path), len(inp),
                                            len(state.files_written),
                                        )
                                except Exception as _regen_exc:
                                    logger.debug("pipeline: write regen failed: %s", _regen_exc)
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
                        _prior_qs = [
                            r.get("input", "") for r in state.results
                            if r.get("tool", "").lower() in ("web_search", "websearch", "webfetch")
                            and r.get("input", "")
                        ]
                        new_steps = await self._replan_after_search(
                            sub["task"], found, available_tools,
                            soul_section=state.soul_section,
                            user_prefs=state.user_prefs,
                            epistemic=_epistemic_block(state),
                            prior_queries=_prior_qs,
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

        # When there are zero results (steps=[] — conversational/direct query),
        # don't pass synthesis_ctx to the synthesizer. The 0.6b model echoes
        # injected context (cwd, user prefs) instead of answering when there's
        # nothing to synthesize. Give it a clean call with only soul guidance.
        has_real_results = any(
            not r.get("outer") and r.get("result") is not None
            for r in state.results
        )
        # Always pass conversation context — synthesis_ctx is history + graph recall,
        # NOT env/cwd. The dual synthesizer system prompts (_SYSTEM_WITH_RESULTS vs
        # _SYSTEM_NO_RESULTS) handle framing; context helps the model answer in-thread
        # (e.g. "do you think you're alive?" benefits from the prior philosophy turns).
        synth_ctx = state.synthesis_ctx

        result_summary = "; ".join(r.get("summary", "")[:60] for r in state.results[-3:])
        _tracker.tree_synthesizer_running(state.task_id, result_summary)
        t0 = _time.time()
        answer = await synthesize(
            state.query,
            state.soul_section,
            state.user_prefs,
            state.results,
            self.client,
            context=synth_ctx,
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
        epistemic: str = "",
        prior_queries: list[str] | None = None,
    ) -> list[dict]:
        """Derive follow-up execution steps from an actual search/memory result.

        Replaces pre-guessed steps with steps grounded in what was found.
        Soul guidance and user preferences shape which steps are chosen and how.
        Returns [] if the result already answers the task (no execution needed),
        or on any failure (caller keeps original steps in that case).

        prior_queries: list of search queries already executed this session —
        passed to the model so it never repeats a query verbatim.
        """
        logger.debug("pipeline: replan from search result for task=%r", task[:60])
        soul_block = f"\nPersonality guidance (follow this):\n{soul_section[:300]}" if soul_section else ""
        prefs_block = f"\nUser preferences:\n{user_prefs[:200]}" if user_prefs else ""
        ep_block = f"\n\n{epistemic}" if epistemic else ""
        already_block = ""
        if prior_queries:
            qs = "\n".join(f"  - {q}" for q in prior_queries[:8])
            already_block = f"\n\nAlready searched (do NOT repeat these queries):\n{qs}"
        try:
            r = await self.client.generate(
                [
                    {"role": "system", "content": _REPLAN_SYSTEM + soul_block + prefs_block},
                    {"role": "user",   "content": f"Task: {task}{ep_block}{already_block}\n\nFound:\n{search_result[:3000]}"},
                ],
                max_tokens=500,
                temperature=0.1,
                response_format={"type": "json_object"},
                stream=False,
                thinking=True,
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
                state.query, result, file_path,
                epistemic=_epistemic_block(state),
            )
            if new_steps:
                sub["steps"].extend(new_steps)
                _tracker.tree_subtask_replanned(state.task_id, state.current_task_idx, new_steps)
            else:
                logger.warning("pipeline: _generate_edit_steps returned nothing — skipping edit")

        elif tool in ("websearch", "webfetch"):
            # Informational queries with a non-empty result don't need replan —
            # the websearch answer goes straight to synthesis.
            # (Same logic as glob/grep informational guard above.)
            if _is_informational_query(state.query) and result and len(result.strip()) > 50:
                logger.info(
                    "pipeline: %s informational query with results — skipping replan, "
                    "going to synthesis", tool
                )
                return
            # Delegate to the existing search-result replan (derives bash/save steps)
            logger.info("pipeline: %s→replan from web result", tool)
            _prior_qs = [
                r.get("input", "") for r in state.results
                if r.get("tool", "").lower() in ("web_search", "websearch", "webfetch")
                and r.get("input", "")
            ]
            new_steps = await self._replan_after_search(
                state.query, result, available_tools,
                soul_section=state.soul_section,
                user_prefs=state.user_prefs,
                epistemic=_epistemic_block(state),
                prior_queries=_prior_qs,
            )
            if new_steps:
                sub["steps"].extend(new_steps)
                _tracker.tree_subtask_replanned(state.task_id, state.current_task_idx, new_steps)

    # ── Incremental write plan ────────────────────────────────────────────────

    async def _start_write_plan(self, step: dict, state: PipelineState) -> LoopResponse:
        """Run the subtask planner and populate state.wp_* for item-by-item writing.

        Called when a write_plan meta-step is encountered in _execute.
        After populating the plan, returns control to _execute which calls
        _write_plan_next_item for each item.
        """
        from engine.translation.subtask.planner import plan as _subtask_plan

        task      = step.get("input", state.query)
        file_path = step.get("file_path", "")
        file_type = step.get("file_type", "doc")
        run_after = bool(step.get("run_after", False))

        # Resolve file_path against project_dir
        if file_path and state.project_dir and not os.path.isabs(file_path):
            file_path = os.path.join(state.project_dir, file_path)

        # Read whatever is already on disk — verifier will detect completed items
        existing = ""
        if file_path and os.path.isfile(file_path):
            try:
                existing = Path(file_path).read_text(encoding="utf-8")
            except Exception:
                pass

        logger.info("pipeline: write_plan start  goal=%r  file=%s  type=%s",
                    task[:60], file_path, file_type)

        try:
            manifest = await _subtask_plan(
                client=self.client,
                stage_goal=task,
                file_path=file_path,
                file_type=file_type,
                existing_content=existing,
            )
            items = [
                {"anchor": it.anchor, "title": it.title, "min_chars": it.expected_min_chars}
                for it in manifest.items
                if it.status not in ("complete",)  # skip items already on disk
            ]
        except Exception as exc:
            logger.warning("pipeline: write_plan subtask planner failed: %s — single write", exc)
            items = []

        if not items:
            # Fallback: generate whole file in one shot
            from engine.translation.planner import _generate_code
            content = await _generate_code(task, self.client)
            if content and file_path:
                tool_id = f"toolu_{uuid.uuid4().hex[:16]}"
                state.current_step_idx += 1  # advance past this step
                # Ensure path is absolute
                if state.project_dir and not os.path.isabs(file_path):
                    file_path = os.path.join(state.project_dir, file_path)
                state.files_written.append(file_path)
                return LoopResponse(
                    content=[
                        {"type": "thinking", "thinking": f"{_STATE_PREFIX}{state.to_json()}"},
                        {"type": "tool_use", "id": tool_id, "name": "Write",
                         "input": {"file_path": file_path, "content": content}},
                    ],
                    stop_reason="tool_use",
                )
            state.current_step_idx += 1
            return await self._execute(state, [])

        state.wp_items       = items
        state.wp_goal        = task
        state.wp_file        = file_path
        state.wp_ftype       = file_type
        state.wp_idx         = 0
        state.wp_run_after   = run_after
        state.wp_retry_count = 0
        state.wp_resume_ctx  = ""

        logger.info("pipeline: write_plan %d item(s) to write for %s", len(items), file_path)
        return await self._write_plan_next_item(state)

    async def _write_plan_next_item(self, state: PipelineState) -> LoopResponse:
        """Generate content for the current write-plan item and emit a Write outer tool.

        Reads the current file from disk to provide context so each section
        is coherent with what was already written (the + / append path).

        Uses grammar-constrained JSON output (WRITE_ITEM_SCHEMA) so the model
        cannot stop early or output preambles — it must fill the content field.
        Conversation context (synthesis_ctx) is injected so prior turns
        (e.g. a "meaning of life" web search or "are you alive?" reflection)
        inform the writing, just as BirdClaw's planning_context did.
        """
        from engine.translation.planner import parse_format_response

        item_meta = state.wp_items[state.wp_idx]
        anchor    = item_meta["anchor"]
        title     = item_meta["title"]
        min_chars = item_meta["min_chars"]
        file_path = state.wp_file
        file_type = state.wp_ftype

        # ── Read current file to understand what's already on disk ────────────
        current_content = ""
        if file_path and os.path.isfile(file_path):
            try:
                current_content = Path(file_path).read_text(encoding="utf-8")
            except Exception:
                pass

        # ── Build a focused prompt for this one item ──────────────────────────
        tail_lines = current_content.splitlines()[-20:] if current_content else []
        tail = "\n".join(tail_lines)
        file_state = (
            f"Current file tail:\n{tail}"
            if tail else f"[{os.path.basename(file_path)} — empty, start fresh]"
        )

        if file_type == "code":
            anchor_hint = f"Your content MUST start with exactly: def {anchor}( or class {anchor}:"
            type_reminder = ""
        else:
            anchor_hint = f"Your content MUST start with exactly: ## {anchor}"
            type_reminder = "Write PROSE paragraphs — NO code, NO import statements, NO def/class lines."

        done_items = [it["title"] for it in state.wp_items[:state.wp_idx]]
        done_str = ", ".join(done_items) or "none yet"

        # ── Inject conversation context — prior turns inform the writing ──────
        # For code files: do NOT inject synthesis_ctx — it may contain essay/doc
        # context from prior conversation turns (section headings, "Introduction",
        # "Conclusion" etc.) that causes the model to generate functions with those names.
        if file_type == "code":
            conv_ctx = ""
        else:
            conv_ctx = state.synthesis_ctx[:800].strip() if state.synthesis_ctx else ""

        prompt_parts = []
        if conv_ctx:
            prompt_parts.append(f"Conversation context (draw on this for depth and specifics):\n{conv_ctx}")

        # ── Inject verifier resume context when retrying a partial/missing item ─
        if state.wp_resume_ctx:
            prompt_parts.append(
                f"VERIFICATION FEEDBACK (previous attempt was incomplete — fix this):\n"
                f"{state.wp_resume_ctx}"
            )

        if state.wp_retry_count > 0:
            # Retry: the partial content will be stripped and replaced
            section_line = (
                f"RETRY section: {title} — previous attempt was incomplete.\n"
                f"{anchor_hint}\n"
                f"Write a COMPLETE version with at least {min_chars} chars of body text.\n"
                + (f"{type_reminder}\n" if type_reminder else "")
                + "The partial content above will be replaced — write the full version now."
            )
        else:
            section_line = (
                f"Write next section: {title} (minimum {min_chars} chars of body text)\n"
                f"{anchor_hint}\n"
                f"Then write at least {min_chars} chars of detailed body text below the heading.\n"
                + (f"{type_reminder}\n" if type_reminder else "")
                + "Do NOT repeat any content already in the file tail above."
            )
        prompt_parts.append(
            f"Goal: {state.wp_goal}\n"
            f"Sections already written: {done_str}\n\n"
            f"{file_state}\n\n"
            f"{section_line}"
        )
        prompt = "\n\n".join(prompt_parts)

        logger.info("pipeline: write_plan item %d/%d: %r  conv_ctx=%d chars",
                    state.wp_idx + 1, len(state.wp_items), title[:50], len(conv_ctx))

        # ── Generate with grammar-constrained JSON ────────────────────────────
        # thinking=False: Ollama with think=true + json_schema causes the model
        # to spend its budget on reasoning then output minimal JSON content.
        # With thinking=False the model writes directly into the content field.
        content = ""
        _min_acceptable = max(60, min_chars // 3)
        for _attempt in range(2):
            try:
                r = await self.client.generate(
                    [
                        {"role": "system", "content": _WRITE_ITEM_SYSTEM},
                        {"role": "user",   "content": prompt},
                    ],
                    max_tokens=1500,
                    temperature=0.3,
                    response_format=_WRITE_ITEM_SCHEMA,
                    stream=False,
                    thinking=False,
                )
                raw = (r["choices"][0]["message"]["content"] or "").strip()
                parsed = parse_format_response(raw) if raw else None
                if parsed and isinstance(parsed.get("content"), str):
                    candidate = parsed["content"].strip()
                    # Guard: model sometimes double-wraps — {"content": "{\"content\": \"...\"}"}
                    # Detect and unwrap one level.
                    if candidate.startswith('{"content"') or candidate.startswith("{'content'"):
                        inner = parse_format_response(candidate)
                        if inner and isinstance(inner.get("content"), str):
                            candidate = inner["content"].strip()
                    # Strip any wrapping <section ...>...</section> XML tags the model sometimes adds
                    candidate = re.sub(r'^<section[^>]*>\s*', '', candidate)
                    candidate = re.sub(r'\s*</section>\s*$', '', candidate)
                    candidate = candidate.strip()
                    if len(candidate) >= _min_acceptable:
                        content = candidate
                        break
                    elif candidate and _attempt == 1:
                        content = candidate  # accept short on last attempt
                    logger.debug("pipeline: write_plan item %r attempt %d: too short (%d < %d)",
                                 title[:40], _attempt, len(candidate), _min_acceptable)
                # Do NOT write raw JSON to file — only use raw if it looks like prose, not JSON
            except Exception as exc:
                logger.warning("pipeline: write_plan item %r attempt %d failed: %s",
                               title[:40], _attempt, exc)

        if not content:
            # Final fallback: plain text generation without schema constraint
            file_type = state.wp_ftype
            try:
                if file_type == "code":
                    from engine.translation.planner import _generate_code
                    content = await _generate_code(prompt, self.client) or ""
                else:
                    # Doc fallback: direct prose call, no JSON schema
                    r_fb = await self.client.generate(
                        [{"role": "system", "content": _WRITE_ITEM_SYSTEM},
                         {"role": "user", "content": prompt}],
                        max_tokens=800, temperature=0.4,
                        stream=False, thinking=False,
                    )
                    fb_raw = (r_fb["choices"][0]["message"]["content"] or "").strip()
                    # Extract from JSON if model still wraps it
                    fb_parsed = parse_format_response(fb_raw)
                    content = (fb_parsed.get("content", "") if fb_parsed else fb_raw) or ""
            except Exception:
                content = ""

        if not content or len(content) < 20:
            logger.warning("pipeline: write_plan item %r: empty/short output — skipping",
                           title[:40])
            state.wp_idx += 1
            if state.wp_idx < len(state.wp_items):
                return await self._write_plan_next_item(state)
            state.wp_items = []
            state.current_step_idx += 1
            return await self._execute(state, [])

        # ── Deduplicate / retry-merge ─────────────────────────────────────────
        # If the anchor already exists in the file we need to decide:
        #  • body meets min_chars → already complete, skip this item
        #  • body is partial      → strip existing partial, rewrite (retry path)
        if current_content and anchor:
            if file_type == "code":
                _anchor_pat = re.compile(
                    r'^\s*(def|class)\s+' + re.escape(anchor) + r'\b',
                    re.MULTILINE,
                )
            else:
                _anchor_pat = re.compile(
                    r'^#{1,3}\s+' + re.escape(anchor) + r'\s*$',
                    re.MULTILINE | re.IGNORECASE,
                )
            if _anchor_pat.search(current_content):
                # Check if the existing body is already complete
                try:
                    from engine.translation.subtask.verifier import (
                        parse_doc_sections, parse_code_items,
                        _match_key, is_stub_body,
                    )
                    _parsed_ck = (parse_doc_sections(current_content)
                                  if file_type == "doc"
                                  else parse_code_items(current_content))
                    _key_ck = _match_key(anchor, _parsed_ck)
                    _body_ck = _parsed_ck.get(_key_ck, "") if _key_ck else ""
                    _stub_ck = file_type == "code" and is_stub_body(_body_ck)
                    _already_complete = (not _stub_ck and len(_body_ck) >= min_chars)
                except Exception:
                    _already_complete = False  # be conservative — try to write

                if _already_complete:
                    logger.info(
                        "pipeline: write_plan item %r already complete (%dc) — skipping",
                        title[:40], len(_body_ck),
                    )
                    state.wp_idx += 1
                    if state.wp_idx < len(state.wp_items):
                        return await self._write_plan_next_item(state)
                    state.wp_items = []
                    state.current_step_idx += 1
                    return await self._execute(state, [])
                else:
                    # Partial content exists — strip it out so we don't create
                    # a duplicate heading when we append the new content below.
                    logger.info(
                        "pipeline: write_plan item %r partial in file — stripping for rewrite",
                        title[:40],
                    )
                    current_content = _strip_item_from_content(
                        current_content, anchor, file_type
                    )

        # ── Build full file content: existing + separator + new section ───────
        separator = "\n\n" if current_content else ""
        new_full_content = current_content + separator + content

        tool_id = f"toolu_{uuid.uuid4().hex[:16]}"
        _tracker.tree_subtask_step(
            state.task_id, state.current_task_idx,
            "write", f"{title} → {file_path}", "", status="running",
        )
        return LoopResponse(
            content=[
                {"type": "thinking", "thinking": f"{_STATE_PREFIX}{state.to_json()}"},
                {"type": "tool_use", "id": tool_id, "name": "Write",
                 "input": {"file_path": file_path, "content": new_full_content}},
            ],
            stop_reason="tool_use",
        )

    async def _reflect_write_plan(self, goal: str, file_path: str) -> str:
        """One cheap LLM call after all write-plan items complete.

        Reads the written file, compares against the goal, and returns a
        specific deepen instruction if important depth is missing — or '' if
        the content is good enough.  Falls back to '' on any error so the
        gate never blocks the pipeline.
        """
        if not file_path or not goal:
            return ""
        try:
            current = Path(file_path).read_text(encoding="utf-8")
        except Exception:
            return ""
        if not current.strip():
            return ""
        # Cap the content so this stays a cheap call
        content_preview = current[:2000]
        try:
            r = await self.client.generate(
                [
                    {"role": "system", "content": _REFLECT_SYSTEM},
                    {"role": "user",   "content":
                        f"Goal: {goal[:300]}\n\nWritten content:\n{content_preview}"},
                ],
                max_tokens=200,
                temperature=0.1,
                response_format={"type": "json_object"},
                stream=False,
                thinking=False,
            )
            raw  = (r["choices"][0]["message"]["content"] or "").strip()
            from engine.translation.planner import parse_format_response
            data = parse_format_response(raw) or {}
            decision = data.get("decision", "")
            if decision == "deepen" and data.get("goal"):
                return str(data["goal"])[:200]
        except Exception as exc:
            logger.debug("pipeline: _reflect_write_plan failed: %s", exc)
        return ""

    async def _generate_edit_steps(
        self, task: str, file_content: str, file_path: str,
        epistemic: str = "",
    ) -> list[dict]:
        """Given a task and current file content, produce Edit or Write steps.

        Two-phase: small decision call first (edit vs write + old/new strings),
        then _generate_code/write_plan for the write path — avoids asking the LLM to output
        an entire file in a single constrained JSON call.

        Doc files (.md, .txt, .rst) use write_plan rather than _generate_code to avoid
        the code generator writing Python into document files.
        """
        from engine.translation.planner import parse_format_response, _generate_code
        _DOC_EXTS = {".md", ".txt", ".rst", ".html"}
        _is_doc = Path(file_path).suffix.lower() in _DOC_EXTS
        ep_note = f"\n\n{epistemic}" if epistemic else ""
        try:
            r = await self.client.generate(
                [
                    {"role": "system", "content": _EDIT_SYSTEM},
                    {"role": "user", "content":
                        f"Task: {task}{ep_note}\n\nFile:\n{file_content[:3000]}"},
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

            # write mode (or edit fallback)
            logger.info("pipeline: write mode — regenerating %s", file_path)

            # Doc files: use write_plan (subtask writer) not _generate_code, which outputs Python
            if _is_doc:
                logger.info("pipeline: doc file — routing to write_plan for %s", file_path)
                steps: list[dict] = [{"tool": "write_plan", "input": task,
                                      "file_path": file_path, "file_type": "doc"}]
                # If task mentions converting to .doc/.docx, append a bash conversion after
                if re.search(r'\bconvert\b', task, re.I) and re.search(r'\b\.?docx?\b', task, re.I):
                    stem = Path(file_path).stem
                    conv_cmd = (
                        f"python -c \""
                        f"import glob, os; "
                        f"src = {file_path!r}; "
                        f"out = os.path.splitext(src)[0]+'.docx'; "
                        f"import docx; d=docx.Document(); "
                        f"[d.add_paragraph(l.rstrip()) for l in open(src, encoding='utf-8')]; "
                        f"d.save(out); print('Saved', out)\""
                    )
                    steps.append({"tool": "bash", "input": conv_cmd})
                    logger.info("pipeline: appended .docx conversion after write_plan")
                return steps

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
        """Save web search result summary directly to the knowledge graph.

        No LLM call — raw result is stored as-is.  The dream cycle consolidates
        and distils reusable knowledge during off-peak hours.
        """
        if not self.graph or not result:
            return
        try:
            summary = result[:500].replace("\n", " ").strip()
            self.graph.upsert_node(
                name=query[:80],
                node_type="research",
                summary=summary,
                sources=["web_search"],
            )
            self.graph.save()
            logger.debug("pipeline: saved search result to graph for %r", query[:40])
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
            raw = []
            try:
                raw = await _web_search(inp, max_results=4)
                content = format_results(raw) if raw else "No results."
            except Exception as exc:
                content = f"Search failed: {exc}"
            # ── Auto-fetch page content when snippets are thin ────────────────
            # Jina AI tier (is_ai_synthesized=True) already returns rich content.
            # DDG / SearXNG tiers return short snippets (≤500 chars) that are
            # often too thin for the model to act on. Fetch the top pages to give
            # the same deep-content behaviour BirdClaw's web_fetch tool provided.
            if raw and not any(getattr(r, "is_ai_synthesized", False) for r in raw):
                fetched_parts: list[str] = []
                for r in raw[:3]:
                    url = getattr(r, "url", "") or ""
                    if not url.startswith("http"):
                        continue
                    try:
                        page = await _web_fetch_page(url, goal=inp, client=self.client)
                        if page and len(page) > 150 and not page.startswith("("):
                            fetched_parts.append(f"[{getattr(r, 'title', url)}]\n{url}\n{page[:2000]}")
                    except Exception as _fe:
                        logger.debug("auto-fetch failed for %s: %s", url[:60], _fe)
                    if len(fetched_parts) >= 2:
                        break
                if fetched_parts:
                    content += "\n\n### Page Content\n\n" + "\n\n---\n\n".join(fetched_parts)
            # Extract and save reusable procedural knowledge (not raw data) to graph.
            # Live data (status, metrics) is filtered out by the extraction LLM.
            if content and "No results" not in content and "failed" not in content:
                await self._save_research_to_graph(inp, content)
            return {"tool": tool, "input": inp, "result": content, "summary": content[:2000]}

        if tool in ("web_fetch", "fetch_url"):
            # Fetch a specific URL and return its cleaned text content.
            # Triggered by replan steps like "web_fetch:https://docs.example.com/page"
            # when search snippets are too thin to act on directly.
            url = inp.strip()
            if not url.startswith("http"):
                return {"tool": tool, "input": url, "result": "(invalid URL — must start with http)",
                        "summary": "invalid URL"}
            current_goal = (
                state.sub_tasks[state.current_task_idx].get("task", "")
                if state.sub_tasks and state.current_task_idx < len(state.sub_tasks)
                else ""
            )
            try:
                page = await _web_fetch_page(url, goal=current_goal or inp, client=self.client)
                if page and not page.startswith("("):
                    logger.info("pipeline: web_fetch ok  url=%s  chars=%d", url[:60], len(page))
                    return {"tool": tool, "input": url, "result": page,
                            "summary": page[:200]}
                return {"tool": tool, "input": url, "result": page or "(empty page)",
                        "summary": "empty or unsupported page"}
            except Exception as exc:
                logger.warning("pipeline: web_fetch error  url=%s  %s", url[:60], exc)
                return {"tool": tool, "input": url, "result": f"(fetch error: {exc})",
                        "summary": f"fetch error: {exc}"}

        if tool in ("save_memory", "remember"):
            # user_prefs are BirdClaw's domain — CLAUDE.md handles prefs in Claude Code.
            # Just write to graph so the fact is recalled during future requests.
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


def _extract_birdclaw_plan(raw_history: list[dict]) -> "tuple[str, list[dict]] | None":
    """Detect a pre-made plan injected by BirdClaw's generate_plan() call.

    BirdClaw inserts a user message of the form:
        [BirdClaw-Plan]
        outcome: <one-sentence success criteria>
        steps: step1 | step2 | step3

    before delegating to Sisyphean.  When found, return (outcome, stages) so
    _start() can skip think_decompose() and avoid a duplicate planning call.
    Returns None if no such block is found (standalone Sisyphean / Claude Code).
    """
    _MARKER = "[BirdClaw-Plan]"
    for msg in raw_history:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        if _MARKER not in content:
            continue
        # Parse outcome and steps lines
        outcome = ""
        steps_raw = ""
        for line in content.splitlines():
            line = line.strip()
            if line.lower().startswith("outcome:"):
                outcome = line[len("outcome:"):].strip()
            elif line.lower().startswith("steps:"):
                steps_raw = line[len("steps:"):].strip()
        if not steps_raw:
            return None
        plain_steps = [s.strip() for s in steps_raw.split("|") if s.strip()]
        if not plain_steps:
            return None
        stages: list[dict] = [
            {"type": infer_stage_type(step), "goal": step}
            for step in plain_steps
        ]
        if not outcome:
            outcome = plain_steps[0]
        logger.info("pipeline: BirdClaw plan detected — outcome=%r stages=%d", outcome[:60], len(stages))
        return outcome, stages
    return None


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


def _file_head(path: str, n: int = 5) -> str:
    """Return the first *n* non-blank lines of *path*, or '' on any error."""
    try:
        lines: list[str] = []
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                stripped = line.rstrip()
                if stripped:
                    lines.append(stripped)
                    if len(lines) >= n:
                        break
        return "\n".join(lines)
    except OSError:
        return ""


def _workspace_snapshot(workspace: str, max_files: int = 15) -> str:
    """Return a compact file-tree with first-line previews for non-empty files.

    Skips hidden dirs, __pycache__, .venv, .git, *.pyc.
    Truncates at *max_files* entries so the snapshot stays token-lean.
    """
    if not workspace or not os.path.isdir(workspace):
        return ""
    try:
        lines: list[str] = [f"Workspace ({workspace}):"]
        count = 0
        for root, dirs, files in os.walk(workspace):
            dirs[:] = sorted(
                d for d in dirs
                if not d.startswith(".")
                and d not in ("__pycache__", ".git", "node_modules", ".venv", "venv")
            )
            rel_root = os.path.relpath(root, workspace)
            depth = 0 if rel_root == "." else rel_root.count(os.sep) + 1
            indent = "  " * (depth + 1)
            for fname in sorted(files):
                if fname.startswith(".") or fname.endswith(".pyc"):
                    continue
                fpath = os.path.join(root, fname)
                rel_path = os.path.relpath(fpath, workspace)
                head = _file_head(fpath, 3)
                if head:
                    preview = head.replace("\n", " / ")[:100]
                    lines.append(f"{indent}{rel_path} → {preview}")
                else:
                    lines.append(f"{indent}{rel_path} (empty)")
                count += 1
                if count >= max_files:
                    lines.append(f"{indent}... ({count} files shown)")
                    return "\n".join(lines)
        return "\n".join(lines) if count > 0 else ""
    except Exception:
        return ""


def _epistemic_block(state: "PipelineState") -> str:
    """Format the current epistemic state as a compact context string.

    Injected into LLM calls so the model reasons from *what has actually been
    done* rather than relying on imperfect memory of prior turns.
    Returns '' when there is nothing worth reporting.
    """
    parts: list[str] = []

    if state.files_read:
        items = []
        for r in state.files_read[-8:]:
            p = r.get("path", "?")
            h = r.get("head", "")
            if h:
                items.append(f"  {p} → \"{h[:100].replace(chr(10), ' / ')}\"")
            else:
                items.append(f"  {p}")
        parts.append("Files read:\n" + "\n".join(items))

    if state.files_written:
        parts.append("Files written:\n" + "\n".join(
            f"  {p}" for p in state.files_written[-10:]
        ))

    if state.commands_run:
        items = [
            f"  {r.get('cmd','')[:60]} → {r.get('brief','')[:70]}"
            for r in state.commands_run[-5:]
        ]
        parts.append("Commands run:\n" + "\n".join(items))

    if not parts:
        return ""

    # Append a live workspace snapshot when we have written files — gives the
    # model import signatures and function stubs already on disk.
    if state.project_dir and state.files_written:
        snap = _workspace_snapshot(state.project_dir)
        if snap:
            parts.append(snap)

    return "[Current session state]\n" + "\n\n".join(parts)


def _strip_item_from_content(content: str, anchor: str, file_type: str) -> str:
    """Remove an existing partial item (section or function) from file content.

    Used when retrying a write-plan item: strips the old partial body so that
    the newly generated content can be appended without creating a duplicate heading.

    For docs: removes from '## anchor' to the next '##' heading (or end of file).
    For code: removes the def/class block for anchor up to the next top-level def/class.
    """
    if not content or not anchor:
        return content

    if file_type == "doc":
        # Match from ## anchor to next ## heading or end-of-string
        pattern = re.compile(
            r'^#{1,3}\s+' + re.escape(anchor) + r'[^\n]*\n.*?(?=^#{1,3}\s|\Z)',
            re.MULTILINE | re.DOTALL | re.IGNORECASE,
        )
        return pattern.sub("", content).rstrip()

    else:  # code
        lines = content.splitlines(keepends=True)
        result: list[str] = []
        skip = False
        for line in lines:
            if re.match(r'^(def |class )' + re.escape(anchor) + r'\b', line):
                skip = True
                continue
            if skip:
                # End of the current def/class block: next top-level def/class or non-blank non-indented
                if line.rstrip() and not line[0].isspace() and re.match(r'^(def |class |\S)', line):
                    skip = False
            if not skip:
                result.append(line)
        return "".join(result).rstrip()


def _build_verify_resume_ctx(
    wp_items: list[dict], wp_idx: int, file_content: str, file_type: str
) -> str:
    """Build a human-readable verification context showing item status + file tail.

    Injected into the retry prompt so the model knows exactly what is missing
    and where to continue writing.  Mirrors verifier._build_resume_context but
    works directly from wp_items without a full SubtaskManifest object.
    """
    try:
        from engine.translation.subtask.verifier import (
            parse_doc_sections, parse_code_items, _match_key, is_stub_body,
        )
        parsed = (parse_doc_sections(file_content)
                  if file_type == "doc" else parse_code_items(file_content))
    except Exception:
        parsed = {}

    lines: list[str] = []
    for i, item in enumerate(wp_items):
        anchor    = item["anchor"]
        title     = item["title"]
        min_chars = item["min_chars"]
        key = _match_key(anchor, parsed) if parsed else None

        if key:
            body = parsed[key]
            stub = file_type == "code" and is_stub_body(body)
            if not stub and len(body) >= min_chars:
                lines.append(f"  ✓ [{i}] {title}  ({len(body)}c)")
            else:
                status = "stub" if stub else f"{len(body)}c (needs {min_chars}c)"
                arrow  = " ← retry this" if i == wp_idx else ""
                lines.append(f"  ~ [{i}] {title}  — {status}{arrow}")
        else:
            arrow = " ← write this" if i == wp_idx else ""
            lines.append(f"  ✗ [{i}] {title}  — missing{arrow}")

    if wp_idx < len(wp_items):
        item = wp_items[wp_idx]
        lines += [
            "",
            f'Resume at [{wp_idx}]: "{item["anchor"]}"',
            f"Minimum {item['min_chars']} chars of substantive content.",
            "Append the missing content below. Do NOT rewrite earlier sections.",
        ]

    if file_content:
        tail = file_content[-500:].strip()
        lines += ["", "--- current file tail ---", tail, "--- end ---"]

    return "\n".join(lines)


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
