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
import platform as _platform
import re
import time as _time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from engine.core.synthesizer import synthesize
from engine.core.recall import Recall
from engine.core.context_extractor import extract_for_task, filter_tools_for_task
from engine.translation.planner import split_deep, plan_task, think_decompose, infer_stage_type, parse_format_response, resolve_bash_command
from engine.memory.skills import (
    get_skill_index, get_skill_runbook, get_skill_program,
    get_skill_script_path, save_skill_to_disk, save_skill_program_to_graph,
)
from engine.translation.prompts import STAGE_BUDGETS
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
# All recognised spellings of web-search / web-fetch tools.
# Used to strip them from verify stages and to build websearch fallback steps.
_WEB_TOOL_NAMES = frozenset({"websearch", "webfetch", "web_search", "web_fetch"})
_OUTER_INFO_TOOLS = frozenset({"read", "glob", "grep"}) | _WEB_TOOL_NAMES

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
    - If a better query is genuinely needed → plan web_search with the improved query

IF the search returned system diagnostic commands (systeminfo, tasklist, Get-ComputerInfo,
  top, ps, df, etc.) — copy the EXACT command from the result and plan a bash step.
  Always pick the command for the OS that appears in the search result context.

RULES (enforce strictly):
  - NEVER output the exact same search query that was just run — always reformulate
  - A query that already ran once must not appear again in your steps
  - Copy commands VERBATIM from search results — never invent or paraphrase commands

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
    # ── Bash retry state ──────────────────────────────────────────────────────
    bash_retry_count: int = 0  # consecutive bash failures on current step
    last_bash_error:  str = ""  # traceback from most recent bash failure (cleared after fix edit)

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
            "brc": self.bash_retry_count,
            "lbe": self.last_bash_error[:400],
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
        obj.bash_retry_count = d.get("brc", 0)
        obj.last_bash_error  = d.get("lbe", "")
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
        budget_tracker=None,
    ) -> None:
        self.client  = client
        self.soul_path  = policy_path  # kept as soul_path internally for compatibility
        self.prefs_path = prefs_path
        self.graph   = knowledge_graph
        self.workspace = workspace
        self.budget_tracker = budget_tracker
        self.recall  = Recall(graph=knowledge_graph, workspace=workspace or ".")

    async def process(
        self,
        user_message: str,
        raw_history: list[dict],
        available_tools: list[dict],
        system_context: str = "",
        memory_ctx: str = "",
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

        # Gather recall context — pure Python, no LLM call.
        # Returns ≤100 words from relevant history turns + graph + mentioned files.
        recall_ctx = self.recall.gather(user_message, raw_history)

        recent_turns_text = _format_history(raw_history[-4:])   # last 2 user+asst pairs

        return await self._start(user_message, available_tools, session_id=session_id,
                                 history_text=history_text, project_ctx=project_ctx,
                                 project_dir=project_dir, bc_plan=bc_plan,
                                 recall_ctx=recall_ctx, memory_ctx=memory_ctx,
                                 recent_turns_text=recent_turns_text)

    # ── Fresh request ─────────────────────────────────────────────────────────

    async def _start(self, query: str, available_tools: list[dict], session_id: str = "",
                     history_text: str = "", project_ctx: str = "",
                     project_dir: str = "",
                     bc_plan: "tuple[str, list[dict]] | None" = None,
                     recall_ctx: str = "",
                     memory_ctx: str = "",
                     recent_turns_text: str = "") -> LoopResponse:
        from engine.policy.router import load_user_prefs
        from engine.soul.router import parse_soul_sections, match_soul_section

        task_id = _tracker.start_task(session_id=session_id, user_message=query)

        if self.soul_path.exists():
            _soul_sections = parse_soul_sections(self.soul_path)
            _, soul_section = match_soul_section(query, _soul_sections)
            if not soul_section:
                # No section matched — fall back to first 400 chars of full soul
                soul_section = self.soul_path.read_text(encoding="utf-8").strip()[:400]
        else:
            soul_section = ""
        user_prefs   = load_user_prefs(self.prefs_path)

        logger.info("pipeline.start: query=%r history=%d chars", query[:60], len(history_text))

        # ── Skill index — compact list of relevant skills for this query ──────
        # Layer 1 of progressive disclosure: name + 60-char summary per match.
        # Injected into think_decompose + plan_task so the model sees "I have a
        # skill for this" without loading any full runbook token cost.
        skill_index = get_skill_index(query, self.graph) if self.graph else ""
        if skill_index:
            logger.debug("pipeline: skill index hit for %r", query[:40])

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
                workspace=self.workspace,
                skill_index=skill_index,
            )
            _tracker.tree_context_done(task_id, f"decomposed into {len(stages)} stage(s)")
            logger.info("pipeline: outcome=%r stages=%s", outcome[:60],
                        [(s["type"], s["goal"][:40]) for s in stages])

        # Fall back only if think_decompose raised an exception (it now explicitly
        # returns a "direct" stage when the model decides steps="").
        if not stages:
            stages = [{"type": infer_stage_type(query), "goal": query}]

        # No hardcoded routing overrides — think_decompose() owns all routing decisions.

        # ── Project expansion: write_project stages → per-file write_code stages ──
        # When think_decompose returns a single write_project/write_code stage for a
        # project-scale task (e.g. "build a todo app"), call the project planner to
        # break it into ordered per-file stages.  Each file is then handled by the
        # existing write_plan pipeline — cross-file sigs flow via the memory graph.
        #
        # Only triggers when:
        #   a) at least one stage is write_project, OR
        #   b) the query looks like a project AND there's just one write_code stage
        has_write_project = any(s.get("type") == "write_project" for s in stages)
        write_code_stages  = [s for s in stages if s.get("type") in ("write_code", "write_project")]
        if has_write_project or (
            _is_project_query(query) and len(write_code_stages) == 1
        ):
            from engine.translation.project.planner import plan_project as _plan_project
            # Use the write_project/write_code stage's goal as the project description;
            # fall back to the full query if the stage goal is too vague (≤6 words).
            _proj_goal = (
                write_code_stages[0]["goal"]
                if write_code_stages and len(write_code_stages[0]["goal"].split()) > 6
                else query
            )
            try:
                _proj_files = await _plan_project(self.client, _proj_goal, self.workspace)
            except Exception as _pe:
                logger.warning("pipeline: project planner failed (%s) — single-stage fallback", _pe)
                _proj_files = []

            if len(_proj_files) >= 2:
                # Keep non-write stages (research, verify, etc.), replace write stages
                _non_write = [s for s in stages if s.get("type") not in ("write_code", "write_project")]
                _expanded  = [
                    {"type": "write_code",
                     "goal": f"Write {f['filename']}: {f['purpose']}"}
                    for f in _proj_files
                ]
                stages = _non_write + _expanded
                logger.info(
                    "pipeline: project expansion → %d files: %s",
                    len(_proj_files),
                    [f["filename"] for f in _proj_files],
                )

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
            if ws_snap:
                ws_block = f"[Workspace — write ALL task files here]\n{ws_snap}"
            elif self.workspace:
                ws_block = f"[Workspace — write ALL task files here: {self.workspace}]"
            else:
                ws_block = ""

            # Prepend the original query so plan_task always has the full user request,
            # even when the stage goal is a vague placeholder like "check system status".
            original_ctx = f"Original request: {query}" if task != query else ""
            # recall_ctx: ≤100-word summary from recent history + graph + files
            # placed first so it's never truncated by downstream context limits
            recall_block = f"[Recall]\n{recall_ctx}" if recall_ctx else ""
            task_ctx = "\n\n".join(p for p in (recall_block, original_ctx, task_history, graph_block, ws_block, project_ctx) if p)
            all_extracted.append(task_history)

            if graph_recall:
                logger.info("pipeline: graph recall hit for task=%r (%d chars)", task[:50], len(graph_recall))

            # ── Tool filtering (keyword score, synchronous) ───────────────────
            relevant_tools = filter_tools_for_task(task, available_tools)

            # ── Planning stage ────────────────────────────────────────────────
            # Stage type from think_decompose guides tool selection so plan_task
            # can skip the LLM for well-understood stage types.
            stage_type = stage.get("type", "")

            # Stage-type tool restriction — show only tools relevant to the stage.
            # Reduces model confusion on small models (BirdClaw A2 principle).
            _WRITE_TOOL_NAMES = frozenset({"write", "edit"})
            _READ_TOOL_NAMES  = frozenset({"read", "glob", "grep"})
            if stage_type == "verify":
                # Verify: bash only — no web search, no file writes
                relevant_tools = [t for t in relevant_tools
                                  if t.get("name", "").lower() not in
                                  _WEB_TOOL_NAMES | _WRITE_TOOL_NAMES]
            elif stage_type == "research":
                # Research: web + file reads — no writes or bash
                relevant_tools = [t for t in relevant_tools
                                  if t.get("name", "").lower() not in
                                  {"bash", "powershell"} | _WRITE_TOOL_NAMES]
            elif stage_type in ("write_code", "write_doc", "edit"):
                # Write/edit: bash + file tools — no web search
                relevant_tools = [t for t in relevant_tools
                                  if t.get("name", "").lower() not in _WEB_TOOL_NAMES]

            filtered_names = [t.get("name", "") for t in relevant_tools]
            logger.info("pipeline: task=%r filtered_tools=%s", task[:60], filtered_names)

            # For "direct" stages (social messages, trivial answers) skip planning.
            # For "save_memory" stages, extract the fact and save it directly.
            if stage_type == "direct":
                steps = []
            elif stage_type == "save_memory":
                # Extract fact from goal: strip leading verb/prefix
                goal_text = stage.get("goal", task)
                fact = goal_text
                for prefix in ("save: ", "save_memory:", "save_memory ", "save "):
                    if fact.lower().startswith(prefix):
                        fact = fact[len(prefix):].strip()
                        break
                # Strip literal placeholder words if model echoed them
                for junk in ("FACT:", "fact:", "<verbatim fact>", "<fact>"):
                    if fact.startswith(junk):
                        fact = fact[len(junk):].strip()
                if self.graph and fact:
                    try:
                        _save_fact_to_graph(self.graph, fact)
                        _write_pref_to_file(self.prefs_path, fact)
                        logger.info("pipeline: save_memory stage → stored %r", fact[:60])
                    except Exception as exc:
                        logger.warning("pipeline: save_memory stage failed: %s", exc)
                steps = [{"tool": "save_memory", "input": fact, "result": "saved",
                          "summary": f"saved: {fact[:80]}"}]
            elif task.strip().lower().startswith("run ") or re.match(
                r'^command:\s*run\s+', task.strip(), re.IGNORECASE
            ):
                # Stage goal is already a concrete command — skip plan_task,
                # extract the command and emit a bash step directly.
                # Strip "COMMAND: Run " prefix that small models echo from the format description.
                _task_stripped = re.sub(r'^command:\s*', '', task.strip(), flags=re.IGNORECASE)
                if _task_stripped.lower().startswith("run "):
                    _task_stripped = _task_stripped[4:].strip()
                cmd = _sub_workspace(_normalize_python_c(_task_stripped), self.workspace)
                logger.info("pipeline: direct-run stage → bash: %s", cmd[:80])
                steps = [{"tool": "bash", "input": cmd}]
            elif stage_type == "verify":
                # Tool already decided by think_decompose: bash.
                # Only ask "what command?" — don't re-do tool selection via plan_task.
                cmd = await resolve_bash_command(task, self.client, context=task_ctx)
                if cmd:
                    cmd = _sub_workspace(_normalize_python_c(cmd), self.workspace)
                    logger.info("pipeline: verify stage → bash: %s", cmd[:80])
                    steps = [{"tool": "bash", "input": cmd}]
                else:
                    steps = await plan_task(task, relevant_tools or available_tools, self.client,
                                            context=task_ctx, soul_section=soul_section,
                                            user_prefs=user_prefs, workspace=self.workspace,
                                            skill_index=skill_index)
            else:
                steps = await plan_task(task, relevant_tools or available_tools, self.client,
                                        context=task_ctx,
                                        soul_section=soul_section,
                                        user_prefs=user_prefs,
                                        workspace=self.workspace,
                                        skill_index=skill_index)

            # ── Websearch fallback — if plan is empty and no graph recall ──────
            if not steps and stage_type == "research" and not graph_recall:
                all_names = {t.get("name", "").lower() for t in (relevant_tools or available_tools)}
                ws_tool = next((n for n in all_names if n in _WEB_TOOL_NAMES), "web_search")
                steps = [{"tool": ws_tool, "input": task}]
                logger.info("pipeline: empty plan → websearch fallback for %r", task[:60])
            elif not steps and stage_type in ("write_code", "write_doc"):
                # Write stage with no plan — synthesise a write_plan step directly
                # so the model can't escape by returning an empty response.
                _ft = "code" if stage_type == "write_code" else "doc"
                _fn = _extract_filename_from_task(task, _ft)
                if not _fn:
                    _fn = _extract_filename_from_task(query, _ft)
                if _fn:
                    if self.workspace and not os.path.isabs(_fn):
                        _fn = os.path.join(self.workspace, _fn)
                    steps = [{"tool": "write_plan", "input": task,
                              "file_path": _fn, "file_type": _ft}]
                    logger.info("pipeline: empty plan for %s → write_plan fallback: %s",
                                stage_type, _fn)
                else:
                    logger.warning("pipeline: empty plan for %s and no filename in %r",
                                   stage_type, task[:60])
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

        # ── Recent turns — always included, Jaccard-independent ───────────────
        # extract_for_task() filters by lexical similarity; follow-up questions
        # ("based on this, do you think you are alive?") have near-zero overlap
        # with the previous turn's topic ("meaning of life") so the context gets
        # dropped even though it's essential.  Always include the last 2 turns
        # verbatim so conversational continuity is never broken by the filter.
        recent_turns = recent_turns_text   # last 2 user+asst pairs (computed in process())

        # memory_ctx first — soul policy, user knowledge, self-concept.
        # recent_turns second — takes precedence over Jaccard-filtered history.
        synthesis_ctx = "\n\n".join(
            p for p in (memory_ctx, recent_turns, synthesis_history, project_ctx) if p
        )

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

        # ── Write plan continuation ─────────────────────────────────────────────
        # Write plan is active: the outer tool result was a Write (new item) or
        # Edit (retry of partial item).  Verify actual file content; retry if needed.
        if state.wp_items:
            tool_results = _extract_tool_results(raw_history)
            # Item that was just written/edited
            written_item = state.wp_items[state.wp_idx] if state.wp_idx < len(state.wp_items) else {}
            item_title  = written_item.get("title",  f"item {state.wp_idx + 1}")
            item_anchor = written_item.get("anchor", item_title)
            item_min    = written_item.get("min_chars", 200)

            # Always track the file — Write returns empty result, Edit returns a message.
            # We track regardless so epistemic state stays accurate.
            if state.wp_file and state.wp_file not in state.files_written:
                state.files_written.append(state.wp_file)

            for tr in tool_results:
                logger.info("pipeline.wp_continue: item=%d/%d result=%s",
                            state.wp_idx, len(state.wp_items), tr[:60])
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
        # Propagate _fix_error annotation so _replan_from_outer_result can use it
        outer_fix_error = outer_step.get("_fix_error", "")

        # Extract tool results from the last user message
        tool_results = _extract_tool_results(raw_history)
        for tr in tool_results:
            # Empty stdout from bash/write means the command succeeded silently
            if not tr and outer_input:
                tr = f"Completed successfully: {outer_input[:80]}"
            result_entry = {
                "tool": outer_tool,
                "input": outer_input,
                "result": tr,
                "summary": tr[:120],
            }
            if outer_fix_error:
                result_entry["_fix_error"] = outer_fix_error
            state.results.append(result_entry)
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

                # ── Bash failure retry ────────────────────────────────────────
                # Two strategies, tried in order:
                #   1. Smart fix loop: Python traceback with a named file →
                #      insert Read(broken_file) so _replan_from_outer_result
                #      calls _generate_edit_steps with the error as context,
                #      then re-runs the original bash command.
                #   2. Command correction: no file reference → ask the LLM
                #      for a corrected command (existing _bash_correct path).
                _MAX_BASH_RETRIES = 2
                if _is_bash_failure(tr) and state.bash_retry_count < _MAX_BASH_RETRIES:
                    tb_info = _parse_traceback(tr)
                    fix_file = _resolve_fix_file(
                        tb_info["file"] if tb_info else "",
                        state.project_dir, self.workspace,
                    ) if tb_info else ""

                    if fix_file:
                        # Smart fix: Read broken file → LLM generates Edit → re-run
                        logger.warning(
                            "pipeline: traceback in %s (attempt %d) → smart fix loop",
                            os.path.basename(fix_file), state.bash_retry_count + 1,
                        )
                        state.last_bash_error = tr[:400]
                        try:
                            sub = state.sub_tasks[state.current_task_idx]
                            ins = state.current_step_idx  # insert BEFORE step-idx advance
                            # Re-run the original bash after the fix
                            sub["steps"].insert(ins + 1, {"tool": "bash", "input": outer_input})
                            # Read the broken file first (replanner will generate Edit)
                            sub["steps"].insert(ins + 1, {
                                "tool": "read", "input": fix_file, "_fix_error": tr[:400],
                            })
                        except (IndexError, KeyError):
                            pass
                        state.bash_retry_count += 1
                        # Do NOT decrement step_idx — _continue advances to Read next
                        state.results.pop()   # discard failed result so _execute re-runs
                    else:
                        # Command-level error (wrong path, typo, OS difference) →
                        # ask the LLM for a corrected command
                        corrected = await _bash_correct(outer_input, tr, self.client)
                        if corrected:
                            logger.warning(
                                "pipeline: bash failed — retrying with corrected cmd "
                                "(attempt %d): %s",
                                state.bash_retry_count + 1, corrected[:80],
                            )
                            try:
                                sub = state.sub_tasks[state.current_task_idx]
                                sub["steps"][state.current_step_idx]["input"] = corrected
                            except (IndexError, KeyError):
                                pass
                            state.bash_retry_count += 1
                            state.current_step_idx -= 1  # re-run same step
                            state.results.pop()          # discard failed result
                        else:
                            state.bash_retry_count = 0
                else:
                    state.bash_retry_count = 0

            # Mark outer tool as done in dashboard.
            # Use display-friendly type for special steps so the dashboard
            # can colour them distinctly: import_check (green) and fix_read (orange).
            _display_tool = outer_tool
            if outer_step.get("_import_verify"):
                _display_tool = "import_check"
            elif outer_step.get("_fix_error"):
                _display_tool = "fix_read"
            _tracker.tree_subtask_step(
                state.task_id, state.current_task_idx,
                _display_tool, outer_input, tr[:200], status="done",
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
                return await self._write_plan_next_item(state, available_tools)
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

                # ── Cross-file sig injection (project builds) ─────────────────
                # After a code file is fully written, extract def/class signatures
                # and store them in the memory graph so subsequent files in the same
                # project can import or reference them.  Only runs for code files
                # (not docs) and only when the graph is available.
                if wp_ftype_was == "code" and wp_file_was and self.graph:
                    _inject_file_sigs_to_graph(wp_file_was, self.graph)

                # ── Per-file import verify (project builds) ───────────────────
                # After each code file is written, verify its imports resolve
                # with a quick `python -c "import X"` check.  If this fails,
                # the smart fix loop (traceback → Read → Edit → re-run) kicks in
                # automatically via the bash failure handler above.
                # Only runs for clean module names (no hyphens, no path separators).
                if wp_ftype_was == "code" and wp_file_was:
                    _mod = os.path.splitext(os.path.basename(wp_file_was))[0]
                    _ws  = self.workspace or state.project_dir or ""
                    if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', _mod) and _ws:
                        _verify_cmd = (
                            f'python -c "import sys; sys.path.insert(0, r\'{_ws}\'); '
                            f'import {_mod}; print(\'import OK: {_mod}\')"'
                        )
                        _verify_step = {"tool": "bash", "input": _verify_cmd,
                                        "_import_verify": True}
                        try:
                            sub = state.sub_tasks[state.current_task_idx]
                            sub["steps"].insert(state.current_step_idx + 1, _verify_step)
                            logger.info("pipeline: import verify queued for %s", _mod)
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
            stage_type = sub.get("type", "")
            # Per-stage budget: hard cap on internal step count.  Falls back to 12
            # for unknown stage types.  External tool calls (bash, write, read…) that
            # return as tool_use blocks don't count — they exit _execute() immediately
            # and re-enter via _continue().  This cap prevents a single stuck
            # research / verify stage from consuming context indefinitely.
            _stage_budget = STAGE_BUDGETS.get(stage_type, 12)
            _consec_search_fail = 0   # reset per task; drives research loop guard

            while state.current_step_idx < len(steps):
                # ── Per-stage budget cap ──────────────────────────────────────
                # Second line of defence after the research loop guard.
                if state.current_step_idx >= _stage_budget:
                    logger.warning(
                        "pipeline: stage budget exhausted (%d/%d) for %s stage %r "
                        "— proceeding to synthesis",
                        state.current_step_idx, _stage_budget,
                        stage_type, sub["task"][:60],
                    )
                    _log("stage", "budget-cap",
                         f"{stage_type} hit {_stage_budget}-step cap",
                         session_id=state.task_id,
                         data={"stage": "budget-cap", "stage_type": stage_type,
                               "budget": _stage_budget, "task": sub["task"][:80]})
                    break

                step = steps[state.current_step_idx]
                tool  = step.get("tool", "").strip().lower()
                inp   = step.get("input", "").strip()

                # ── Graph-first: warm path ────────────────────────────────────
                # Serve from graph if available. Recency scoring ranks fresh nodes
                # higher — no time-gate. If model judged memory stale/partial and
                # re-plans a web_search, that request reaches here, runs, and
                # upserts the node with fresh data (targeted enrichment).
                if tool in ("web_search", "websearch") and self.graph:
                    mem = _search_graph(inp, self.graph)
                    if mem and len(mem.strip()) > 40:
                        logger.info(
                            "pipeline: graph warm hit for %r — serving from memory",
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

                # ── Normalise shell-like tool names to "bash" ────────────────
                # Note: web_search and web_fetch are always handled internally
                # (SearXNG → Jina). We do NOT promote them to outer tools even
                # when Claude Code offers websearch/webfetch — Claude Code's
                # built-in search does not work with a custom API provider.
                # Small models sometimes plan "shell", "terminal", "cmd", "mkdir",
                # "run", "execute" instead of "bash". Map them so the outer-tool
                # dispatch fires correctly instead of silently skipping the step.
                _BASH_ALIASES = frozenset({
                    "shell", "terminal", "cmd", "command", "run", "execute",
                    "mkdir", "powershell", "sh", "zsh",
                })
                if tool in _BASH_ALIASES and "bash" in outer_tool_names:
                    logger.info("pipeline: normalising tool %r → bash", tool)
                    tool = "bash"
                    step["tool"] = "bash"

                # ── run_skill → resolve stored program → convert to bash ────
                # Layer 3: the model chose to re-execute a previously built
                # program instead of re-planning from scratch.
                # 1. Look up program in graph.
                # 2. Ensure it exists on disk (write if missing).
                # 3. Substitute this step with bash execution of the script.
                # If the skill has no program, fall through to _run_internal
                # which returns a "not found" result and lets synthesis explain.
                if tool == "run_skill":
                    _prog = get_skill_program(inp, self.graph)
                    if _prog:
                        _script = get_skill_script_path(inp)
                        # Write to disk if missing or stale
                        if not _script.exists():
                            _script = save_skill_to_disk(inp, _prog) or _script
                        if _script and _script.exists():
                            _cmd = f"python {_script}"
                            logger.info(
                                "pipeline: run_skill %r → bash: %s", inp[:40], _cmd
                            )
                            step["tool"]  = "bash"
                            step["input"] = _cmd
                            tool = "bash"
                            inp  = _cmd
                            # Fall through — bash is an outer tool and will be dispatched below
                        else:
                            logger.warning("pipeline: run_skill %r — could not write script", inp[:40])
                            # Fall through to _run_internal for graceful "not found" response

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
                    # Parse "FILENAME|TASK DESCRIPTION" from inp if the model
                    # used the write_plan:file|goal format.  Populate step fields
                    # so _start_write_plan has file_path, file_type and input.
                    if "|" in inp and not step.get("file_path"):
                        _wp_file, _, _wp_task = inp.partition("|")
                        _wp_file = _wp_file.strip()
                        _wp_task = _wp_task.strip() or inp
                        if _wp_file:
                            # Resolve to workspace
                            if self.workspace and not os.path.isabs(_wp_file):
                                _wp_file = os.path.join(self.workspace, _wp_file)
                            _ext = os.path.splitext(_wp_file)[1].lower()
                            step["file_path"] = _wp_file
                            step["input"]     = _wp_task
                            step["file_type"] = "code" if _ext in (".py", ".js", ".ts", ".rb", ".go", ".rs") else "doc"
                            step["run_after"] = step["file_type"] == "code"
                            inp = _wp_task
                    return await self._start_write_plan(step, state, available_tools)

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
                        cmd = _sub_workspace(_normalize_python_c(inp), self.workspace)
                        # Strip "COMMAND" prefix that models echo from format descriptions
                        cmd = re.sub(r'^COMMAND[\s:]+', '', cmd, flags=re.IGNORECASE).strip()
                        # Safety net: if cmd writes/runs relative files, cd to workspace first
                        if self.workspace:
                            cmd = _apply_workspace_to_cmd(cmd, self.workspace)
                        tool_input = {"command": cmd}
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

                # ── Persist research to notes.md ──────────────────────────────
                # Mirrors BirdClaw_Old: every informational tool result is
                # appended to notes.md so the write phase can reference what
                # was found, and so results persist across Claude Code turns.
                if tool in _INFO_TOOLS and quality in ("good", "weak") and self.workspace:
                    _npath = _notes_path(self.workspace)
                    _append_to_notes(
                        _npath, tool, inp,
                        result.get("result", ""),
                        task_id=state.task_id,
                    )
                _log("llm", f"{tool} → {quality}", summary[:120], session_id=state.task_id,
                     data={"action": tool, "input": inp[:120], "result": summary[:200],
                           "quality": quality, "elapsed_ms": elapsed_ms})

                # ── Dynamic replan after info-gathering ───────────────────────
                # The search result IS the reasoning — replace any pre-guessed
                # follow-up steps with steps derived from what was actually found.
                if tool in _INFO_TOOLS:
                    # Track consecutive failures for the research loop guard.
                    # Reset on any non-empty result (even weak) so a single bad
                    # step between two good ones doesn't trip the guard.
                    if quality in ("empty", "error"):
                        _consec_search_fail += 1
                    else:
                        _consec_search_fail = 0

                    # ── Research loop guard ───────────────────────────────────
                    # _replan_after_search() can schedule more web_search steps
                    # even after a failure, creating an infinite search loop.
                    # After 2 consecutive empty/error results on this task,
                    # stop replanning and fall through to synthesis with what
                    # was gathered so far.
                    if _consec_search_fail >= 2:
                        logger.warning(
                            "pipeline: research loop guard — %d consecutive search "
                            "failures on task %r; skipping replan, proceeding to synthesis",
                            _consec_search_fail, sub["task"][:60],
                        )
                        _log("stage", "loop-guard",
                             f"stopped after {_consec_search_fail} consecutive failures",
                             session_id=state.task_id,
                             data={"stage": "loop-guard", "task": sub["task"][:80],
                                   "fail_count": _consec_search_fail, "last_tool": tool})
                        state.current_step_idx += 1
                        break  # exit step loop for this task → synthesis

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
            # Log actual step count for P75 budget learning.
            # current_step_idx == steps taken (0-based index ≈ count at task end).
            if self.budget_tracker and stage_type and state.current_step_idx > 0:
                self.budget_tracker.log(
                    stage_type, state.current_step_idx, sub.get("task", "")
                )
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

        # ── Write task log (mirrors BirdClaw_Old's BIRDCLAW.md post-task write) ─
        _write_task_log(
            workspace=self.workspace,
            task_id=state.task_id,
            query=state.query,
            sub_tasks=state.sub_tasks,
            files_written=state.files_written,
            answer_preview=answer[:200],
        )

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
                temperature=0.1,
                response_format={"type": "json_object"},  # "{" prefix on llama.cpp
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
            # If this Read was triggered by a bash traceback (smart fix loop),
            # use the error message as the task so the LLM knows exactly what to fix.
            # Prefer the per-result annotation; fall back to state.last_bash_error.
            fix_error = last.get("_fix_error", "") or state.last_bash_error
            if fix_error:
                edit_task = (
                    f"Fix this Python error in {os.path.basename(file_path)}:\n"
                    f"{fix_error[:350]}"
                )
                logger.info("pipeline: read→fix-edit for %s: %s",
                            os.path.basename(file_path), fix_error[:60])
                # Clear the stored error — it's been consumed
                state.last_bash_error = ""
            else:
                edit_task = state.query
                logger.info("pipeline: read→edit replan for %s", file_path)
            new_steps = await self._generate_edit_steps(
                edit_task, result, file_path,
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

    async def _start_write_plan(self, step: dict, state: PipelineState,
                               available_tools: list[dict] | None = None) -> LoopResponse:
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

        # If the task input is just a bare filename / path (model used the filename
        # as the input instead of the actual goal description), fall back to the
        # original user query so the subtask planner gets a meaningful description.
        _task_words = task.split() if task else []
        _looks_like_bare_path = (
            task
            and len(_task_words) <= 3
            and ("/" in task or os.sep in task or "." in os.path.basename(task))
        )
        if not task or _looks_like_bare_path:
            task = state.query
            logger.info("pipeline: _start_write_plan: bare path as task — using query instead")

        # Resolve file_path against project_dir
        if file_path and state.project_dir and not os.path.isabs(file_path):
            file_path = os.path.join(state.project_dir, file_path)

        # Guard: empty file_path OR model gave a directory (both common mistakes).
        # Try to recover a filename from the task text or fall back to project_dir.
        _fp_is_dir = file_path and os.path.isdir(file_path)
        if not file_path or _fp_is_dir:
            _extracted = _extract_filename_from_task(task or state.query, file_type)
            if _extracted:
                _base = file_path if _fp_is_dir else (state.project_dir or "")
                file_path = os.path.join(_base, _extracted) if _base else _extracted
                logger.info("pipeline: write_plan file_path fix (%s) → %s",
                            "dir" if _fp_is_dir else "empty", file_path)
            else:
                logger.warning("pipeline: write_plan has no usable file_path "
                               "and could not extract filename from task")

        # Infer file_type from extension — override even "doc" default when the
        # extension clearly indicates code.  Covers three cases:
        #   • step.get("file_type") is None  (plan_task didn't set it)
        #   • step.get("file_type") == "doc" (pipe-parser defaulted; deepen inherited)
        #   • file_path was empty and just got resolved above
        _CODE_EXTS = {".py", ".js", ".ts", ".rb", ".go", ".rs", ".java", ".c", ".cpp", ".sh"}
        if file_path:
            _ext = os.path.splitext(file_path)[1].lower()
            if _ext in _CODE_EXTS and file_type != "code":
                file_type = "code"
                run_after = True
                logger.info("pipeline: _start_write_plan: inferred file_type=code from %s", _ext)

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
        return await self._write_plan_next_item(state, available_tools)

    async def _write_plan_next_item(self, state: PipelineState,
                                    available_tools: list[dict] | None = None) -> LoopResponse:
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

        # Failsafe: if wp_file is empty (can happen if the LLM omitted the filename
        # and the _start_write_plan guard couldn't extract it), recover from the
        # original user query before giving up.
        if not state.wp_file:
            _fb = _extract_filename_from_task(state.query, state.wp_ftype or "doc")
            if _fb:
                state.wp_file = (
                    os.path.join(state.project_dir, _fb)
                    if state.project_dir else _fb
                )
                logger.warning(
                    "pipeline: _write_plan_next_item: empty wp_file — recovered → %s",
                    state.wp_file,
                )
            else:
                logger.error(
                    "pipeline: _write_plan_next_item: empty wp_file, cannot extract "
                    "filename from query %r — aborting write plan", state.query[:80]
                )
                state.wp_items = []
                state.current_step_idx += 1
                return await self._execute(state, [])

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
        # 4-step progressive disclosure: CLAUDE.md → exact section →
        # relevant lines → continuation point (last header → EOF fallback).
        from engine.translation.subtask.line_search import (
            find_section as _find_section,
            find_continuation_point as _find_continuation,
            search_relevant as _search_relevant,
        )

        def _build_file_state(fp: str, item_title: str, ftype: str, ws: str) -> str:
            parts: list[str] = []
            # Step 1: CLAUDE.md in workspace root
            if ws:
                _cmd = Path(ws) / "CLAUDE.md"
                if _cmd.is_file():
                    _ws_ctx = _search_relevant(item_title, [_cmd], context_lines=1, max_results=3)
                    if _ws_ctx:
                        parts.append(f"[CLAUDE.md]\n{_ws_ctx}")
            # Step 2: exact section in the file
            _sec = _find_section(fp, item_title, ftype)
            if _sec:
                parts.append(f"[{os.path.basename(fp)} — {item_title}]\n{_sec}")
                return "\n\n".join(parts)
            # Step 3: relevant lines scattered through the file
            _rel = _search_relevant(item_title, [fp], context_lines=2)
            if _rel:
                parts.append(f"[{os.path.basename(fp)}]\n{_rel}")
                return "\n\n".join(parts)
            # Step 4: last section → EOF
            _cont = _find_continuation(fp, ftype)
            if _cont:
                parts.append(f"[{os.path.basename(fp)}]\n{_cont}")
            return "\n\n".join(parts)

        _ctx_str = _build_file_state(file_path, title, file_type, self.workspace) if file_path else ""
        file_state = _ctx_str if _ctx_str else f"[{os.path.basename(file_path)} — empty, start fresh]"

        if file_type == "code":
            anchor_hint = f"Your content MUST start with exactly: def {anchor}( or class {anchor}:"
            type_reminder = ""
        else:
            anchor_hint = f"Your content MUST start with exactly: ## {anchor}"
            type_reminder = "Write PROSE paragraphs — NO code, NO import statements, NO def/class lines."

        done_items = [it["title"] for it in state.wp_items[:state.wp_idx]]
        done_str = ", ".join(done_items) or "none yet"

        # ── Inject conversation context + session research notes ─────────────
        # For code files: do NOT inject synthesis_ctx — it may contain essay/doc
        # context from prior conversation turns (section headings, "Introduction",
        # "Conclusion" etc.) that causes the model to generate functions with those names.
        # Instead inject recent bash output (ls, tests, etc.) so the model knows
        # what files already exist in the workspace before writing.
        if file_type == "code":
            recent_bash = [
                r for r in state.results
                if r.get("tool") == "bash" and r.get("result", "").strip()
            ]
            if recent_bash:
                bash_lines = "\n".join(
                    f"$ {r['input']}\n{r['result'][:300]}" for r in recent_bash[-3:]
                )
                conv_ctx = f"[Recent shell output — shows what already exists]\n{bash_lines}"
            else:
                conv_ctx = ""
            # ── Cross-file signature injection ──────────────────────────────
            # For multi-file project builds: inject def/class signatures that
            # were extracted from previously-written project files.  This lets
            # qwen3:0.6b know what functions are already available to import,
            # keeping the cross-file context to ~50 chars/symbol (not full file).
            _sigs = _get_graph_sigs(state.wp_goal, self.graph)
            if _sigs:
                conv_ctx = (_sigs + ("\n\n" + conv_ctx if conv_ctx else "")).strip()
                logger.debug("write_plan: injected %d chars of cross-file sigs for %r",
                             len(_sigs), title[:40])
        else:
            conv_ctx = state.synthesis_ctx[:800].strip() if state.synthesis_ctx else ""
            # Inject session research notes — web_search / web_fetch results
            # written to notes.md during this session's research phase.
            # Uses search_relevant so only the parts pertinent to this section
            # are injected (BirdClaw_Old injected the last 2000 chars verbatim;
            # we use term-matching to stay within context budget).
            if self.workspace:
                _nfile = _notes_path(self.workspace)
                if _nfile and Path(_nfile).is_file():
                    from engine.translation.subtask.line_search import search_relevant as _sr_notes
                    _notes_ctx = _sr_notes(title, [_nfile], context_lines=2, max_results=6)
                    if _notes_ctx:
                        conv_ctx = (conv_ctx + "\n\n[Research notes for this section]\n" + _notes_ctx).strip()
                        logger.debug("write_plan: injected %d chars from notes.md for item %r",
                                     len(_notes_ctx), title[:40])

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
                return await self._write_plan_next_item(state, available_tools)
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
                        return await self._write_plan_next_item(state, available_tools)
                    state.wp_items = []
                    state.current_step_idx += 1
                    return await self._execute(state, [])
                else:
                    # Partial content exists — prefer Edit-in-place (Claude Code pattern)
                    # to avoid rewriting the entire file.  Falls back to strip+Write if
                    # Edit is unavailable or the old text cannot be located.
                    _has_edit = (
                        available_tools
                        and any(t.get("name", "").lower() == "edit" for t in available_tools)
                    )
                    if _has_edit:
                        _old_text = _get_item_text_from_content(current_content, anchor, file_type)
                        if _old_text:
                            _new_text = _rebuild_item_content(_old_text, content)
                            tool_id = f"toolu_{uuid.uuid4().hex[:16]}"
                            logger.info(
                                "pipeline: write_plan item %r retry — Edit in-place (%dc→%dc)",
                                title[:40], len(_old_text), len(_new_text),
                            )
                            _tracker.tree_subtask_step(
                                state.task_id, state.current_task_idx,
                                "edit", f"{title} → {file_path}", "", status="running",
                            )
                            return LoopResponse(
                                content=[
                                    {"type": "thinking",
                                     "thinking": f"{_STATE_PREFIX}{state.to_json()}"},
                                    {"type": "tool_use", "id": tool_id, "name": "Edit",
                                     "input": {"file_path": file_path,
                                               "old_string": _old_text,
                                               "new_string": _new_text}},
                                ],
                                stop_reason="tool_use",
                            )
                    # Fallback: strip partial and write full file
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

        # ── Visible progress for multi-file project builds ────────────────────
        # Show a checklist at the start of each new file (wp_idx == 0) so the
        # user can see overall project progress without looking at the dashboard.
        # Mirrors the replan/subtask cycle: same information, surfaced as text.
        response_content: list[dict] = [
            {"type": "thinking", "thinking": f"{_STATE_PREFIX}{state.to_json()}"},
        ]
        if state.wp_idx == 0:
            progress_text = _render_project_progress(state)
            if progress_text:
                response_content.append({"type": "text", "text": progress_text})

        response_content.append(
            {"type": "tool_use", "id": tool_id, "name": "Write",
             "input": {"file_path": file_path, "content": new_full_content}},
        )
        return LoopResponse(content=response_content, stop_reason="tool_use")

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
                temperature=0.1,
                response_format={"type": "json_object"},  # "{" prefix on llama.cpp
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
                temperature=0.1,
                response_format={"type": "json_object"},  # "{" prefix on llama.cpp
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
        """Upsert a web search result into the knowledge graph.

        Creates a new research node on cold miss, or enriches (overwrites summary
        + bumps last_seen) on a stale hit where the model chose to re-search.
        No LLM call — the dream cycle consolidates and distils during off-peak hours.
        """
        if not self.graph or not result:
            return
        try:
            summary = result[:500].replace("\n", " ").strip()
            node_name = query[:80]
            existing = self.graph.get_node(node_name)
            if existing:
                logger.debug(
                    "pipeline: enriching existing research node %r", node_name[:40]
                )
            else:
                logger.debug(
                    "pipeline: creating new research node %r", node_name[:40]
                )
            self.graph.upsert_node(
                name=node_name,
                node_type="research",
                summary=summary,
                sources=["web_search"],
            )
            self.graph.save()
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
            # ── Graph/web synergy ─────────────────────────────────────────────
            # Warm:  graph already has this research → serve directly, no web cost.
            #        Recency scoring in graph.search() naturally ranks fresh nodes
            #        higher — no time-gate needed here.
            # Cold:  graph miss → full web search → upsert node (create or enrich).
            # Stale: model sees graph content in memory injection and judges it
            #        insufficient → it explicitly plans web_search → we run it
            #        and upsert the node with fresh data (targeted enrichment).
            if self.graph:
                mem = _search_graph(inp, self.graph)
                if mem and len(mem.strip()) > 40:
                    logger.info("pipeline: web_search → graph warm hit for %r", inp[:40])
                    return {"tool": "search_memory", "input": inp, "result": mem,
                            "summary": f"recalled: {mem[:120]}"}
            # ── Multi-angle query expansion ───────────────────────────────────
            # Fire 2 semantically varied queries to get broader, more grounded
            # results — mirrors BirdClaw_Old's multi-angle search behaviour.
            # The second query is generated by a cheap LLM call; falls back to
            # the original if the call fails or produces nothing new.
            queries = [inp]
            try:
                alt = await _expand_query(inp, self.client)
                if alt and alt.lower().strip() != inp.lower().strip():
                    queries.append(alt)
                    logger.info("pipeline: web_search expanded %r → also %r", inp[:40], alt[:40])
            except Exception:
                pass   # expansion is best-effort

            # Nothing in graph — fetch from web (all query angles combined)
            raw = []
            all_snippets: list[str] = []
            for q in queries:
                try:
                    q_raw = await _web_search(q, max_results=3)
                    if q_raw:
                        raw.extend(q_raw)
                        all_snippets.append(format_results(q_raw))
                except Exception as exc:
                    logger.debug("web_search angle %r failed: %s", q[:40], exc)
            content = "\n\n".join(all_snippets) if all_snippets else "No results."
            # ── Auto-fetch page content when snippets are thin ────────────────
            # Jina AI tier (is_ai_synthesized=True) already returns rich content.
            # DDG / SearXNG tiers return short snippets (≤500 chars) that are
            # often too thin for the model to act on. Fetch the top pages to give
            # the same deep-content behaviour BirdClaw's web_fetch tool provided.
            if raw:
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
            if self.graph:
                try:
                    _save_fact_to_graph(self.graph, inp)
                    _write_pref_to_file(self.prefs_path, inp)
                    logger.info("pipeline: save_memory → stored %r", inp[:60])
                except Exception as exc:
                    logger.warning("pipeline: save_memory graph write failed: %s", exc)
            return {"tool": tool, "input": inp, "result": "saved", "summary": f"saved: {inp[:80]}"}

        # ── Skill tools (progressive disclosure Layer 2) ──────────────────────

        if tool == "read_skill":
            # Layer 2: model requested the full runbook for a skill.
            # Also reports whether a runnable program exists so the model can
            # decide to use run_skill:NAME on the next task of the same type.
            runbook = get_skill_runbook(inp, self.graph)
            has_prog = bool(get_skill_program(inp, self.graph)) if self.graph else False
            if runbook:
                prog_note = "\n\n[This skill has a saved program — use run_skill:NAME to re-execute it directly.]" if has_prog else ""
                full = runbook + prog_note
                logger.info("pipeline: read_skill → loaded %r (%d chars)", inp[:40], len(runbook))
                return {
                    "tool": tool, "input": inp,
                    "result": full,
                    "summary": f"skill runbook: {runbook[:120]}",
                }
            logger.info("pipeline: read_skill → no skill found for %r", inp[:40])
            return {
                "tool": tool, "input": inp,
                "result": f"(no skill found for '{inp}')",
                "summary": f"skill not found: {inp[:60]}",
            }

        if tool == "save_skill":
            # Persist a text runbook as a skill node.
            # Input format:  SKILL-NAME|step-by-step runbook
            # Summary = first 60 chars of runbook, or the name if no runbook.
            skill_name = ""
            runbook    = ""
            if "|" in inp:
                skill_name, _, runbook = inp.partition("|")
                skill_name = skill_name.strip()
                runbook    = runbook.strip()
            else:
                skill_name = inp.strip()
            if self.graph and skill_name:
                summary = (runbook[:60].rstrip() if runbook else skill_name[:60])
                self.graph.upsert_node(
                    name=skill_name,
                    node_type="skill",
                    summary=summary,
                    content=runbook,
                    sources=["pipeline"],
                )
                logger.info("pipeline: save_skill → saved %r (%d chars)", skill_name[:40], len(runbook))
            return {
                "tool": tool, "input": inp,
                "result": "skill saved",
                "summary": f"saved skill: {skill_name[:60]}",
            }

        if tool == "save_skill_program":
            # Persist actual program code as a skill node + write to disk.
            # Input format:  SKILL-NAME|complete program code
            # Handles version tracking: old code moved to program_history.
            # On subsequent calls for the same skill, increments version and
            # preserves a snippet of the previous program for BirdClaw's dream cycle.
            skill_name  = ""
            code        = ""
            if "|" in inp:
                skill_name, _, code = inp.partition("|")
                skill_name = skill_name.strip()
                code       = code.strip()
            else:
                skill_name = inp.strip()
            if self.graph and skill_name and code:
                script_path = save_skill_program_to_graph(
                    skill_name=skill_name,
                    code=code,
                    graph=self.graph,
                )
                path_note = f" at {script_path}" if script_path else ""
                logger.info("pipeline: save_skill_program → saved %r%s", skill_name[:40], path_note)
                return {
                    "tool": tool, "input": inp,
                    "result": f"skill program saved{path_note}",
                    "summary": f"saved skill program: {skill_name[:60]}",
                }
            return {
                "tool": tool, "input": inp,
                "result": "(save_skill_program: missing name or code)",
                "summary": "save_skill_program: no-op",
            }

        if tool == "run_skill":
            # run_skill should have been converted to bash in _execute.
            # If we reach here it means the skill has no program — return info.
            runbook = get_skill_runbook(inp, self.graph) if self.graph else ""
            msg = (f"Skill '{inp}' has no saved program. Runbook:\n{runbook}"
                   if runbook else f"Skill '{inp}' not found.")
            return {"tool": tool, "input": inp, "result": msg, "summary": msg[:100]}

        # Unknown tool — log and skip
        logger.warning("pipeline: unknown internal tool %r — skipping", tool)
        return {"tool": tool, "input": inp, "result": "", "summary": f"unknown: {tool}"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _notes_path(workspace: str) -> str:
    """Absolute path to the rolling session notes file."""
    return str(Path(workspace) / "notes.md") if workspace else ""


def _append_to_notes(notes_path: str, tool: str, inp: str, content: str,
                     task_id: str = "") -> None:
    """Append one tool result to notes.md so later steps (and write phases) can see it.

    Called after every informational tool result (web_search, web_fetch,
    search_memory).  The file accumulates during a session exactly like
    BirdClaw_Old's per-task notes.md — a running log of what was found.
    """
    if not notes_path or not content or not content.strip():
        return
    try:
        p = Path(notes_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tag   = f"[{task_id[:8]}] " if task_id else ""
        snippet = content.strip()[:1500]
        entry   = f"\n## {tag}[{tool}] {inp[:80]}\n{snippet}\n"
        with p.open("a", encoding="utf-8") as f:
            f.write(entry)
        logger.debug("notes.md: appended %d chars for %s(%r)", len(snippet), tool, inp[:40])
    except OSError as exc:
        logger.debug("notes.md append failed: %s", exc)


def _write_task_log(workspace: str, task_id: str, query: str,
                    sub_tasks: list[dict], files_written: list[str],
                    answer_preview: str = "") -> None:
    """Append a task-completion entry to task_log.md.

    Mirrors BirdClaw_Old's BIRDCLAW.md post-task write.  Gives a permanent
    audit trail of what Sisyphean did and what files it produced.
    """
    if not workspace:
        return
    try:
        import time as _t2
        p     = Path(workspace) / "task_log.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        ts    = _t2.strftime("%Y-%m-%d %H:%M")
        lines = [f"\n## [{ts}] {query[:100]}"]
        for sub in sub_tasks[:6]:
            stype = sub.get("type", "?")
            stask = sub.get("task", "")[:70]
            nsteps = len(sub.get("steps", []))
            lines.append(f"- [{stype}] {stask} ({nsteps} steps)")
        if files_written:
            names = ", ".join(Path(f).name for f in files_written[:6])
            lines.append(f"Files written: {names}")
        if answer_preview:
            lines.append(f"Answer: {answer_preview[:200]}")
        with p.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except OSError as exc:
        logger.debug("task_log.md write failed: %s", exc)


def _save_fact_to_graph(graph, fact: str) -> None:
    """Upsert a remembered fact — merge into a related node when possible.

    Strategy:
      1. Extract content words from the new fact (4+ chars, not stop words).
      2. Search existing 'user' nodes for ones sharing ≥2 content words.
      3. If found: append the new fact to that node's summary (skip if duplicate).
      4. If not found: create a new node keyed by the fact text.

    This prevents the graph from filling with duplicate preference nodes
    (e.g. "I prefer vim" and "I use vim for editing" → same node).
    """
    import re as _re

    if not graph or not fact:
        return
    fact = fact.strip()

    _STOP = {
        "the", "and", "for", "with", "that", "this", "from", "into",
        "you", "are", "was", "have", "has", "will", "can", "use", "used",
        "using", "prefer", "like", "want", "need", "also", "just", "very",
        "remember", "note", "save", "keep", "mind", "always", "never",
        "all", "its", "own", "let", "get", "set", "put", "new", "old",
    }

    def _content_words(text: str) -> set[str]:
        # 3-char minimum so short but meaningful tokens like "vim", "git",
        # "css", "sql", "cpp" are included alongside longer words.
        return {
            w for w in _re.findall(r"[a-z0-9]+", text.lower())
            if len(w) >= 3 and w not in _STOP
        }

    fact_words = _content_words(fact)

    if fact_words:
        try:
            hits = graph.search(fact, limit=5, node_type="user")
            for hit in hits:
                node_words = _content_words(
                    hit.get("name", "") + " " + hit.get("summary", "")
                )
                overlap = len(fact_words & node_words)
                # threshold 1: any shared content word means same topic
                # (e.g. both mention "vim", "python", "tabs" → same note)
                if overlap >= 1:
                    existing = hit.get("summary", "")
                    # Already captured — just refresh last_seen
                    if fact.lower()[:60] in existing.lower():
                        graph.upsert_node(hit["name"], "user", summary=existing)
                        logger.info("save_memory: already known, refreshed %r", hit["name"][:40])
                        return
                    # Merge: append new fact to existing node
                    merged = f"{existing.rstrip('. ')}. {fact}".strip()[:500]
                    graph.upsert_node(hit["name"], "user", summary=merged)
                    logger.info("save_memory: merged %r into node %r (overlap=%d)",
                                fact[:40], hit["name"][:40], overlap)
                    return
        except Exception as exc:
            logger.debug("save_memory: graph search failed: %s", exc)

    # No related node found — create a new one
    graph.upsert_node(fact[:80], "user", summary=fact)
    logger.info("save_memory: new node %r", fact[:60])


def _write_pref_to_file(prefs_path, fact: str) -> None:
    """Write-back: append a remembered fact to user_prefs.md so the file
    stays in sync with the graph.

    Skipped if the fact (normalised — stripped of trailing punctuation and
    lowercased, first 60 chars) is already present in the file.
    Creates the file and any missing parent directories.
    """
    if not prefs_path or not fact:
        return
    try:
        _pp = Path(prefs_path)
        existing = _pp.read_text(encoding="utf-8") if _pp.exists() else ""
        # Normalise: strip trailing punctuation / whitespace before comparing
        # so "I prefer tabs over spaces." doesn't duplicate "I prefer tabs over spaces"
        fact_norm = fact.rstrip(".!?,;: ").lower()[:60]
        # Also normalise each existing line the same way before checking
        existing_norm = " ".join(
            line.rstrip(".!?,;: ").lower() for line in existing.splitlines()
        )
        if fact_norm in existing_norm:
            return  # already recorded (possibly with different punctuation)
        _pp.parent.mkdir(parents=True, exist_ok=True)
        with open(_pp, "a", encoding="utf-8") as _f:
            _f.write(fact + "\n")
        logger.info("_write_pref_to_file: appended %r to %s", fact[:50], _pp.name)
    except Exception as exc:
        logger.debug("_write_pref_to_file: skipped (%s)", exc)


# Regex patterns used by _apply_workspace_to_cmd — compiled once at module load.
_RELATIVE_FILE_WRITE = re.compile(
    r'>\s*(?![/\\"])(?!\w+:)([^\s<>&|;,\'"()\r\n]+)',  # > relfile  (not abs path)
)
_RELATIVE_PY_RUN = re.compile(
    r'\bpython\d*(?:\.exe)?\s+(?![/\\\-])(?!\w+:)(\S+\.py)\b',
)

# Commands that should NOT be wrapped — they don't need workspace context.
_WORKSPACE_SKIP_RE = re.compile(
    r'^(git |pip\d*\s|npm |conda |systeminfo|ipconfig|ifconfig|nvidia-|wmic |tasklist'
    r'|Get-|Set-|New-|Remove-|Start-|Stop-|Invoke-|Test-|where |which |echo\s+\$)',
    re.IGNORECASE,
)


def _apply_workspace_to_cmd(cmd: str, workspace: str) -> str:
    """Safety net: if a bash command writes/runs a relative file, prefix with
    ``cd "<workspace>" && `` so task-created scripts land in workspace/ rather
    than the engine project root.

    Only activates when:
      • a workspace path is configured
      • the command redirects output to a relative filename  (> calc.py)
      • OR the command runs a relative Python script           (python calc.py)
      • AND the command doesn't already reference the workspace path

    Skips git, pip, system-info commands that are unaffected by CWD.
    """
    if not workspace or not cmd.strip():
        return cmd
    cmd = cmd.strip()

    # Already contains the workspace path — nothing to do.
    ws_norm = workspace.replace("\\", "/").lower()
    if ws_norm in cmd.lower() or workspace.lower() in cmd.lower():
        return cmd

    # Skip infrastructure commands that don't write task files.
    if _WORKSPACE_SKIP_RE.match(cmd):
        return cmd

    needs_cd = False

    def _is_bare_relative(fname: str) -> bool:
        """True only for bare filenames with no directory component (e.g. calc.py).
        Returns False for absolute paths (C:/..., C:\\..., /...) or paths that
        already contain a directory separator (workspace/calc.py, etc.).
        """
        if not fname:
            return False
        if re.match(r'^[a-zA-Z]:[/\\]|^[/\\]', fname):  # absolute
            return False
        if '/' in fname or '\\' in fname:  # already has directory component
            return False
        return True

    # Output redirection to a bare filename?  (> calc.py, >> result.txt)
    for m in _RELATIVE_FILE_WRITE.finditer(cmd):
        if _is_bare_relative(m.group(1)):
            needs_cd = True
            break

    # Running a bare .py filename?  (python calc.py)
    if not needs_cd:
        m = _RELATIVE_PY_RUN.search(cmd)
        if m and _is_bare_relative(m.group(1)):
            needs_cd = True

    if needs_cd:
        return f'cd "{workspace}" && {cmd}'
    return cmd


def _sub_workspace(cmd: str, workspace: str) -> str:
    """Replace the literal token WORKSPACE (or WORKSPACE_PATH) with the actual
    workspace path.  Small models often output these as literal strings instead
    of substituting the path provided in the prompt.

    Uses simple str.replace (case-sensitive for WORKSPACE, case-insensitive for
    workspace_path) — avoids word-boundary regex edge cases on Windows paths.
    """
    if not workspace or "WORKSPACE" not in cmd.upper():
        return cmd
    ws = workspace.replace("\\", "/")
    # Replace longer token first to avoid partial substitution
    cmd = cmd.replace("WORKSPACE_PATH", ws).replace("workspace_path", ws)
    cmd = cmd.replace("WORKSPACE", ws)
    return cmd


def _normalize_python_c(cmd: str) -> str:
    """Fix common model mistakes in `python -c '...'` commands.

    1. Re-wrap single-quoted bodies with double quotes (single quotes break
       when the code contains string literals or imports).
    2. Insert semicolons between adjacent Python statements that the model
       concatenated without a separator (e.g. print(1)print(2) → print(1); print(2)).
    """
    import re as _re

    # Unwrap whichever quoting style was used
    m = _re.match(r'^(python\d*\s+-c\s+)[\'"](.*)[\'"]\s*$', cmd, _re.DOTALL)
    if not m:
        return cmd

    prefix, body = m.group(1), m.group(2)

    # Insert "; " between adjacent statements: identifier/closing-paren
    # immediately followed by an opening token with no whitespace/semicolon.
    # Covers: print(x)print(y), a=1b=2, etc.
    body = _re.sub(r'(\)|\w)(print\s*\(|import\s+|\w+\s*=)', r'\1; \2', body)

    body = body.replace('"', '\\"')
    return f'{prefix}"{body}"'


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


async def _expand_query(query: str, client) -> str:
    """Generate one semantically different reformulation of a search query.

    Used by web_search to fire a second angle on the same topic — e.g.
    "meaning of life" → "purpose and significance of human existence philosophy".
    Returns "" on failure so the caller falls back to the original query only.
    """
    try:
        result = await client.generate(
            [{"role": "user", "content":
                f"Rewrite this search query from a different angle to find complementary results.\n"
                f"Original: {query}\n"
                f"Output only the rewritten query — no explanation, no quotes."}],
            max_tokens=40,
            temperature=0.4,
            stream=False,
            thinking=False,
        )
        alt = (result["choices"][0]["message"]["content"] or "").strip().strip('"\'')
        # Reject if too similar (>80% word overlap) or too long
        orig_words = set(query.lower().split())
        alt_words  = set(alt.lower().split())
        if alt_words and orig_words:
            overlap = len(orig_words & alt_words) / max(len(orig_words), len(alt_words))
            if overlap < 0.8 and len(alt) < 120:
                return alt
    except Exception:
        pass
    return ""


_BASH_FAIL_PREFIXES = (
    "[exit ",      # non-zero exit code from bash tool
    "[timeout",    # command timed out
    "Error:",      # Python exception or tool-level error
    "error:",
)
_BASH_FAIL_SUBSTRINGS = (
    "command not found",
    "is not recognized",   # Windows cmd error
    "No such file or directory",
    "Permission denied",
    "cannot find the path",
    "ModuleNotFoundError",
    "SyntaxError",
    "Traceback (most recent call last)",
)

def _is_bash_failure(result: str) -> bool:
    """Return True if a bash tool result indicates the command failed."""
    r = result.strip()
    if not r or r == "(no output)":
        return False  # silent success
    for prefix in _BASH_FAIL_PREFIXES:
        if r.startswith(prefix):
            return True
    low = r.lower()
    for sub in _BASH_FAIL_SUBSTRINGS:
        if sub.lower() in low:
            return True
    return False


_OS_NAME = _platform.system()   # "Windows", "Linux", "Darwin" — set once at import

_BASH_CORRECT_SYSTEM = (
    "A shell command failed. Output ONLY a corrected JSON object: "
    '{\"command\": \"<fixed shell command>\"}\n'
    "Rules:\n"
    f"- OS is {_OS_NAME}. Fix the command for this OS.\n"
    "- If the error says 'is not recognized' or 'command not found', the command is\n"
    f"  likely wrong for {_OS_NAME}. Replace it with the correct {_OS_NAME} equivalent.\n"
    "  Windows examples: uptime→(Get-Date)-(gcim Win32_OperatingSystem).LastBootUpTime, "
    "ls→Get-ChildItem, grep→Select-String, find→Get-ChildItem -Recurse.\n"
    "- Fix the specific error — wrong path, missing module, syntax error, etc.\n"
    "- Keep the same intent as the original command.\n"
    "- Do not explain. Do not use markdown. Output only the JSON object."
)


async def _bash_correct(original_cmd: str, error: str, client) -> str:
    """Ask the LLM to produce a corrected command given the failure output.

    Returns the corrected command string, or "" if correction fails.
    """
    prompt = (
        f"OS: {_OS_NAME}\n"
        f"Original command: {original_cmd[:300]}\n"
        f"Error output:\n{error[:400]}\n\n"
        "Output the corrected command as JSON."
    )
    try:
        result = await client.generate(
            [
                {"role": "system", "content": _BASH_CORRECT_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=128,
            temperature=0.0,
            stream=False,
            thinking=False,
            response_format={"type": "json_object"},
        )
        raw = (result["choices"][0]["message"]["content"] or "").strip()
        import json as _json
        obj = _json.loads(raw)
        cmd = obj.get("command", "").strip()
        return cmd
    except Exception as exc:
        logger.warning("bash_correct: LLM call failed: %s", exc)
        return ""


# ── Smart fix loop helpers ────────────────────────────────────────────────────

# Matches "  File \"path/to/file.py\", line N" in Python tracebacks
_TB_FILE_RE = re.compile(
    r'File ["\']([^"\']+\.py)["\'],\s*line\s*(\d+)',
    re.IGNORECASE,
)
# Matches the final error line "ErrorType: message"
_TB_ERROR_RE = re.compile(
    r'^([A-Za-z][A-Za-z0-9_]*(?:Error|Exception|Warning)):\s*(.+)',
    re.MULTILINE,
)


def _parse_traceback(stderr: str) -> dict | None:
    """Extract structured info from a Python traceback string.

    Returns {file, line, error_type, message} for the LAST file reference
    in the traceback (the proximate cause), or None if no traceback found.
    Only returns user-owned files (not site-packages / stdlib paths).
    """
    if "Traceback" not in stderr and "Error:" not in stderr:
        return None

    file_matches = _TB_FILE_RE.findall(stderr)
    if not file_matches:
        return None

    # Walk backward to find the last user-owned file (not stdlib / site-packages)
    user_file, user_line = "", "0"
    for fpath, lineno in reversed(file_matches):
        norm = fpath.replace("\\", "/").lower()
        if ("site-packages" in norm or "lib/python" in norm or
                "lib\\python" in norm or "<" in fpath):
            continue
        user_file, user_line = fpath, lineno
        break

    if not user_file:
        # All files are stdlib — use last match anyway
        user_file, user_line = file_matches[-1]

    err_match = _TB_ERROR_RE.search(stderr)
    error_type = err_match.group(1) if err_match else "Error"
    message    = err_match.group(2).strip() if err_match else stderr.strip()[-120:]

    return {
        "file":       user_file,
        "line":       int(user_line),
        "error_type": error_type,
        "message":    message[:200],
    }


def _resolve_fix_file(raw_path: str, project_dir: str, workspace: str) -> str:
    """Turn a raw path from a traceback into an absolute path we can Read.

    Returns "" if the file cannot be resolved to something that exists on disk.
    """
    if not raw_path:
        return ""
    # Already absolute and exists
    if os.path.isabs(raw_path) and os.path.isfile(raw_path):
        return raw_path
    # Try against workspace first, then project_dir
    for base in (workspace, project_dir):
        if base:
            candidate = os.path.join(base, os.path.basename(raw_path))
            if os.path.isfile(candidate):
                return candidate
            candidate2 = os.path.join(base, raw_path)
            if os.path.isfile(candidate2):
                return candidate2
    return ""


# ── Visible progress helper ───────────────────────────────────────────────────

def _render_project_progress(state: "PipelineState") -> str:  # type: ignore[name-defined]
    """Build a one-line-per-file progress checklist for multi-file project builds.

    Shown at the start of each new file so the user can see overall progress.
    Returns "" for single-file tasks (no clutter for simple writes).

    The format mirrors Claude Code's TodoWrite output so it feels familiar:
      ✓ models.py — done
      → api.py — writing now
      ○ main.py
    """
    # Only show for tasks with multiple write_code stages
    write_tasks = [
        (i, s) for i, s in enumerate(state.sub_tasks)
        if s.get("type") in ("write_code", "write_project")
    ]
    if len(write_tasks) <= 1:
        return ""

    checklist = []
    for task_idx, t in write_tasks:
        goal = t.get("task", "")
        # Extract "models.py" from "Write models.py: Todo dataclass..."
        fn_m = re.search(r'\b([A-Za-z0-9_\-]+\.[a-z]{2,5})\b', goal)
        label = fn_m.group(1) if fn_m else goal[:30]
        if task_idx < state.current_task_idx:
            checklist.append(f"[done] {label}")
        elif task_idx == state.current_task_idx:
            checklist.append(f"[ --> ] {label}  (writing)")
        else:
            checklist.append(f"[    ] {label}")
    inner = "\n".join(checklist)
    return f"**Project progress:**\n```\n{inner}\n```"


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
        in_file = any(
            (item["anchor"].strip().lower() in k.strip().lower()
             or k.strip().lower() in item["anchor"].strip().lower())
            for k in (parsed or {})
        )
        lines += [
            "",
            f'Resume item [{wp_idx}]: "{item["anchor"]}"',
            f"  Anchor {'IS already in the file (partial body — write body only, NO def/## line)' if in_file else 'is NOT yet in the file — write from the anchor header'}.",
            f"  Min {item['min_chars']} chars of substantive content.",
        ]

    if file_content:
        tail_lines = file_content.splitlines()[-30:]
        tail = "\n".join(tail_lines)
        lines += [
            "",
            "--- current file tail (seam — continue AFTER the last line below) ---",
            tail,
            "--- end seam ---",
        ]

    return "\n".join(lines)


def _extract_filename_from_task(task: str, file_type: str) -> str:
    """Try to extract a target filename from a task description.

    Used when the model supplies a directory as file_path instead of a file.
    Returns a filename string (possibly with extension) or "" if not found.
    """
    import re as _re
    # Explicit filename patterns: "in foo.py", "to bar.md", "file foo.txt"
    m = _re.search(
        r'\b(?:in|to|file|named?|called?|as)\s+["\']?([A-Za-z0-9_\-]+\.[a-z]{2,5})["\']?',
        task, _re.IGNORECASE,
    )
    if m:
        return m.group(1)
    # Bare filename anywhere in the task
    m = _re.search(r'\b([A-Za-z0-9_\-]+\.(py|md|txt|js|ts|json|yaml|yml|html|css))\b', task)
    if m:
        return m.group(1)
    # Infer extension from file_type and first meaningful word
    ext = ".py" if file_type == "code" else ".md"
    words = _re.findall(r'[A-Za-z][A-Za-z0-9_]+', task)
    skip = {"write", "create", "implement", "make", "generate", "a", "the", "an",
            "short", "simple", "module", "file", "script", "document", "essay", "in"}
    for w in words:
        if w.lower() not in skip and len(w) >= 3:
            return w.lower() + ext
    return ""


def _get_item_text_from_content(content: str, anchor: str, file_type: str) -> str | None:
    """Extract the exact text block of a named item from file content.

    For code: returns the full def/class block (header + body).
    For docs: returns the heading line + body.
    Returns None if the anchor cannot be found.
    Used by the Edit-in-place retry path in _write_plan_next_item.
    """
    try:
        from engine.translation.subtask.verifier import (
            parse_doc_sections, parse_code_items, _match_key,
        )
        parsed = (parse_doc_sections(content) if file_type == "doc"
                  else parse_code_items(content))
        key = _match_key(anchor, parsed)
        if key is None:
            return None
        body = parsed.get(key, "")
        if file_type == "doc":
            for prefix in ("## ", "# ", "### "):
                heading = f"{prefix}{key}"
                if heading in content:
                    return heading + ("\n" + body if body else "")
            return None
        # code: parse_code_items already returns def/class header + body as one string
        return body or None
    except Exception:
        return None


def _rebuild_item_content(existing_text: str, new_body: str) -> str:
    """Keep the header line of existing_text, replace the body with new_body.

    Used to construct new_string for the Edit tool in the retry path.
    Strips accidental header-repetition when the model echoes the def/## line.
    """
    lines = existing_text.splitlines()
    header = lines[0] if lines else ""
    body_lines = new_body.splitlines()
    # If model echoed the header as its first line, strip it
    if body_lines and body_lines[0].strip() == header.strip():
        body_lines = body_lines[1:]
    new_body_clean = "\n".join(body_lines).lstrip("\n")
    return (header + "\n" + new_body_clean) if header else new_body_clean


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


# ── Multi-file project helpers ────────────────────────────────────────────────

# Build verbs + multi-file subject keywords that signal a project-scale task.
# Conservative: require BOTH so "write a Python function" doesn't trigger it.
_PROJECT_BUILD_VERBS = frozenset({"build", "create", "develop", "implement", "make"})
_PROJECT_SUBJECT_KW  = (
    " app", "application", " api", " server", " service",
    " project", " system", " backend", " website", " cli",
    "rest api", "web app", "todo ", "microservice",
)


def _is_project_query(query: str) -> bool:
    """True if the query describes a multi-file project to be built.

    Used in _start to decide whether to expand a single write_code stage
    (or a write_project stage) into per-file stages via the project planner.
    """
    q = query.lower()
    first_word = q.split()[0] if q.split() else ""
    has_verb = (first_word in _PROJECT_BUILD_VERBS or
                any(f" {v} " in q for v in _PROJECT_BUILD_VERBS))
    has_subject = any(kw in q for kw in _PROJECT_SUBJECT_KW)
    return has_verb and has_subject


def _inject_file_sigs_to_graph(file_path: str, graph) -> None:
    """Extract def/class signature lines from a code file and save as skill nodes.

    Called after each file in a project is written so subsequent files can
    import or reference the available functions/classes.  Each signature is
    stored as a graph node (type="skill", kind="code_sig") so it can be
    retrieved by _get_graph_sigs() before writing the next file.
    """
    if graph is None:
        return
    try:
        content = Path(file_path).read_text(encoding="utf-8")
        basename = os.path.basename(file_path)
        # Match "def foo(..." and "class Bar:" at module level or inside classes
        _SIG_RE = re.compile(
            r'^((?:def |class )\w[\w\d_]*(?:\([^)]{0,120}\))?)\s*(?:->.*?)?:',
            re.MULTILINE,
        )
        saved = 0
        saved_labels: list[str] = []
        for m in _SIG_RE.finditer(content):
            sig = m.group(1).strip()
            if not (3 < len(sig) < 120):
                continue
            # Skip dunder methods — unlikely to be useful cross-file
            if sig.startswith("def __") and not sig.startswith("def __init__"):
                continue
            label       = sig
            content_str = f"{sig}  # {basename}"
            graph.upsert_node(
                label, "skill",
                summary=content_str,
                metadata={"file": basename, "kind": "code_sig"},
            )
            # produced edge: file → skill sig (weight 1.0 each time it's saved)
            try:
                graph.upsert_edge(basename, "produced", label, weight=1.0)
            except Exception:
                pass
            saved_labels.append(label)
            saved += 1

        # related_to edges between all sigs in the same file (co-defined)
        for i in range(len(saved_labels)):
            for j in range(i + 1, len(saved_labels)):
                try:
                    graph.upsert_edge(saved_labels[i], "related_to", saved_labels[j], weight=0.5)
                except Exception:
                    pass

        if saved:
            logger.info("pipeline: saved %d sig(s) from %s to graph", saved, basename)
    except Exception as exc:
        logger.warning("pipeline: sig extraction failed for %s: %s", file_path, exc)


def _get_graph_sigs(goal: str, graph, max_sigs: int = 8) -> str:
    """Return a compact snippet of relevant def/class signatures from the graph.

    Used in _write_plan_next_item to inject cross-file context so the model
    knows what functions/classes are already available in sibling files.
    Returns "" when no signatures are found or the graph is unavailable.
    """
    if graph is None:
        return ""
    try:
        nodes = graph.search(goal[:80], top_n=max_sigs, node_types=["skill"])
        sigs = [
            n["content"] for n in nodes
            if n.get("metadata", {}).get("kind") == "code_sig"
        ]
        if not sigs:
            return ""
        return "# Available from other project files:\n" + "\n".join(sigs[:max_sigs])
    except Exception as exc:
        logger.debug("pipeline: _get_graph_sigs failed: %s", exc)
        return ""


