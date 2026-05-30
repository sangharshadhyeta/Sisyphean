"""Unified task context — grounded gather + feedback-driven interpretation loop.

Flow
----
1. gather_grounded()       — deterministic Python, no LLM
       workspace listing, last N history turns, merged memory

2. interpret_context()     — one focused LLM call
       intent / done / failed / constraints / state
       accepts feedback from failed stages so re-runs are progressively richer

3. Context loop (pipeline-driven)
       Each failed stage appends structured feedback:
         {stage, received, attempted, failed, needed}
       Pipeline re-calls interpret_context(grounded, feedback) until:
         a) context is sufficient for planning, OR
         b) MAX_CTX_ITER reached

4. format_for_*(ctx)       — zero-cost slicing for each downstream stage
       format_for_decompose()  → think_decompose
       format_for_planning()   → plan_task
       format_for_synthesis()  → synthesizer

The key invariant: stages only ever see clean distilled context, never raw history.
Raw results from gap-filling feed back into interpret_context, not directly to stages.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_CTX_ITER = 2   # maximum context rebuild iterations per query

# ---------------------------------------------------------------------------
# Interpret system prompt
# ---------------------------------------------------------------------------
# Fields are historical facts extracted from conversation — NOT what the
# planner computes.  intent/done/failed/constraints/state give the planner
# grounded input so it can decide what to do next without attending over
# the full raw history.

_INTERPRET_SYSTEM = """\
Extract structured task context from the data below. Output JSON only.

Fields:
  intent      — what the user ultimately wants (one sentence, from full conversation)
  done        — what has already been completed successfully (empty string if nothing yet)
  failed      — what was attempted and broke, so planner avoids repeating it (empty if first try)
  constraints — explicit rules or limits stated by the user (empty string if none)
  state       — the current blocking issue or gap; empty string if clear to proceed"""

# ---------------------------------------------------------------------------
# Grounded step — deterministic, no LLM
# ---------------------------------------------------------------------------

def _list_workspace(workspace: str, max_entries: int = 20) -> list[str]:
    try:
        p = Path(workspace)
        if not p.exists():
            return []
        entries = []
        for item in sorted(p.iterdir()):
            if item.name.startswith("."):
                continue
            entries.append(item.name + ("/" if item.is_dir() else ""))
            if len(entries) >= max_entries:
                break
        return entries
    except Exception:
        return []


def _format_turns(raw_history: list[dict], n: int = 4, max_chars: int = 400) -> str:
    parts = []
    for turn in raw_history[-n:]:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    content = block.get("text", "")
                    break
            else:
                content = ""
        if role and content:
            parts.append(f"{role}: {str(content)[:120].replace(chr(10), ' ')}")
    return "\n".join(parts)[:max_chars]


def gather_grounded(
    query: str,
    raw_history: list[dict],
    memory_ctx: str,
    recall_ctx: str,
    workspace: str,
    project_ctx: str,
    extra: dict | None = None,   # additional data for re-interpretation (e.g. results summary)
) -> dict:
    """Collect all deterministic context — no LLM, no async."""
    history_turns   = _format_turns(raw_history)
    workspace_files = _list_workspace(workspace)

    mem_parts = []
    if memory_ctx:
        mem_parts.append(memory_ctx[:400])
    if recall_ctx and recall_ctx not in memory_ctx:
        mem_parts.append(recall_ctx[:200])
    combined_memory = "\n".join(mem_parts)

    grounded = {
        "query":           query,
        "history_turns":   history_turns,
        "memory":          combined_memory,
        "workspace_root":  workspace,
        "workspace_files": workspace_files,
        "project_ctx":     project_ctx,
    }
    if extra:
        grounded.update(extra)
    return grounded


# ---------------------------------------------------------------------------
# Interpret step — one LLM call
# ---------------------------------------------------------------------------

async def interpret_context(
    query: str,
    grounded: dict,
    client,
    feedback: list[dict] | None = None,
) -> dict:
    """One LLM call: distil grounded data into structured context.

    feedback: list of {stage, received, attempted, failed, needed} dicts from
    previously failed stages.  Included in the prompt so each re-run extracts
    more targeted information from history.
    """
    files_str = ", ".join(grounded.get("workspace_files", [])[:10]) or "(empty)"
    history   = grounded.get("history_turns", "")[:250]
    memory    = grounded.get("memory", "")[:250]
    ws_root   = grounded.get("workspace_root", "")
    extra_done = grounded.get("done_summary", "")     # set by pipeline on re-runs
    extra_results = grounded.get("results_summary", "")

    parts = [f"Query: {query[:200]}"]
    if ws_root:
        parts.append(f"Workspace: {ws_root}  [{files_str}]")
    if history:
        parts.append(f"Recent:\n{history}")
    if memory:
        parts.append(f"Memory:\n{memory}")
    if extra_done:
        parts.append(f"Completed this session:\n{extra_done}")
    if extra_results:
        parts.append(f"Tool results so far:\n{extra_results}")

    # Structured feedback from failed stages — this is the core of the loop.
    # Each re-run sees what was tried, what broke, and what was needed so it
    # can extract more targeted information from the full history.
    if feedback:
        fb_lines = []
        for f in feedback[-3:]:
            line = f"[{f.get('stage', '?')}]"
            if f.get("received"):
                line += f"\n  context it received: {f['received'][:120]}"
            if f.get("attempted"):
                line += f"\n  attempted: {f['attempted'][:80]}"
            if f.get("failed"):
                line += f"\n  failed: {f['failed'][:120]}"
            if f.get("needed"):
                line += f"\n  needed: {f['needed'][:80]}"
            fb_lines.append(line)
        parts.append("Previous attempts (use to extract more targeted context):\n"
                     + "\n".join(fb_lines))

    user_msg = "\n".join(parts)

    try:
        result = await client.generate(
            [
                {"role": "system", "content": _INTERPRET_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=128,
            temperature=0.0,
            stream=False,
            thinking=False,
            response_format={"type": "json_object"},
        )
        raw = (result["choices"][0]["message"]["content"] or "").strip()
        obj  = json.loads(raw)
        return {
            "intent":      str(obj.get("intent",      ""))[:200],
            "done":        str(obj.get("done",        ""))[:200],
            "failed":      str(obj.get("failed",      ""))[:200],
            "constraints": str(obj.get("constraints", ""))[:150],
            "state":       str(obj.get("state",       ""))[:150],
        }
    except Exception as exc:
        logger.warning("context_builder: interpret failed: %s", exc)
        return {
            "intent":      query[:100],
            "done":        "",
            "failed":      "",
            "constraints": "",
            "state":       "",
        }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def generate_task_context(
    query: str,
    raw_history: list[dict],
    memory_ctx: str,
    recall_ctx: str,
    workspace: str,
    project_ctx: str,
    client,
    route: str = "",
    feedback: list[dict] | None = None,
    extra_grounded: dict | None = None,
) -> dict:
    """Build the unified TaskContext dict.

    Returns a plain dict (JSON-serialisable) — stored in PipelineState and
    survives the thinking-block round-trip.

    On re-runs (feedback is non-empty), extra_grounded carries live session
    data (files written, results summary) so interpret sees the current state.
    """
    grounded = gather_grounded(
        query=query,
        raw_history=raw_history,
        memory_ctx=memory_ctx,
        recall_ctx=recall_ctx,
        workspace=workspace,
        project_ctx=project_ctx,
        extra=extra_grounded,
    )
    interpreted = await interpret_context(query, grounded, client, feedback=feedback)

    ctx = {**grounded, **interpreted}
    logger.info(
        "context_builder: intent=%r done=%r failed=%r state=%r (feedback=%d)",
        ctx.get("intent", "")[:50],
        ctx.get("done", "")[:40],
        ctx.get("failed", "")[:40],
        ctx.get("state", "")[:40],
        len(feedback) if feedback else 0,
    )
    return ctx


# ---------------------------------------------------------------------------
# Format functions — each stage gets the slice it needs
# ---------------------------------------------------------------------------

def format_for_decompose(ctx: dict, max_chars: int = 800) -> str:
    """Slice for think_decompose: intent + recent history + memory + workspace."""
    parts = []
    if ctx.get("intent"):
        parts.append(f"Intent: {ctx['intent']}")
    if ctx.get("failed"):
        parts.append(f"Previously failed: {ctx['failed']}")
    if ctx.get("history_turns"):
        parts.append(f"[Recent]\n{ctx['history_turns'][:300]}")
    if ctx.get("memory"):
        parts.append(f"[Memory]\n{ctx['memory'][:250]}")
    if ctx.get("workspace_files"):
        parts.append("Workspace: " + ", ".join(ctx["workspace_files"][:8]))
    return "\n\n".join(p for p in parts if p.strip())[:max_chars]


def format_for_planning(ctx: dict, stage_type: str = "", max_chars: int = 600) -> str:
    """Slice for plan_task: intent + done/failed/constraints + workspace state."""
    parts = []
    if ctx.get("intent"):
        parts.append(f"Intent: {ctx['intent']}")
    if ctx.get("done"):
        parts.append(f"Done: {ctx['done']}")
    if ctx.get("failed"):
        parts.append(f"Failed (avoid repeating): {ctx['failed']}")
    if ctx.get("constraints"):
        parts.append(f"Constraints: {ctx['constraints']}")
    if ctx.get("state"):
        parts.append(f"Current blocker: {ctx['state']}")
    if ctx.get("workspace_root"):
        files = ", ".join(ctx.get("workspace_files", [])[:8]) or "(empty)"
        parts.append(f"Workspace: {ctx['workspace_root']}\nFiles: {files}")
    if ctx.get("memory"):
        parts.append(f"Context: {ctx['memory'][:200]}")
    return "\n\n".join(p for p in parts if p.strip())[:max_chars]


def format_for_synthesis(
    ctx: dict,
    files_written: list[str] | None = None,
    commands_run: list[dict] | None = None,
    synthesis_history: str = "",
    max_chars: int = 1500,
) -> str:
    """Slice for synthesizer: full picture + live session activity."""
    parts = []
    if ctx.get("memory"):
        parts.append(f"[Memory]\n{ctx['memory'][:400]}")
    if ctx.get("history_turns"):
        parts.append(f"[History]\n{ctx['history_turns'][:300]}")
    if synthesis_history:
        parts.append(f"[Related]\n{synthesis_history[:300]}")
    if ctx.get("project_ctx"):
        parts.append(f"[Project]\n{ctx['project_ctx'][:200]}")
    if ctx.get("intent"):
        parts.append(f"Intent: {ctx['intent']}")
    if ctx.get("done"):
        parts.append(f"Done: {ctx['done']}")

    activity: list[str] = []
    if files_written:
        activity.extend(f"wrote: {f}" for f in files_written[-4:])
    if commands_run:
        activity.extend(
            f"ran: {r.get('cmd', '')[:60]} → {r.get('brief', '')[:40]}"
            for r in commands_run[-4:]
        )
    if activity:
        parts.append("[Session]\n" + "\n".join(activity))

    return "\n\n".join(p for p in parts if p.strip())[:max_chars]
