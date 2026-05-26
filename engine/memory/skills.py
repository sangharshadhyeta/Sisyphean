"""Skill injection utilities — progressive disclosure for learned approaches.

Design philosophy
-----------------
Skills are not just runbooks.  A skill is a living artifact:
  - summary   → 60-char description for the compact planning-time index
  - content   → full markdown runbook (approach, steps, caveats)
  - program   → actual executable code that implements the skill
  - program_path → on-disk path in ~/.sisyphean/skill_scripts/ (always valid if present)
  - version   → incremented each time the program is improved
  - status    → "draft" (first run), "accepted" (user satisfied), "improved" (revised)

When a skill has a program, the compact index shows [runnable] so the model
knows it can use run_skill:NAME to re-execute without re-planning from scratch.

Progressive disclosure layers
------------------------------
  Layer 1  get_skill_index()       Planning time, always injected.
                                   Jaccard-scores skill nodes against task.
                                   Returns name + 60-char description + [runnable] tag.

  Layer 2  get_skill_runbook()     Execution time, model-requested.
                                   Loads full markdown runbook (content field).
                                   Triggered by read_skill:NAME internal tool.

  Layer 3  get_skill_program()     Execution time, automatic.
                                   Loads actual program code for run_skill dispatch.
                                   Called by pipeline._execute when tool==run_skill.

Version tracking
----------------
  When save_skill_program() is called on an existing skill node, the current
  program is moved to program_history (capped at 3 versions) before being
  replaced.  This preserves the improvement trail without graph bloat.

  program_history is a JSON-encoded list of {version: N, code: "...snippet..."}.
  Only the first 400 chars of each historical version are kept — enough for
  BirdClaw's dream cycle to understand what changed, not enough to bloat the graph.

Skill scripts directory
-----------------------
  ~/.sisyphean/skill_scripts/<skill-name>.py

  Programs are written here when first saved and re-used on run_skill.
  The on-disk file is the canonical execution artifact; the graph node's
  'program' field is the backup / transport mechanism.

Hermes design parallels
-----------------------
  Hermes Agent: YAML-frontmatter markdown + compact index + skill_manage tool
  Our design:   graph node          + compact index + read_skill / save_skill / run_skill
  Key addition beyond Hermes: program field + version history + run_skill execution.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # graph type imported lazily

logger = logging.getLogger(__name__)

# ── Storage ───────────────────────────────────────────────────────────────────

SKILL_SCRIPTS_DIR = Path.home() / ".sisyphean" / "skill_scripts"

# ── Tuning ────────────────────────────────────────────────────────────────────

_INDEX_TOP_N      = 4    # max skills in compact index
_SUMMARY_LEN      = 60   # max chars per compact-index bullet
_MIN_SCORE        = 1    # min token overlap to appear in index
_MAX_HISTORY      = 3    # how many old versions to keep in program_history
_HISTORY_SNIPPET  = 400  # chars of old code preserved per history entry


# ── Internal helpers ──────────────────────────────────────────────────────────

def _token_overlap(a: str, b: str) -> int:
    """Count of shared lowercase tokens between two strings."""
    return len(set(a.lower().split()) & set(b.lower().split()))


def _safe_slug(name: str) -> str:
    """Convert a skill name to a filesystem-safe slug."""
    slug = re.sub(r"[^\w\-]", "-", name.lower().strip())
    return re.sub(r"-{2,}", "-", slug).strip("-")[:80]


def _resolve_node(skill_name: str, graph) -> dict | None:
    """Return the graph node for skill_name (exact then fuzzy fallback).

    Fuzzy fallback requires >= 2 token overlap so that generic words that
    happen to appear in many skill names (e.g. "skill", "tool", "run")
    don't cause an unrelated skill to be returned for a completely
    different query like "no-skill-here".
    """
    node = graph.get_node(skill_name)
    if not node:
        hits = graph.search(skill_name, top_n=3, node_type="skill")
        # Require the best match to share >= 2 tokens with the query
        # to avoid false positives on single common words.
        if hits:
            query_tokens = set(re.findall(r"[a-z0-9]+", skill_name.lower()))
            for hit in hits:
                node_tokens = set(
                    re.findall(r"[a-z0-9]+",
                               (hit.get("name", "") + " " + hit.get("summary", "")).lower())
                )
                if len(query_tokens & node_tokens) >= 2:
                    node = hit
                    break
    return node


# ── Layer 1 — Compact index ───────────────────────────────────────────────────

def get_skill_index(task: str, graph, top_n: int = _INDEX_TOP_N,
                    stage_type: str = "") -> str:
    """Compact skill index for skills relevant to *task*.

    Scores all 'skill' graph nodes by token overlap with the task text.
    Returns one bullet per top match:
        "  • name: 60-char summary"           (text-only skill)
        "  • name: 60-char summary [runnable]" (skill with program)

    stage_type: when provided, skills whose stored stage_type field does NOT
    match are excluded.  Skills without a stage_type field are always included
    (backwards compatibility with nodes saved before this field existed).

    Empty string when no relevant skills exist or graph is None.
    Injected into plan_task at planning time (Layer 1).
    """
    if not graph:
        return ""
    try:
        nodes = graph.all_nodes(node_type="skill")
        if not nodes:
            return ""

        scored: list[tuple[int, str, str, bool, str]] = []
        for node in nodes:
            name         = node.get("name", "")
            summary      = node.get("summary", "")
            content      = str(node.get("content", "") or "")[:200]
            has_program  = bool(node.get("program") or node.get("program_path"))
            context_hint = (node.get("context_hint") or "").strip()
            # Stage-type gate: if the node carries a stage_type, only include it
            # when the requested stage matches.  No field = always eligible.
            node_stage = (node.get("stage_type") or "").strip().lower()
            if stage_type and node_stage and node_stage != stage_type.lower():
                continue
            score = _token_overlap(task, f"{name} {summary} {content}")
            if score >= _MIN_SCORE:
                scored.append((score, name, summary, has_program, context_hint))

        if not scored:
            return ""

        scored.sort(reverse=True)
        lines = []
        for _, name, summary, has_prog, ctx_hint in scored[:top_n]:
            desc = summary[:_SUMMARY_LEN].rstrip()
            tag  = " [runnable]" if has_prog else ""
            # Append context hint so the model knows what input to pass
            hint = f" — needs: {ctx_hint}" if ctx_hint else ""
            lines.append(f"  • {name}: {desc}{tag}{hint}")

        logger.debug("skill_index: %d match(es) for %r", len(lines), task[:40])
        return "\n".join(lines)

    except Exception as exc:
        logger.debug("get_skill_index failed: %s", exc)
        return ""


# ── Layer 2 — Full runbook ────────────────────────────────────────────────────

def get_skill_runbook(skill_name: str, graph) -> str:
    """Load the full markdown runbook for a named skill (Layer 2).

    Returns the 'content' field (full runbook) if present, otherwise the
    summary.  Empty string if the skill is not found.

    Called by the read_skill internal tool in pipeline._run_internal.
    """
    if not graph or not skill_name:
        return ""
    try:
        node = _resolve_node(skill_name, graph)
        if not node:
            logger.debug("get_skill_runbook: no skill found for %r", skill_name[:40])
            return ""
        content = node.get("content") or node.get("runbook") or node.get("summary", "")
        result  = str(content).strip()
        logger.debug("get_skill_runbook: %r → %d chars", skill_name[:40], len(result))
        return result
    except Exception as exc:
        logger.debug("get_skill_runbook failed: %s", exc)
        return ""


# ── Layer 3 — Program retrieval ───────────────────────────────────────────────

def get_skill_program(skill_name: str, graph) -> str:
    """Load the executable program code for a named skill (Layer 3).

    Returns the 'program' field (full source code) if present.
    Empty string if the skill has no program or is not found.

    Called by pipeline._execute when tool == run_skill.
    """
    if not graph or not skill_name:
        return ""
    try:
        node = _resolve_node(skill_name, graph)
        if not node:
            return ""
        program = node.get("program", "")
        return str(program).strip() if program else ""
    except Exception as exc:
        logger.debug("get_skill_program failed: %s", exc)
        return ""


def get_skill_script_path(skill_name: str) -> Path:
    """Return the canonical on-disk path for a skill's script.

    Always returns a Path in SKILL_SCRIPTS_DIR — even if the file
    doesn't exist yet.  Callers check existence before reading.
    """
    return SKILL_SCRIPTS_DIR / f"{_safe_slug(skill_name)}.py"


def save_skill_to_disk(skill_name: str, code: str) -> Path | None:
    """Write skill program code to ~/.sisyphean/skill_scripts/<name>.py.

    Creates the directory if needed.  Returns the written Path, or None
    on failure.  Called by save_skill_program_to_graph and by run_skill
    when the on-disk file is missing but the graph has the code.
    """
    if not code or not skill_name:
        return None
    try:
        SKILL_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        path = get_skill_script_path(skill_name)
        path.write_text(code, encoding="utf-8")
        logger.info("skills: wrote %s (%d chars)", path.name, len(code))
        return path
    except Exception as exc:
        logger.warning("skills: could not write skill script: %s", exc)
        return None


# ── Persistence — save to graph + disk ───────────────────────────────────────

def save_skill_program_to_graph(
    skill_name: str,
    code: str,
    graph,
    runbook: str = "",
    summary: str = "",
    stage_type: str = "",
    context_hint: str = "",
) -> Path | None:
    """Upsert a skill node with program code and write script to disk.

    stage_type:   which pipeline stage type this skill belongs to
                  (run | write_code | write_doc | research | edit).
                  Used by get_skill_index() to filter skills by stage.
    context_hint: brief note on what input the skill needs, e.g.
                  "file path", "search query", "url".
                  Surfaced in the tool list so the model knows what to pass.

    Version tracking:
      - Reads the existing node (if any).
      - Moves current 'program' snippet → program_history (capped at _MAX_HISTORY).
      - Increments 'version'.
      - Sets status to "improved" (if existing) or "draft" (if new).
      - Writes code to ~/.sisyphean/skill_scripts/<name>.py.
      - Upserts graph node with all new values.

    Returns the on-disk Path of the written script, or None on disk failure.
    """
    if not graph or not skill_name or not code:
        return None

    try:
        existing  = graph.get_node(skill_name) or {}
        old_prog  = existing.get("program", "")
        old_ver   = int(existing.get("version", 0))
        old_hist  = existing.get("program_history", "[]")

        # Parse history (safe — may be missing or malformed)
        try:
            history: list[dict] = json.loads(old_hist) if isinstance(old_hist, str) else list(old_hist)
        except Exception:
            history = []

        # Append previous program to history before replacing
        if old_prog and old_prog.strip():
            history.append({
                "version": old_ver,
                "snippet": old_prog[:_HISTORY_SNIPPET],
            })
            # Keep only the most recent _MAX_HISTORY entries
            history = history[-_MAX_HISTORY:]

        new_version = old_ver + 1
        new_status  = "improved" if old_prog else "draft"

        # Build summary from first line of runbook or code, if not provided
        if not summary:
            summary = (runbook.splitlines()[0].lstrip("# ").strip()[:_SUMMARY_LEN]
                       if runbook
                       else f"program v{new_version}")

        # Write to disk
        script_path = save_skill_to_disk(skill_name, code)
        path_str    = str(script_path) if script_path else ""

        # Preserve existing stage_type / context_hint if caller didn't provide new ones
        _stage_type    = stage_type    or existing.get("stage_type", "")
        _context_hint  = context_hint  or existing.get("context_hint", "")

        graph.upsert_node(
            name=skill_name,
            node_type="skill",
            summary=summary[:_SUMMARY_LEN],
            content=runbook or existing.get("content", ""),
            sources=["pipeline"],
            # Program fields
            program=code,
            program_path=path_str,
            program_history=json.dumps(history),
            version=new_version,
            status=new_status,
            # Routing metadata — used by get_skill_index stage filter
            stage_type=_stage_type,
            context_hint=_context_hint,
        )
        logger.info(
            "skills: saved program for %r  v%d  %d chars  status=%s",
            skill_name[:40], new_version, len(code), new_status,
        )
        return script_path

    except Exception as exc:
        logger.warning("save_skill_program_to_graph failed: %s", exc)
        return None


def mark_skill_accepted(skill_name: str, graph) -> None:
    """Mark a skill as accepted by the user.

    Called when the user's follow-up indicates satisfaction with the outcome
    (e.g. BirdClaw dream cycle detects no correction in the next turn).
    Safe no-op if the skill does not exist.
    """
    if not graph or not skill_name:
        return
    try:
        node = graph.get_node(skill_name)
        if node:
            graph.upsert_node(skill_name, "skill",
                               summary=node.get("summary", ""),
                               status="accepted")
            logger.info("skills: %r marked accepted", skill_name[:40])
    except Exception as exc:
        logger.debug("mark_skill_accepted failed: %s", exc)
