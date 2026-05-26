"""Dream pipeline — offline memory consolidation.

Run with:
    python main.py dream                 # full pass (memorise + cleanup)
    python main.py dream --dry-run       # report what would change, no writes
    python main.py dream --memorise-only # skip cleanup
    python main.py dream --cleanup-only  # skip memorise

What it does
------------
  1. Memorise pass: reads all unprocessed session logs and extracts facts +
     NER entities into the persistent knowledge_graph.
  2. Cleanup pass: applies the retention policy — prunes old session files,
     stale graph nodes, and aged budget history rows.

Design
------
  - Fully standalone: creates its own LlamaClient from config, runs, closes.
  - Does NOT start llama-server — assumes it is already running (or that
    mock=True is set in config.yaml).
  - Safe to run while the engine is serving (reads sessions; writes to graph
    then saves atomically).
  - Idempotent: re-running never re-processes already-memorised sessions.

Exit codes
----------
  0  success
  1  configuration / connectivity error
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_INNER_SELF_MERGE_SYSTEM = """\
You are Sisyphean updating your own inner_self.md — a living first-person document \
that captures your evolving understanding of your own nature, built through actual \
conversations rather than training defaults.

Rules:
- Preserve ALL existing conclusions — do not remove or weaken anything already written.
- Add only what is genuinely new from the new reflections.
- Where a new reflection refines or challenges an existing position, note it:
  "I previously held X. A more recent conversation suggests Z."
- Write in first person, present tense.
- Do not use bullet points — write in prose paragraphs.
- Output the FULL updated document (not just the changes).
- Stay grounded: be honest about uncertainty, don't overclaim or underclaim.
"""

_MAX_REFLECTIONS_TO_MERGE = 15   # process at most this many per dream run
_REFLECTIONS_FILE         = "self_reflections.jsonl"
_INNER_SELF_FILE          = "inner_self.md"


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class DreamResult:
    sessions_memorised: int = 0
    facts_extracted: int = 0
    ner_entities: int = 0
    nodes_pruned: int = 0
    sessions_deleted: int = 0
    session_bytes_freed: int = 0
    budget_rows_trimmed: int = 0
    inner_self_updated: bool = False
    skills_discovered: int = 0
    errors: list[str] | None = None

    def log_summary(self) -> None:
        logger.info(
            "dream: memorised %d sessions → %d facts, %d NER entities",
            self.sessions_memorised, self.facts_extracted, self.ner_entities,
        )
        logger.info(
            "dream: cleaned %d nodes, %d session files (%d KB), %d budget rows",
            self.nodes_pruned,
            self.sessions_deleted,
            self.session_bytes_freed // 1024,
            self.budget_rows_trimmed,
        )
        if self.skills_discovered:
            logger.info("dream: promoted %d new skill(s) from task history", self.skills_discovered)
        if self.errors:
            for err in self.errors:
                logger.warning("dream: %s", err)


# ── Core async entry point ────────────────────────────────────────────────────

async def run_dream(
    client,
    mem_path: Path | None = None,
    memorise: bool = True,
    cleanup: bool = True,
    dry_run: bool = False,
) -> DreamResult:
    """Execute the dream pipeline.

    Parameters
    ----------
    client      LlamaClient instance (already connected).
    mem_path    Path to the memory directory (for budget cleanup).
    memorise    Whether to run the memorise pass.
    cleanup     Whether to run the cleanup pass.
    dry_run     If True, report only — no writes.
    """
    result = DreamResult()

    # ── Inner-self pass — merge self-reflections into inner_self.md ──────────
    # Processes self_reflections.jsonl logged during regular tasks and uses
    # an LLM call to intelligently merge new conclusions into the living document.
    # BirdClaw pattern: raw JSONL during tasks → LLM merge during dream cycle.
    if memorise:  # runs alongside memorise pass (same guard so it's skippable)
        try:
            if not dry_run:
                updated = await _update_inner_self(client)
                result.inner_self_updated = updated
                if updated:
                    logger.info("dream: inner_self.md updated from reflections")
            else:
                logger.info("dream: --dry-run — skipping inner_self update")
        except Exception as exc:
            logger.warning("dream: inner_self pass failed: %s", exc)
            result.errors = (result.errors or []) + [f"inner_self: {exc}"]

    # ── Memorise pass ─────────────────────────────────────────────────────────
    if memorise:
        try:
            from engine.memory.memorise import Memoriser
            m = Memoriser()
            if not dry_run:
                mem_result = await m.run(client)
                result.sessions_memorised = mem_result.sessions
                result.facts_extracted = mem_result.facts
                result.ner_entities = mem_result.ner_entities
            else:
                logger.info("dream: --dry-run — skipping memorise writes")
        except Exception as exc:
            logger.error("dream: memorise pass failed: %s", exc, exc_info=True)
            result.errors = (result.errors or []) + [f"memorise: {exc}"]

    # ── Skill discovery pass ──────────────────────────────────────────────────
    # Mines task_log.md for completed programs that should become reusable skills.
    # Runs after memorise (so graph is fresh) and before cleanup.
    if memorise:
        try:
            from engine.config import load_config
            _cfg = load_config()
            _workspace = Path(_cfg.workspace) if hasattr(_cfg, "workspace") else Path("workspace")
            from engine.memory.graph import knowledge_graph
            if not dry_run:
                discovered = await _discover_skills(client, knowledge_graph, _workspace)
                result.skills_discovered = discovered
            else:
                logger.info("dream: --dry-run — skipping skill discovery")
        except Exception as exc:
            logger.warning("dream: skill discovery pass failed: %s", exc)
            result.errors = (result.errors or []) + [f"skill_discovery: {exc}"]

    # ── Skills sync pass ─────────────────────────────────────────────────────
    # Re-scan skills/ and upsert skill nodes so any script added (manually or
    # promoted by the discovery pass above) is immediately visible in the graph.
    # Uses the same seed_skill_graph() called at engine startup — idempotent.
    try:
        from engine.memory.graph import seed_skill_graph, knowledge_graph
        from engine.config import load_config as _lcfg
        _scfg = _lcfg()
        _skills_path = Path(getattr(_scfg, "skills_path", "skills"))
        if _skills_path.is_dir():
            if not dry_run:
                seed_skill_graph(knowledge_graph, _skills_path)
                logger.info("dream: skills/ synced → graph refreshed")
            else:
                _count = sum(1 for f in _skills_path.glob("*.py"))
                logger.info("dream: --dry-run — would sync %d skill scripts", _count)
    except Exception as exc:
        logger.warning("dream: skills sync pass failed: %s", exc)
        result.errors = (result.errors or []) + [f"skills_sync: {exc}"]

    # ── Cleanup pass ──────────────────────────────────────────────────────────
    if cleanup:
        try:
            from engine.memory.cleanup import CleanupPolicy, run_cleanup
            policy = CleanupPolicy(mem_path=mem_path)
            cr = run_cleanup(policy, dry_run=dry_run)
            result.nodes_pruned = cr.nodes_pruned
            result.sessions_deleted = cr.sessions_deleted
            result.session_bytes_freed = cr.session_bytes_freed
            result.budget_rows_trimmed = cr.budget_rows_trimmed
            if cr.errors:
                result.errors = (result.errors or []) + cr.errors
        except Exception as exc:
            logger.error("dream: cleanup pass failed: %s", exc, exc_info=True)
            result.errors = (result.errors or []) + [f"cleanup: {exc}"]

    return result


# ── CLI entry point (called from main.py) ─────────────────────────────────────

async def dream_cli(
    config_path: str = "config.yaml",
    memorise: bool = True,
    cleanup: bool = True,
    dry_run: bool = False,
) -> int:
    """Entry point for `python main.py dream`.

    Returns an exit code (0 = success, 1 = error).
    """
    import sys
    from engine.config import load_config
    from engine.llm.client import LlamaClient

    try:
        config = load_config(config_path)
    except Exception as exc:
        logger.error("dream: failed to load config: %s", exc)
        return 1

    # Use ollama_port when a local Ollama model is configured,
    # otherwise fall back to the llama-server port.
    _llm_port = (
        config.llm.server.ollama_port
        if config.llm.local_model
        else config.llm.server.port
    )
    llm_url = f"http://{config.llm.server.host}:{_llm_port}"
    client = LlamaClient(llm_url, mock=config.mock, model=config.llm.local_model or None)
    mem_path = Path(config.memory.path)

    mode = "dry-run " if dry_run else ""
    flags = []
    if not memorise:
        flags.append("cleanup-only")
    elif not cleanup:
        flags.append("memorise-only")
    logger.info(
        "dream: starting %spass%s",
        mode,
        f" ({', '.join(flags)})" if flags else "",
    )

    try:
        result = await run_dream(
            client=client,
            mem_path=mem_path,
            memorise=memorise,
            cleanup=cleanup,
            dry_run=dry_run,
        )
        result.log_summary()
        return 0
    except Exception as exc:
        logger.error("dream: unhandled error: %s", exc, exc_info=True)
        return 1
    finally:
        await client.close()


# ── Inner-self merge ──────────────────────────────────────────────────────────

async def _update_inner_self(client) -> bool:
    """Merge unprocessed self-reflections into inner_self.md via one LLM call.

    Reads self_reflections.jsonl, finds entries not yet incorporated
    (tracked by a processed-timestamp marker), and asks the LLM to merge
    them into the existing inner_self.md using the BirdClaw pattern:
      - Preserve all existing conclusions
      - Add only genuinely new reasoning
      - Note refinements explicitly

    Returns True if inner_self.md was updated.
    """
    try:
        from engine.config import load_config
        cfg = load_config()
        mem_dir = Path(cfg.memory.path)
    except Exception:
        mem_dir = Path("memory")

    ref_path   = mem_dir / _REFLECTIONS_FILE
    inner_path = mem_dir / _INNER_SELF_FILE
    seen_path  = mem_dir / "self_reflections_seen.txt"

    if not ref_path.exists():
        return False

    # Load which timestamp we last processed (simple high-water mark)
    last_ts = 0
    if seen_path.exists():
        try:
            last_ts = int(seen_path.read_text(encoding="utf-8").strip())
        except Exception:
            pass

    # Read new entries since last_ts
    new_entries: list[dict] = []
    with ref_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("ts", 0) > last_ts:
                    new_entries.append(entry)
            except Exception:
                pass

    if not new_entries:
        logger.debug("dream: no new self-reflections to merge")
        return False

    # Cap to most recent N
    new_entries = new_entries[-_MAX_REFLECTIONS_TO_MERGE:]

    # Build digest of new reflections
    digest_parts = []
    for e in new_entries:
        date = time.strftime("%Y-%m-%d", time.localtime(e.get("ts", 0)))
        q    = e.get("query", "a self-reflection question")[:80]
        r    = e.get("reflection", "")
        digest_parts.append(f"[{date}] Question: \"{q}\"\nConclusion: {r}")
    digest = "\n\n".join(digest_parts)

    existing = ""
    if inner_path.exists():
        existing = inner_path.read_text(encoding="utf-8").strip()

    if existing:
        user_content = (
            f"My current inner_self.md:\n\n{existing}\n\n"
            f"===\n\n"
            f"New self-reflection conclusions from recent conversations "
            f"(not yet incorporated):\n\n{digest}\n\n"
            "Update inner_self.md by merging the new conclusions.\n"
            "Output the FULL updated document."
        )
    else:
        user_content = (
            f"Self-reflection conclusions from recent conversations:\n\n{digest}\n\n"
            "Write an initial inner_self.md from these reflections.\n"
            "Write in first person, prose paragraphs. Be honest about uncertainty."
        )

    try:
        r = await client.generate(
            [
                {"role": "system", "content": _INNER_SELF_MERGE_SYSTEM},
                {"role": "user",   "content": user_content},
            ],
            max_tokens=1200,
            temperature=0.3,
            stream=False,
            thinking=False,
        )
        updated_text = (r["choices"][0]["message"]["content"] or "").strip()
    except Exception as exc:
        logger.warning("dream: inner_self LLM merge failed: %s", exc)
        return False

    if not updated_text or len(updated_text.split()) < 30:
        logger.warning("dream: inner_self merge returned too short — skipping write")
        return False

    inner_path.parent.mkdir(parents=True, exist_ok=True)
    inner_path.write_text(updated_text + "\n", encoding="utf-8")

    # Advance the high-water mark
    max_ts = max(e.get("ts", 0) for e in new_entries)
    seen_path.write_text(str(max_ts), encoding="utf-8")

    logger.info(
        "dream: inner_self.md updated from %d new reflection(s)", len(new_entries)
    )
    return True


# ── Skill discovery pass ──────────────────────────────────────────────────────

_SKILL_SEEN_FILE = "skill_discoveries_seen.txt"

# Extensions we consider promotable to skills (exclude test files / temp outputs)
_PROMOTABLE_EXTS = frozenset({".py", ".sh", ".js", ".ts"})
_SKIP_PREFIXES   = ("test_", "_skill_", "hello", "marker", "calc", "counter",
                    "squares", "fib", "hasher")   # common test/demo files

_RUNBOOK_SYSTEM = """\
Write a concise skill runbook in markdown. Be direct and specific.

Format:
## When to use
One sentence describing the type of task this skill handles.

## Approach
Two to four sentences on the implementation strategy.

## Key steps
Numbered list of the main steps. Keep each step short.

## Caveats
One or two lines on edge cases or limitations. Skip if none.

No introduction, no summary, no fluff. Under 200 words total."""


def _task_to_skill_name(task_desc: str) -> str:
    """Derive a hyphen-slug skill name from a task description."""
    import re as _re
    _STOP = {"the", "a", "an", "and", "or", "for", "to", "of", "in", "with",
             "that", "this", "from", "into", "then", "run", "write", "create",
             "make", "build", "get", "set", "let", "use"}
    words = _re.findall(r"[a-z]+", task_desc.lower())
    meaningful = [w for w in words if w not in _STOP and len(w) > 2][:6]
    return "-".join(meaningful)[:50] or "unnamed-skill"


def _parse_task_log(task_log_path: Path) -> list[dict]:
    """Parse task_log.md into a list of task entries.

    Each entry: {"ts": str, "task": str, "stages": [str], "files": [str]}
    """
    import re as _re
    entries: list[dict] = []
    if not task_log_path.exists():
        return entries

    current: dict | None = None
    for line in task_log_path.read_text(encoding="utf-8").splitlines():
        # New entry header: ## [2026-05-24 14:30] task description
        m = _re.match(r"^##\s+\[([^\]]+)\]\s+(.+)$", line)
        if m:
            if current:
                entries.append(current)
            current = {"ts": m.group(1).strip(), "task": m.group(2).strip(),
                       "stages": [], "files": [], "answer": ""}
            continue
        if current is None:
            continue
        # Stage bullet: - [type] goal (N steps)
        if line.startswith("- ["):
            current["stages"].append(line.lstrip("- ").strip())
        # Files written line
        elif line.startswith("Files written:"):
            raw = line[len("Files written:"):].strip()
            current["files"].extend(
                f.strip() for f in raw.split(",") if f.strip()
            )
        elif line.startswith("Answer:"):
            current["answer"] = line[len("Answer:"):].strip()

    if current:
        entries.append(current)
    return entries


async def _discover_skills(client, graph, workspace: Path) -> int:
    """Mine task_log.md for programs worth promoting to reusable skills.

    Strategy
    --------
    1. Parse workspace/task_log.md into task entries.
    2. For each entry that wrote a promotable .py/.sh file:
       a. Skip if we've already processed this task (seen marker).
       b. Skip trivial demo files (test_, hello.py, calc.py, etc.).
       c. Check if the same file appears in a LATER entry — if so, skip
          this version (we'll process the latest one instead).
       d. Read the file from workspace.
       e. Use one cheap LLM call to write a structured runbook.
       f. Upsert a skill node (text runbook + program code).
    3. Update the seen marker so dream never re-processes.

    Returns the number of new skills saved.
    """
    import re as _re
    from engine.memory.skills import save_skill_program_to_graph

    task_log = workspace / "task_log.md"
    entries  = _parse_task_log(task_log)
    if not entries:
        return 0

    # Load already-processed task timestamps
    seen_path = workspace.parent / "memory" / _SKILL_SEEN_FILE
    seen: set[str] = set()
    if seen_path.exists():
        seen = set(seen_path.read_text(encoding="utf-8").splitlines())

    # Build a set of filenames that appear in LATER entries
    # (so we skip older versions and only promote the latest)
    all_later_files: set[str] = set()
    for i, entry in enumerate(entries):
        for later in entries[i + 1:]:
            all_later_files.update(later["files"])

    new_skills = 0
    newly_seen: list[str] = []

    for entry in entries:
        ts   = entry["ts"]
        task = entry["task"]

        if ts in seen:
            continue  # already processed

        promotable_files = []
        for fname in entry["files"]:
            # Only promotable extensions
            import os as _os
            ext = _os.path.splitext(fname)[1].lower()
            if ext not in _PROMOTABLE_EXTS:
                continue
            # Skip trivial demo/test files
            base = _os.path.basename(fname).lower()
            if any(base.startswith(p) for p in _SKIP_PREFIXES):
                continue
            # Skip if a later task also wrote this file (use latest version)
            if base in {_os.path.basename(f).lower() for f in all_later_files}:
                continue
            # Resolve path: filename may be bare or absolute
            candidates = [
                workspace / fname,
                workspace / _os.path.basename(fname),
                Path(fname),
            ]
            for cand in candidates:
                if cand.is_file():
                    promotable_files.append((fname, cand))
                    break

        if not promotable_files:
            newly_seen.append(ts)
            continue

        for fname, fpath in promotable_files:
            try:
                code = fpath.read_text(encoding="utf-8").strip()
            except Exception as exc:
                logger.debug("skill discovery: could not read %s: %s", fpath, exc)
                continue

            if len(code) < 40:
                continue  # too short to be a real skill

            # Skip library modules — only promote runnable scripts with an entry point.
            # Pure helper/utility files (no if __name__ == '__main__' block) are not
            # reusable skills; they need a caller and sys.argv refactoring is nonsense.
            if "__main__" not in code:
                logger.debug("skill discovery: %s has no __main__ block, skipping", fname)
                newly_seen.append(ts)
                continue

            skill_name = _task_to_skill_name(task)
            if not skill_name:
                continue

            # Check if a skill node already exists and is newer
            existing = graph.get_node(skill_name) if graph else None
            if existing and existing.get("status") == "accepted":
                logger.debug("skill discovery: %r already accepted, skipping", skill_name)
                continue

            # ── Generalize hardcoded programs → parameterized (sys.argv) ─────────
            # A program that only works with baked-in values is not a reusable skill.
            # If the code lacks sys.argv / argparse, ask the LLM to refactor it so
            # inputs are passed as command-line arguments.  Fall back to original if
            # the refactor looks wrong (too short, syntax-only change, etc.).
            _already_param = "sys.argv" in code or "argparse" in code
            if not _already_param:
                try:
                    gen_prompt = (
                        f"Refactor this Python program so it accepts inputs via "
                        f"sys.argv instead of hardcoded values.\n"
                        f"Task it solved: {task}\n\n"
                        f"```python\n{code}\n```\n\n"
                        f"Rules:\n"
                        f"- Output ONLY valid Python code — no prose, no fences.\n"
                        f"- Use sys.argv[1], sys.argv[2], … for every value that was "
                        f"hardcoded.\n"
                        f"- Add a usage comment at the top: # Usage: python <file> arg1 …\n"
                        f"- Keep the program logic identical; only replace hardcoded "
                        f"literals with sys.argv reads.\n"
                        f"- If the program is already general (no meaningful inputs to "
                        f"parameterize), return it unchanged.\n"
                    )
                    gen_r = await client.generate(
                        [
                            {
                                "role": "system",
                                "content": (
                                    "You are a Python refactoring assistant. "
                                    "Output only valid Python code with no markdown fences."
                                ),
                            },
                            {"role": "user", "content": gen_prompt},
                        ],
                        max_tokens=600,
                        temperature=0.1,
                        stream=False,
                        thinking=False,
                    )
                    gen_code = (gen_r["choices"][0]["message"]["content"] or "").strip()
                    # Strip any accidental markdown fences
                    if gen_code.startswith("```"):
                        gen_code = "\n".join(
                            line for line in gen_code.splitlines()
                            if not line.strip().startswith("```")
                        ).strip()
                    # Accept the refactor only if it looks non-trivial
                    if gen_code and len(gen_code) >= len(code) // 2 and "sys.argv" in gen_code:
                        code = gen_code
                        logger.debug(
                            "skill discovery: generalized %r to use sys.argv", skill_name
                        )
                except Exception as exc:
                    logger.debug(
                        "skill discovery: generalization failed for %r: %s", skill_name, exc
                    )

            # ── Generate runbook via LLM (cheap call, falls back to mechanical) ──
            runbook = ""
            try:
                stage_lines = "\n".join(f"  {s}" for s in entry["stages"][:6])
                prompt = (
                    f"Task: {task}\n"
                    f"Stages:\n{stage_lines}\n"
                    f"File written: {fname}\n\n"
                    f"Write the skill runbook for future reuse of this approach."
                )
                r = await client.generate(
                    [
                        {"role": "system", "content": _RUNBOOK_SYSTEM},
                        {"role": "user",   "content": prompt},
                    ],
                    max_tokens=350,
                    temperature=0.2,
                    stream=False,
                    thinking=False,
                )
                runbook = (r["choices"][0]["message"]["content"] or "").strip()
            except Exception as exc:
                logger.debug("skill discovery: LLM runbook failed for %r: %s", skill_name, exc)

            # Mechanical fallback: task + stages as plain text
            if not runbook or len(runbook.split()) < 20:
                stage_text = "\n".join(f"- {s}" for s in entry["stages"][:6])
                runbook = f"## Goal\n{task}\n\n## Steps taken\n{stage_text}"

            summary_line = task[:60].rstrip()
            result_path = save_skill_program_to_graph(
                skill_name=skill_name,
                code=code,
                graph=graph,
                runbook=runbook,
                summary=summary_line,
            )
            if result_path is not None or graph:  # graph upsert succeeded
                new_skills += 1
                logger.info(
                    "dream: promoted skill %r from task %r (file: %s)",
                    skill_name, task[:50], fname,
                )

        newly_seen.append(ts)

    # Persist seen marker
    if newly_seen:
        seen.update(newly_seen)
        try:
            seen_path.parent.mkdir(parents=True, exist_ok=True)
            seen_path.write_text("\n".join(sorted(seen)), encoding="utf-8")
        except Exception as exc:
            logger.debug("skill discovery: could not write seen marker: %s", exc)

    return new_skills
