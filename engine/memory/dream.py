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

    llm_url = f"http://{config.llm.server.host}:{config.llm.server.port}"
    client = LlamaClient(llm_url, mock=config.mock)
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
