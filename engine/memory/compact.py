"""History compaction — prevents token-limit crashes in long sessions.

Sisyphean is fully stateless: the entire conversation history is sent by
Claude Code on every request.  As sessions grow the history accumulates:
  - Thinking blocks with old SISYPHEAN_STATE snapshots (obsolete — only the
    most recent state matters, and it lives in the last thinking block)
  - tool_result blocks with large stdout / file contents (already consumed)
  - tool_use blocks (lightweight, keep — they show what happened)
  - text blocks (keep — summaries / answers reference earlier answers)

Strategy
--------
1. Keep the last `preserve_turns` user+assistant pairs verbatim.
2. For all older turns, strip:
     - thinking blocks entirely (SISYPHEAN_STATE is stale; reasoning is noise)
     - tool_result blocks with content longer than `max_result_chars`
       (replace with a stub that notes the result was compacted)
3. Returns (compacted_history, was_compacted: bool).

This runs at the START of every handle_messages() call so the loop always
works on a slimmed history.  The raw history Claude Code manages is
unchanged — compaction is applied in-process only.

Design notes
------------
- We never drop turns; we only slim their block content.
  The model still sees the full conversation shape (which tools were called),
  just not the massive blobs inside old tool results.
- The most recent SISYPHEAN_STATE thinking block is always preserved because
  it's in the last assistant turn, which is inside the "keep recent" window.
- For compaction to fire, the history must exceed both `min_turns` and
  `token_estimate_threshold`.  Short sessions are untouched.
"""
from __future__ import annotations

import logging
from typing import NamedTuple

logger = logging.getLogger(__name__)

# ── Tunables ──────────────────────────────────────────────────────────────────

# Number of (user, assistant) round-trips preserved verbatim at the tail.
_PRESERVE_TURNS: int = 4

# Minimum number of turns before compaction fires at all.
_MIN_TURNS_TO_COMPACT: int = 6

# tool_result content longer than this gets replaced with a stub.
_MAX_RESULT_CHARS: int = 800

# Rough token budget that triggers compaction (chars ÷ 4).
_TOKEN_THRESHOLD: int = 6_000

_COMPACTED_STUB = "[tool result compacted — content was consumed in earlier turn]"
_STATE_PREFIX = "SISYPHEAN_STATE:"


# ── Public types ──────────────────────────────────────────────────────────────

class CompactionResult(NamedTuple):
    history: list[dict]       # the (possibly) slimmed history
    was_compacted: bool       # True if any blocks were removed/stubbed
    turns_compacted: int      # how many old turns were slimmed
    tokens_saved: int         # rough estimate of chars saved ÷ 4


# ── Token estimation ─────────────────────────────────────────────────────────

def _estimate_tokens(history: list[dict]) -> int:
    total = 0
    for turn in history:
        content = turn.get("content", [])
        if isinstance(content, str):
            total += len(content) // 4 + 1
        else:
            for block in content:
                for key in ("text", "thinking"):
                    val = block.get(key)
                    if isinstance(val, str):
                        total += len(val) // 4 + 1
                # tool_result content may be a list of blocks or a raw string
                if block.get("type") == "tool_result":
                    rc = block.get("content", "")
                    if isinstance(rc, str):
                        total += len(rc) // 4 + 1
                    elif isinstance(rc, list):
                        for rb in rc:
                            total += len(rb.get("text", "")) // 4 + 1
    return total


# ── Block-level helpers ───────────────────────────────────────────────────────

def _slim_block(block: dict) -> tuple[dict | None, int]:
    """Return (slimmed_block_or_None, chars_saved).

    Returns None to drop the block entirely.
    """
    btype = block.get("type", "")

    # Drop all thinking blocks in old turns — SISYPHEAN_STATE is stale,
    # reasoning is noise, and thinking blocks are often the largest items.
    if btype == "thinking":
        saved = len(block.get("thinking", ""))
        return None, saved

    # Stub out large tool_result content; keep small ones verbatim.
    if btype == "tool_result":
        rc = block.get("content", "")
        if isinstance(rc, str):
            if len(rc) > _MAX_RESULT_CHARS:
                saved = len(rc) - len(_COMPACTED_STUB)
                return {**block, "content": _COMPACTED_STUB}, saved
        elif isinstance(rc, list):
            # List of content blocks (text/image).  Compact text-only.
            full_len = sum(len(b.get("text", "")) for b in rc)
            if full_len > _MAX_RESULT_CHARS:
                stub_block = [{"type": "text", "text": _COMPACTED_STUB}]
                return {**block, "content": stub_block}, full_len - len(_COMPACTED_STUB)
        return block, 0

    # Everything else (tool_use, text) — keep as-is.
    return block, 0


def _slim_turn(turn: dict) -> tuple[dict, int]:
    """Slim a single turn's content blocks. Returns (slimmed_turn, chars_saved)."""
    content = turn.get("content", [])
    if isinstance(content, str):
        return turn, 0  # plain-string content — nothing to strip

    new_blocks: list[dict] = []
    total_saved = 0
    for block in content:
        slimmed, saved = _slim_block(block)
        total_saved += saved
        if slimmed is not None:
            new_blocks.append(slimmed)

    if not new_blocks:
        # Never return a turn with empty content — keep a minimal text stub
        # so the history shape is preserved (model sees a turn happened).
        new_blocks = [{"type": "text", "text": "[turn compacted]"}]

    return {**turn, "content": new_blocks}, total_saved


# ── Public API ────────────────────────────────────────────────────────────────

def compact_history(
    raw_history: list[dict],
    preserve_turns: int = _PRESERVE_TURNS,
    min_turns: int = _MIN_TURNS_TO_COMPACT,
    token_threshold: int = _TOKEN_THRESHOLD,
) -> CompactionResult:
    """Slim old turns in raw_history to keep the token count manageable.

    Parameters
    ----------
    raw_history:
        The full message list as returned by _messages_to_dicts() in
        engine/compat/anthropic.py.  Each entry is {"role": ..., "content": [...]}
        where content is a list of typed blocks.
    preserve_turns:
        Number of recent (user, assistant) pairs to keep verbatim.
        Default 4 → 8 messages preserved at the tail.
    min_turns:
        Don't compact unless the history has at least this many entries.
    token_threshold:
        Don't compact unless the estimated token count exceeds this.

    Returns
    -------
    CompactionResult with the slimmed history (or original if no compaction needed).
    """
    n = len(raw_history)

    # Quick bail-outs
    if n < min_turns:
        return CompactionResult(raw_history, False, 0, 0)

    estimated = _estimate_tokens(raw_history)
    if estimated < token_threshold:
        return CompactionResult(raw_history, False, 0, 0)

    # The tail we preserve verbatim: last preserve_turns * 2 messages
    # (each "turn" = 1 user message + 1 assistant message).
    keep_count = min(preserve_turns * 2, n)
    cut_at = n - keep_count       # index of first kept turn

    if cut_at <= 0:
        return CompactionResult(raw_history, False, 0, 0)

    compacted: list[dict] = []
    total_chars_saved = 0
    turns_slimmed = 0

    for i, turn in enumerate(raw_history):
        if i < cut_at:
            slimmed, saved = _slim_turn(turn)
            compacted.append(slimmed)
            if saved > 0:
                total_chars_saved += saved
                turns_slimmed += 1
        else:
            compacted.append(turn)

    was_compacted = turns_slimmed > 0
    tokens_saved = total_chars_saved // 4

    if was_compacted:
        logger.info(
            "[compact] slimmed %d/%d turns  ~%d tokens saved  (preserve_tail=%d)",
            turns_slimmed, cut_at, tokens_saved, keep_count,
        )

    return CompactionResult(
        history=compacted,
        was_compacted=was_compacted,
        turns_compacted=turns_slimmed,
        tokens_saved=tokens_saved,
    )


def should_compact(raw_history: list[dict]) -> bool:
    """Quick check — True if compact_history() would do work."""
    if len(raw_history) < _MIN_TURNS_TO_COMPACT:
        return False
    return _estimate_tokens(raw_history) >= _TOKEN_THRESHOLD
