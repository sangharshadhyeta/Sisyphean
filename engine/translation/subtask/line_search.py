"""Progressive context retrieval for the subtask writer.

Ported from BirdClaw's tools/line_search.py — self-contained, no BirdClaw deps.

Three public entry points:

  search_relevant(goal, paths)
      Goal-driven search: extracts key terms from goal, finds the highest-scoring
      lines across the given files. Returns formatted string for LLM injection.

  find_section(path, title, file_type)
      Finds the section/function whose name best matches 'title' and returns
      that block (heading → next heading). For targeted continuation writes.

  find_continuation_point(path, file_type)
      Returns from the LAST section/function header to EOF — the natural
      re-entry point when no exact section match is found.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

_MAX_INJECT_CHARS = 1200   # hard cap on total injected text per call
_MIN_TERM_LEN     = 3
_STOP_WORDS       = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "into",
    "write", "create", "make", "add", "get", "set", "use", "run",
    "build", "file", "code", "function", "class", "method", "data",
    "next", "new", "all", "each", "any", "not", "its", "are", "was",
})


# ── Internal types ───────────────────────────────────────────────────────────

@dataclass
class _Match:
    path:    Path
    line_no: int          # 1-based
    line:    str
    context: list[str] = field(default_factory=list)
    score:   int = 0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def _extract_terms(goal: str) -> list[str]:
    """Extract meaningful search terms from a goal/title string."""
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|[a-z]+", goal.lower())
    terms: list[str] = []
    seen: set[str] = set()
    for t in tokens:
        if len(t) >= _MIN_TERM_LEN and t not in _STOP_WORDS and t not in seen:
            seen.add(t)
            terms.append(t)
    # Also expand snake_case / camelCase fragments
    extras: list[str] = []
    for t in terms:
        parts = re.findall(r"[a-z]+", re.sub(r"([A-Z])", r"_\1", t).lower())
        for p in parts:
            if len(p) >= _MIN_TERM_LEN and p not in _STOP_WORDS and p not in seen:
                seen.add(p)
                extras.append(p)
    return terms + extras


def _format_matches(matches: list[_Match], cap: int = _MAX_INJECT_CHARS) -> str:
    if not matches:
        return ""
    parts: list[str] = []
    total = 0
    for m in matches:
        block = f"[{m.path.name}:{m.line_no}] {m.line}"
        for c in m.context:
            block += f"\n  {c}"
        total += len(block)
        if total > cap:
            break
        parts.append(block)
    return "\n".join(parts)


# ── Public API ────────────────────────────────────────────────────────────────

def search_relevant(
    goal: str,
    paths: Sequence[str | Path],
    context_lines: int = 2,
    max_results: int = 6,
) -> str:
    """Goal-driven search: which lines in these files are relevant to this goal?

    Extracts key terms from goal, scores each line by term overlap, returns
    the highest-scoring matches formatted for LLM injection.
    Returns empty string if nothing relevant found.
    """
    if not goal or not paths:
        return ""
    terms = _extract_terms(goal)
    if not terms:
        return ""
    term_set = set(terms)

    def _score(line: str) -> int:
        words = set(re.findall(r"[a-z0-9_]+", line.lower()))
        return len(term_set & words)

    candidates: list[_Match] = []
    for raw_path in paths:
        p = Path(raw_path)
        if not p.is_file():
            continue
        lines = _read_lines(p)
        for i, line in enumerate(lines):
            s = _score(line)
            if s > 0:
                lo = max(0, i - context_lines)
                hi = min(len(lines), i + context_lines + 1)
                ctx = lines[lo:i] + lines[i + 1:hi]
                candidates.append(_Match(path=p, line_no=i + 1, line=line, context=ctx, score=s))

    if not candidates:
        logger.debug("[line_search] relevant goal=%r terms=%d hits=0", goal[:40], len(terms))
        return ""

    candidates.sort(key=lambda m: (-m.score, m.path.name, m.line_no))
    seen_lines: set[tuple[Path, int]] = set()
    top: list[_Match] = []
    for m in candidates:
        key = (m.path, m.line_no)
        if key not in seen_lines:
            seen_lines.add(key)
            top.append(m)
        if len(top) >= max_results:
            break

    # Re-sort by file + line for readable output
    top.sort(key=lambda m: (m.path.name, m.line_no))
    logger.debug(
        "[line_search] relevant goal=%r terms=%d candidates=%d top=%d",
        goal[:40], len(terms), len(candidates), len(top),
    )
    return _format_matches(top)


def find_section(path: str | Path, title: str, file_type: str) -> str:
    """Find the section or function most relevant to 'title' and return that block.

    For docs (md): finds the ## heading whose words best overlap with title,
                   returns from that heading to just before the next heading.
    For code (py): finds the def/class whose name best overlaps with title terms,
                   returns from that line to just before the next top-level def/class.

    Returns empty string if nothing matches (score == 0) or file is empty.
    Caps returned block at 60 lines to avoid huge sections.
    """
    p = Path(path)
    lines = _read_lines(p)
    if not lines:
        return ""

    if file_type == "doc" or str(path).endswith(".md"):
        header_re = re.compile(r"^(#{1,3}) (.+)")
        name_group = 2
    elif file_type == "code" or str(path).endswith(".py"):
        header_re = re.compile(r"^(?:async def |def |class )(\w+)")
        name_group = 1
    else:
        return ""

    terms = set(_extract_terms(title))
    if not terms:
        return ""

    headers: list[tuple[int, int, str]] = []
    for i, line in enumerate(lines):
        m = header_re.match(line)
        if m:
            name_words = set(re.findall(r"[a-z0-9]+", m.group(name_group).lower()))
            score = len(terms & name_words)
            headers.append((i, score, line))

    if not headers:
        return ""

    best_idx, best_score, best_line = max(headers, key=lambda h: h[1])
    if best_score == 0:
        logger.debug("[line_search] find_section title=%r path=%s result=no-match",
                     title[:40], Path(path).name)
        return ""

    # Section ends at the next header (or EOF)
    end_idx = len(lines)
    for (i, _, _) in headers:
        if i > best_idx:
            end_idx = i
            break

    start = max(0, best_idx - 1)
    block = lines[start:end_idx]
    if len(block) > 60:
        block = block[:60]

    logger.debug(
        "[line_search] find_section title=%r path=%s matched=%r lines=%d score=%d",
        title[:40], Path(path).name, best_line.strip()[:50], len(block), best_score,
    )
    return "\n".join(block)


def find_continuation_point(path: str | Path, file_type: str) -> str:
    """Return from the last section/function header to EOF — the natural re-entry point.

    Used as a fallback when find_section finds nothing relevant.
      Docs (md):  last ## or ### heading → EOF
      Code (py):  last def / class / async def → EOF
      Fallback:   last 30 lines

    Returns empty string if file doesn't exist or is empty.
    """
    p = Path(path)
    lines = _read_lines(p)
    if not lines:
        return ""

    if file_type == "doc" or str(path).endswith(".md"):
        header_re = re.compile(r"^#{1,3} ")
    elif file_type == "code" or str(path).endswith(".py"):
        header_re = re.compile(r"^(?:async def |def |class )")
    else:
        header_re = None

    if header_re:
        last_idx = -1
        for i, line in enumerate(lines):
            if header_re.match(line):
                last_idx = i
        if last_idx >= 0:
            start = max(0, last_idx - 2)
            logger.debug("[line_search] continuation path=%s last_header=line%d",
                         Path(path).name, last_idx + 1)
            return "\n".join(lines[start:])

    logger.debug("[line_search] continuation path=%s fallback=last30", Path(path).name)
    return "\n".join(lines[-30:])
