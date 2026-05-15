"""Engine policy router — semantic section search on engine_policy.md and user_prefs.md.

No LLM call. Parses engine_policy.md into tagged sections, then uses Jaccard bigram
similarity to match the query to the most relevant section.

Jaccard(query, section) = |ngrams(query) ∩ ngrams(section)| / |ngrams(query) ∪ ngrams(section)|
where ngrams = word unigrams + consecutive bigrams.

The result feeds into the consolidator as context.
The planner decides what tools to call — the router just surfaces the right policy guidance.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_SECTION_RE = re.compile(r"^\[([^\]]+)\]", re.MULTILINE)
_WORD_RE    = re.compile(r"\b[a-z']+\b")

# Minimum similarity score to accept a section match.
# Below this the query has too little overlap — return empty, let the planner decide.
_MATCH_THRESHOLD = 0.10

# Blend weight: α * containment + (1-α) * jaccard
# Containment = fraction of query ngrams found in section (handles short queries well).
# Jaccard = symmetric overlap (handles longer queries well).
_ALPHA = 0.65

# Common English words that carry no topical signal — stripped from query ngrams
# so they don't create spurious overlap with sections that happen to use them.
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "ought",
    "i", "me", "my", "myself", "you", "your", "yourself",
    "he", "she", "it", "we", "they", "them", "their",
    "this", "that", "these", "those",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "and", "or", "but", "if", "not", "nor", "so", "yet", "for",
    "at", "by", "from", "in", "of", "on", "to", "up", "as", "with",
    "about", "into", "through", "before", "after",
    "just", "there", "here", "any", "all", "no", "its", "out", "s",
    "please", "tell", "me", "give", "let", "help",
})


# ── Bigram similarity ─────────────────────────────────────────────────────────

def _ngrams(text: str, strip_stops: bool = False) -> frozenset[tuple[str, ...]]:
    """Word unigrams + bigrams from lowercased text.

    strip_stops=True removes stopwords before building ngrams — use for queries
    so common words don't create false overlap with every section.
    """
    words = _WORD_RE.findall(text.lower())
    if strip_stops:
        words = [w for w in words if w not in _STOPWORDS]
    uni = {(w,) for w in words}
    bi  = {(words[i], words[i + 1]) for i in range(len(words) - 1)}
    return frozenset(uni | bi)


def _score(query_ng: frozenset, section_ng: frozenset) -> float:
    """Blended containment + Jaccard score.

    containment = |q ∩ s| / |q|   — what fraction of query is covered by section
    jaccard     = |q ∩ s| / |q ∪ s| — symmetric overlap

    Short queries (1-2 words) are dominated by containment; longer queries get
    more Jaccard weight naturally via the union size.
    """
    if not query_ng or not section_ng:
        return 0.0
    inter       = len(query_ng & section_ng)
    containment = inter / len(query_ng)
    jaccard     = inter / len(query_ng | section_ng)
    return _ALPHA * containment + (1 - _ALPHA) * jaccard


# Extra phrases that signal a section — not in content but commonly used by users.
# These are appended to the section's ngram signature to boost matching.
_SECTION_ALIASES: dict[str, str] = {
    "identity":      "who are you what are you tell me about yourself introduce yourself",
    "capabilities":  "what can you do what do you do what are your capabilities help me",
    "greeting":      "hi hey hello morning evening howdy good morning good evening greet",
    "communication": "how do you respond talk style terse verbose communicate answer",
    "character":     "personality traits how do you behave think feel",
}


def _section_ngrams(name: str, content: str) -> frozenset[tuple[str, ...]]:
    """Build the ngram signature for a section.

    Section name words are repeated 4× to boost their weight.
    Aliases are appended (2×) to cover common user phrasings.
    Only the first 400 chars of content are used.
    """
    weighted_name = (name.replace("-", " ") + " ") * 4
    aliases       = (_SECTION_ALIASES.get(name, "") + " ") * 2
    return _ngrams(weighted_name + aliases + content[:400])


# ── Policy parsing ────────────────────────────────────────────────────────────

def parse_policy_sections(policy_path: Path) -> dict[str, str]:
    """Parse engine_policy.md into {section_name: content} dict."""
    if not policy_path.exists():
        return {}
    text = policy_path.read_text(encoding="utf-8")
    sections: dict[str, str] = {}
    matches = list(_SECTION_RE.finditer(text))
    for i, m in enumerate(matches):
        name = m.group(1).strip().lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[name] = text[start:end].strip()
    return sections


def match_policy_section(query: str, sections: dict[str, str]) -> tuple[str, str]:
    """Return (section_name, content) most relevant to query via Jaccard bigrams.

    Falls back to ("", "") when nothing clears the threshold — the planner
    will then route purely via its tool choices.
    """
    if not sections or not query.strip():
        return "", ""

    q_ng = _ngrams(query, strip_stops=True)
    if not q_ng:
        # All words were stopwords (e.g. "who are you") — retry without filtering
        q_ng = _ngrams(query, strip_stops=False)
    if not q_ng:
        return "", ""

    best_name    = ""
    best_content = ""
    best_score   = 0.0

    for name, content in sections.items():
        s_ng  = _section_ngrams(name, content)
        score = _score(q_ng, s_ng)
        if score > best_score:
            best_score   = score
            best_name    = name
            best_content = content

    if best_score >= _MATCH_THRESHOLD:
        logger.debug("policy match: %r → [%s] score=%.3f", query[:50], best_name, best_score)
        return best_name, best_content

    logger.debug("policy: no match for %r (best=[%s] %.3f < %.3f)",
                 query[:50], best_name, best_score, _MATCH_THRESHOLD)
    return "", ""


# ── User prefs ────────────────────────────────────────────────────────────────

def load_user_prefs(prefs_path: Path) -> str:
    """Load user_prefs.md as a plain string (one fact per line)."""
    if not prefs_path.exists():
        return ""
    text = prefs_path.read_text(encoding="utf-8").strip()
    lines = [l.strip() for l in text.splitlines()
             if l.strip() and not l.startswith("#")]
    return "\n".join(lines)


def save_user_pref(note: str, prefs_path: Path) -> None:
    """Append a new preference to user_prefs.md (with deduplication).

    Skips the write if a very similar preference already exists — substring
    match in either direction catches both exact duplicates and minor variations
    like "I prefer vim" vs "the user prefers vim".
    """
    note_clean = note.strip()
    if not note_clean:
        return

    existing = prefs_path.read_text(encoding="utf-8") if prefs_path.exists() else "# User Preferences\n"

    # Deduplicate: normalise whitespace, check both directions
    note_norm = re.sub(r"\s+", " ", note_clean.lower())
    for line in existing.splitlines():
        line_norm = re.sub(r"\s+", " ", line.strip().lower()).lstrip("- ")
        if line_norm and (note_norm in line_norm or line_norm in note_norm):
            logger.debug("save_user_pref: already known — %r", note_clean[:60])
            return

    with prefs_path.open("a", encoding="utf-8") as f:
        if not existing.endswith("\n"):
            f.write("\n")
        f.write(f"{note_clean}\n")
    logger.info("saved user pref: %s", note_clean[:80])


# ── Public API ────────────────────────────────────────────────────────────────

def load_inner_self(policy_path: Path) -> str:
    """Load inner_self.md relative to engine_policy.md (BirdClaw harness feature)."""
    candidates = [
        policy_path.parent / "memory" / "inner_self.md",
        policy_path.parent / "inner_self.md",
        Path("memory/inner_self.md"),
    ]
    for p in candidates:
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    return ""


def route(query: str, policy_path: Path, prefs_path: Path) -> dict:
    """Load engine policy. No section routing — full policy always.

    user_prefs is BirdClaw's domain (CLAUDE.md handles prefs in Claude Code).
    Soul / engine_policy.md is loaded when present — if BirdClaw is the caller
    it will set policy_path to its own soul; standalone Sisyphean uses its own
    engine_policy.md; if neither exists the model gets no soul guidance.
    """
    policy_text = policy_path.read_text(encoding="utf-8").strip() if policy_path.exists() else ""
    sections    = parse_policy_sections(policy_path)

    return {
        "policy_section_name": "policy",
        "policy_section":      policy_text,
        "user_prefs":          "",   # BirdClaw's domain — not Sisyphean's
        "all_sections":        sections,
    }


# ── Backward-compat aliases (old engine/soul imports still work) ──────────────

parse_soul_sections  = parse_policy_sections
match_soul_section   = match_policy_section


# ── Legacy: SoulDecision (kept for any callers that still import it) ──────────

class SoulDecision:
    __slots__ = ("action", "note")

    def __init__(self, action: str, note: str = "") -> None:
        self.action = action
        self.note = note


async def soul_route(message: str, client, soul_text: str = "", memory_context: str = "") -> SoulDecision:
    """Legacy shim — returns SoulDecision(action='task') unconditionally."""
    return SoulDecision(action="task")


async def handle_remember(message: str, note: str, client, knowledge_graph=None) -> str:
    """Store a user fact into the knowledge graph.

    user_prefs.md is BirdClaw's domain — CLAUDE.md handles preferences in Claude Code.
    Sisyphean writes to the graph so the fact is recalled during future requests.
    """
    if note:
        if knowledge_graph is not None:
            try:
                knowledge_graph.upsert_node(
                    name=note[:80], node_type="user",
                    summary=note, sources=["policy:remember"],
                )
                knowledge_graph.save()
            except Exception as exc:
                logger.warning("handle_remember: graph persist failed: %s", exc)

    return f"Noted: {note}" if note else "Got it."
