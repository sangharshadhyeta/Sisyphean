"""Graph retrieval — query → subgraph → compact text for context injection.

Ported and adapted from BirdClaw birdclaw/memory/retrieval.py.

Flow:
    1. Tokenise query into keyword set
    2. Search session_graph then knowledge_graph (session takes priority)
    3. BFS depth-2 from top seed nodes
    4. Render subgraph as compact bullet text, hard-capped at CHAR_CAP

Also exposes extract_and_index() — NER over tool result text → knowledge_graph.
Both functions are called inline from the executor/loop after web tool calls.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from engine.memory.graph import knowledge_graph, session_graph
from engine.translation.pruner import keyword_prune

logger = logging.getLogger(__name__)

# Hard cap on rendered output — ~300 tokens at 4 chars/token
TOKEN_CAP = 300
CHAR_CAP = TOKEN_CAP * 4

MAX_SEEDS = 3
BFS_DEPTH = 2

# Hard cap: don't extract more than this many entities per text block
_MAX_ENTITIES = 20

# Patterns for common entities found in tool results
_NER_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("file_path",   re.compile(r"(?:^|[\s'\"(])(/(?:[\w.\-]+/)*[\w.\-]+\.\w{1,6})", re.MULTILINE)),
    ("function",    re.compile(r"\bdef\s+(\w{3,})\s*\(", re.MULTILINE)),
    ("class_name",  re.compile(r"\bclass\s+(\w{3,})\s*[:(]", re.MULTILINE)),
    ("error_type",  re.compile(r"\b(\w+(?:Error|Exception|Warning))\b")),
    ("import_path", re.compile(r"(?:from|import)\s+([\w.]{4,})", re.MULTILINE)),
    ("url",         re.compile(r"https?://[\w.\-/?=&%#@:+]{10,80}")),
    ("package",     re.compile(r"\b([a-z][a-z0-9_\-]{2,})==[\d.]+")),
]


# ── Scored node ────────────────────────────────────────────────────────────────

@dataclass
class ScoredNode:
    key: str
    name: str
    node_type: str
    summary: str
    score: int
    source: str  # "session" or "knowledge"


# ── Merged search ──────────────────────────────────────────────────────────────

def _tokenise(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


def _search_merged(query: str, limit: int = 10) -> list[ScoredNode]:
    """Search both graphs; session takes priority on key collisions."""
    tokens = _tokenise(query)
    if not tokens:
        return []

    seen: dict[str, ScoredNode] = {}
    for graph, source in ((session_graph, "session"), (knowledge_graph, "knowledge")):
        for node in graph.search(query, limit=limit):
            key = node.get("key", "")
            name = node.get("name", key)
            score = len(tokens & (_tokenise(name) | _tokenise(node.get("summary", ""))))
            sn = ScoredNode(
                key=key,
                name=name,
                node_type=node.get("type", "entity"),
                summary=node.get("summary", ""),
                score=score,
                source=source,
            )
            if key not in seen or source == "session":
                seen[key] = sn

    return sorted(seen.values(), key=lambda n: n.score, reverse=True)[:limit]


# ── Render subgraph ────────────────────────────────────────────────────────────

def _render_node(node: dict, neighbors: list[dict]) -> str:
    name = node.get("name", node.get("key", "?"))
    ntype = node.get("type", "")
    summary = node.get("summary", "")
    lines = [f"[{ntype}] {name}" + (f" — {summary}" if summary else "")]
    for nb in neighbors[:3]:
        rel = nb.get("relation", "→")
        direction = nb.get("direction", "out")
        arrow = f"→ {rel} →" if direction == "out" else f"← {rel} ←"
        lines.append(f"  {arrow} {nb.get('name', '?')}")
    return "\n".join(lines)


def _render_subgraph(bfs_nodes: list[dict], char_cap: int = CHAR_CAP) -> str:
    parts: list[str] = []
    total = 0
    for node in bfs_nodes:
        neighbors = node.pop("neighbors", [])
        block = _render_node(node, neighbors)
        if total + len(block) > char_cap:
            break
        parts.append(block)
        total += len(block) + 1
    return "\n\n".join(parts)


# ── Public retrieval API ───────────────────────────────────────────────────────

def retrieve(query: str, top_n: int = MAX_SEEDS) -> str:
    """Main retrieval entry point.

    Returns a compact text block (≤ ~300 tokens) of relevant graph context.
    Empty string if nothing relevant is found.
    """
    candidates = _search_merged(query, limit=top_n * 2)
    if not candidates:
        return ""

    seeds = [c.name for c in candidates[:top_n]]
    logger.debug("retrieval seeds: %s", seeds)

    session_nodes = {n["key"]: n for n in session_graph.bfs(seeds, depth=BFS_DEPTH)}
    knowledge_nodes = {n["key"]: n for n in knowledge_graph.bfs(seeds, depth=BFS_DEPTH)}
    merged: dict[str, dict] = {**knowledge_nodes, **session_nodes}  # session wins
    bfs_result = list(merged.values())

    if not bfs_result:
        # Seeds found but BFS empty — render seeds directly
        bfs_result = [
            {"key": c.key, "name": c.name, "type": c.node_type, "summary": c.summary, "neighbors": []}
            for c in candidates[:top_n]
        ]

    rendered = _render_subgraph(bfs_result)
    if len(rendered) > 400 and query:
        rendered = keyword_prune(rendered, goal=query, max_chars=CHAR_CAP)

    logger.debug("retrieval output: %d chars", len(rendered))
    return rendered


def retrieve_top_nodes(query: str, n: int = 3) -> list[str]:
    """Return top-N node name strings — lighter than full BFS."""
    candidates = _search_merged(query, limit=n)
    return [c.name for c in candidates[:n]]


# ── NER extraction ────────────────────────────────────────────────────────────

def extract_and_index(text: str, context: str = "") -> int:
    """Extract named entities from text and upsert into knowledge_graph.

    Uses structural regex patterns for syntactic entities: file paths,
    function names, class names, import paths, error types, URLs, packages.

    Returns the number of new/updated nodes.
    """
    if not text or len(text) < 20:
        return 0

    entities: dict[str, dict] = {}

    for entity_type, pattern in _NER_PATTERNS:
        for m in pattern.finditer(text):
            name = m.group(1) if pattern.groups else m.group(0)
            name = name.strip("'\"() ")
            if len(name) < 3 or len(name) > 120:
                continue
            key = f"{entity_type}:{name}"
            if key not in entities:
                entities[key] = {
                    "name": name,
                    "type": entity_type,
                    "summary": context[:80] if context else "",
                }
            if len(entities) >= _MAX_ENTITIES:
                break
        if len(entities) >= _MAX_ENTITIES:
            break

    for node in entities.values():
        try:
            knowledge_graph.upsert_node(
                name=node["name"],
                node_type=node["type"],
                summary=node["summary"],
            )
        except Exception:
            pass

    if entities:
        logger.debug("NER: %d entities extracted from %d chars", len(entities), len(text))
    return len(entities)
