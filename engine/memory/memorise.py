"""Memory consolidation — batch-drain session logs into the knowledge graph.

The 'memorise' pass is the offline half of the memory pipeline.  During a
live session the extractor saves individual exchange facts inline (hot path).
The memorise pass runs after the fact — typically nightly via the dream command
— and does a deeper sweep:

  1. Finds every session JSONL not yet marked as memorised.
  2. For each session, groups (user, assistant) exchange pairs.
  3. Runs an LLM extraction call per exchange (same prompt as extractor.py).
  4. Runs NER over stage_done summaries to pick up file paths, functions, etc.
  5. Upserts all extracted facts/entities into the persistent knowledge_graph.
  6. Marks the session as memorised (writes to a watermark file).

Skipped events
--------------
  tool_call / tool_result / compaction / plan events are not fed to the LLM —
  they are noisy and most useful entities are already in stage_done summaries.

Watermark
---------
  ~/.sisyphean/sessions/.memorised  — JSON file: {session_id → ISO timestamp}
  Processed sessions are never re-processed (idempotent).

Limits
------
  _MAX_EXCHANGES_PER_SESSION = 20  — avoid LLM overload for giant sessions.
  _MAX_SESSIONS_PER_RUN      = 50  — cap one dream pass.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.llm.client import LlamaClient

logger = logging.getLogger(__name__)

_SESSIONS_DIR = Path.home() / ".sisyphean" / "sessions"
_WATERMARK_FILE = _SESSIONS_DIR / ".memorised"

_MAX_EXCHANGES_PER_SESSION: int = 20
_MAX_SESSIONS_PER_RUN: int = 50

_EXTRACT_PROMPT = """\
Analyze this conversation exchange and extract information worth remembering long-term.

USER: {user_msg}
ASSISTANT: {asst_msg}

Return ONLY a JSON object:
{{
  "facts": [
    {{"label": "short unique name", "content": "specific fact (1-2 sentences)", "type": "fact|concept|project|preference"}}
  ]
}}

Rules:
- Only include genuinely NEW, specific information.
- Omit greetings, filler, generic statements, tool output noise.
- Return empty lists if nothing notable.
- Return valid JSON only."""


# ── Watermark helpers ─────────────────────────────────────────────────────────

def _load_watermark() -> dict[str, str]:
    if _WATERMARK_FILE.exists():
        try:
            return json.loads(_WATERMARK_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_watermark(wm: dict[str, str]) -> None:
    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _WATERMARK_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(wm, indent=2), encoding="utf-8")
    tmp.replace(_WATERMARK_FILE)


# ── Exchange extraction ───────────────────────────────────────────────────────

def _build_exchanges(events: list[dict]) -> list[tuple[str, str]]:
    """Pair up user_message + assistant_message events into exchange tuples."""
    exchanges: list[tuple[str, str]] = []
    pending_user: str | None = None
    for evt in events:
        etype = evt.get("type", "")
        data = evt.get("data", {})
        if etype == "user_message":
            pending_user = data.get("content", "").strip()
        elif etype == "assistant_message" and pending_user:
            asst = data.get("content", "").strip()
            if asst:
                exchanges.append((pending_user, asst))
            pending_user = None
    return exchanges


def _stage_summaries(events: list[dict]) -> list[str]:
    """Collect stage_done summaries for NER pass."""
    summaries = []
    for evt in events:
        if evt.get("type") == "stage_done":
            summary = evt.get("data", {}).get("summary", "").strip()
            if summary:
                summaries.append(summary)
    return summaries


def _load_session_events(path: Path) -> list[dict]:
    events = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    return events


# ── JSON parsing ──────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        return None
    fragment = text[start:end]
    try:
        return json.loads(fragment)
    except json.JSONDecodeError:
        cleaned = re.sub(r",\s*([}\]])", r"\1", fragment)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


# ── Memoriser ─────────────────────────────────────────────────────────────────

class Memoriser:
    """Batch-processes unmemorised session logs into the knowledge graph."""

    def __init__(self) -> None:
        self._watermark: dict[str, str] = _load_watermark()

    async def run(self, client: "LlamaClient") -> "MemoriseResult":
        """Process all pending sessions. Returns a summary result."""
        from engine.memory.graph import knowledge_graph
        try:
            from engine.memory.retrieval import extract_and_index as _ner
        except ImportError:
            _ner = None

        if not _SESSIONS_DIR.exists():
            return MemoriseResult(0, 0, 0)

        # Collect unprocessed .jsonl files (not archives, not the watermark)
        pending: list[Path] = []
        for p in sorted(_SESSIONS_DIR.glob("*.jsonl")):
            session_id = p.stem
            if session_id not in self._watermark:
                pending.append(p)

        pending = pending[:_MAX_SESSIONS_PER_RUN]
        if not pending:
            logger.info("memorise: no new sessions to process")
            return MemoriseResult(0, 0, 0)

        sessions_done = 0
        total_facts = 0
        total_ner = 0

        for session_path in pending:
            session_id = session_path.stem
            facts_this, ner_this = await self._process_session(
                session_path, session_id, client, knowledge_graph, _ner
            )
            total_facts += facts_this
            total_ner += ner_this
            sessions_done += 1
            # Mark as processed immediately so a crash doesn't re-process
            from datetime import datetime, timezone
            self._watermark[session_id] = datetime.now(timezone.utc).isoformat()
            _save_watermark(self._watermark)

        # Persist graph once at the end
        try:
            knowledge_graph.save()
        except Exception as exc:
            logger.warning("memorise: graph save failed: %s", exc)

        logger.info(
            "memorise: processed %d sessions → %d facts, %d NER entities",
            sessions_done, total_facts, total_ner,
        )
        return MemoriseResult(sessions_done, total_facts, total_ner)

    async def _process_session(
        self,
        path: Path,
        session_id: str,
        client: "LlamaClient",
        kg,
        ner_fn,
    ) -> tuple[int, int]:
        events = _load_session_events(path)
        if not events:
            return 0, 0

        exchanges = _build_exchanges(events)[:_MAX_EXCHANGES_PER_SESSION]
        summaries = _stage_summaries(events)

        facts_saved = 0
        ner_saved = 0

        # LLM extraction per exchange
        for user_msg, asst_msg in exchanges:
            try:
                new_facts = await self._extract_exchange(user_msg, asst_msg, client, kg)
                facts_saved += new_facts
            except Exception as exc:
                logger.debug("memorise: exchange extraction failed (%s): %s", session_id, exc)

        # NER pass over stage summaries
        if ner_fn and summaries:
            combined = "\n".join(summaries)
            try:
                count = ner_fn(combined, context="dream")
                ner_saved = count or 0
            except Exception as exc:
                logger.debug("memorise: NER pass failed (%s): %s", session_id, exc)

        # ── Build session timeline node ───────────────────────────────────────
        # Creates a 'session' node for this conversation and wires it into the
        # chronological chain anchored at 'session_timeline'.
        # Injected into context when the user asks "what did we do recently?"
        # or "when did we last work on X?"
        try:
            _build_timeline_node(session_id, events, facts_saved + ner_saved, kg)
        except Exception as exc:
            logger.debug("memorise: timeline node failed (%s): %s", session_id, exc)

        logger.debug(
            "memorise: session %s → %d facts, %d NER",
            session_id, facts_saved, ner_saved,
        )
        return facts_saved, ner_saved

    async def _extract_exchange(
        self,
        user_msg: str,
        asst_msg: str,
        client: "LlamaClient",
        kg,
    ) -> int:
        """Run one LLM extraction call. Returns number of facts saved."""
        prompt = _EXTRACT_PROMPT.format(
            user_msg=user_msg[:600],
            asst_msg=asst_msg[:1200],
        )
        result = await client.generate(
            [{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.1,
            stream=False,
            thinking=False,
        )
        raw = result["choices"][0]["message"]["content"].strip()
        data = _parse_json(raw)
        if not data:
            return 0

        saved = 0
        for item in data.get("facts", []):
            label = (item.get("label") or "").strip()
            content = (item.get("content") or "").strip()
            ftype = item.get("type", "fact")
            if label and content:
                kg.upsert_node(label, ftype, summary=content, sources=["dream:memorise"])
                saved += 1
        return saved


# ── Session timeline builder ──────────────────────────────────────────────────

def _build_timeline_node(session_id: str, events: list[dict], facts_count: int, kg) -> None:
    """Create / upsert a 'session' graph node and wire it into the timeline.

    Structure produced
    ------------------
    session_timeline  (type=timeline)
        │ contains
        ▼
    session:{date}:{id8}  (type=session)   ◄── summary: date + first user msg + tools
        │ precedes
        ▼
    session:{date}:{id8}  (type=session)   ← next session (linked later)

    The serial 'precedes' chain is built by finding the most recent existing
    session node and pointing it at the new one.  On first run the timeline
    root is the only anchor.
    """
    from datetime import datetime, timezone

    # ── Derive metadata from events ───────────────────────────────────────────
    first_ts: str = ""
    first_user: str = ""
    tools_used: list[str] = []
    for evt in events:
        etype = evt.get("type", "")
        data  = evt.get("data", {})
        if not first_ts and evt.get("ts"):
            first_ts = evt["ts"]
        if not first_user and etype == "user_message":
            first_user = data.get("content", "").strip()
        if etype == "tool_call":
            name = data.get("name", "")
            if name and name not in tools_used:
                tools_used.append(name)

    # Date from first event timestamp (fallback: today)
    if first_ts:
        try:
            date_str = datetime.fromisoformat(first_ts).strftime("%Y-%m-%d")
        except ValueError:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    else:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    short_id  = session_id[:8]
    node_name = f"session:{date_str}:{short_id}"

    tools_str = ", ".join(tools_used[:6]) if tools_used else "none"
    summary   = (
        f"{date_str} | {first_user[:120] or '(no message)'} "
        f"| tools: {tools_str} | facts extracted: {facts_count}"
    )

    # ── Upsert this session node ──────────────────────────────────────────────
    kg.upsert_node(node_name, "session", summary=summary,
                   sources=[f"session:{session_id}"])

    # ── Ensure timeline root exists ───────────────────────────────────────────
    _TL = "session_timeline"
    if not kg.get_node(_TL):
        kg.upsert_node(
            _TL, "timeline",
            summary=(
                "Chronological history of all Sisyphean sessions. "
                "Each session node links outward to what was done in that conversation. "
                "Follow 'precedes' edges to navigate the timeline."
            ),
        )

    # ── timeline → session ────────────────────────────────────────────────────
    kg.upsert_edge(_TL, "contains", node_name)

    # ── Find the most recent session node and chain it ────────────────────────
    # All session nodes are sorted by name (date is first so lexicographic = chrono).
    session_nodes = sorted(
        [n for n in kg.all_nodes(node_type="session")
         if n.get("name", "").startswith("session:") and n.get("name") != node_name],
        key=lambda n: n.get("name", ""),
    )
    if session_nodes:
        prev_name = session_nodes[-1].get("name", "")  # lexically last = most recent
        if prev_name and prev_name < node_name:
            kg.upsert_edge(prev_name, "precedes", node_name)

    logger.debug("memorise: timeline node %r (%d tools, %d facts)", node_name, len(tools_used), facts_count)


# ── Result type ───────────────────────────────────────────────────────────────

class MemoriseResult:
    __slots__ = ("sessions", "facts", "ner_entities")

    def __init__(self, sessions: int, facts: int, ner_entities: int) -> None:
        self.sessions = sessions
        self.facts = facts
        self.ner_entities = ner_entities

    def __repr__(self) -> str:
        return (
            f"MemoriseResult(sessions={self.sessions}, "
            f"facts={self.facts}, ner={self.ner_entities})"
        )
