"""Memory consolidation — batch-drain session logs into the knowledge graph.

Design (v2)
-----------
  ONE LLM call per session produces a prose summary stored as full text on
  the session node.  No per-exchange extraction, no fact nodes from conversation.

  What goes into the KG
  ---------------------
  - session node    — full prose summary as the .summary field (searchable)
  - session edges   — precedes, contains (timeline chain)
  - NER entities    — file paths, functions, imports, URLs from stage_done logs
                      (regex, zero LLM calls)

  What does NOT go into the KG
  ----------------------------
  - Arithmetic results, greetings, generic Q&A — these live in the session
    summary text, not as separate nodes.

Watermark
---------
  ~/.sisyphean/sessions/.memorised  — JSON: {session_id → ISO timestamp}
  Processed sessions are never re-processed (idempotent).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.llm.client import LlamaClient

logger = logging.getLogger(__name__)

_SESSIONS_DIR = Path.home() / ".sisyphean" / "sessions"
_WATERMARK_FILE = _SESSIONS_DIR / ".memorised"

_MAX_EXCHANGES_PER_SESSION: int = 12
_MAX_SESSIONS_PER_RUN: int = 50

# Simple summary prompt — small models handle "summarize in 2-3 sentences" reliably.
_SUMMARY_PROMPT = """\
Summarize this conversation in 2-3 sentences.
Say what the user asked, what was done, and what the outcome was.

{exchanges_text}

Write only the summary. Be specific."""


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


# ── Event parsing ─────────────────────────────────────────────────────────────

def _build_exchanges(events: list[dict]) -> list[tuple[str, str]]:
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
    summaries = []
    for evt in events:
        if evt.get("type") == "stage_done":
            s = evt.get("data", {}).get("summary", "").strip()
            if s:
                summaries.append(s)
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


# ── Session summary (one LLM call per session) ────────────────────────────────

async def _summarise_session(
    exchanges: list[tuple[str, str]],
    client: "LlamaClient",
) -> str:
    """Produce a 2-3 sentence prose summary of the session.

    Caps input to keep the prompt short enough for small models.
    Falls back to empty string on any failure — the caller uses the first
    user message as a fallback label.
    """
    if not exchanges:
        return ""

    lines: list[str] = []
    total = 0
    for user, asst in exchanges[:_MAX_EXCHANGES_PER_SESSION]:
        u_clip = user[:200]
        a_clip = asst[:300]
        lines.append(f"User: {u_clip}")
        lines.append(f"Assistant: {a_clip}")
        total += len(u_clip) + len(a_clip)
        if total > 1200:
            break

    text = "\n".join(lines)
    try:
        r = await client.generate(
            [{"role": "user", "content": _SUMMARY_PROMPT.format(exchanges_text=text)}],
            max_tokens=150,
            temperature=0.1,
            stream=False,
            thinking=False,
        )
        summary = (r["choices"][0]["message"]["content"] or "").strip()
        # Sanity: reject if model just echoed the prompt or returned garbage
        if len(summary.split()) < 5 or "exchanges_text" in summary:
            return ""
        return summary
    except Exception as exc:
        logger.debug("memorise: summary call failed: %s", exc)
        return ""


# ── Memoriser ─────────────────────────────────────────────────────────────────

class Memoriser:
    """Batch-processes unmemorised session logs into the knowledge graph."""

    def __init__(self) -> None:
        self._watermark: dict[str, str] = _load_watermark()

    async def run(self, client: "LlamaClient") -> "MemoriseResult":
        from engine.memory.graph import knowledge_graph
        try:
            from engine.memory.retrieval import extract_and_index as _ner
        except ImportError:
            _ner = None

        if not _SESSIONS_DIR.exists():
            return MemoriseResult(0, 0, 0)

        pending: list[Path] = [
            p for p in sorted(_SESSIONS_DIR.glob("*.jsonl"))
            if p.stem not in self._watermark
        ]
        pending = pending[:_MAX_SESSIONS_PER_RUN]

        if not pending:
            logger.info("memorise: no new sessions to process")
            return MemoriseResult(0, 0, 0)

        sessions_done = 0
        total_ner = 0

        for session_path in pending:
            session_id = session_path.stem
            ner_this = await self._process_session(
                session_path, session_id, client, knowledge_graph, _ner
            )
            total_ner += ner_this
            sessions_done += 1
            self._watermark[session_id] = datetime.now(timezone.utc).isoformat()
            _save_watermark(self._watermark)

        try:
            knowledge_graph.save()
        except Exception as exc:
            logger.warning("memorise: graph save failed: %s", exc)

        logger.info(
            "memorise: processed %d sessions → %d NER entities",
            sessions_done, total_ner,
        )
        return MemoriseResult(sessions_done, 0, total_ner)

    async def _process_session(
        self,
        path: Path,
        session_id: str,
        client: "LlamaClient",
        kg,
        ner_fn,
    ) -> int:
        events = _load_session_events(path)
        if not events:
            return 0

        exchanges = _build_exchanges(events)
        stage_texts = _stage_summaries(events)

        # ── ONE LLM call: prose summary of the whole session ──────────────────
        llm_summary = ""
        summary_confidence: float | None = None
        if exchanges:
            try:
                llm_summary = await _summarise_session(exchanges, client)
                if llm_summary:
                    # Score how well the summary is grounded in the actual exchanges
                    from engine.memory.graph import faithfulness as _faith
                    src_text = " ".join(f"{u} {a}" for u, a in exchanges[:8])
                    summary_confidence = _faith(src_text, llm_summary)
                    logger.debug("memorise: summary faithfulness=%.2f (%s)",
                                 summary_confidence, session_id)
            except Exception as exc:
                logger.debug("memorise: summary failed (%s): %s", session_id, exc)

        # ── NER pass over stage_done logs (regex, zero LLM calls) ────────────
        ner_count = 0
        if ner_fn and stage_texts:
            combined = "\n".join(stage_texts)
            try:
                ner_count = ner_fn(combined, context=session_id) or 0
            except Exception as exc:
                logger.debug("memorise: NER pass failed (%s): %s", session_id, exc)

        # ── Session timeline node (summary = LLM prose or fallback) ──────────
        try:
            _build_timeline_node(session_id, events, llm_summary, kg,
                                 summary_confidence)
        except Exception as exc:
            logger.debug("memorise: timeline node failed (%s): %s", session_id, exc)

        logger.debug(
            "memorise: session %s → summary=%d chars, %d NER",
            session_id, len(llm_summary), ner_count,
        )
        return ner_count


# ── Session timeline builder ──────────────────────────────────────────────────

def _build_timeline_node(
    session_id: str,
    events: list[dict],
    llm_summary: str,
    kg,
    confidence: float | None = None,
) -> None:
    """Create / upsert a session node with a prose summary and wire it into the timeline.

    The session node's .summary field is the primary text store for what happened
    in that conversation.  It is searched by keyword during retrieval.
    """
    first_ts: str = ""
    first_user: str = ""
    tools_used: list[str] = []

    for evt in events:
        etype = evt.get("type", "")
        data = evt.get("data", {})
        if not first_ts and evt.get("ts"):
            first_ts = evt["ts"]
        if not first_user and etype == "user_message":
            first_user = data.get("content", "").strip()
        if etype == "tool_call":
            name = data.get("name", "")
            if name and name not in tools_used:
                tools_used.append(name)

    if first_ts:
        try:
            date_str = datetime.fromisoformat(first_ts).strftime("%Y-%m-%d")
        except ValueError:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    else:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    short_id = session_id[:8]
    node_name = f"session:{date_str}:{short_id}"
    tools_str = ", ".join(tools_used[:6]) if tools_used else "none"

    # Summary: LLM prose first, fall back to first user message as label
    if llm_summary:
        summary = f"{date_str} | {llm_summary} | tools: {tools_str}"
    else:
        label = first_user[:120] or "(no message)"
        summary = f"{date_str} | {label} | tools: {tools_str}"

    kg.upsert_node(node_name, "session", summary=summary,
                   sources=[f"session:{session_id}"],
                   confidence=confidence)

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

    kg.upsert_edge(_TL, "contains", node_name)

    # Chain sessions chronologically via 'precedes' edges
    session_nodes = sorted(
        [n for n in kg.all_nodes(node_type="session")
         if n.get("name", "").startswith("session:") and n.get("name") != node_name],
        key=lambda n: n.get("name", ""),
    )
    if session_nodes:
        prev_name = session_nodes[-1].get("name", "")
        if prev_name and prev_name < node_name:
            kg.upsert_edge(prev_name, "precedes", node_name)

    # ── Link session → produced → facts + user → has_session → session ──────
    _link_session_findings(session_id, node_name, events, kg)

    logger.debug("memorise: session node %r (tools: %s)", node_name, tools_str)


def _link_session_findings(
    session_id: str,
    session_node_name: str,
    events: list[dict],
    kg,
) -> None:
    """Wire all facts extracted during a session back to the session node.

    Finds every fact/concept/entity/research/url node whose created_at
    falls within the session's event time window, adds:
        session_node → produced → fact_node

    Then links the user anchor upward:
        user_anchor → has_session → session_node

    This turns the dangling isolated fact leaves into a proper hierarchy:
        user → has_session → session → produced → [all search results]

    No LLM calls — pure timestamp matching and graph traversal.
    """
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td

    # ── Determine session time window from logged event timestamps ────────────
    _SKIP_TYPES = frozenset({
        "soul", "session", "anchor", "user", "skill",
        "timeline", "system", "policy", "self",
    })
    _LINK_TYPES = frozenset({
        "fact", "concept", "entity", "research", "url", "preference",
    })

    ts_vals = [evt.get("ts", "") for evt in events if evt.get("ts")]
    if not ts_vals:
        return

    try:
        parsed = []
        for raw in ts_vals:
            t = _dt.fromisoformat(str(raw))
            if t.tzinfo is None:
                t = t.replace(tzinfo=_tz.utc)
            parsed.append(t)
        t_start = min(parsed) - _td(seconds=60)   # 60-s buffer before first event
        t_end   = max(parsed) + _td(minutes=5)    # 5-min buffer after last event
    except Exception:
        return

    # ── Link session → produced → every fact in that time window ─────────────
    produced = 0
    try:
        all_nodes = kg.all_nodes()
    except Exception:
        return

    for node in all_nodes:
        ntype = node.get("type", "")
        if ntype not in _LINK_TYPES:
            continue
        node_name = node.get("name", "")
        if not node_name or node_name == session_node_name:
            continue

        ts_raw = node.get("created_at") or node.get("last_seen") or ""
        if not ts_raw:
            continue
        try:
            node_ts = _dt.fromisoformat(str(ts_raw))
            if node_ts.tzinfo is None:
                node_ts = node_ts.replace(tzinfo=_tz.utc)
        except Exception:
            continue

        if not (t_start <= node_ts <= t_end):
            continue

        try:
            kg.upsert_edge(session_node_name, "produced", node_name, weight=1.0)
            produced += 1
        except Exception:
            pass

    if produced:
        logger.debug(
            "memorise: linked %d fact(s) → session %r", produced, session_node_name
        )

    # ── Link user anchor → has_session → session ──────────────────────────────
    # Look for an anchor/user node to root the hierarchy.
    # Priority: anchor type → user type → any node whose name is "user" or "sisyphean".
    user_node_name = ""
    try:
        for ntype in ("anchor", "user"):
            candidates = [
                n.get("name", "") for n in kg.all_nodes(node_type=ntype)
                if n.get("name")
            ]
            if candidates:
                # Prefer node literally named "user" or "sisyphean"; fall back to first
                for preferred in ("user", "sisyphean", "self"):
                    if preferred in candidates:
                        user_node_name = preferred
                        break
                if not user_node_name:
                    user_node_name = candidates[0]
                break
    except Exception:
        pass

    if user_node_name:
        try:
            kg.upsert_edge(user_node_name, "has_session", session_node_name, weight=1.0)
        except Exception:
            pass
    else:
        logger.debug("memorise: no user/anchor node found — skipping has_session edge")


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
