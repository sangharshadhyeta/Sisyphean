"""Memory extractor — post-response fact and artifact harvesting.

After every conversation turn, this runs a small focused LLM call to
pull out anything worth remembering:
  - Discrete facts → KnowledgeGraph nodes (type: fact / concept / project)
  - Code, files, decisions, outputs → ArtifactStore entries

The extraction prompt is intentionally tight (< 512 tokens response)
so it does not eat into the main context budget.  It also runs
asynchronously after the response is returned, so it never blocks
the user.

Design notes
------------
- Updates existing graph nodes by label (avoids duplicate facts).
- Falls back silently on any parse error — extraction is best-effort.
- Uses temperature=0.1 to keep JSON output stable.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.llm.client import LlamaClient
    from .graph import KnowledgeGraph
    from .store import ArtifactStore

logger = logging.getLogger(__name__)

# Import lazily to avoid circular imports at module load time.
# We use the shared knowledge_graph GraphStore (name-keyed, retrieval-compatible)
# instead of the old KnowledgeGraph (UUID-keyed) for extracted facts.
def _knowledge_graph():
    from engine.memory.graph import knowledge_graph
    return knowledge_graph

def _extract_and_index():
    from engine.memory.retrieval import extract_and_index
    return extract_and_index

_PROMPT = """\
Analyze this conversation turn and extract new information worth remembering.

USER: {user_msg}
ASSISTANT: {asst_msg}

Return ONLY a JSON object:
{{
  "facts": [
    {{"label": "short unique name", "content": "specific fact (1-2 sentences)", "type": "<pick ONE: fact | concept | project | preference>"}}
  ],
  "artifacts": [
    {{"type": "<pick ONE: code | file | decision | output>", "summary": "what it is in <15 words", "content": "key content or path"}}
  ]
}}

Type guide for facts — pick the single best fit:
  fact        — a specific thing that is true (name, version, date, result)
  concept     — a reusable idea, pattern, or explanation
  project     — something being built or worked on
  preference  — how the user wants things done

Rules:
- Only include genuinely NEW, specific information not already obvious.
- Omit greetings, filler, or generic statements.
- Return empty lists if nothing notable.
- Return valid JSON only, no explanation."""

# Valid node types the extractor may write — enforced in code as a hard allowlist.
# Any composite or unknown type the model produces is normalised to "fact".
_VALID_FACT_TYPES = frozenset({"fact", "concept", "project", "preference", "entity", "skill"})


def _normalise_type(raw: str) -> str:
    """Map any LLM-output type string to a single valid type.

    Handles model mistakes like "fact|concept", "fact|concept|project|preference",
    or leading/trailing whitespace.  Falls back to "fact" if nothing matches.
    """
    raw = (raw or "").strip().lower()
    if raw in _VALID_FACT_TYPES:
        return raw
    # Model wrote a pipe- or comma-separated composite — take first valid token
    for part in raw.replace(",", "|").split("|"):
        part = part.strip()
        if part in _VALID_FACT_TYPES:
            return part
    return "fact"


# Labels that the model sometimes emits verbatim from the prompt template —
# these are placeholder examples, not real facts, and must be rejected.
_JUNK_LABELS: frozenset[str] = frozenset({
    "short unique name", "specific fact", "short name", "fact label",
    "label", "name", "entity", "concept name", "fact", "item",
    "information", "detail", "data", "knowledge", "thing",
    "pick one", "type", "content", "summary", "value",
    "a", "an", "the", "is", "it", "this", "that",
})


def _is_junk_label(label: str) -> bool:
    """Return True when *label* is clearly a prompt-template placeholder or trivial."""
    s = label.lower().strip().strip("\"'")
    return (
        s in _JUNK_LABELS
        or len(s) < 3
        or s.startswith("<")   # <label>
        or s.startswith("[")   # [label]
        or s.startswith("{")   # {label}
    )


def _jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _find_similar_node(graph, label: str, content: str, threshold: float = 0.75) -> str | None:
    """Return the name of an existing graph node similar to *label*, or None.

    Similarity is computed as token-level Jaccard on the node name.  When label
    similarity is moderate (≥ 0.5) the content is also compared — two facts
    about the same topic but stated differently still count as duplicates if
    their content overlaps substantially.
    """
    try:
        candidates = graph.search(label, top_k=5)
        for node in candidates:
            existing_name = node.get("name", "")
            if not existing_name:
                continue
            # Exact match — let upsert_node handle it
            if existing_name.lower().strip() == label.lower().strip():
                return existing_name
            label_sim = _jaccard(label, existing_name)
            if label_sim >= threshold:
                return existing_name
            # Moderate label overlap: check content too
            if label_sim >= 0.5:
                content_sim = _jaccard(content, node.get("summary", ""))
                if content_sim >= 0.6:
                    return existing_name
    except Exception:
        pass
    return None


class MemoryExtractor:

    def __init__(
        self,
        graph: KnowledgeGraph,
        store: ArtifactStore,
        client: LlamaClient,
    ) -> None:
        self.graph = graph
        self.store = store
        self.client = client

    def extract_later(self, user_message: str, assistant_response: str) -> None:
        """Fire-and-forget: schedule extraction without blocking the response."""
        asyncio.create_task(self._extract(user_message, assistant_response))

    async def extract(self, user_message: str, assistant_response: str) -> None:
        """Await extraction directly (use for non-streaming responses)."""
        await self._extract(user_message, assistant_response)

    # ── Internal ─────────────────────────────────────────────────────────────

    async def _extract(self, user_message: str, assistant_response: str) -> None:
        # Skip short/social exchanges — the small model ignores "omit greetings"
        _SOCIAL = {
            "hi", "hello", "hey", "thanks", "thank you", "cheers", "bye",
            "goodbye", "ok", "okay", "yes", "no", "yep", "nope", "cool",
            "great", "got it", "noted", "sure", "alright", "sounds good",
            "thanks, that helps", "that helps", "thx",
        }
        _stripped = user_message.strip().lower().rstrip("!.,?")
        if len(_stripped) < 12 or _stripped in _SOCIAL:
            return

        # Skip if the exchange is too short to contain memorable facts
        combined_words = len(user_message.split()) + len(assistant_response.split())
        if combined_words < 30:
            return

        # Skip pure-code responses — no conversational facts to extract.
        # Run NER only (already handles file paths, function names, imports).
        _asst_stripped = assistant_response.lstrip()
        _first_line = _asst_stripped.split("\n")[0].strip()
        _is_code_block = _asst_stripped.startswith("```") or _asst_stripped.startswith("~~~")
        _is_code_def   = bool(_first_line.startswith(("def ", "class ", "import ", "from ")))
        if _is_code_block or _is_code_def:
            combined = f"{user_message}\n{assistant_response}"
            try:
                ner_count = _extract_and_index()(combined, context="conversation")
                if ner_count:
                    logger.debug("NER indexed %d entities (code-only skip)", ner_count)
            except Exception as ner_exc:
                logger.debug("NER extraction skipped: %s", ner_exc)
            return

        prompt = _PROMPT.format(
            user_msg=user_message[:600],
            asst_msg=assistant_response[:1200],
        )
        try:
            result = await self.client.generate(
                [{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.1,
                response_format={"type": "json_object"},  # "{" prefix on llama.cpp
                stream=False,
                thinking=False,
            )
            raw = result["choices"][0]["message"]["content"].strip()
            data = _parse_json(raw)
            if not data:
                return

            saved_facts, saved_arts = 0, 0
            kg = _knowledge_graph()
            # Tracks (final_label, ftype) for every fact actually written —
            # used below to create graph edges between related facts.
            saved: list[tuple[str, str]] = []

            for item in data.get("facts", []):
                label = (item.get("label") or "").strip()
                content = (item.get("content") or "").strip()
                ftype = _normalise_type(item.get("type", "fact"))
                if not label or not content:
                    continue

                # ── Quality guard ─────────────────────────────────────────────
                # Reject labels that are verbatim prompt-template placeholders
                # (the model sometimes copies "short unique name" etc. literally).
                if _is_junk_label(label):
                    logger.debug("extractor: dropped junk label %r", label[:40])
                    continue

                # ── Fuzzy dedup ───────────────────────────────────────────────
                # Exact-name match is already handled by upsert_node's key lookup.
                # Also check for near-duplicate labels (Jaccard ≥ 0.75) so the
                # same fact stated slightly differently enriches the existing node
                # instead of creating a separate one.
                existing_name = _find_similar_node(kg, label, content)
                if existing_name and existing_name.lower().strip() != label.lower().strip():
                    # Enrich: append new content only if not already captured
                    existing_node = kg.get_node(existing_name) or {}
                    existing_summary = existing_node.get("summary", "")
                    if content.lower()[:50] not in existing_summary.lower():
                        merged = f"{existing_summary} | {content}"[:500]
                    else:
                        merged = existing_summary
                    kg.upsert_node(existing_name, ftype, summary=merged)
                    final_label = existing_name
                    logger.debug(
                        "extractor: merged %r → existing node %r",
                        label[:40], existing_name[:40],
                    )
                else:
                    # Exact match or new node — upsert handles both
                    kg.upsert_node(label, ftype, summary=content)
                    final_label = label
                saved_facts += 1
                saved.append((final_label, ftype))

            # ── Create graph edges ────────────────────────────────────────────
            # (1) Type-specific edges: anchor preferences/projects/concepts to
            #     the "user" node so the graph has a meaningful hub structure.
            for final_label, ftype in saved:
                try:
                    if ftype == "preference":
                        kg.upsert_edge("user", "has_preference", final_label, weight=1.0)
                    elif ftype == "project":
                        kg.upsert_edge("user", "works_on", final_label, weight=1.0)
                    elif ftype == "concept":
                        kg.upsert_edge("user", "knows_about", final_label, weight=0.6)
                    elif ftype == "fact":
                        # Facts about a visible project get linked there too
                        pass  # handled by co-occurrence below
                except Exception as _edge_exc:
                    logger.debug("extractor: type-edge failed: %s", _edge_exc)

            # (2) Co-occurrence edges: facts extracted from the same exchange
            #     are semantically related — wire them together (window of 4).
            for i in range(len(saved)):
                for j in range(i + 1, min(i + 4, len(saved))):
                    a_lbl, a_type = saved[i]
                    b_lbl, b_type = saved[j]
                    # Only link if sharing a meaningful type or one is a hub type
                    if (a_type == b_type
                            or a_type in ("concept", "project")
                            or b_type in ("concept", "project")):
                        try:
                            kg.upsert_edge(a_lbl, "related_to", b_lbl, weight=0.5)
                        except Exception:
                            pass

            for item in data.get("artifacts", []):
                atype = item.get("type", "output")
                summary = (item.get("summary") or "").strip()
                content = (item.get("content") or "").strip()
                if summary and content:
                    self.store.save(atype, content, summary=summary)
                    saved_arts += 1

            # NER pass: index file paths, functions, imports, URLs, etc. directly
            # from the raw conversation text so they're searchable via retrieval.py.
            combined = f"{user_message}\n{assistant_response}"
            try:
                ner_count = _extract_and_index()(combined, context="conversation")
                if ner_count:
                    logger.debug("NER indexed %d entities", ner_count)
            except Exception as ner_exc:
                logger.debug("NER extraction skipped: %s", ner_exc)

            if saved_facts or saved_arts:
                logger.debug("Extracted %d facts, %d artifacts", saved_facts, saved_arts)

            # Persist new knowledge_graph nodes to disk
            if saved_facts:
                try:
                    kg.save()
                except Exception as save_exc:
                    logger.debug("knowledge_graph save skipped: %s", save_exc)

        except Exception as exc:
            logger.warning("Memory extraction failed: %s", exc)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict | None:
    """Tolerant JSON parser — finds the first {...} block in model output."""
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None
