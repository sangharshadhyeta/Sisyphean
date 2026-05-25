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
- Context-aware extraction: existing graph nodes related to the current
  conversation are injected into the prompt BEFORE the LLM call.  The
  model can then reuse an existing label instead of inventing a new one,
  preventing semantic duplicates at the source rather than post-hoc.
- Exact-name dedup is handled by upsert_node's key lookup.
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


# ── Prompt ────────────────────────────────────────────────────────────────────

# {existing_block} is injected at call time with existing graph nodes relevant
# to this conversation.  When the block is non-empty, the model sees what labels
# already exist and is instructed to reuse them — this is the primary defence
# against duplicate nodes, replacing any post-hoc fuzzy matching.
_PROMPT = """\
Analyze this conversation turn and extract new information worth remembering.

USER: {user_msg}
ASSISTANT: {asst_msg}
{existing_block}
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
- If an EXISTING NODE above already covers this information, use that EXACT label.
  Never create a new node for the same concept under a different name.
- Only include genuinely NEW information not already captured by existing nodes.
- Omit greetings, filler, or generic statements.
- Return empty lists if nothing notable.
- Return valid JSON only, no explanation."""


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _build_existing_context(kg, conversation: str) -> str:
    """Query the graph for nodes relevant to this conversation and format them
    for injection into the extraction prompt.

    The LLM sees these existing labels and is instructed to reuse them rather
    than inventing new names — this prevents semantic duplicates at the source.
    Returns an empty string when the graph is empty or the query fails.
    """
    try:
        hits = kg.search(conversation[:400], top_k=8)
        if not hits:
            return ""
        lines = []
        for h in hits:
            name = h.get("name") or h.get("label", "")
            summary = (h.get("summary") or h.get("content", ""))[:120]
            ntype = h.get("type", "?")
            if name and summary:
                lines.append(f'  "{name}" [{ntype}]: {summary}')
        if not lines:
            return ""
        return (
            "\nExisting memory nodes — if the conversation covers the same topic as any "
            "of these, use that EXACT label (do not create a new node):\n"
            + "\n".join(lines)
            + "\n"
        )
    except Exception:
        return ""


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


# ── Extractor class ───────────────────────────────────────────────────────────

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

        # ── Resolve graph and build context block BEFORE the LLM call ────────
        # Injecting existing relevant nodes lets the model reuse existing labels
        # rather than inventing new ones — the primary dedup mechanism.
        kg = _knowledge_graph()
        existing_block = _build_existing_context(
            kg, f"{user_message} {assistant_response}"
        )

        prompt = _PROMPT.format(
            user_msg=user_message[:600],
            asst_msg=assistant_response[:1200],
            existing_block=existing_block,
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
            # Tracks (final_label, ftype) for every fact actually written —
            # used below to create graph edges between related facts.
            saved: list[tuple[str, str]] = []

            for item in data.get("facts", []):
                label = (item.get("label") or "").strip()
                content = (item.get("content") or "").strip()
                ftype = _normalise_type(item.get("type", "fact"))
                if not label or not content:
                    continue

                # Reject labels that are verbatim prompt-template placeholders.
                if _is_junk_label(label):
                    logger.debug("extractor: dropped junk label %r", label[:40])
                    continue

                # upsert_node merges into the existing node when the label matches
                # exactly — that handles the happy path where the model correctly
                # reused an existing label from the injected context block.
                kg.upsert_node(label, ftype, summary=content)
                saved_facts += 1
                saved.append((label, ftype))
                logger.debug("extractor: saved [%s] %r", ftype, label[:40])

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
                except Exception as _edge_exc:
                    logger.debug("extractor: type-edge failed: %s", _edge_exc)

            # (2) Co-occurrence edges: facts extracted from the same exchange
            #     are semantically related — wire them together (window of 4).
            for i in range(len(saved)):
                for j in range(i + 1, min(i + 4, len(saved))):
                    a_lbl, a_type = saved[i]
                    b_lbl, b_type = saved[j]
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
