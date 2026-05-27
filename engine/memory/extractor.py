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
import re
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
    {{"label": "short unique name", "content": "≤15 words", "type": "<fact|concept|project|preference>"}}
  ],
  "relations": [
    {{"from": "entity A", "relation": "verb_phrase", "to": "entity B"}}
  ],
  "artifacts": [
    {{"type": "<code|file|decision|output>", "summary": "<15 words", "content": "key content or path"}}
  ]
}}

facts: entity names only — short labels, minimal content.
relations: explicit connections between entities. Use short verb phrases:
  is_part_of | created_by | used_for | depends_on | runs_on | version_of | answers
  Only state what is EXPLICITLY said. 2-4 relations max.

Rules:
- Use EXACT label from existing nodes above when the topic matches.
- New info only. Skip greetings and filler.
- Empty lists if nothing notable.
- Valid JSON only."""


# ── Helpers ───────────────────────────────────────────────────────────────────

# Valid node types the extractor may write — enforced in code as a hard allowlist.
# Any composite or unknown type the model produces is normalised to "fact".
_VALID_FACT_TYPES = frozenset({"fact", "concept", "project", "preference", "entity", "skill"})

# URL pattern — labels matching this are always saved as "url" type regardless
# of what the LLM outputs, preventing URLs from being stored as "skill" nodes.
_URL_LABEL_RE = re.compile(r'^https?://', re.IGNORECASE)


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


async def _build_existing_context(
    kg,
    conversation: str,
    embed_client: "EmbeddingClient | None" = None,
) -> str:
    """Query the graph for nodes relevant to this conversation and format them
    for injection into the extraction prompt.

    Uses semantic (embedding) search when available so the LLM sees the most
    conceptually relevant existing nodes, not just keyword matches.
    Falls back to keyword search when the embedding client is unavailable.

    The LLM sees these existing labels and is instructed to reuse them rather
    than inventing new names — this prevents semantic duplicates at the source.
    Returns an empty string when the graph is empty or the query fails.
    """
    try:
        if embed_client is not None:
            hits = await kg.search_by_embedding(
                conversation[:400], embed_client, limit=8,
            )
        else:
            hits = kg.search(conversation[:400], limit=8)
        if not hits:
            return ""
        lines = []
        for h in hits:
            name    = h.get("name") or h.get("label", "")
            summary = (h.get("summary") or h.get("content", ""))[:120]
            ntype   = h.get("type", "?")
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


def _jaccard_match_label(kg, label: str) -> str:
    """Jaccard fallback dedup — used when the embedding client is unavailable.

    Threshold 0.75 on ≥3-char tokens.  Single-token labels are left unchanged
    (one token is too lossy for Jaccard to be reliable).
    """
    label_toks = {w for w in re.findall(r"[a-z0-9]+", label.lower()) if len(w) >= 3}
    if len(label_toks) < 2:
        return label
    best_name, best_j = label, 0.0
    for node in kg.all_nodes():
        name = node.get("name") or ""
        if not name or name.lower() == label.lower():
            continue
        name_toks = {w for w in re.findall(r"[a-z0-9]+", name.lower()) if len(w) >= 3}
        if not name_toks:
            continue
        j = len(label_toks & name_toks) / len(label_toks | name_toks)
        if j > best_j:
            best_j, best_name = j, name
    return best_name if best_j >= 0.75 else label


async def _semantic_match_label(
    kg,
    label: str,
    embed_client: "EmbeddingClient | None",
) -> str:
    """Return the name of an existing node that semantically matches *label*.

    Strategy
    --------
    1. Embed the new label via the embedding client.
    2. Compare against cached embeddings of all existing nodes.
       Each node is represented as "<name>: <summary>" (first 200 chars).
    3. If cosine similarity ≥ 0.82, return the matching node's name so the
       caller merges into it rather than creating a duplicate.
    4. Fall back to Jaccard (threshold 0.75) if embeddings are unavailable.

    Threshold 0.82 is intentionally tight — semantic embedding space is rich
    enough that 0.82 means very close paraphrases, not just topic overlap.
    Example hits:
      'prefers dark mode'  ↔  'dark mode preference'          (sim ≈ 0.91)
      'Python dev'         ↔  'Python developer'              (sim ≈ 0.95)
    Example misses (correctly rejected):
      'vim editor'         ↔  'text editor productivity'      (sim ≈ 0.74)
    """
    if embed_client is None:
        return _jaccard_match_label(kg, label)

    from engine.llm.embeddings import cosine_similarity

    label_vec = await embed_client.embed(label)
    if label_vec is None:
        # Embedding service down — fall back silently
        return _jaccard_match_label(kg, label)

    best_name, best_sim = label, 0.0
    for node in kg.all_nodes():
        name = node.get("name") or ""
        if not name or name.lower() == label.lower():
            continue
        summary   = (node.get("summary") or "")[:200]
        node_text = f"{name}: {summary}" if summary else name
        node_key  = node.get("key", name.lower())
        node_vec  = await embed_client.embed(node_text, cache_key=node_key)
        if node_vec is None:
            continue
        sim = cosine_similarity(label_vec, node_vec)
        if sim > best_sim:
            best_sim, best_name = sim, name

    if best_sim >= 0.82:
        logger.debug(
            "extractor: semantic-merged %r → %r (sim=%.3f)",
            label[:40], best_name[:40], best_sim,
        )
        return best_name
    return label


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
        embed_client: "EmbeddingClient | None" = None,
    ) -> None:
        self.graph        = graph
        self.store        = store
        self.client       = client
        self.embed_client = embed_client   # None → Jaccard fallback throughout

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
        existing_block = await _build_existing_context(
            kg, f"{user_message} {assistant_response}",
            embed_client=self.embed_client,
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
                # Skip URL labels entirely — the extractor has no useful context
                # to attach to them (summary would just be "conversation").
                # Researched URLs are saved with real content by _save_research_to_graph.
                if _URL_LABEL_RE.match(label):
                    logger.debug("extractor: skipping URL label %r", label[:60])
                    continue
                if not label or not content:
                    continue

                # Reject labels that are verbatim prompt-template placeholders.
                if _is_junk_label(label):
                    logger.debug("extractor: dropped junk label %r", label[:40])
                    continue

                # Semantic dedup: find an existing node that means the same thing.
                # Uses embedding cosine similarity (≥0.82) when the client is
                # available; falls back to Jaccard token overlap (≥0.75) otherwise.
                deduped = await _semantic_match_label(kg, label, self.embed_client)
                if deduped != label:
                    label = deduped

                # Merge content into existing node rather than overwriting.
                # When the LLM reuses an existing label (the dedup happy path),
                # the new content should enrich, not replace, the stored summary.
                # Append only if the new content isn't already captured.
                existing = kg.get_node(label)
                if existing:
                    existing_summary = existing.get("summary", "")
                    if content.lower()[:60] not in existing_summary.lower():
                        content = f"{existing_summary} | {content}"[:500]
                kg.upsert_node(label, ftype, summary=content)
                saved_facts += 1
                saved.append((label, ftype))
                logger.debug("extractor: saved [%s] %r", ftype, label[:40])

            # ── Create graph edges ────────────────────────────────────────────
            # (1) Type-specific edges: anchor node to the right hub based on type.
            for final_label, ftype in saved:
                try:
                    if ftype == "preference":
                        kg.upsert_edge("user", "has_preference", final_label, weight=1.0)
                    elif ftype == "project":
                        kg.upsert_edge("user", "works_on", final_label, weight=1.0)
                    elif ftype == "concept":
                        kg.upsert_edge("user", "knows_about", final_label, weight=0.6)
                    elif ftype == "skill":
                        kg.upsert_edge("sisyphean", "has_skill", final_label, weight=0.8)
                except Exception as _edge_exc:
                    logger.debug("extractor: type-edge failed: %s", _edge_exc)

            # (2) removed — regex noun extraction created low-confidence stub
            #     entity nodes that polluted retrieval.  Explicit model-stated
            #     relations (block 4 below) are the only non-anchor edges written.

            # (3) removed — co-occurrence edges ("these two facts appeared in the
            #     same conversation") are semantically meaningless and the primary
            #     source of graph noise.  All edges now come from anchor hub rules
            #     (block 1) or explicit model-stated relations (block 4).

            # ── (4) Explicit relations from the model ─────────────────────────
            # The model now outputs a "relations" array of {from, relation, to}
            # triples.  These are first-class edges — more reliable than the
            # heuristic noun extraction above because the model stated them.
            saved_rels = 0
            for rel in data.get("relations", []):
                r_from     = (rel.get("from") or "").strip()
                r_relation = (rel.get("relation") or "").strip().lower()
                r_to       = (rel.get("to") or "").strip()
                if not r_from or not r_relation or not r_to:
                    continue
                if _is_junk_label(r_from) or _is_junk_label(r_to):
                    continue
                # Only wire an edge when both nodes already exist.
                # Stub creation here caused the graph to fill up with one-word
                # "entity" nodes that were never recalled (no summary, no confidence).
                if not kg.get_node(r_from) or not kg.get_node(r_to):
                    logger.debug(
                        "extractor: skipping relation %r -[%s]-> %r (endpoint missing)",
                        r_from[:30], r_relation, r_to[:30],
                    )
                    continue
                # Normalise relation to snake_case
                r_relation = re.sub(r'\s+', '_', r_relation)[:40]
                kg.upsert_edge(r_from, r_relation, r_to, weight=0.8)
                saved_rels += 1
                logger.debug("extractor: relation %r -[%s]-> %r", r_from[:30], r_relation, r_to[:30])
            if saved_rels:
                logger.debug("extractor: %d explicit relations saved", saved_rels)

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
