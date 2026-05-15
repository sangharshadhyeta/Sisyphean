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
    {{"label": "short unique name", "content": "specific fact (1-2 sentences)", "type": "fact|concept|project|preference"}}
  ],
  "artifacts": [
    {{"type": "code|file|decision|output", "summary": "what it is in <15 words", "content": "key content or path"}}
  ]
}}

Rules:
- Only include genuinely NEW, specific information not already obvious.
- Omit greetings, filler, or generic statements.
- Return empty lists if nothing notable.
- Return valid JSON only, no explanation."""


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
                response_format={"type": "json_object"},
                stream=False,
                thinking=False,
            )
            raw = result["choices"][0]["message"]["content"].strip()
            data = _parse_json(raw)
            if not data:
                return

            saved_facts, saved_arts = 0, 0
            kg = _knowledge_graph()

            for item in data.get("facts", []):
                label = (item.get("label") or "").strip()
                content = (item.get("content") or "").strip()
                ftype = item.get("type", "fact")
                if not label or not content:
                    continue
                # upsert_node is idempotent: merges if node already exists,
                # creates if not. Replaces old find_by_label/update_node/add_node
                # trio that used UUID keys incompatible with retrieval.py's search().
                kg.upsert_node(label, ftype, summary=content)
                saved_facts += 1

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
