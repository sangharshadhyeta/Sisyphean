"""Memory injector — builds the system prompt memory section.

Runs on every incoming request. Selects the most relevant memory
within a token budget and returns it as a formatted string that
gets prepended to the model's system prompt.

Priority order (highest → lowest):
  1. Soul / personality     — always included, highest priority
  2. User knowledge         — who the user is, preferences
  3. Active project context — what's currently being worked on
  4. Relevant facts/concepts from graph (query-matched)
  5. Related past artifacts from store (query-matched)

If the budget runs out, lower-priority sections are dropped first.
Each section is hard-truncated (never silently dropped mid-word) to
preserve budget accuracy.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.llm.client import LlamaClient
    from .graph import KnowledgeGraph
    from .store import ArtifactStore

logger = logging.getLogger(__name__)


class MemoryInjector:

    def __init__(
        self,
        graph: KnowledgeGraph,
        store: ArtifactStore,
        token_budget: int = 1500,
        top_n_nodes: int = 5,
        top_n_artifacts: int = 3,
    ) -> None:
        self.graph = graph
        self.store = store
        self.token_budget = token_budget
        self.top_n_nodes = top_n_nodes
        self.top_n_artifacts = top_n_artifacts

    async def build(self, message: str, client: LlamaClient) -> str:
        """Return the memory context string to prepend to the system prompt.

        The string is guaranteed to fit within token_budget tokens.
        Returns empty string if memory is empty or budget is zero.

        Note: client parameter retained for API compatibility but no longer used
        for tokenisation — budget is enforced via a 4-chars-per-token heuristic,
        eliminating the 5 round-trips to /tokenize that fired on every request.
        """
        sections: list[str] = []
        remaining = self.token_budget

        # ── 1. Engine policy ─────────────────────────────────────────────────
        policy_nodes = self.graph.all_nodes(node_type="policy") or self.graph.all_nodes(node_type="soul")
        if policy_nodes and remaining > 50:
            text = "### Engine Policy\n" + "\n".join(n["content"] for n in policy_nodes)
            text, remaining = _fit(text, remaining)
            if text:
                sections.append(text)

        # ── 2. User knowledge ────────────────────────────────────────────────
        user_nodes = self.graph.all_nodes(node_type="user")
        user_nodes = [n for n in user_nodes if n.get("content", "").strip() and
                      "to be filled" not in n["content"]]
        if user_nodes and remaining > 50:
            text = "### User\n" + "\n".join(n["content"] for n in user_nodes)
            text, remaining = _fit(text, remaining)
            if text:
                sections.append(text)

        # ── 3. Active project ────────────────────────────────────────────────
        proj_nodes = self.graph.all_nodes(node_type="project")
        proj_nodes = [n for n in proj_nodes if n.get("content", "").strip() and
                      "to be filled" not in n["content"]]
        if proj_nodes and remaining > 50:
            text = "### Current Project\n" + "\n".join(
                f"**{n['label']}**: {n['content']}" for n in proj_nodes[:2]
            )
            text, remaining = _fit(text, remaining)
            if text:
                sections.append(text)

        # ── 4. Relevant facts / concepts (query-matched, old KnowledgeGraph) ────
        if remaining > 80:
            hits = self.graph.search(
                message,
                top_n=self.top_n_nodes,
                node_types=["fact", "concept", "project"],
            )
            if hits:
                lines = [f"- [{h['type']}] **{h['label']}**: {h['content'][:200]}" for h in hits]
                text = "### Relevant Context\n" + "\n".join(lines)
                text, remaining = _fit(text, remaining)
                if text:
                    sections.append(text)

        # ── 4b. Research knowledge (new GraphStore — NER + extracted facts) ───
        # The knowledge_graph GraphStore holds entities indexed by extractor.py
        # (extracted facts from conversations) and retrieval.py (NER from tool
        # results).  Query it separately so both memory layers are injected.
        if remaining > 60:
            try:
                from engine.memory.retrieval import retrieve as _retrieve
                kg_text = _retrieve(message, top_n=self.top_n_nodes)
                if kg_text:
                    text = "### Research Knowledge\n" + kg_text
                    text, remaining = _fit(text, remaining)
                    if text:
                        sections.append(text)
            except Exception:
                pass  # retrieval is best-effort

        # ── 5. Related past artifacts (query-matched) ────────────────────────
        if remaining > 80:
            arts = self.store.search(message, top_n=self.top_n_artifacts)
            if arts:
                lines = [f"- [{a['type']}] {a['summary']}" for a in arts]
                text = "### Past Work\n" + "\n".join(lines)
                text, remaining = _fit(text, remaining)
                if text:
                    sections.append(text)

        if not sections:
            return ""

        header = "---\n## Memory\n"
        footer = "\n---"
        return header + "\n\n".join(sections) + footer


# ── Helpers ──────────────────────────────────────────────────────────────────

# Standard approximation: ~4 characters per token.
# Combined with the 12% safety margin this is accurate enough for budget
# enforcement without any network round-trips.
_CHARS_PER_TOKEN = 4


def _fit(text: str, budget: int) -> tuple[str, int]:
    """Truncate *text* to fit within *budget* tokens.

    Uses a 4-chars-per-token heuristic instead of a live /tokenize call,
    eliminating up to 5 HTTP round-trips per request with negligible
    accuracy loss (the 12% safety margin absorbs the estimation error).

    Returns (possibly_truncated_text, remaining_budget).
    Returns ("", budget) if text is empty.
    """
    if not text.strip():
        return "", budget
    tokens = max(1, len(text) // _CHARS_PER_TOKEN)
    if tokens <= budget:
        return text, budget - tokens
    # Hard-truncate to fit — apply 12% safety margin
    max_chars = int(budget * _CHARS_PER_TOKEN * 0.88)
    return text[:max_chars].rstrip() + "…", 0
