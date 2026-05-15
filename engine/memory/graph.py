"""GraphRAG knowledge graph — personality, user context, projects, concepts.

Persisted as JSON with atomic writes (write-to-temp + os.replace).
Search is keyword-based by default; swap _score() for embedding similarity
once the embedding server is wired in Stage 3.

Node types
----------
soul        Personality traits, values, communication style
user        Who the user is — role, expertise, preferences
project     Active projects, goals, current status
concept     Technical / domain concepts and their relationships
fact        Discrete facts learned from conversations
artifact    Pointer to an entry in the ArtifactStore (JSONL)

Edge relations
--------------
knows_about   user → concept
works_on      user → project
related_to    concept ↔ concept  /  project ↔ concept
produced      project → artifact
exemplifies   artifact → concept
has_pref      user → concept  (preference edge)
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)


class KnowledgeGraph:

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._g: nx.DiGraph = nx.DiGraph()
        self._load()

    # ── Node CRUD ────────────────────────────────────────────────────────────

    def add_node(
        self,
        type: str,
        label: str,
        content: str,
        metadata: dict | None = None,
    ) -> str:
        node_id = str(uuid.uuid4())
        now = _now()
        self._g.add_node(node_id, **{
            "id": node_id,
            "type": type,
            "label": label,
            "content": content,
            "created_at": now,
            "updated_at": now,
            "metadata": metadata or {},
        })
        self._save()
        return node_id

    def update_node(self, node_id: str, **kwargs) -> bool:
        if node_id not in self._g:
            return False
        kwargs["updated_at"] = _now()
        self._g.nodes[node_id].update(kwargs)
        self._save()
        return True

    def get_node(self, node_id: str) -> dict | None:
        if node_id not in self._g:
            return None
        return dict(self._g.nodes[node_id])

    def find_by_label(self, label: str, type: str | None = None) -> dict | None:
        for _, data in self._g.nodes(data=True):
            if data.get("label", "").lower() == label.lower():
                if type is None or data.get("type") == type:
                    return dict(data)
        return None

    def all_nodes(self, node_type: str | None = None) -> list[dict]:
        return [
            dict(data) for _, data in self._g.nodes(data=True)
            if node_type is None or data.get("type") == node_type
        ]

    # ── Edges ────────────────────────────────────────────────────────────────

    def add_edge(self, source_id: str, target_id: str, relation: str, weight: float = 1.0) -> None:
        if source_id not in self._g or target_id not in self._g:
            return
        self._g.add_edge(source_id, target_id, relation=relation, weight=weight)
        self._save()

    def get_neighbors(self, node_id: str, relation: str | None = None) -> list[dict]:
        if node_id not in self._g:
            return []
        result = []
        for _, target, data in self._g.out_edges(node_id, data=True):
            if relation is None or data.get("relation") == relation:
                node = self.get_node(target)
                if node:
                    result.append(node)
        return result

    # ── Search ───────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_n: int = 5,
        node_types: list[str] | None = None,
    ) -> list[dict]:
        """Keyword-scored search. Returns top-N most relevant nodes."""
        q_tokens = set(_tokenize(query))
        if not q_tokens:
            return []

        scored: list[tuple[float, dict]] = []
        for _, data in self._g.nodes(data=True):
            if node_types and data.get("type") not in node_types:
                continue
            text = f"{data.get('label', '')} {data.get('content', '')}"
            n_tokens = set(_tokenize(text))
            overlap = q_tokens & n_tokens
            if overlap:
                score = len(overlap) / len(q_tokens)
                scored.append((score, dict(data)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_n]]

    # ── Persistence (atomic) ─────────────────────────────────────────────────

    def _load(self) -> None:
        if not self.path.exists():
            logger.info("No graph at %s — starting fresh", self.path)
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            for node in data.get("nodes", []):
                self._g.add_node(node["id"], **node)
            for edge in data.get("edges", []):
                self._g.add_edge(
                    edge["source"], edge["target"],
                    relation=edge.get("relation", ""),
                    weight=edge.get("weight", 1.0),
                )
            logger.info("Graph loaded: %d nodes, %d edges", self._g.number_of_nodes(), self._g.number_of_edges())
        except Exception as exc:
            logger.error("Failed to load graph (%s) — starting fresh", exc)

    def _save(self) -> None:
        """Atomic write: serialise to .tmp then os.replace()."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        payload = {
            "nodes": [dict(d) for _, d in self._g.nodes(data=True)],
            "edges": [
                {"source": s, "target": t, **d}
                for s, t, d in self._g.edges(data=True)
            ],
        }
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp, self.path)


# ── GraphStore — BirdClaw-compatible API with upsert + bfs ────────────────────

def _node_key(name: str) -> str:
    return name.lower().strip()


class GraphStore:
    """Lightweight graph store with upsert-based node management and BFS retrieval.

    Two usage modes:
        GraphStore()            — session graph (in-memory, not persisted)
        GraphStore(path)        — knowledge graph (persisted to JSON with .bak)

    Exposes the same API as BirdClaw's memory.graph.GraphStore so the
    retrieval and NER modules work unchanged.
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        self._path = persist_path
        self._graph: nx.DiGraph = nx.DiGraph()
        if persist_path and persist_path.exists():
            self._load()

    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self) -> None:
        assert self._path is not None
        bak = self._path.with_suffix(self._path.suffix + ".bak")
        for candidate in (self._path, bak):
            if not candidate.exists():
                continue
            try:
                raw = json.loads(candidate.read_text(encoding="utf-8"))
                # Support both node-link format and old flat format
                if "graph" in raw or "nodes" in raw:
                    try:
                        self._graph = nx.node_link_graph(raw, directed=True, multigraph=False, edges="edges")
                    except Exception:
                        # Fallback: reconstruct from flat nodes/edges lists
                        self._graph = nx.DiGraph()
                        for n in raw.get("nodes", []):
                            key = n.pop("key", n.get("id", _node_key(n.get("name", "?"))))
                            self._graph.add_node(key, **n)
                        for e in raw.get("edges", []):
                            self._graph.add_edge(e["source"], e["target"], **{k: v for k, v in e.items() if k not in ("source", "target")})
                logger.info("GraphStore loaded from %s (%d nodes)", candidate, self.node_count())
                return
            except Exception as exc:
                logger.warning("GraphStore load failed (%s: %s) — trying backup", candidate, exc)
        self._graph = nx.DiGraph()

    def save(self) -> None:
        """Atomic write with .bak — no-op for in-memory graphs."""
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self._graph, edges="edges")
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        bak = self._path.with_suffix(self._path.suffix + ".bak")
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        if self._path.exists():
            try:
                self._path.replace(bak)
            except OSError:
                pass
        tmp.replace(self._path)

    # ── Node upsert ────────────────────────────────────────────────────────────

    def upsert_node(
        self,
        name: str,
        node_type: str,
        summary: str = "",
        sources: list[str] | None = None,
        **extra,
    ) -> str:
        key = _node_key(name)
        ts = _now()
        if self._graph.has_node(key):
            node = self._graph.nodes[key]
            existing = set(node.get("sources", []))
            existing.update(sources or [])
            node["sources"] = list(existing)
            node["last_seen"] = ts
            if summary:
                node["summary"] = summary
            node.update(extra)
        else:
            self._graph.add_node(
                key,
                name=name,
                type=node_type,
                summary=summary,
                sources=list(sources or []),
                created_at=ts,
                last_seen=ts,
                **extra,
            )
        return key

    def upsert_edge(self, subject: str, relation: str, obj: str, weight: float = 1.0) -> None:
        s_key = _node_key(subject)
        o_key = _node_key(obj)
        for key, name in ((s_key, subject), (o_key, obj)):
            if not self._graph.has_node(key):
                self._graph.add_node(key, name=name, type="entity", summary="", sources=[], last_seen=_now())
        if self._graph.has_edge(s_key, o_key):
            self._graph.edges[s_key, o_key]["weight"] = self._graph.edges[s_key, o_key].get("weight", 1.0) + weight
        else:
            self._graph.add_edge(s_key, o_key, relation=relation, weight=weight)

    # ── Query ──────────────────────────────────────────────────────────────────

    def get_node(self, name: str) -> dict | None:
        key = _node_key(name)
        if self._graph.has_node(key):
            return {"key": key, **dict(self._graph.nodes[key])}
        return None

    def search(self, query: str, limit: int = 10, node_type: str | None = None) -> list[dict]:
        """Token-overlap search. Returns list of dicts with 'key' field."""
        import re
        def _tok(t: str) -> set[str]:
            return {w for w in re.findall(r"[a-z0-9]+", t.lower()) if len(w) > 2}

        tokens = _tok(query)
        if not tokens:
            return []
        scored: list[tuple[int, dict]] = []
        for key, data in self._graph.nodes(data=True):
            if node_type and data.get("type") != node_type:
                continue
            score = len(tokens & (_tok(data.get("name", "")) | _tok(data.get("summary", ""))))
            if score > 0:
                scored.append((score, {"key": key, **data}))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:limit]]

    def bfs(self, seeds: list[str], depth: int = 2) -> list[dict]:
        """BFS from seed names. Returns reachable nodes within depth with neighbour lists."""
        visited: set[str] = set()
        frontier: set[str] = set()
        for name in seeds:
            key = _node_key(name)
            if self._graph.has_node(key):
                frontier.add(key)

        result: list[dict] = []
        for _ in range(depth):
            next_frontier: set[str] = set()
            for key in frontier:
                if key in visited:
                    continue
                visited.add(key)
                data = dict(self._graph.nodes[key])
                neighbors = []
                for n in self._graph.successors(key):
                    edge = self._graph.edges[key, n]
                    neighbors.append({"name": self._graph.nodes[n].get("name", n), "relation": edge.get("relation", ""), "direction": "out"})
                for n in self._graph.predecessors(key):
                    edge = self._graph.edges[n, key]
                    neighbors.append({"name": self._graph.nodes[n].get("name", n), "relation": edge.get("relation", ""), "direction": "in"})
                result.append({"key": key, "neighbors": neighbors, **data})
                next_frontier.update(self._graph.successors(key))
                next_frontier.update(self._graph.predecessors(key))
            frontier = next_frontier - visited
        return result

    def remove_node(self, name: str) -> bool:
        key = _node_key(name)
        if self._graph.has_node(key):
            self._graph.remove_node(key)
            return True
        return False

    def merge_from(self, other: "GraphStore") -> None:
        """Merge all nodes and edges from another GraphStore (used by dreaming)."""
        for key, data in other._graph.nodes(data=True):
            if not self._graph.has_node(key):
                self._graph.add_node(key, **data)
            else:
                existing = set(self._graph.nodes[key].get("sources", []))
                existing.update(data.get("sources", []))
                self._graph.nodes[key]["sources"] = list(existing)
        for s, t, data in other._graph.edges(data=True):
            if not self._graph.has_edge(s, t):
                self._graph.add_edge(s, t, **data)


# ── Module-level singletons ────────────────────────────────────────────────────

def _knowledge_graph_path() -> Path:
    """Sisyphean's canonical graph path.

    Sisyphean owns the graph. BirdClaw (and any other consumer) reads from
    and writes to this path so there is only one knowledge graph on the machine.

    Override by setting SISYPHEAN_GRAPH_PATH environment variable.
    """
    import os
    override = os.environ.get("SISYPHEAN_GRAPH_PATH", "").strip()
    if override:
        return Path(override)
    return Path.home() / ".sisyphean" / "memory" / "knowledge_graph.json"


# session_graph: ephemeral, in-memory only — cleared on restart
session_graph = GraphStore()

# knowledge_graph: Sisyphean owns this; BirdClaw shares it
knowledge_graph = GraphStore(_knowledge_graph_path())


# ── Seed ─────────────────────────────────────────────────────────────────────

def seed_graph(graph: KnowledgeGraph, policy_text: str) -> None:
    """Populate a fresh graph with the engine policy node and empty user/project stubs."""
    if graph.find_by_label("policy") or graph.find_by_label("soul"):
        return  # already seeded (check both names for backward compat)
    policy_id = graph.add_node("policy", "policy", policy_text)
    user_id   = graph.add_node("user", "user", "User profile — to be filled in through conversation.")
    proj_id   = graph.add_node("project", "active_project", "Current project — to be filled in.")
    graph.add_edge(user_id, proj_id, "works_on")
    graph.add_edge(user_id, policy_id, "guided_by")
    logger.info("Graph seeded with soul, user, and project nodes")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in text.split() if len(w) > 2]
