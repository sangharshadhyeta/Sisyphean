"""One-shot backfill: add edges to the existing orphaned knowledge-graph nodes.

Run from the Sisyphean project root:
    python backfill_edges.py

What it does
------------
- For every preference node  →  user --has_preference--> node
- For every project node     →  user --works_on-->       node
- For every concept node     →  user --knows_about-->    node
- For every fact/research node that shares at least one label token with a
  project node  →  project --contains_fact--> fact
- Co-occurrence within the same 'last_seen' second bucket (nodes added in the
  same extraction call) → node --related_to--> node

After running, restart the Sisyphean engine so the dashboard picks up the
fresh graph.
"""
from __future__ import annotations

import sys
import os
import time
from collections import defaultdict

# Add project root to path so engine imports work
sys.path.insert(0, os.path.dirname(__file__))

from engine.memory.graph import knowledge_graph as kg


def main() -> None:
    nodes = kg.all_nodes()
    print(f"Loaded {len(nodes)} nodes")

    edges_before = kg._graph.number_of_edges()

    # ── (1) Type-specific hub edges ──────────────────────────────────────────
    added = 0
    for n in nodes:
        name = n.get("name") or n.get("key", "")
        ntype = n.get("type", "")
        if not name or not ntype:
            continue
        try:
            if ntype == "preference":
                kg.upsert_edge("user", "has_preference", name, weight=1.0)
                added += 1
            elif ntype == "project":
                kg.upsert_edge("user", "works_on", name, weight=1.0)
                added += 1
            elif ntype == "concept":
                kg.upsert_edge("user", "knows_about", name, weight=0.6)
                added += 1
            elif ntype == "research":
                kg.upsert_edge("user", "researched", name, weight=0.4)
                added += 1
        except Exception as e:
            print(f"  WARN type-edge for {name!r}: {e}")

    print(f"  Type-specific edges added: {added}")

    # ── (2) Cross-link facts that share a token with a project node ──────────
    project_nodes = [n for n in nodes if n.get("type") == "project"]
    fact_nodes    = [n for n in nodes if n.get("type") in ("fact", "research")]

    def tokens(s: str) -> set[str]:
        return {t.lower() for t in s.replace("-", " ").replace("_", " ").split() if len(t) > 3}

    proj_tokens: dict[str, set[str]] = {}
    for p in project_nodes:
        name = p.get("name", "")
        proj_tokens[name] = tokens(name) | tokens(p.get("summary", ""))

    cross = 0
    for f in fact_nodes:
        fname = f.get("name", "")
        ftok  = tokens(fname) | tokens(f.get("summary", ""))
        for pname, ptok in proj_tokens.items():
            if ftok & ptok:          # shared meaningful token
                try:
                    kg.upsert_edge(pname, "contains_fact", fname, weight=0.4)
                    cross += 1
                except Exception:
                    pass

    print(f"  Project->fact cross-links added: {cross}")

    # ── (3) Co-occurrence within the same last_seen bucket (± 2 seconds) ─────
    # Nodes extracted in the same call share virtually the same timestamp.
    def _ts(val) -> int:
        """Convert a last_seen value to a Unix-timestamp int (handles both int and ISO str)."""
        if val is None:
            return 0
        if isinstance(val, (int, float)):
            return int(val)
        try:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(str(val))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except Exception:
            return 0

    bucket_map: dict[int, list[str]] = defaultdict(list)
    for n in nodes:
        ts = _ts(n.get("last_seen", 0))
        # Round to 5-second bucket so near-simultaneous writes group together
        bucket_map[(ts // 5) * 5].append(n.get("name") or n.get("key", ""))

    co = 0
    for bucket_names in bucket_map.values():
        if len(bucket_names) < 2:
            continue
        # Wire up to 5 pairs per bucket to avoid a combinatorial explosion
        for i in range(min(len(bucket_names), 5)):
            for j in range(i + 1, min(len(bucket_names), 5)):
                a, b = bucket_names[i], bucket_names[j]
                if a and b:
                    try:
                        kg.upsert_edge(a, "related_to", b, weight=0.3)
                        co += 1
                    except Exception:
                        pass

    print(f"  Co-occurrence edges added:      {co}")

    # ── Save ─────────────────────────────────────────────────────────────────
    kg.save()
    edges_after = kg._graph.number_of_edges()
    print(f"\nDone. Edges: {edges_before} -> {edges_after} (+{edges_after - edges_before})")
    print("Restart the Sisyphean engine to pick up the new graph.")


if __name__ == "__main__":
    main()
