"""Reset the knowledge graph and session store to a clean state.

Run from the Sisyphean project root:
    python scripts/reset_graph.py

What this does
--------------
1. Deletes ~/.sisyphean/memory/knowledge_graph.json (and .bak/.tmp)
2. Deletes ALL session JSONL files from ~/.sisyphean/sessions/
3. Clears the watermark / seen-marker files so dream re-processes nothing stale
4. Re-seeds the graph with clean anchor nodes (user, active_project, inner_self,
   world_model, system) and all skill nodes from skills/

Safe to re-run — the seed guard in seed_knowledge_graph is skipped here
(we are explicitly requesting a fresh start).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MEM_DIR      = Path.home() / ".sisyphean" / "memory"
SESSIONS_DIR = Path.home() / ".sisyphean" / "sessions"
GRAPH_PATH   = MEM_DIR / "knowledge_graph.json"


# ── Step 1: wipe graph files ──────────────────────────────────────────────────

def _wipe_graph() -> None:
    for suffix in ("", ".bak", ".bak2", ".tmp", ".lock"):
        p = GRAPH_PATH.with_suffix(GRAPH_PATH.suffix + suffix) if suffix else GRAPH_PATH
        if p.exists():
            p.unlink()
            print(f"  deleted {p.name}")
        # Handle double-suffix like .json.bak
        p2 = Path(str(GRAPH_PATH) + suffix)
        if p2 != p and p2.exists():
            p2.unlink()
            print(f"  deleted {p2.name}")


# ── Step 2: wipe sessions ─────────────────────────────────────────────────────

def _wipe_sessions() -> int:
    if not SESSIONS_DIR.exists():
        return 0
    count = 0
    for p in SESSIONS_DIR.iterdir():
        if p.suffix in (".jsonl",) or p.name == ".memorised":
            p.unlink()
            count += 1
    return count


# ── Step 3: wipe seen-markers so dream starts clean ──────────────────────────

def _wipe_markers() -> None:
    for name in (
        "self_reflections_seen.txt",
        "skill_discoveries_seen.txt",
        ".memorised",
    ):
        for d in (MEM_DIR, SESSIONS_DIR):
            p = d / name
            if p.exists():
                p.unlink()
                print(f"  deleted marker: {p}")


# ── Step 4: re-seed ───────────────────────────────────────────────────────────

def _reseed() -> None:
    from engine.memory.graph import GraphStore, seed_knowledge_graph, seed_skill_graph

    # Fresh store pointed at the standard graph path
    graph = GraphStore(persist_path=GRAPH_PATH)

    # Force-bypass the "already seeded" guard by calling upsert directly
    # (the guard checks for user+inner_self — we just deleted the file so both
    # are absent; calling seed_knowledge_graph() is sufficient).
    seed_knowledge_graph(graph, policy_text="")

    # Seed skill nodes from skills/
    skills_path = ROOT / "skills"
    if skills_path.is_dir():
        seed_skill_graph(graph, skills_path)
        skill_count = sum(1 for _ in skills_path.glob("*.py"))
        print(f"  seeded {skill_count} skill node(s) from skills/")
    else:
        print("  [warn] skills/ directory not found — skipping skill nodes")

    graph.save()
    print(f"  graph saved: {graph.node_count()} nodes, {graph.edge_count()} edges")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n=== Sisyphean graph reset ===\n")

    print("[1] Wiping graph files...")
    _wipe_graph()

    print("\n[2] Wiping session logs...")
    n = _wipe_sessions()
    print(f"  deleted {n} session file(s)")

    print("\n[3] Wiping seen-markers...")
    _wipe_markers()

    print("\n[4] Re-seeding graph...")
    _reseed()

    print("\n=== Done. Graph is clean and re-seeded. ===\n")


if __name__ == "__main__":
    main()
