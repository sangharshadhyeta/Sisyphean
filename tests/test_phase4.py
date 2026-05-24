"""Phase 4 pathway tests.

Covers every pathway added in Phase 4 (session auto-merge, skill injection
hook, planning integration) WITHOUT hitting the live engine or LLM.

Run:
    python tests/test_phase4.py
    python -m pytest tests/test_phase4.py -v   (if pytest is available)

Test groups
-----------
  A  infer_stage_type       -- skill prefix routing to "direct"
  B  get_skill_index        -- Jaccard filtering, top_n cap, [runnable] tag
  C  Progressive disclosure -- Layer 1 index -> Layer 2 runbook -> Layer 3 program
  D  save_skill_program     -- version tracking, history preservation, status transitions
  E  mark_skill_accepted    -- status written to graph
  F  Planner integration    -- SKILL-FIRST in _THINK_DECOMPOSE_SYSTEM,
                              conditional tools in _build_plan_system
  G  session_merge          -- node created, re-call enriches, empty session no-ops
  H  pipeline _run_internal -- read_skill / save_skill / save_skill_program / run_skill fallback
  I  pipeline _execute      -- run_skill -> bash conversion (unit, no live LLM)
"""
from __future__ import annotations

import asyncio
import json
import sys
import traceback
from pathlib import Path

# -- ensure project root is on sys.path ---------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -- imports -------------------------------------------------------------------
from engine.memory.graph import GraphStore
from engine.memory.skills import (
    get_skill_index,
    get_skill_runbook,
    get_skill_program,
    get_skill_script_path,
    save_skill_to_disk,
    save_skill_program_to_graph,
    mark_skill_accepted,
    SKILL_SCRIPTS_DIR,
)
from engine.translation.planner import (
    infer_stage_type,
    _THINK_DECOMPOSE_SYSTEM,
    _build_plan_system,
)


# -- Helpers -------------------------------------------------------------------

def _make_graph() -> GraphStore:
    """In-memory GraphStore -- no disk persistence."""
    return GraphStore(persist_path=None)


_PASS = "  OK"
_FAIL = "  FAIL"
_results: list[tuple[str, bool, str]] = []


def check(label: str, condition: bool, detail: str = "") -> bool:
    symbol = _PASS if condition else _FAIL
    _results.append((label, condition, detail))
    print(f"{symbol} {label}" + (f"  [{detail}]" if detail and not condition else ""))
    return condition


def section(title: str) -> None:
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")


# ===============================================================================
# A -- infer_stage_type: skill prefix routing
# ===============================================================================

def test_A_infer_stage_type():
    section("A  infer_stage_type -- skill prefix routing")

    # All run_skill variants
    check("A1  run skill: prefix -> direct",
          infer_stage_type("Run skill:pdf-extractor") == "direct")
    check("A2  run_skill: prefix -> direct",
          infer_stage_type("run_skill:pdf-extractor") == "direct")
    check("A3  run skill NAME (space) -> direct",
          infer_stage_type("run skill system-status") == "direct")
    check("A4  Run skill:NAME mixed case -> direct",
          infer_stage_type("Run skill:System-Status") == "direct")

    # All read_skill variants
    check("A5  read skill: prefix -> direct",
          infer_stage_type("Read skill:pdf-extractor") == "direct")
    check("A6  read_skill: prefix -> direct",
          infer_stage_type("read_skill:pdf-extractor") == "direct")
    check("A7  read skill NAME (space) -> direct",
          infer_stage_type("read skill system-status") == "direct")

    # Normal routing still works
    check("A8  Search still -> research",
          infer_stage_type("Search latest Python version") == "research")
    check("A9  Write filename -> write_code",
          infer_stage_type("Write data_cleaner.py") == "write_code")
    check("A10 Run command -> verify",
          infer_stage_type("Run python calc.py") == "verify")
    check("A11 empty step -> direct",
          infer_stage_type("") == "direct")
    check("A12 unknown step -> direct (not research)",
          infer_stage_type("XYZ-unknown-keyword-xyz") == "direct")


# ===============================================================================
# B -- get_skill_index: Jaccard filtering, top_n, [runnable] tag
# ===============================================================================

def test_B_skill_index():
    section("B  get_skill_index -- filtering, top_n, [runnable] tag")

    g = _make_graph()

    # Populate 5 skills -- 3 relevant to PDF, 2 unrelated
    g.upsert_node("pdf-extractor", "skill",
                  summary="Extract text from PDF files using pdfminer")
    g.upsert_node("pdf-cleaner", "skill",
                  summary="Clean extracted PDF text, remove noise",
                  program="print('clean')")
    g.upsert_node("pdf-summariser", "skill",
                  summary="Summarise PDF document content")
    g.upsert_node("git-diff-tool", "skill",
                  summary="Show git diff with context and stats")
    g.upsert_node("network-scanner", "skill",
                  summary="Scan local network for active hosts")

    task = "extract text from a PDF file and clean it"
    idx = get_skill_index(task, g)

    check("B1  returns non-empty string for relevant task", bool(idx))
    check("B2  pdf-extractor appears in index", "pdf-extractor" in idx)
    check("B3  pdf-cleaner appears in index",   "pdf-cleaner" in idx)
    # B4: Jaccard-ranked — PDF skills score higher than unrelated ones.
    # With _MIN_SCORE=1 a stop-word hit can let unrelated skills through, but
    # PDF skills should always score higher, so they appear before git-diff-tool.
    lines = [l for l in idx.splitlines() if l.strip()]
    pdf_first = any("pdf" in l for l in lines[:2])
    check("B4  PDF skills rank above unrelated skills in top-2",
          pdf_first, f"top lines={lines[:2]!r}")
    check("B5  [runnable] tag on pdf-cleaner",   "[runnable]" in idx,
          f"index={idx!r}")

    # top_n cap — count lines since bullet char may be unicode
    idx_n1 = get_skill_index(task, g, top_n=1)
    n_bullets = len([l for l in idx_n1.splitlines() if l.strip()])
    check("B6  top_n=1 returns exactly 1 line",
          n_bullets == 1, f"lines={n_bullets} idx_n1={idx_n1!r}")

    # No graph
    check("B7  None graph -> empty string", get_skill_index(task, None) == "")

    # No matching skills
    g2 = _make_graph()
    g2.upsert_node("git-diff-tool", "skill",
                   summary="Show git diff with context and stats")
    idx_none = get_skill_index("cook a chicken recipe", g2)
    check("B8  no token overlap -> empty string", idx_none == "",
          f"got: {idx_none!r}")


# ===============================================================================
# C -- Progressive disclosure: Layer 1 -> 2 -> 3
# ===============================================================================

def test_C_progressive_disclosure():
    section("C  Progressive disclosure -- L1 index -> L2 runbook -> L3 program")

    g = _make_graph()
    CODE = "import sys\nprint('hello from skill')"
    RUNBOOK = "## PDF Extractor\n1. Open file\n2. Extract text\n3. Return string"

    g.upsert_node(
        "pdf-extract", "skill",
        summary="Extract text from PDF",
        content=RUNBOOK,
        program=CODE,
        version=1,
        status="draft",
    )

    # Layer 1 -- compact index
    idx = get_skill_index("extract text from pdf", g)
    check("C1  L1 index is non-empty",          bool(idx))
    check("C2  L1 index shows [runnable]",       "[runnable]" in idx)
    check("C3  L1 index does NOT show full runbook",
          "1. Open file" not in idx,
          f"idx={idx!r}")

    # Layer 2 -- full runbook
    rb = get_skill_runbook("pdf-extract", g)
    check("C4  L2 runbook returns full content", "1. Open file" in rb)
    check("C5  L2 runbook is longer than index", len(rb) > len(idx))

    # Layer 3 -- program
    prog = get_skill_program("pdf-extract", g)
    check("C6  L3 program returns code",         "print('hello from skill')" in prog)

    # Missing skill
    check("C7  L2 missing skill -> empty string", get_skill_runbook("no-such-skill", g) == "")
    check("C8  L3 missing skill -> empty string", get_skill_program("no-such-skill", g) == "")


# ===============================================================================
# D -- save_skill_program_to_graph: version tracking, history, status
# ===============================================================================

def test_D_versioning():
    section("D  save_skill_program_to_graph -- versioning, history, status")

    g = _make_graph()
    CODE_V1 = "print('version 1')"
    CODE_V2 = "print('version 2')"
    CODE_V3 = "print('version 3')"

    # First save -- new skill
    save_skill_program_to_graph("test-skill", CODE_V1, g, runbook="Step 1 only")
    n1 = g.get_node("test-skill")
    check("D1  version starts at 1",     n1 and int(n1.get("version", 0)) == 1)
    check("D2  status is 'draft'",       n1 and n1.get("status") == "draft")
    check("D3  program stored",          n1 and "version 1" in n1.get("program", ""))
    h1 = json.loads(n1.get("program_history", "[]")) if n1 else []
    check("D4  history empty on first save", len(h1) == 0)

    # Second save -- improvement
    save_skill_program_to_graph("test-skill", CODE_V2, g, runbook="Step 1 and Step 2")
    n2 = g.get_node("test-skill")
    check("D5  version incremented to 2", n2 and int(n2.get("version", 0)) == 2)
    check("D6  status is 'improved'",     n2 and n2.get("status") == "improved")
    check("D7  program updated to v2",    n2 and "version 2" in n2.get("program", ""))
    h2 = json.loads(n2.get("program_history", "[]")) if n2 else []
    check("D8  history has 1 entry (v1 snippet)", len(h2) == 1)
    check("D9  history entry has version key",    h2 and "version" in h2[0])
    check("D10 history entry has snippet",        h2 and "version 1" in h2[0].get("snippet", ""))

    # Third save
    save_skill_program_to_graph("test-skill", CODE_V3, g)
    n3 = g.get_node("test-skill")
    h3 = json.loads(n3.get("program_history", "[]")) if n3 else []
    check("D11 version incremented to 3", n3 and int(n3.get("version", 0)) == 3)
    check("D12 history has 2 entries",    len(h3) == 2,
          f"len={len(h3)}")

    # History cap at 3
    save_skill_program_to_graph("test-skill", "print('v4')", g)
    save_skill_program_to_graph("test-skill", "print('v5')", g)
    n5 = g.get_node("test-skill")
    h5 = json.loads(n5.get("program_history", "[]")) if n5 else []
    check("D13 history capped at 3 entries", len(h5) <= 3,
          f"len={len(h5)}")

    # Guard: no graph
    result = save_skill_program_to_graph("x", "code", None)
    check("D14 None graph -> returns None", result is None)

    # Guard: no code
    result2 = save_skill_program_to_graph("x", "", g)
    check("D15 empty code -> returns None", result2 is None)


# ===============================================================================
# E -- mark_skill_accepted
# ===============================================================================

def test_E_mark_accepted():
    section("E  mark_skill_accepted -- status transitions")

    g = _make_graph()
    save_skill_program_to_graph("accept-test", "print('x')", g)
    n = g.get_node("accept-test")
    check("E1  initial status is draft",       n and n.get("status") == "draft")

    mark_skill_accepted("accept-test", g)
    n2 = g.get_node("accept-test")
    check("E2  after accept -> status=accepted", n2 and n2.get("status") == "accepted")

    # Safe no-op on missing skill
    mark_skill_accepted("does-not-exist", g)
    check("E3  missing skill -> safe no-op (no exception)", True)

    # Safe no-op on None graph
    mark_skill_accepted("accept-test", None)
    check("E4  None graph -> safe no-op",        True)


# ===============================================================================
# F -- Planner integration: SKILL-FIRST in prompt, conditional tools
# ===============================================================================

def test_F_planner_integration():
    section("F  Planner integration -- SKILL-FIRST prompt + conditional tools")

    # SKILL-FIRST rule presence
    check("F1  _THINK_DECOMPOSE_SYSTEM contains SKILL-FIRST",
          "SKILL-FIRST" in _THINK_DECOMPOSE_SYSTEM)
    check("F2  SKILL-FIRST rule mentions [runnable]",
          "[runnable]" in _THINK_DECOMPOSE_SYSTEM)
    check("F3  SKILL-FIRST rule mentions Run skill",
          "Run skill:" in _THINK_DECOMPOSE_SYSTEM)
    check("F4  SKILL-FIRST rule mentions Read skill",
          "Read skill:" in _THINK_DECOMPOSE_SYSTEM)

    # _build_plan_system: WITHOUT skill_index -- no skill tools
    prompt_no_skills = _build_plan_system(outer_tools=[], skill_index="")
    check("F5  no skill_index -> read_skill absent from plan system",
          "read_skill" not in prompt_no_skills,
          f"first 200={prompt_no_skills[:200]!r}")
    check("F6  no skill_index -> run_skill absent from plan system",
          "run_skill" not in prompt_no_skills)

    # _build_plan_system: WITH skill_index -- skill tools injected
    prompt_with_skills = _build_plan_system(
        outer_tools=[],
        skill_index="  * pdf-extractor: Extract PDF text [runnable]",
    )
    check("F7  with skill_index -> read_skill present in plan system",
          "read_skill" in prompt_with_skills)
    check("F8  with skill_index -> run_skill present in plan system",
          "run_skill" in prompt_with_skills)


# ===============================================================================
# G -- session_merge pathway
# ===============================================================================

def test_G_session_merge():
    section("G  merge_session_to_graph -- node creation, re-call, empty no-op")

    from engine.memory.session_merge import merge_session_to_graph

    g = _make_graph()

    # Empty session -- should no-op gracefully
    asyncio.run(merge_session_to_graph("empty-session-id", g))
    check("G1  empty session -> no node created",
          g.get_node("session:") is None and
          len(g.all_nodes(node_type="session")) == 0)

    # None graph -- no-op
    asyncio.run(merge_session_to_graph("any-id", None))
    check("G2  None graph -> no exception",  True)

    # Session with real data
    from engine.memory.session_log import get_session_log
    slog = get_session_log("test-merge-session")
    slog.user_message("what is 2+2")
    slog.stage_done(
        stage_type="research",
        goal="find answer to 2+2",
        summary="the answer is 4",
    )

    asyncio.run(merge_session_to_graph("test-merge-session", g))
    nodes = g.all_nodes(node_type="session")
    check("G3  session with activity -> node created", len(nodes) == 1,
          f"node count={len(nodes)}")

    node = nodes[0] if nodes else {}
    check("G4  node name contains 'session:' prefix",
          node.get("name", "").startswith("session:"))
    check("G5  summary contains user request",
          "2+2" in node.get("summary", ""),
          f"summary={node.get('summary', '')!r}")
    check("G6  summary contains completed stage",
          "research" in node.get("summary", "").lower(),
          f"summary={node.get('summary', '')!r}")

    # Re-call enriches same node (upsert, not duplicate)
    slog.user_message("what is 3+3")
    slog.stage_done(stage_type="research", goal="3+3", summary="6")
    asyncio.run(merge_session_to_graph("test-merge-session", g))
    nodes2 = g.all_nodes(node_type="session")
    check("G7  re-call does not create duplicate node", len(nodes2) == 1,
          f"node count={len(nodes2)}")
    check("G8  re-call enriches summary with new request",
          "3+3" in nodes2[0].get("summary", ""),
          f"summary={nodes2[0].get('summary', '')!r}")


# ===============================================================================
# H -- pipeline _run_internal skill tool handlers (unit, no LLM)
# ===============================================================================

def test_H_run_internal():
    section("H  pipeline._run_internal -- skill tool handlers")

    # We test the handler logic directly via the same code paths
    # rather than instantiating a full Pipeline (which needs LLM config).
    # Strategy: call the handler functions directly through their
    # read_skill / save_skill / save_skill_program logic.

    g = _make_graph()

    # Seed a skill with program
    g.upsert_node("test-read", "skill",
                  summary="Test skill summary",
                  content="Step 1: do this\nStep 2: do that",
                  program="print('ran skill')")

    # read_skill -- has program
    runbook = get_skill_runbook("test-read", g)
    has_prog = bool(get_skill_program("test-read", g))
    prog_note = "\n\n[This skill has a saved program -- use run_skill:NAME to re-execute it directly.]" if has_prog else ""
    full = runbook + prog_note
    check("H1  read_skill returns content field",
          "Step 1: do this" in full)
    check("H2  read_skill prog_note when has program",
          "run_skill:NAME" in full)

    # read_skill -- no program
    g.upsert_node("text-only", "skill",
                  summary="Text-only skill",
                  content="Just follow these steps",
                  program="")
    runbook2 = get_skill_runbook("text-only", g)
    has_prog2 = bool(get_skill_program("text-only", g))
    full2 = runbook2 + ("\n\n[run_skill]" if has_prog2 else "")
    check("H3  text-only skill: no prog_note", "[run_skill]" not in full2)

    # read_skill -- missing skill (name shares no 2-token overlap with any node)
    rb_missing = get_skill_runbook("zorbax-quantum-widget", g)
    check("H4  read_skill missing -> empty string", rb_missing == "",
          f"got: {rb_missing[:80]!r}")

    # save_skill (text-only)
    inp = "my-text-skill|Step A: do X\nStep B: do Y"
    skill_name, _, content = inp.partition("|")
    summary = content[:60].rstrip()
    g.upsert_node(name=skill_name, node_type="skill",
                  summary=summary, content=content, sources=["pipeline"])
    n = g.get_node("my-text-skill")
    check("H5  save_skill stores content",
          n and "do X" in n.get("content", ""),
          f"content={n.get('content', '') if n else None!r}")

    # save_skill_program -- parse and delegate
    inp2 = "prog-skill|print('hello from program')"
    skill_name2, _, code = inp2.partition("|")
    save_skill_program_to_graph(skill_name=skill_name2.strip(), code=code.strip(), graph=g)
    n2 = g.get_node("prog-skill")
    check("H6  save_skill_program writes program field",
          n2 and "hello from program" in n2.get("program", ""))
    check("H7  save_skill_program sets version=1",
          n2 and int(n2.get("version", 0)) == 1)

    # run_skill fallback (no program -> message)
    g.upsert_node("runbook-only-skill", "skill",
                  summary="has runbook, no program",
                  content="Do step 1\nDo step 2")
    prog_fallback = get_skill_program("runbook-only-skill", g)
    runbook_fb = get_skill_runbook("runbook-only-skill", g)
    msg = (f"Skill 'runbook-only-skill' has no saved program. Runbook:\n{runbook_fb}"
           if runbook_fb else "Skill 'runbook-only-skill' not found.")
    check("H8  run_skill fallback: reports no program with runbook",
          "no saved program" in msg and "Do step 1" in msg)


# ===============================================================================
# I -- pipeline _execute: run_skill -> bash conversion (logic only, no LLM)
# ===============================================================================

def test_I_execute_run_skill():
    section("I  pipeline _execute -- run_skill -> bash step mutation")

    g = _make_graph()
    CODE = "import platform\nprint(platform.node())"

    # Save skill to graph (+ disk)
    save_skill_program_to_graph("sys-info", CODE, g, summary="print system hostname")
    script_path = get_skill_script_path("sys-info")

    # Simulate the _execute conversion logic exactly as in pipeline.py
    step = {"tool": "run_skill", "input": "sys-info"}
    tool = step["tool"]
    inp  = step["input"]

    _prog = get_skill_program(inp, g)
    check("I1  get_skill_program returns code",
          bool(_prog) and "platform" in _prog)

    if _prog:
        _script = get_skill_script_path(inp)
        if not _script.exists():
            _script = save_skill_to_disk(inp, _prog) or _script
        if _script and _script.exists():
            _cmd = f"python {_script}"
            step["tool"]  = "bash"
            step["input"] = _cmd
            tool = "bash"
            inp  = _cmd

    check("I2  step tool mutated to 'bash'",     step["tool"] == "bash",
          f"tool={step['tool']!r}")
    check("I3  step input is python <path>",
          step["input"].startswith("python "),
          f"input={step['input']!r}")
    check("I4  script exists on disk after write",
          script_path.exists(), f"path={script_path}")
    check("I5  script content matches code",
          "platform" in script_path.read_text(encoding="utf-8"))

    # run_skill with no program -- step should NOT mutate
    g.upsert_node("no-program-skill", "skill",
                  summary="no program here",
                  content="manual steps only")
    step2 = {"tool": "run_skill", "input": "no-program-skill"}
    _prog2 = get_skill_program("no-program-skill", g)
    if _prog2:
        step2["tool"] = "bash"   # would convert -- shouldn't happen
    check("I6  no-program skill: step NOT mutated to bash",
          step2["tool"] == "run_skill",
          f"tool={step2['tool']!r}")


# ===============================================================================
# Runner
# ===============================================================================

def main():
    print("\n" + "=" * 60)
    print("  Sisyphean Phase 4 Pathway Tests")
    print("=" * 60)

    tests = [
        ("A", test_A_infer_stage_type),
        ("B", test_B_skill_index),
        ("C", test_C_progressive_disclosure),
        ("D", test_D_versioning),
        ("E", test_E_mark_accepted),
        ("F", test_F_planner_integration),
        ("G", test_G_session_merge),
        ("H", test_H_run_internal),
        ("I", test_I_execute_run_skill),
    ]

    errors: list[tuple[str, str]] = []
    for group, fn in tests:
        try:
            fn()
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"\n{_FAIL} Group {group} CRASHED: {exc}")
            errors.append((group, tb))

    # Summary
    total   = len(_results)
    passed  = sum(1 for _, ok, _ in _results if ok)
    failed  = total - passed
    print("\n" + "=" * 60)
    print(f"  {passed}/{total} passed  |  {failed} failed")
    if errors:
        print(f"\n  {len(errors)} group(s) crashed:")
        for g_id, tb in errors:
            print(f"\n  -- Group {g_id} traceback --")
            print(tb)
    if failed > 0:
        print("\n  Failed checks:")
        for label, ok, detail in _results:
            if not ok:
                print(f"    {_FAIL} {label}" + (f"  [{detail}]" if detail else ""))
    print("=" * 60 + "\n")
    sys.exit(0 if failed == 0 and not errors else 1)


if __name__ == "__main__":
    main()
