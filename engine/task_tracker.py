"""In-memory task tracker — publishes live plan state to the dashboard.

The translation loop calls track_* functions as steps progress.
The dashboard polls /api/tasks to render the flowchart.
State is ephemeral (resets on server restart) — just enough for live display.

Tree schema (task["tree"]):
  {
    "extractor": {status, input, output},
    "planner":         {status, output, sub_tasks: [{text, status, steps: [{type,input,output,status}]}]},
    "synthesizer":     {status, input, output},
  }
"""
from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from typing import Literal

StepStatus = Literal["pending", "running", "done", "failed"]
TaskStatus = Literal["running", "done", "failed"]

_MAX_TASKS = 20   # keep last N completed tasks for display

_tasks: OrderedDict[str, dict] = OrderedDict()


# ── Public API ────────────────────────────────────────────────────────────────

def start_task(session_id: str, user_message: str) -> str:
    """Register a new task. Returns the task_id."""
    task_id = uuid.uuid4().hex[:12]
    _tasks[task_id] = {
        "id": task_id,
        "session": session_id[-8:] if session_id else "",
        "user_message": user_message[:120],
        "status": "running",
        "started_at": round(time.time()),
        "finished_at": None,
        "steps": [],
        # Pipeline tree — populated by tree_* helpers below
        "tree": {
            "extractor": {"status": "pending", "input": "", "output": ""},
            "planner":         {"status": "pending", "output": "", "sub_tasks": []},
            "synthesizer":     {"status": "pending", "input": "", "output": ""},
        },
    }
    _evict()
    return task_id


def set_plan(task_id: str, steps: list[dict]) -> None:
    """Set the plan steps for a task. Each step: {id, type, text}."""
    task = _tasks.get(task_id)
    if not task:
        return
    task["steps"] = [
        {
            "id": s.get("id") or f"step-{i}",
            "type": s.get("type", "execute"),
            "text": s.get("text", "")[:100],
            "status": "pending",
            "started_at": None,
            "finished_at": None,
            "input": "",
            "output": "",
        }
        for i, s in enumerate(steps)
    ]


def step_running(task_id: str, step_index: int, input_text: str = "") -> None:
    """Mark a step as currently running."""
    step = _get_step(task_id, step_index)
    if step:
        step["status"] = "running"
        step["started_at"] = round(time.time())
        step["input"] = input_text[:300]


def step_done(task_id: str, step_index: int, output_text: str = "") -> None:
    """Mark a step as completed."""
    step = _get_step(task_id, step_index)
    if step:
        step["status"] = "done"
        step["finished_at"] = round(time.time())
        step["output"] = output_text[:300]


def step_failed(task_id: str, step_index: int, error: str = "") -> None:
    """Mark a step as failed."""
    step = _get_step(task_id, step_index)
    if step:
        step["status"] = "failed"
        step["finished_at"] = round(time.time())
        step["output"] = error[:300]


def finish_task(task_id: str, status: TaskStatus = "done") -> None:
    """Mark the whole task as done or failed."""
    task = _tasks.get(task_id)
    if task:
        task["status"] = status
        task["finished_at"] = round(time.time())
        # Mark any still-pending/running steps as done or failed
        for step in task["steps"]:
            if step["status"] in ("pending", "running"):
                step["status"] = "done" if status == "done" else "failed"


def add_inline_step(task_id: str, step_type: str, text: str,
                    input_text: str = "", output_text: str = "",
                    status: StepStatus = "done") -> None:
    """Add a single tool-call step that happened outside the formal plan
    (e.g. memory search, think, web_search triggered mid-execution)."""
    task = _tasks.get(task_id)
    if not task:
        return
    now = round(time.time())
    task["steps"].append({
        "id": f"inline-{len(task['steps'])}",
        "type": step_type,
        "text": text[:100],
        "status": status,
        "started_at": now,
        "finished_at": now if status != "running" else None,
        "input": input_text[:300],
        "output": output_text[:300],
    })


# ── Pipeline tree helpers ─────────────────────────────────────────────────────

def tree_context_running(task_id: str, query: str) -> None:
    t = _tree(task_id)
    if t:
        t["extractor"]["status"] = "running"
        t["extractor"]["input"] = query[:120]


def tree_context_done(task_id: str, summary: str = "") -> None:
    t = _tree(task_id)
    if t:
        t["extractor"]["status"] = "done"
        t["extractor"]["output"] = summary[:200]
        t["planner"]["status"] = "running"


def tree_plan_done(task_id: str, sub_tasks: list[dict]) -> None:
    """Called once the planner has split + planned all sub-tasks.

    sub_tasks: list of {"task": str, "steps": [{"tool": str, "input": str}]}
    Registers all planned steps upfront so the dashboard shows the full plan
    before execution begins.
    """
    t = _tree(task_id)
    if not t:
        return
    t["planner"]["status"] = "done"
    # Show actual stage goals, not a generic count
    if len(sub_tasks) == 1:
        t["planner"]["output"] = sub_tasks[0]["task"][:120]
    else:
        t["planner"]["output"] = "  →  ".join(st["task"][:35] for st in sub_tasks[:5])
    now = round(time.time())
    t["planner"]["sub_tasks"] = [
        {
            "text": st["task"][:120],
            "status": "pending",
            "steps": [
                {
                    # For run_skill steps, use the skill name as the display type
                    # so the dashboard shows "arxiv" / "youtube" (pink) rather than
                    # the generic "run_skill" label.
                    "type": s["input"].strip() if s["tool"] == "run_skill" else s["tool"],
                    "is_skill": s["tool"] == "run_skill",
                    "input": s["input"][:200],
                    "output": "", "status": "pending", "ts": now,
                }
                for s in st.get("steps", [])
            ],
        }
        for st in sub_tasks
    ]
    t["synthesizer"]["status"] = "pending"


def tree_subtask_replanned(task_id: str, task_idx: int,
                           new_steps: list[dict]) -> None:
    """Called when _replan_after_search() adds steps mid-execution.

    Appends the new planned steps (pending) to the sub-task's step list so the
    dashboard shows the full updated plan, not just steps already executed.
    """
    t = _tree(task_id)
    if not t:
        return
    subs = t["planner"]["sub_tasks"]
    if not (0 <= task_idx < len(subs)):
        return
    now = round(time.time())
    for s in new_steps:
        _is_skill = s["tool"] == "run_skill"
        subs[task_idx]["steps"].append({
            "type": s["input"].strip() if _is_skill else s["tool"],
            "is_skill": _is_skill,
            "input": s["input"][:200],
            "output": "", "status": "pending", "ts": now,
            "replan": True,   # mark so the dashboard can show a "↩ Replanned" divider
        })


def tree_subtask_step(task_id: str, task_idx: int, step_type: str,
                      input_text: str, output_text: str,
                      status: StepStatus = "done") -> None:
    """Upsert the most-recent step for a sub-task (running or done)."""
    t = _tree(task_id)
    if not t:
        return
    subs = t["planner"]["sub_tasks"]
    if not (0 <= task_idx < len(subs)):
        return
    sub = subs[task_idx]
    sub["status"] = "running"  # sub-task stays running until tree_subtask_done()
    now = round(time.time())
    # 1. Try to update an existing step of the same type that is running or pending
    for existing in reversed(sub["steps"]):
        if existing["type"] == step_type and existing["status"] in ("running", "pending"):
            existing["input"]  = input_text[:200]
            existing["output"] = output_text[:200]
            existing["status"] = status
            existing["ts"]     = now
            return
    # 2. No match — append a new step (handles replanned steps not pre-registered)
    sub["steps"].append({
        "type": step_type,
        "input": input_text[:200],
        "output": output_text[:200],
        "status": status,
        "ts": now,
    })


def tree_subtask_done(task_id: str, task_idx: int) -> None:
    t = _tree(task_id)
    if not t:
        return
    subs = t["planner"]["sub_tasks"]
    if 0 <= task_idx < len(subs):
        subs[task_idx]["status"] = "done"


def tree_subtask_failed(task_id: str, task_idx: int) -> None:
    t = _tree(task_id)
    if not t:
        return
    subs = t["planner"]["sub_tasks"]
    if 0 <= task_idx < len(subs):
        subs[task_idx]["status"] = "failed"


def tree_synthesizer_running(task_id: str, input_summary: str = "") -> None:
    t = _tree(task_id)
    if t:
        t["synthesizer"]["status"] = "running"
        t["synthesizer"]["input"] = input_summary[:200]


def tree_synthesizer_done(task_id: str, output: str = "") -> None:
    t = _tree(task_id)
    if t:
        t["synthesizer"]["status"] = "done"
        t["synthesizer"]["output"] = output[:300]


def active_tasks(n: int = 10) -> list[dict]:
    """Return all running tasks + most-recently-finished tasks, up to n total."""
    _expire_stale()
    all_tasks = list(_tasks.values())
    running  = [t for t in all_tasks if t["status"] == "running"]
    finished = sorted(
        [t for t in all_tasks if t["status"] != "running"],
        key=lambda t: t.get("finished_at") or t.get("started_at") or 0,
        reverse=True,
    )
    # All running tasks first, then fill remaining slots with finished
    result = list(running)
    slots_left = max(0, n - len(result))
    result.extend(finished[:slots_left])
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _expire_stale(max_age: int = 900) -> None:
    """Mark tasks that have been 'running' for over max_age seconds as failed."""
    now = round(time.time())
    for task in _tasks.values():
        if task["status"] == "running" and (now - task["started_at"]) > max_age:
            task["status"] = "failed"
            task["finished_at"] = now
            for step in task["steps"]:
                if step["status"] in ("pending", "running"):
                    step["status"] = "failed"


def _get_step(task_id: str, step_index: int) -> dict | None:
    task = _tasks.get(task_id)
    if not task:
        return None
    steps = task.get("steps", [])
    if 0 <= step_index < len(steps):
        return steps[step_index]
    return None


def _tree(task_id: str) -> dict | None:
    task = _tasks.get(task_id)
    return task["tree"] if task else None


def _evict() -> None:
    """Keep only the last _MAX_TASKS tasks.

    Eviction candidates: finished tasks first, then tasks that have been
    "running" long enough to have been expired by _expire_stale().  Pure
    running tasks (started recently) are never evicted — they would appear
    to vanish from the dashboard mid-execution.
    """
    _expire_stale()  # age out any hung tasks before eviction decisions
    evictable = [k for k, v in _tasks.items() if v["status"] != "running"]
    while len(_tasks) > _MAX_TASKS and evictable:
        del _tasks[evictable.pop(0)]
