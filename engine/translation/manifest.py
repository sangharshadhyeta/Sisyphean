"""Task manifest — the structured result of the conversion step.

A TaskManifest is a list of self-contained Instructions.  Each Instruction
carries everything the executor needs to act on it: what to do, what type
of action it is, and whether the previous step's result is relevant.

The word "self-contained" is intentional: the instruction text is written
by the decomposer to be followable without reading any prior history.
For example:
  BAD:  "Continue the scraper work"
  GOOD: "Write scraper.py using bs4: fetch article titles from a URL,
         handle HTTP errors, return a list of strings."

This means the executor's LLM call only ever needs:
  1. The manifest progress block (goal + one-line summaries of done steps)
  2. The current instruction (fully self-contained)
  3. The last tool result (only when needs_prev=True)

No history.  No accumulated tool results.  No old thinking blocks.
Context loss is architecturally impossible — there is no context to lose.

Serialisation
-------------
The entire manifest serialises to / from a compact JSON dict that lives in
the SISYPHEAN_STATE thinking block.  Sizes on a typical 4-step task:
  - goal            ~15 words
  - 4 instructions  ~30 words each (self-contained but still brief)
  - 4 summaries     ~15 words each (after completion)
Total: ~200 words → ~300 tokens in the thinking block.  Well within budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ── Instruction ───────────────────────────────────────────────────────────────

@dataclass
class Instruction:
    idx: int
    type: str        # research | write_code | write_doc | verify | reflect | direct
    text: str        # fully self-contained instruction text
    needs_prev: bool = False   # True → inject last_result into execution prompt
    status: str = "pending"    # pending | done | failed
    summary: str = ""          # 1–2 line outcome written after completion


# ── TaskManifest ──────────────────────────────────────────────────────────────

@dataclass
class TaskManifest:
    """Ordered list of self-contained instructions for a single user task."""

    goal: str
    steps: list[Instruction] = field(default_factory=list)
    current_idx: int = 0
    last_result: str = ""   # compressed most-recent tool result (≤ 300 chars)

    # ── Navigation ───────────────────────────────────────────────────────────

    @property
    def current(self) -> Instruction | None:
        if self.current_idx < len(self.steps):
            return self.steps[self.current_idx]
        return None

    @property
    def is_complete(self) -> bool:
        return self.current_idx >= len(self.steps)

    @property
    def n_done(self) -> int:
        return sum(1 for s in self.steps if s.status == "done")

    def advance(self, summary: str) -> None:
        """Mark the current step done, store its summary, move to next.

        Also updates last_result to the summary so the next needs_prev step
        gets a clean 1-2 sentence finding rather than raw tool output.
        """
        if self.current_idx < len(self.steps):
            step = self.steps[self.current_idx]
            step.status = "done"
            step.summary = summary.strip()[:160]  # keep summaries short
        self.current_idx += 1
        # Propagate summary as last_result for the next needs_prev step.
        # This is more useful than the raw tool output (which could be thousands
        # of chars of scraped HTML) — the summary is already distilled.
        self.last_result = summary.strip()[:280]

    def fail_current(self, reason: str) -> None:
        """Mark the current step failed and skip it."""
        if self.current_idx < len(self.steps):
            self.steps[self.current_idx].status = "failed"
            self.steps[self.current_idx].summary = f"failed: {reason[:80]}"
        self.current_idx += 1

    # ── Prompt helpers ────────────────────────────────────────────────────────

    def progress_block(self) -> str:
        """Compact progress context injected into every execution prompt.

        Shows the overall goal + one-line outcome per completed step +
        the current instruction.  Stays under ~200 words regardless of
        how many steps have been completed.
        """
        lines: list[str] = [f"Objective: {self.goal}"]

        done = [s for s in self.steps if s.status in ("done", "failed")]
        if done:
            lines.append("Completed:")
            for s in done:
                icon = "[done]" if s.status == "done" else "[fail]"
                lines.append(f"  {icon} [{s.type}] {s.summary or s.text[:60]}")

        cur = self.current
        if cur:
            total = len(self.steps)
            # Do NOT show the step type (direct/research/verify/etc.) to the
            # executor model — it's an internal routing hint, not a directive.
            # Showing "(direct)" causes qwen3 to skip tools and just answer.
            lines.append(
                f"Current step [{cur.idx + 1}/{total}]: {cur.text}"
            )

        return "\n".join(lines)

    def execution_prompt(self) -> str:
        """Full context block for the execution LLM call.

        Includes progress_block() + last_result (if the current step
        needs it).  This is the entire context the executor sees — no
        history, no accumulated results.
        """
        cur = self.current
        parts = [self.progress_block()]

        if cur and cur.needs_prev and self.last_result:
            parts.append(f"Previous result:\n{self.last_result}")

        return "\n\n".join(parts)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "steps": [
                {
                    "idx": s.idx,
                    "type": s.type,
                    "text": s.text,
                    "needs_prev": s.needs_prev,
                    "status": s.status,
                    "summary": s.summary,
                }
                for s in self.steps
            ],
            "current_idx": self.current_idx,
            "last_result": self.last_result[:300],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TaskManifest":
        steps = [
            Instruction(
                idx=s["idx"],
                type=s.get("type", "research"),
                text=s["text"],
                needs_prev=s.get("needs_prev", False),
                status=s.get("status", "pending"),
                summary=s.get("summary", ""),
            )
            for s in d.get("steps", [])
        ]
        return cls(
            goal=d.get("goal", ""),
            steps=steps,
            current_idx=d.get("current_idx", 0),
            last_result=d.get("last_result", ""),
        )

    # ── Final assembly ────────────────────────────────────────────────────────

    def assemble_answer(self) -> str:
        """Produce a final answer from the completed step summaries.

        If only one step, return its summary directly.
        Otherwise, weave summaries into a coherent response.
        """
        done = [s for s in self.steps if s.status == "done" and s.summary]
        if not done:
            return f"Task completed: {self.goal}"
        if len(done) == 1:
            return done[0].summary
        parts = [f"**{self.goal}**\n"]
        for s in done:
            parts.append(f"- {s.summary}")
        return "\n".join(parts)
