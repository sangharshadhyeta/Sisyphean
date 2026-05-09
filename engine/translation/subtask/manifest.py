"""Subtask manifest — tracks per-item write progress for write_code/write_doc stages."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Literal, Optional


ItemStatus = Literal["pending", "in_progress", "complete", "partial", "missing", "regressed"]


@dataclass
class SubtaskItem:
    index: int
    title: str                          # "reverse_string" or "## Introduction"
    anchor: str                         # function/class name or heading text (without ##)
    kind: Literal["section", "function", "class", "test"]
    expected_min_chars: int = 200

    # filled by verifier after each pass
    status: ItemStatus = "pending"
    actual_chars: int = 0
    is_stub: bool = False
    summary: str = ""                   # first 120 chars; feeds retry context
    content_hash: str = ""              # sha256[:16]; detects regression

    def mark_complete(self, body: str) -> None:
        self.status = "complete"
        self.actual_chars = len(body)
        self.content_hash = hashlib.sha256(body[:200].encode()).hexdigest()[:16]
        self.summary = body[:120].replace("\n", " ")

    def mark_partial(self, body: str) -> None:
        self.status = "partial"
        self.actual_chars = len(body)
        self.summary = body[:120].replace("\n", " ") if body else ""


@dataclass
class SubtaskManifest:
    stage_goal: str
    file_path: str
    file_type: Literal["doc", "code"]
    items: list[SubtaskItem] = field(default_factory=list)
    file_content_hash: str = ""
    created_at: float = field(default_factory=time.time)

    @property
    def current_item(self) -> Optional[SubtaskItem]:
        for it in self.items:
            if it.status in ("pending", "in_progress", "partial", "missing", "regressed"):
                return it
        return None

    @property
    def done_count(self) -> int:
        return sum(1 for it in self.items if it.status == "complete")

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def all_done(self) -> bool:
        return all(it.status in ("complete", "partial") for it in self.items)

    @property
    def progress_line(self) -> str:
        cur = self.current_item
        cur_label = cur.title if cur else "done"
        icons = " ".join(
            "✓" if it.status == "complete"
            else "~" if it.status == "partial"
            else "✗" if it.status in ("missing", "regressed")
            else "▶" if it.status == "in_progress"
            else "·"
            for it in self.items
        )
        return f"[{self.done_count}/{self.total}] {icons}  now: {cur_label}"

    def update_file_hash(self, content: str) -> None:
        self.file_content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class SubtaskDiff:
    complete: list[SubtaskItem] = field(default_factory=list)
    partial: list[SubtaskItem] = field(default_factory=list)
    missing: list[SubtaskItem] = field(default_factory=list)
    regressed: list[SubtaskItem] = field(default_factory=list)
    seam_index: int = 0
    resume_context: str = ""

    @property
    def needs_resume(self) -> bool:
        return bool(self.partial or self.missing or self.regressed)

    @property
    def summary(self) -> str:
        parts = []
        if self.missing:
            parts.append(f"missing: {', '.join(i.title for i in self.missing)}")
        if self.partial:
            parts.append(f"partial: {', '.join(i.title for i in self.partial)}")
        if self.regressed:
            parts.append(f"regressed: {', '.join(i.title for i in self.regressed)}")
        return "; ".join(parts) if parts else "ok"
