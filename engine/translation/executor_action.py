"""Action and StageResult dataclasses for the executor."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from engine.translation.planner import Stage


@dataclass
class Action:
    type: str           # "tool" | "answer"
    reasoning: str = ""
    tool_name: str = ""
    tool_id: str = field(default_factory=lambda: f"toolu_{uuid.uuid4().hex[:16]}")
    tool_input: dict = field(default_factory=dict)
    content: str = ""   # final answer text when type="answer"


@dataclass
class StageResult:
    stage: Stage
    output: str
    steps_taken: int
    searches_done: list[str] = field(default_factory=list)
