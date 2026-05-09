"""Per-stage step budget tracking — ported from BirdClaw agent/budget.py.

Learns realistic budgets from empirical run history (P75 of past step counts).
Falls back to config defaults on first run or unknown stage types.

Key change vs BirdClaw: history stored in Sisyphean's memory path,
not ~/.birdclaw/. Path injected at init so there's no global config dependency.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from engine.translation.prompts import STAGE_BUDGETS

logger = logging.getLogger(__name__)


class BudgetTracker:

    def __init__(self, memory_path: Path) -> None:
        self._history_path = Path(memory_path) / "stage_history.jsonl"

    def get(self, stage_type: str) -> int:
        """Return P75 step count for stage_type from history, or default.

        Requires ≥3 samples before trusting empirical data.
        Caps at 200 to prevent runaway loops.
        """
        default = STAGE_BUDGETS.get(stage_type, 10)
        if not self._history_path.exists():
            return default

        samples: list[int] = []
        try:
            for line in self._history_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("type") == stage_type and isinstance(rec.get("steps"), int):
                    samples.append(rec["steps"])
        except Exception as exc:
            logger.debug("stage_history read failed: %s", exc)
            return default

        if len(samples) < 3:
            return default

        samples.sort()
        p75_idx = int(len(samples) * 0.75)
        p75 = samples[min(p75_idx, len(samples) - 1)]
        return max(default, min(p75, 200))

    def log(self, stage_type: str, steps_taken: int, goal: str) -> None:
        """Append a completion record — feeds future get() calls."""
        try:
            self._history_path.parent.mkdir(parents=True, exist_ok=True)
            rec = {
                "type": stage_type,
                "steps": steps_taken,
                "goal_len": len(goal),
                "ts": time.time(),
            }
            with open(self._history_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        except Exception as exc:
            logger.debug("stage_history write failed: %s", exc)
