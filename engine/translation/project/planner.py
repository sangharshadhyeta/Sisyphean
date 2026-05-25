"""Project planner — one LLM call that breaks a project description into an
ordered list of files to create.

Each file becomes a separate write_code stage in the pipeline.
Files are returned in dependency order (dependencies first) so later files
can import from earlier ones.

Context budget: prompt + response < 600 tokens — fits qwen3:0.6b.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

_PROJECT_SYSTEM = """\
You are a project architect. List the files needed to build the described project.

Rules:
- Output ONLY valid JSON. No explanation.
- List files in dependency order (utility/model files first, entry point last).
- Each file's "purpose" must name the specific functions or classes it will contain.
- 3-6 files maximum. Keep the structure minimal and buildable.
- Use short filenames without subdirectories.

Output format:
{"files": [
  {"filename": "models.py", "purpose": "Todo dataclass: id, title, done; save_todos, load_todos"},
  {"filename": "api.py", "purpose": "FastAPI app: create_todo, list_todos, delete_todo routes"},
  {"filename": "main.py", "purpose": "entry point: start uvicorn on port 8000"}
]}"""


# ── Public API ────────────────────────────────────────────────────────────────

async def plan_project(
    client: Any,
    task: str,
    workspace: str = "",
) -> list[dict]:
    """Return an ordered list of {filename, purpose} dicts for the project.

    Falls back gracefully — either a single generic file or two generic files
    so the caller always has something to iterate over.
    """
    prompt = f"Project: {task[:400]}"

    messages = [
        {"role": "system", "content": _PROJECT_SYSTEM},
        {"role": "user",   "content": prompt},
    ]
    try:
        result = await client.generate(
            messages,
            max_tokens=350,
            temperature=0.2,
            response_format={"type": "json_object"},
            stream=False,
            thinking=False,
        )
        raw = result["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.warning("project_planner: LLM error: %s", exc)
        return _fallback(task)

    parsed = _parse(raw)
    if len(parsed) < 2:
        logger.warning(
            "project_planner: got %d file(s) — falling back. raw: %.120s",
            len(parsed), raw,
        )
        return _fallback(task)

    logger.info("project_planner: %d files for %r", len(parsed), task[:60])
    return parsed


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse(raw: str) -> list[dict]:
    """Extract the files list from a JSON response."""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end <= start:
        return []
    try:
        data  = json.loads(raw[start:end])
        files = data.get("files", [])
        if not isinstance(files, list):
            return []
        result = []
        for f in files[:6]:
            filename = str(f.get("filename", "")).strip()
            purpose  = str(f.get("purpose",  "")).strip()
            # Must look like a real filename (has a dot extension)
            if filename and re.search(r'\.[a-z]{2,5}$', filename):
                result.append({"filename": filename, "purpose": purpose})
        return result
    except json.JSONDecodeError as exc:
        logger.warning("project_planner: JSON error: %s — raw: %.120s", exc, raw)
        return []


def _fallback(task: str) -> list[dict]:
    """Two-file fallback when planning fails."""
    # Extract a Python filename from the task, or default to "app.py"
    m = re.search(r'\b([a-z][a-z0-9_]+\.py)\b', task, re.IGNORECASE)
    main_file = m.group(1) if m else "app.py"
    stem = main_file.rsplit(".", 1)[0]
    return [
        {"filename": f"{stem}_models.py", "purpose": f"data models for {stem}"},
        {"filename": main_file,            "purpose": task[:80]},
    ]
