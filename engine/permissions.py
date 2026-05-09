"""Permission guard — protected path and dangerous command detection.

Design intent:
  - Claude Code handles actual tool execution and shows its own permission prompts.
  - This guard adds an ENGINE-LAYER warning on top: when the agent tries to touch
    a protected file or run a dangerous bash command, the loop prepends a visible
    warning text block to the response BEFORE the tool_use block.
  - The user sees the warning in the Claude Code UI, Claude Code still asks for
    permission, and the user makes the final call.
  - Critical during self-update: the engine cannot overwrite its own core files
    without explicit user approval surfaced through both layers.

Why not silently block (BirdClaw's failure mode)?
  Silent blocking left the agent confused — it thought it wrote the file but
  nothing happened. Showing the warning + still forwarding the tool_use means:
    1. The user understands WHY the agent paused.
    2. Claude Code's own permission prompt fires.
    3. If the user approves, execution proceeds normally.
    4. If the user denies, the agent gets a clear tool_result explaining why.
"""
from __future__ import annotations

import fnmatch
import json
import logging
import re

logger = logging.getLogger(__name__)

# These are always protected regardless of config — the engine must never
# overwrite its own compatibility layer or startup code during self-update.
_HARDCODED_PROTECTED: list[str] = [
    "engine/compat/**",
    "engine/api/app.py",
    "engine/__init__.py",
    "main.py",
]

# Tool names that write files — checked against protected paths
_WRITE_TOOL_NAMES: frozenset[str] = frozenset({
    "write_file", "str_replace_editor", "create_file",
    "text_editor", "edit_file",
})

# Input keys that may contain a file path
_PATH_KEYS: tuple[str, ...] = ("path", "file_path", "filename", "target", "dest")


class PermissionGuard:
    """Checks tool actions against protected paths and dangerous commands.

    Usage:
        guard = PermissionGuard.from_config(config.permissions)
        warning = guard.check(action.tool_name, action.tool_input)
        if warning:
            # prepend warning text block before the tool_use block
    """

    def __init__(
        self,
        protected_paths: list[str] | None = None,
        dangerous_commands: list[str] | None = None,
    ) -> None:
        self._protected = list(_HARDCODED_PROTECTED) + (protected_paths or [])
        self._dangerous = dangerous_commands or []

    @classmethod
    def from_config(cls, cfg) -> "PermissionGuard":
        return cls(
            protected_paths=getattr(cfg, "protected_paths", []),
            dangerous_commands=getattr(cfg, "dangerous_commands", []),
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def check(self, tool_name: str, tool_input: dict) -> str | None:
        """Return a warning string if the action needs user attention, else None.

        The caller should prepend this as a text block before the tool_use block.
        The tool_use itself is NOT blocked — that's Claude Code's job.
        """
        if tool_name in _WRITE_TOOL_NAMES:
            path = self._extract_path(tool_input)
            if path:
                pattern = self._matches_protected(path)
                if pattern:
                    return (
                        f"**[PROTECTED FILE] Write blocked pending approval**\n\n"
                        f"The agent is attempting to write to: `{path}`\n"
                        f"This path matches protected pattern: `{pattern}`\n\n"
                        f"Core engine files should not be modified without explicit approval. "
                        f"Review the change below before allowing."
                    )

        if tool_name == "bash":
            command = tool_input.get("command", "")
            hit = self._matches_dangerous(command)
            if hit:
                return (
                    f"**[DANGEROUS COMMAND] Review before allowing**\n\n"
                    f"The agent wants to run a bash command matching: `{hit}`\n"
                    f"Command preview: `{command[:120]}`\n\n"
                    f"Review carefully before allowing."
                )

        return None

    # ── Internal ───────────────────────────────────────────────────────────────

    def _extract_path(self, tool_input: dict) -> str | None:
        for key in _PATH_KEYS:
            val = tool_input.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        # str_replace_editor / text_editor pass content inside a nested dict
        content = tool_input.get("content") or tool_input.get("new_str") or ""
        if isinstance(content, dict):
            for key in _PATH_KEYS:
                val = content.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
        return None

    def _matches_protected(self, path: str) -> str | None:
        # Normalise: strip leading ./ and convert backslashes
        normalised = path.lstrip("./").replace("\\", "/")
        for pattern in self._protected:
            if fnmatch.fnmatch(normalised, pattern) or fnmatch.fnmatch(path, pattern):
                return pattern
        return None

    def _matches_dangerous(self, command: str) -> str | None:
        low = command.lower()
        for pattern in self._dangerous:
            if pattern.lower() in low:
                return pattern
        return None
