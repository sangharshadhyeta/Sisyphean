"""Anthropic Messages API compatibility layer.

Request path (new tool_use design):
  Claude Code → POST /v1/messages → handle_messages()
    → MemoryInjector builds memory context
    → TranslationLoop.process() decides next action
    → Returns thinking blocks + tool_use blocks (stop_reason="tool_use")
    → Claude Code executes tools, sends tool_result back
    → Loop continues until stop_reason="end_turn"
    → MemoryExtractor saves new facts

The inner loop state is encoded in thinking blocks as SISYPHEAN_STATE:<json>.
Claude Code manages the conversation history; the server is fully stateless.

The full loop is visible in Claude Code's UI:
  - Thinking blocks show reasoning and plan state
  - Tool use blocks show what tools were called and why
  - Text blocks show final answers
"""
from __future__ import annotations

import asyncio
import json
import re
import uuid
from typing import Any, AsyncIterator

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from engine.llm.client import LlamaClient
from engine.llm.context import ContextManager
from engine.memory.injector import MemoryInjector
from engine.memory.extractor import MemoryExtractor
from engine.memory.session_log import get_session_log
from engine.translation.loop import TranslationLoop


# ── Request models ────────────────────────────────────────────────────────────

class ContentBlock(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str
    text: str | None = None
    # thinking blocks
    thinking: str | None = None
    # tool_use blocks
    id: str | None = None
    name: str | None = None
    input: dict | None = None
    # tool_result blocks
    tool_use_id: str | None = None
    content: Any | None = None


class AnthropicMessage(BaseModel):
    role: str
    content: str | list[ContentBlock]


class Tool(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    description: str | None = None
    input_schema: dict | None = None


_MAX_TOKENS_LIMIT = 32768  # hard cap — prevents accidental OOM from malformed requests


class AnthropicRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    messages: list[AnthropicMessage]
    max_tokens: int = 1024
    system: str | list[ContentBlock] | None = None
    temperature: float = 0.7
    stream: bool = False
    stop_sequences: list[str] | None = None
    tools: list[Tool] | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.max_tokens > _MAX_TOKENS_LIMIT:
            self.max_tokens = _MAX_TOKENS_LIMIT

    def system_text(self) -> str | None:
        """Return system as a plain string regardless of input format."""
        if self.system is None:
            return None
        if isinstance(self.system, str):
            return self.system
        return "".join(b.text or "" for b in self.system if b.type == "text")


# ── Format helpers ────────────────────────────────────────────────────────────

_SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL | re.IGNORECASE)
_STATE_PREFIX = "SISYPHEAN_STATE:"
_PIPELINE_STATE_PREFIX = "PIPELINE_STATE:"


def _strip_stale_states(history: list[dict]) -> list[dict]:
    """Remove state-carrying thinking blocks from all messages in history.

    Both SISYPHEAN_STATE (legacy) and PIPELINE_STATE blocks are stripped for
    fresh requests — they are only valid during an active tool_result continuation
    and cause false continuation triggers if left in history.
    """
    cleaned = []
    for msg in history:
        content = msg.get("content", "")
        if not isinstance(content, list):
            cleaned.append(msg)
            continue
        new_content = [
            b for b in content
            if not (
                isinstance(b, dict)
                and b.get("type") == "thinking"
                and (
                    _STATE_PREFIX in b.get("thinking", "")
                    or _PIPELINE_STATE_PREFIX in b.get("thinking", "")
                )
            )
        ]
        cleaned.append({**msg, "content": new_content} if new_content != content else msg)
    return cleaned

def _flatten_content(content: str | list[ContentBlock]) -> str:
    if isinstance(content, str):
        text = content
    else:
        text = "".join(b.text or "" for b in content if b.type == "text")
    # Strip <system-reminder> injection blocks that Claude Code appends to
    # user messages — they are meta-instructions for Claude, not the user's task.
    return _SYSTEM_REMINDER_RE.sub("", text).strip()


def _last_user_message(req: AnthropicRequest) -> str:
    for m in reversed(req.messages):
        if m.role == "user":
            return _flatten_content(m.content)
    return ""


def _messages_to_dicts(messages: list[AnthropicMessage]) -> list[dict]:
    """Convert Pydantic AnthropicMessage list to plain dicts with typed content blocks."""
    result = []
    for m in messages:
        if isinstance(m.content, str):
            block_content = [{"type": "text", "text": m.content}]
        else:
            block_content = [
                b.model_dump(exclude_none=True) for b in m.content
            ]
        result.append({"role": m.role, "content": block_content})
    return result


def _tools_to_dicts(tools: list[Tool] | None) -> list[dict]:
    if not tools:
        return []
    return [t.model_dump(exclude_none=True) for t in tools]


def _is_tool_result_message(messages: list[AnthropicMessage]) -> bool:
    """True if the last message contains tool_result blocks (mid-task continuation)."""
    if not messages:
        return False
    last = messages[-1]
    if last.role != "user":
        return False
    content = last.content
    if isinstance(content, str):
        return False
    return any(
        isinstance(b, ContentBlock) and b.type == "tool_result"
        for b in content
    )


def _first_user_turn_key(messages: list[AnthropicMessage]) -> str:
    """Derive a stable session key from the first user message content.

    Claude Code resends the full history on every turn, so the first user
    message is constant throughout a session — a good stable key.
    Truncated and sanitised to be filesystem-safe.
    """
    for m in messages:
        if m.role == "user":
            text = _flatten_content(m.content)[:48].strip()
            # Replace characters that aren't alphanumeric/hyphen/underscore
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in text)
            return safe[:40] or "session"
    return "session"


def _map_finish_reason(oai_reason: str | None) -> str:
    return {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }.get(oai_reason or "", "end_turn")


# ── System prompt translation ─────────────────────────────────────────────────

def _extract_claude_md(system: str) -> str:
    """Extract CLAUDE.md project context from Claude Code's system prompt.

    Claude Code injects CLAUDE.md as:
      "Contents of /path/CLAUDE.md:\n<content>"
    or inside XML-like tags. We extract the content block and cap it so the
    small model gets project context without drowning in boilerplate.
    """
    # Pattern 1: "Contents of /path/CLAUDE.md:\n..." (most common)
    m = re.search(
        r"Contents of [^\n]*CLAUDE\.md[^\n]*:\n(.*?)(?=\nContents of |\Z)",
        system, re.DOTALL | re.IGNORECASE,
    )
    if m:
        content = m.group(1).strip()
        # Strip the file header line if present ("# CLAUDE.md\n...")
        content = re.sub(r"^#\s*CLAUDE\.md\s*\n", "", content).strip()
        return content[:2500]

    # Pattern 2: inside <claude_md> or similar tags
    m = re.search(r"<claude[_\-]?md>(.*?)</claude[_\-]?md>", system, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()[:2500]

    return ""


def translate_system(system: str | None) -> str:
    """Extract useful context from a large system prompt (e.g. Claude Code's).

    Returns two parts joined by newlines:
      1. Env facts line: "cwd: ... | os: ..."
      2. Project context block from CLAUDE.md (Option B)

    Everything else in Claude Code's system prompt (its own instructions,
    tool schemas, meta-instructions) is discarded — we replace those with
    our own filtered tool menus and soul sections.
    """
    if not system:
        return ""

    parts: list[str] = []

    # ── Env facts ─────────────────────────────────────────────────────────────
    facts: list[str] = []

    for pattern in (
        r"<cwd>(.*?)</cwd>",
        r"current working directory[:\s]+([^\n]+)",
        r"cwd[:\s]+([^\n,;]+)",
        r"working directory[:\s]+([^\n]+)",
    ):
        m = re.search(pattern, system, re.IGNORECASE)
        if m:
            cwd = m.group(1).strip()
            if cwd:
                cwd_fwd = cwd.replace("\\", "/")
                cwd_fwd = re.sub(r"^/([a-zA-Z])/", lambda x: f"{x.group(1).upper()}:/", cwd_fwd)
                facts.append(f"cwd: {cwd_fwd}")
            break

    m = re.search(r"git branch[:\s]+([^\n]+)", system, re.IGNORECASE)
    if m:
        facts.append(f"branch: {m.group(1).strip()}")

    m = re.search(r"<(?:platform|os)>(.*?)</(?:platform|os)>", system, re.IGNORECASE)
    if m:
        facts.append(f"os: {m.group(1).strip()}")

    if facts:
        parts.append(" | ".join(facts))

    # ── Project context (CLAUDE.md) — Option B ────────────────────────────────
    claude_md = _extract_claude_md(system)
    if claude_md:
        parts.append(f"[Project context]\n{claude_md}")

    return "\n\n".join(parts)


# ── Response builders ─────────────────────────────────────────────────────────

def _sync_response(content: list[dict], model: str, msg_id: str, stop_reason: str) -> dict:
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 0,
            "output_tokens": sum(
                len((b.get("text") or b.get("thinking") or json.dumps(b.get("input") or {})).split())
                for b in content
            ),
        },
    }


def _sse(event: str, data: Any) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


async def _stream_loop_response(
    content_blocks: list[dict],
    stop_reason: str,
    model: str,
    msg_id: str,
) -> AsyncIterator[bytes]:
    """Stream LoopResponse content blocks as Anthropic SSE events."""
    yield _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant",
            "content": [], "model": model,
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    output_tokens = 0

    for idx, block in enumerate(content_blocks):
        btype = block.get("type")

        if btype == "thinking":
            thinking_text = block.get("thinking", "")
            yield _sse("content_block_start", {
                "type": "content_block_start", "index": idx,
                "content_block": {"type": "thinking", "thinking": ""},
            })
            # Stream thinking in chunks of ~80 chars
            for i in range(0, len(thinking_text), 80):
                chunk = thinking_text[i:i + 80]
                output_tokens += 1
                yield _sse("content_block_delta", {
                    "type": "content_block_delta", "index": idx,
                    "delta": {"type": "thinking_delta", "thinking": chunk},
                })
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": idx})

        elif btype == "tool_use":
            tool_id = block.get("id", f"toolu_{uuid.uuid4().hex[:16]}")
            tool_name = block.get("name", "")
            tool_input = block.get("input", {})
            yield _sse("content_block_start", {
                "type": "content_block_start", "index": idx,
                "content_block": {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tool_name,
                    "input": {},
                },
            })
            input_json = json.dumps(tool_input)
            output_tokens += 1
            yield _sse("content_block_delta", {
                "type": "content_block_delta", "index": idx,
                "delta": {"type": "input_json_delta", "partial_json": input_json},
            })
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": idx})

        elif btype == "text":
            text = block.get("text", "")
            yield _sse("content_block_start", {
                "type": "content_block_start", "index": idx,
                "content_block": {"type": "text", "text": ""},
            })
            # Stream text word by word
            words = text.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                if chunk:
                    output_tokens += 1
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta", "index": idx,
                        "delta": {"type": "text_delta", "text": chunk},
                    })
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": idx})

    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })
    yield _sse("message_stop", {"type": "message_stop"})


# ── Route handler ─────────────────────────────────────────────────────────────

import logging as _logging
import os as _os
_req_logger = _logging.getLogger("sisyphean.request_dump")

def _dump_request(req: AnthropicRequest) -> None:
    """Write full incoming request to a file for inspection."""
    dump_path = _os.path.join(_os.path.dirname(__file__), "..", "..", "last_request.json")
    try:
        import json as _json
        payload = {
            "system": req.system_text(),
            "tools": [t.model_dump(exclude_none=True) for t in (req.tools or [])],
            "messages": [
                {
                    "role": m.role,
                    "content": m.content if isinstance(m.content, str)
                               else [b.model_dump(exclude_none=True) for b in m.content],
                }
                for m in req.messages
            ],
        }
        with open(dump_path, "w", encoding="utf-8") as f:
            _json.dump(payload, f, indent=2, ensure_ascii=False)
        _req_logger.info("Request dumped to last_request.json (%d msgs, %d tools, sys=%d chars)",
                         len(req.messages), len(req.tools or []),
                         len(req.system_text() or ""))
    except Exception as e:
        _req_logger.warning("Failed to dump request: %s", e)


async def handle_messages(
    req: AnthropicRequest,
    client: LlamaClient,
    ctx: ContextManager,
    injector: MemoryInjector | None = None,
    extractor: MemoryExtractor | None = None,
    translation_loop: TranslationLoop | None = None,
    soul_text: str = "",
    knowledge_graph=None,
    http_request: Request | None = None,
) -> JSONResponse | StreamingResponse:
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    user_message = _last_user_message(req)
    is_continuation = _is_tool_result_message(req.messages)

    # Dump the first non-continuation request for inspection
    if not is_continuation and user_message:
        _dump_request(req)

    # Ignore empty messages and Claude Code slash-command artefacts (/clear etc.)
    # Tool-result continuations always have empty user text — never skip them.
    if not is_continuation and (not user_message or user_message.startswith("/")):
        empty = [{"type": "text", "text": ""}]
        if req.stream:
            return StreamingResponse(
                _stream_loop_response(empty, "end_turn", req.model, msg_id),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return JSONResponse(_sync_response(empty, req.model, msg_id, "end_turn"))

    raw_history = _messages_to_dicts(req.messages)
    available_tools = _tools_to_dicts(req.tools)
    system_context = translate_system(req.system_text())

    # Strip stale SISYPHEAN_STATE thinking blocks for fresh requests.
    # They are only meaningful during an active tool_result continuation;
    # for new questions they cause false continuation triggers.
    if not is_continuation:
        raw_history = _strip_stale_states(raw_history)

    # ── Session log ───────────────────────────────────────────────────────────
    session_key = _first_user_turn_key(req.messages)
    slog = get_session_log(session_key)
    if user_message:  # continuations have empty user_message — don't log duplicates
        slog.user_message(user_message)

    # Run translation loop — it builds static context itself via _build_static_context
    if translation_loop:
        loop_task = asyncio.ensure_future(translation_loop.process(
            user_message=user_message,
            raw_history=raw_history,
            available_tools=available_tools,
            system_context=system_context,
        ))

        async def _watch_disconnect():
            while True:
                await asyncio.sleep(0.5)
                if http_request and await http_request.is_disconnected():
                    return True
                if loop_task.done():
                    return False

        watcher = asyncio.ensure_future(_watch_disconnect())
        done, _ = await asyncio.wait([loop_task, watcher], return_when=asyncio.FIRST_COMPLETED)

        if watcher in done and watcher.result():
            # Client disconnected — cancel the loop and mark any running tracker task as failed
            loop_task.cancel()
            try:
                import engine.task_tracker as _tracker
                _tracker._expire_stale(max_age=0)
            except Exception:
                pass
            return JSONResponse({"error": "cancelled"}, status_code=499)

        watcher.cancel()
        try:
            loop_resp = await loop_task
        except asyncio.CancelledError:
            return JSONResponse({"error": "cancelled"}, status_code=499)

        # Fire-and-forget memory extraction for end_turn responses
        if loop_resp.stop_reason == "end_turn":
            final_texts = [
                b.get("text", "") for b in loop_resp.content
                if b.get("type") == "text"
            ]
            final_text = "\n".join(final_texts).strip()
            if final_text:
                slog.assistant_message(final_text)
                if extractor:
                    extractor.extract_later(user_message, final_text)

        if req.stream:
            return StreamingResponse(
                _stream_loop_response(
                    loop_resp.content,
                    loop_resp.stop_reason,
                    req.model,
                    msg_id,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        return JSONResponse(
            _sync_response(loop_resp.content, req.model, msg_id, loop_resp.stop_reason)
        )

    # Fallback: direct passthrough (no translation loop configured)
    flat_history = [
        {"role": m.role, "content": _flatten_content(m.content)}
        for m in req.messages[:-1]
    ]
    system_str = req.system_text()
    if system_str:
        flat_history = [{"role": "system", "content": system_str}] + flat_history

    messages = await ctx.fit(
        flat_history + [{"role": "user", "content": user_message}],
        system=system_str,
    )
    result = await client.generate(
        messages, stream=False,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        stop=req.stop_sequences,
        thinking=False,  # fallback passthrough — 0.6b gains nothing from thinking
    )
    text = result["choices"][0]["message"].get("content") or ""
    if extractor:
        await extractor.extract(user_message, text)
    return JSONResponse(
        _sync_response(
            [{"type": "text", "text": text}],
            req.model, msg_id, "end_turn",
        )
    )
