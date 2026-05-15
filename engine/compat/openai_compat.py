"""OpenAI Chat Completions compatibility layer.

llama-server already speaks the OpenAI format, so this is a thin pass-through
that routes through our LlamaClient (giving us health-check, retry, and context
management) rather than hitting llama-server directly.

Endpoint: POST /v1/chat/completions
"""
from __future__ import annotations

from typing import AsyncIterator

from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from engine.llm.client import LlamaClient
from engine.llm.context import ContextManager


# ── Request model ─────────────────────────────────────────────────────────────

class OAIMessage(BaseModel):
    role: str
    content: str | None = None


class OAIRequest(BaseModel):
    model: str
    messages: list[OAIMessage]
    max_tokens: int | None = 1024
    temperature: float | None = 0.7
    stream: bool | None = False
    stop: list[str] | str | None = None


# ── Streaming wrapper ─────────────────────────────────────────────────────────

async def _oai_sse_stream(oai_iter: AsyncIterator[str]) -> AsyncIterator[bytes]:
    async for raw in oai_iter:
        yield f"data: {raw}\n\n".encode()
    yield b"data: [DONE]\n\n"


# ── Route handler ─────────────────────────────────────────────────────────────

async def handle_chat_completions(
    req: OAIRequest,
    client: LlamaClient,
    ctx: ContextManager,
) -> JSONResponse | StreamingResponse:
    messages = [m.model_dump(exclude_none=True) for m in req.messages]
    messages = await ctx.fit(messages)

    stop = [req.stop] if isinstance(req.stop, str) else req.stop

    if req.stream:
        oai_stream = await client.generate(
            messages,
            stream=True,
            max_tokens=req.max_tokens or 1024,
            temperature=req.temperature or 0.7,
            stop=stop,
            thinking=False,
        )
        return StreamingResponse(
            _oai_sse_stream(oai_stream),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        result = await client.generate(
            messages,
            stream=False,
            max_tokens=req.max_tokens or 1024,
            temperature=req.temperature or 0.7,
            stop=stop,
            thinking=False,
        )
    except Exception as exc:
        import httpx as _httpx
        status = 502
        if isinstance(exc, _httpx.HTTPStatusError):
            status = exc.response.status_code
        return JSONResponse(
            {"error": {"type": "upstream_error", "message": str(exc)}},
            status_code=status,
        )

    # Plain-text rescue: if content is empty, promote answer from reasoning fields.
    # The Anthropic path intentionally skips plain-text rescue (to avoid leaking
    # reasoning chains), but the OAI compat endpoint is a direct passthrough —
    # the caller just wants a string answer.
    for choice in result.get("choices", []):
        msg = choice.get("message")
        if isinstance(msg, dict) and not (msg.get("content") or "").strip():
            for field in ("reasoning", "reasoning_content", "thinking"):
                val = (msg.get(field) or "").strip()
                if val:
                    msg["content"] = val
                    break

    return JSONResponse(result)
