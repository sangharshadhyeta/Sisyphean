from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import AsyncIterator

import httpx

logger = logging.getLogger(__name__)


class LlamaClient:
    """Async HTTP client for llama-server's OpenAI-compatible API.

    Three modes:
      1. mock=True          — smart context-aware mock; no network, exercises
                              full engine pipeline with realistic responses.
      2. api_key set        — external OpenAI-compatible provider (OpenRouter,
                              Groq, Google AI Studio, etc.).  base_url should
                              be the provider root; model is added to every
                              payload.
      3. default            — local llama-server at base_url.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 600.0,
        mock: bool = False,
        api_key: str = "",
        model: str = "",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.mock = mock
        self.api_key = api_key
        self.model = model

        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(connect=10.0, read=timeout, write=30.0, pool=10.0),
            headers=headers,
        )

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def close(self) -> None:
        await self._http.aclose()

    # ── Health ───────────────────────────────────────────────────────────────

    async def health_check(
        self,
        retries: int = 30,
        interval: float = 2.0,
        ollama: bool = False,
    ) -> bool:
        """Poll backend until healthy. Returns False on timeout.

        ollama=True  — tries Ollama's endpoints (GET / returns "Ollama is running",
                        or GET /api/health returns 200).
        ollama=False — polls llama-server's GET /health for {"status": "ok"}.
        """
        if self.mock or self.api_key:
            logger.info("Skipping backend health check (mock=%s, external=%s)",
                        self.mock, bool(self.api_key))
            return True

        for attempt in range(retries):
            try:
                if ollama:
                    # Try /api/health first (Ollama >=0.1.47), fall back to /
                    try:
                        r = await self._http.get("/api/health")
                        if r.status_code == 200:
                            logger.info("Ollama is ready (/api/health)")
                            return True
                    except Exception:
                        pass
                    r = await self._http.get("/")
                    if r.status_code == 200 and "ollama" in r.text.lower():
                        logger.info("Ollama is ready (/)")
                        return True
                else:
                    r = await self._http.get("/health")
                    if r.status_code == 200 and r.json().get("status") == "ok":
                        logger.info("llama-server is ready")
                        return True
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError):
                pass

            if attempt < retries - 1:
                backend = "Ollama" if ollama else "llama-server"
                logger.debug("Waiting for %s (%d/%d)...", backend, attempt + 1, retries)
                await asyncio.sleep(interval)

        backend = "Ollama" if ollama else "llama-server"
        logger.error("%s did not become ready after %d attempts", backend, retries)
        return False

    # ── Token counting ───────────────────────────────────────────────────────

    async def tokenize(self, text: str) -> int:
        """Return the token count for *text*. Falls back to len/4 approximation."""
        if self.mock or self.api_key:
            return max(1, len(text) // 4)
        try:
            r = await self._http.post("/tokenize", json={"content": text})
            if r.status_code == 404:
                # Ollama doesn't expose /tokenize — silently use approximation
                return max(1, len(text) // 4)
            r.raise_for_status()
            return len(r.json().get("tokens", []))
        except Exception:
            return max(1, len(text) // 4)

    # ── Generation ───────────────────────────────────────────────────────────

    async def generate(
        self,
        messages: list[dict],
        *,
        stream: bool = False,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: list[str] | None = None,
        response_format: dict | None = None,
        thinking: bool = False,
    ) -> dict | AsyncIterator[str]:
        """Generate a completion.

        - stream=False  → awaitable, returns the full response dict.
        - stream=True   → returns an async iterator of raw JSON chunk strings.
        - response_format → e.g. {"type": "json_object"} for structured output.
        """
        if self.mock:
            if stream:
                return self._mock_stream(messages)
            return await self._mock_sync(messages)

        # External APIs (OpenRouter, Groq, etc.) don't accept Anthropic-specific
        # content block types like "thinking" or "tool_use" in message history.
        # Strip them down to plain text before forwarding.
        if self.api_key:
            messages = _flatten_messages_for_oai(messages)

        payload: dict = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if self.model:
            payload["model"] = self.model
        if stop:
            payload["stop"] = stop
        # response_format (json_object) is only supported by some models/providers.
        # For external APIs we skip it — the engine's _parse_json() is tolerant
        # enough to handle freeform JSON embedded in prose.
        if response_format and not self.api_key:
            payload["response_format"] = response_format
        # thinking=True  → free-form answer calls (_direct) where reasoning helps
        # thinking=False → all structured JSON calls (soul, decompose, executor)
        #                  Ollama/llama.cpp: thinking + response_format=json_object
        #                  produces reasoning-only output with no JSON — must be False.
        # Ollama honours "think": bool on /v1/chat/completions since v0.9.
        if not self.api_key and not self.mock:
            payload["think"] = thinking

        if stream:
            return self._stream(payload)
        return await self._sync(payload)

    # ── Internal: non-streaming ──────────────────────────────────────────────

    @property
    def _completions_path(self) -> str:
        """Endpoint path for chat completions.

        External APIs (api_key set) have base_url that already contains /v1,
        e.g. https://openrouter.ai/api/v1.  httpx joins base_url + relative
        path correctly: base/v1 + chat/completions = base/v1/chat/completions.

        Local llama-server has base_url = http://host:port (no /v1), so we
        need the full path /v1/chat/completions.
        """
        return "chat/completions" if self.api_key else "/v1/chat/completions"

    async def _sync(self, payload: dict, _attempt: int = 0) -> dict:
        try:
            r = await self._http.post(self._completions_path, json=payload)
            # Retry on rate-limit (429) and transient upstream errors
            if r.status_code in (429, 502, 503, 504) and _attempt < 4:
                wait = min(2 ** _attempt * 3, 30)   # 3 → 6 → 12 → 24s
                logger.warning(
                    "LLM HTTP %s — retrying in %ds (attempt %d/4)",
                    r.status_code, wait, _attempt + 1,
                )
                await asyncio.sleep(wait)
                return await self._sync(payload, _attempt + 1)
            r.raise_for_status()
            data = r.json()
            _strip_thinking(data)
            return data
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
            if _attempt >= 2:
                raise
            wait = 2 ** _attempt
            logger.warning("LLM request failed (%s), retrying in %ds", exc, wait)
            await asyncio.sleep(wait)
            return await self._sync(payload, _attempt + 1)

    # ── Internal: streaming ──────────────────────────────────────────────────

    async def _stream(self, payload: dict) -> AsyncIterator[str]:
        for attempt in range(3):
            try:
                async with self._http.stream(
                    "POST", self._completions_path, json=payload
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        if line == "data: [DONE]":
                            return
                        if line.startswith("data: "):
                            yield line[6:]
                return
            except (httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                if attempt >= 2:
                    raise
                wait = 2 ** attempt
                logger.warning("Stream connect failed (%s), retrying in %ds", exc, wait)
                await asyncio.sleep(wait)

    # ── Smart mock ───────────────────────────────────────────────────────────

    async def _mock_sync(self, messages: list[dict]) -> dict:
        content = self._mock_decide(messages)
        return {
            "id": "mock-001",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
        }

    async def _mock_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        text = self._mock_decide(messages)
        words = text.split()
        for word in words:
            chunk = {
                "id": "mock-stream-001",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": word + " "}, "finish_reason": None}],
            }
            yield json.dumps(chunk)
            await asyncio.sleep(0.02)
        done = {
            "id": "mock-stream-001",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield json.dumps(done)

    def _mock_decide(self, messages: list[dict]) -> str:  # noqa: C901
        """Context-aware mock response.

        Inspects the message contents to detect which engine component is
        calling, then returns a realistic response that exercises the full
        pipeline — decomposer, executor, soul router, condenser — without
        needing a real model.
        """
        user_content = " ".join(
            str(m.get("content") or "") for m in messages if m.get("role") == "user"
        )
        all_content = " ".join(str(m.get("content") or "") for m in messages)

        # ── 1. Soul routing ──────────────────────────────────────────────────
        # Prompt contains: {"action": "task|remember", ...}
        if "task|remember" in user_content:
            msg_match = re.search(r"User message:\s*(.+?)(?:\n|$)", user_content)
            if msg_match:
                msg = msg_match.group(1).strip()
                if re.search(
                    r"\b(remember (that|i|this)|always (use|do)|my name is|i prefer|note that|store that)\b",
                    msg,
                    re.I,
                ):
                    note = re.sub(r"^(remember (that|this|:)\s*)+", "", msg, flags=re.I).strip()
                    return json.dumps({"action": "remember", "note": note[:120]})
            return json.dumps({"action": "task"})

        # ── 2. Remember acknowledgement ──────────────────────────────────────
        if "You have stored this fact" in user_content:
            fact_match = re.search(r"You have stored this fact:\s*(.+?)(?:\n|$)", user_content)
            fact = fact_match.group(1).strip() if fact_match else "your preference"
            return f"Got it — I've remembered that {fact}."

        # ── 3. Decomposition ─────────────────────────────────────────────────
        # Prompt contains: "USER REQUEST:" and "step-by-step plan"
        if "USER REQUEST:" in user_content or "step-by-step plan" in user_content:
            task_match = re.search(r"USER REQUEST:\s*(.+?)(?:\n|$)", user_content)
            task = task_match.group(1).strip() if task_match else "complete the request"
            task60 = task[:60]
            task80 = task[:80]
            # Detect task type for smarter step planning
            is_code = any(kw in task.lower() for kw in ("write", "code", "script", "function", "implement", "build"))
            is_question = task.rstrip("?").endswith("?") or any(
                task.lower().startswith(kw) for kw in ("what", "how", "why", "when", "who", "where", "is ", "are ")
            )

            if is_question:
                # Simple questions need just one direct step
                return json.dumps({
                    "goal": f"Answer the question: {task60}",
                    "steps": [
                        {"type": "reflect", "text": f"Answer this question thoroughly: {task80}", "needs_prev": False}
                    ],
                })
            elif is_code:
                return json.dumps({
                    "goal": f"Produce working code for: {task60}",
                    "steps": [
                        {"type": "research", "text": f"Identify requirements and approach for: {task80}", "needs_prev": False},
                        {"type": "write_code", "text": f"Write the code implementation for: {task80}", "needs_prev": True},
                        {"type": "verify", "text": f"Run the code and confirm it works for: {task80}", "needs_prev": True},
                    ],
                })
            else:
                return json.dumps({
                    "goal": f"Successfully complete: {task60}",
                    "steps": [
                        {"type": "research", "text": f"Gather information about: {task80}", "needs_prev": False},
                        {"type": "reflect", "text": f"Synthesise findings and produce the final answer for: {task80}", "needs_prev": True},
                    ],
                })

        # ── 4. Execution — fill-in-the-blank ─────────────────────────────────
        # Prompt contains: "Choose ONE action. Reply with ONLY the JSON"
        if "Choose ONE action" in user_content:
            step_match = re.search(r"Step:\s*(\d+)/(\d+)", user_content)
            step = int(step_match.group(1)) if step_match else 1
            budget = int(step_match.group(2)) if step_match else 5

            # Extract what tools are on offer (Option X - ToolName pattern)
            offered = re.findall(r"Option [A-Z] \xe2\x80\x93 (\w+)", user_content) or \
                      re.findall(r'Option [A-Z] [-–] (\w+)', user_content) or \
                      re.findall(r'"tool":\s*"(\w+)"', user_content)

            # Extract the current instruction for context
            instr_match = re.search(r"Current instruction:\s*(.+?)(?:\n|$)", user_content) or \
                          re.search(r"Goal:\s*(.+?)(?:\n|$)", user_content)
            instr = instr_match.group(1).strip() if instr_match else ""

            # Near budget exhaustion or step >= 2 for simple tasks -> answer
            if step >= max(2, budget - 1):
                return json.dumps({
                    "tool": "Answer",
                    "summary": (
                        f"Completed: {instr[:120]}"
                        if instr else
                        "Research complete. All relevant information has been gathered and analysed."
                    ),
                })

            # Pick the best tool for this step
            tool_lower = [t.lower() for t in offered]

            if "websearch" in tool_lower:
                query = instr[:80] if instr else "relevant information"
                return json.dumps({"tool": "WebSearch", "query": query})

            if "bash" in tool_lower:
                # Pick command based on instruction
                if any(kw in instr.lower() for kw in ("list", "directory", "files", "ls")):
                    cmd = "ls -la"
                elif any(kw in instr.lower() for kw in ("run", "execute", "test")):
                    cmd = "python --version && echo 'ready'"
                elif any(kw in instr.lower() for kw in ("install", "pip")):
                    cmd = "pip list --format=columns | head -20"
                else:
                    cmd = "ls -la && pwd"
                return json.dumps({"tool": "Bash", "command": cmd})

            if "read" in tool_lower:
                return json.dumps({"tool": "Read", "file_path": "./README.md"})

            if "write" in tool_lower:
                return json.dumps({
                    "tool": "Write",
                    "file_path": "./output.md",
                    "content": f"# Result\n\n{instr[:200]}\n",
                })

            # Default: answer
            return json.dumps({"tool": "Answer", "summary": "Task completed successfully."})

        # ── 5. Content distiller ─────────────────────────────────────────────
        # Prompt: "You are a content distiller"
        if "content distiller" in all_content.lower():
            goal_match = re.search(r"Goal:\s*(.+?)(?:\n|$)", user_content)
            goal = goal_match.group(1).strip()[:80] if goal_match else "the task"
            return json.dumps({
                "cleaned": f"Summary of content relevant to: {goal}. Key points extracted and noise removed.",
                "notes": f"Relevant information found for: {goal}. Contains useful data for completing the request.",
            })

        # ── 6. Default echo ──────────────────────────────────────────────────
        last = (user_content or "")[:80]
        return f"[MOCK] Echo: {last}"


# ── Module-level helper: strip <think> blocks from qwen3/deepseek-r1 output ──

# Standard thinking-block tokens emitted by various models:
# <think>…</think>                     — Qwen3, DeepSeek-R1, Llama variants
# <thinking>…</thinking>               — some Ollama builds
# <|thinking|>…<|/thinking|>           — llama.cpp / some quantised models
# <|channel>thought\n…<channel|>       — Gemma 4 (e2b/e4b ignore think:false, tokens leak)
_THINK_RE = re.compile(
    r"<think>.*?</think>"
    r"|<thinking>.*?</thinking>"
    r"|<\|thinking\|>.*?<\|/thinking\|>"
    r"|<\|channel>thought\n.*?<channel\|>",
    re.DOTALL | re.IGNORECASE,
)

_REPEAT_MIN_LEN = 12   # ignore lines shorter than this (blanks, punctuation)
_REPEAT_THRESHOLD = 3  # cut after a line appears this many times


def _truncate_repetitive(text: str) -> str:
    """Truncate text where qwen3 enters a repetition loop.

    Keeps ONE instance of the repeated line (clean prefix), then looks for
    any JSON object after the repetition block and splices it back in.
    This lets the executor still parse a valid action even when the model
    looped before finishing its JSON.
    """
    lines = text.splitlines()
    counts: dict[str, int] = {}
    truncate_at: int | None = None
    repeated_line = ""

    for i, line in enumerate(lines):
        s = line.strip()
        if len(s) < _REPEAT_MIN_LEN:
            continue
        counts[s] = counts.get(s, 0) + 1
        if counts[s] >= _REPEAT_THRESHOLD:
            # Find the second occurrence — that's where the loop starts
            seen = 0
            for j, l in enumerate(lines):
                if l.strip() == s:
                    seen += 1
                    if seen == 2:
                        truncate_at = j
                        repeated_line = s
                        break
            break

    if truncate_at is None:
        return text

    clean_prefix = "\n".join(lines[:truncate_at]).strip()

    # Try to rescue a JSON object from the tail after the repetition block
    # (model sometimes finishes its JSON after the loop)
    tail_lines = lines[truncate_at:]
    tail = "\n".join(tail_lines)
    rescued_json = ""
    json_start = tail.find("{")
    if json_start >= 0:
        json_end = tail.rfind("}") + 1
        if json_end > json_start:
            candidate = tail[json_start:json_end].strip()
            # Only keep it if it looks like a real action JSON
            if candidate not in clean_prefix:
                rescued_json = candidate

    result = clean_prefix
    if rescued_json:
        result = (clean_prefix + "\n" + rescued_json).strip()

    logger.warning(
        "_truncate_repetitive: %r repeated %dx — kept %d/%d chars%s",
        repeated_line[:40], counts[repeated_line], len(result), len(text),
        " (JSON rescued)" if rescued_json else "",
    )
    return result


def _last_json(text: str) -> str:
    """Return the last complete JSON object in text (supports one level of nesting).

    Thinking models emit their final answer at the END of their reasoning block.
    Scanning from the right finds the answer rather than an intermediate fragment.
    Returns "" if no complete JSON object is found.
    """
    best = ""
    # Walk backwards through all '{' positions and try to find a closing '}'
    start = len(text) - 1
    while start >= 0:
        pos = text.rfind("{", 0, start + 1)
        if pos < 0:
            break
        # Find matching closing brace (handle one level of nesting)
        depth = 0
        end = -1
        for i in range(pos, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end > pos:
            candidate = text[pos:end + 1]
            # Accept if it has at least one key-value pair
            if re.search(r'"[^"]+"\s*:', candidate):
                best = candidate
                break  # found the last valid JSON, stop
        start = pos - 1
    return best


def _strip_thinking(data: dict) -> None:
    """Remove <think>…</think> blocks and handle thinking-model output quirks.

    Handles three backends:
      - Ollama: thinking in "reasoning" field, content may be empty
      - llama.cpp (external): thinking in "reasoning_content" field, content has the answer
      - Inline: model emits <think>...</think> tags inside "content"

    If after stripping, content is empty but JSON was inside the think block,
    rescues that JSON so structured callers don't get empty responses.
    Mutates the response dict in-place.
    """
    for choice in data.get("choices", []):
        msg = choice.get("message")
        if not isinstance(msg, dict):
            continue
        original_content = msg.get("content") or ""
        content = original_content

        # Strip inline <think> blocks (some models emit these in content)
        if "<think>" in content:
            cleaned = _THINK_RE.sub("", content).strip()
            if cleaned != content:
                logger.debug("stripped <think> block (%d chars removed)", len(content) - len(cleaned))
            content = cleaned

        # Strip bare /think and /no_think tokens that qwen3 emits
        for tok in ("/think", "/no_think"):
            content = content.replace(tok, "").strip()

        msg["content"] = content

        # Guard against repetition loops in generated text
        if content:
            deduped = _truncate_repetitive(content)
            if deduped != content:
                msg["content"] = deduped
                content = deduped

        # If content is empty after stripping, try rescue sources in priority order:
        if not content.strip():
            # 1. JSON was inside <think> block — rescue it (gemma4 via llama.cpp)
            if "<think>" in original_content:
                m = re.search(r"<think>(.*?)</think>", original_content, re.DOTALL | re.IGNORECASE)
                if m:
                    rescued = _last_json(m.group(1))
                    if rescued:
                        msg["content"] = rescued
                        logger.debug("rescued JSON from <think> block (%d chars)", len(rescued))
                        continue

            # 2. llama.cpp external puts thinking in "reasoning_content", answer in "content"
            for field in ("reasoning_content", "reasoning"):
                thinking = msg.get(field) or ""
                if thinking.strip():
                    # Only rescue structured JSON from reasoning fields (plan calls).
                    # Never promote plain-text reasoning as an answer — that leaks
                    # the thinking chain. Plain-text callers (synthesizer) handle
                    # empty content via their own retry/fallback logic.
                    rescued = _last_json(thinking)
                    if rescued:
                        msg["content"] = rescued
                        logger.debug("rescued JSON from %s field (%d chars)", field, len(rescued))
                        break


# ── Module-level helper: flatten Anthropic content blocks to plain OAI text ──

def _flatten_messages_for_oai(messages: list[dict]) -> list[dict]:
    """Convert Anthropic-format message content to plain OpenAI strings.

    Anthropic messages can have a list of typed content blocks:
      [{"type": "thinking", "thinking": "..."},
       {"type": "text", "text": "..."},
       {"type": "tool_use", ...},
       {"type": "tool_result", ...}]

    External OpenAI-compatible APIs only accept {"role": ..., "content": str}.
    This function:
      - Drops "thinking" blocks entirely (internal state, not needed by LLM)
      - Drops "tool_use" / "tool_result" blocks (can't be replayed via OAI format)
      - Concatenates remaining text blocks into a single string
      - Passes through messages that already have string content unchanged
    """
    flattened: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                btype = block.get("type", "")
                if btype == "text":
                    parts.append(block.get("text", ""))
                elif btype in ("thinking", "tool_use", "tool_result", "image"):
                    pass  # drop — not representable in OAI string format
                else:
                    # Unknown block — try to stringify it
                    raw = block.get("text") or block.get("content") or ""
                    if raw:
                        parts.append(str(raw))
            text = "\n".join(p for p in parts if p.strip())
            if text or msg.get("role") in ("system", "user"):
                flattened.append({**msg, "content": text})
            # If assistant message is now empty (was all thinking/tool blocks),
            # skip it to avoid sending an empty assistant turn.
        else:
            flattened.append(msg)
    return flattened
