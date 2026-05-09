from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client import LlamaClient

logger = logging.getLogger(__name__)

_SUMMARIZE_PROMPT = (
    "Summarize the following conversation segment in 2–3 concise sentences. "
    "Preserve all key facts, decisions, file paths, and outcomes:\n\n{turns}"
)


class ContextManager:
    """Fits a message list into the model's context window.

    Instead of silently dropping messages (the BirdClaw v1 bug), we
    iteratively summarize the oldest turns until the token count is
    within budget.  The summary is injected as a user message so the
    model always has full narrative continuity.
    """

    def __init__(
        self,
        client: LlamaClient,
        context_size: int,
        reserve: int = 512,
    ) -> None:
        self.client = client
        self.context_size = context_size
        self.reserve = reserve  # tokens kept free for the model's response

    # ── Public API ───────────────────────────────────────────────────────────

    async def fit(
        self,
        messages: list[dict],
        system: str | None = None,
    ) -> list[dict]:
        """Return a message list that fits within context_size - reserve tokens.

        Args:
            messages: Conversation history in OpenAI role/content format.
            system:   Optional system prompt.  Ignored if the first message
                      already has role='system'.
        """
        built = _prepend_system(messages, system)

        while True:
            total = await self._count_tokens(built)
            budget = self.context_size - self.reserve

            if total <= budget:
                return built

            non_sys = [m for m in built if m["role"] != "system"]
            if len(non_sys) <= 2:
                # Cannot reduce any further without losing the current exchange.
                logger.warning(
                    "Context at limit with only %d non-system messages; "
                    "returning as-is (response may be cut short)",
                    len(non_sys),
                )
                return built

            # Summarize the oldest quarter of non-system turns (min 2 messages).
            n = max(2, len(non_sys) // 4)
            to_summarize = non_sys[:n]
            summary_text = await self._summarize(to_summarize)
            summary_msg = {
                "role": "user",
                "content": f"[Summary of earlier conversation: {summary_text}]",
            }

            # Rebuild: system + summary + remaining messages
            summarized_ids = {id(m) for m in to_summarize}
            sys_msgs = [m for m in built if m["role"] == "system"]
            rest = [
                m for m in built
                if m["role"] != "system" and id(m) not in summarized_ids
            ]
            built = sys_msgs + [summary_msg] + rest
            logger.info("Summarized %d old messages to reclaim context space", n)

    # ── Internals ────────────────────────────────────────────────────────────

    async def _count_tokens(self, messages: list[dict]) -> int:
        text = " ".join(m.get("content") or "" for m in messages)
        return await self.client.tokenize(text)

    async def _summarize(self, messages: list[dict]) -> str:
        turns = "\n".join(
            f"{m['role'].upper()}: {m.get('content') or ''}" for m in messages
        )
        result = await self.client.generate(
            [{"role": "user", "content": _SUMMARIZE_PROMPT.format(turns=turns)}],
            max_tokens=256,
            temperature=0.3,
            stream=False,
        )
        return result["choices"][0]["message"]["content"].strip()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _prepend_system(messages: list[dict], system: str | None) -> list[dict]:
    """Prepend a system message if one isn't already present."""
    if not system:
        return list(messages)
    if messages and messages[0].get("role") == "system":
        return list(messages)
    return [{"role": "system", "content": system}] + list(messages)
