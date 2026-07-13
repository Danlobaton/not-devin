"""LangChain callback handler that logs LLM and tool call detail."""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, messages_to_dict
from langchain_core.outputs import LLMResult


class EventLogHandler(BaseCallbackHandler):
    """Logs LLM and tool call start/end events to an event log writer."""

    def __init__(self, writer: Any) -> None:
        self._writer = writer
        self._started_at: dict[UUID, float] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._started_at[run_id] = time.monotonic()
        flat_messages = [message for batch in messages for message in batch]
        self._writer.write("llm_start", messages=messages_to_dict(flat_messages))

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        latency_ms = self._latency_ms(run_id)
        message = response.generations[0][0].message
        self._writer.write(
            "llm_end", response=messages_to_dict([message])[0], latency_ms=latency_ms
        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        latency_ms = self._latency_ms(run_id)
        self._writer.write("llm_end", error=str(error), latency_ms=latency_ms)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._started_at[run_id] = time.monotonic()
        self._writer.write(
            "tool_start",
            tool_name=serialized.get("name", "unknown"),
            args=inputs if inputs is not None else input_str,
        )

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        latency_ms = self._latency_ms(run_id)
        self._writer.write(
            "tool_end",
            tool_call_id=getattr(output, "tool_call_id", None),
            result=getattr(output, "content", output),
            latency_ms=latency_ms,
        )

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        latency_ms = self._latency_ms(run_id)
        self._writer.write("tool_end", error=str(error), latency_ms=latency_ms)

    def _latency_ms(self, run_id: UUID) -> float | None:
        started_at = self._started_at.pop(run_id, None)
        if started_at is None:
            return None
        return round((time.monotonic() - started_at) * 1000, 2)
