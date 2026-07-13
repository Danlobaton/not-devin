"""Runtime-owned provider retries, deadlines, and budget accounting."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol

import anthropic
import httpx
import openai
from langchain_core.messages import AIMessage, AnyMessage

from .budget import UsageLedger, preflight, record_response
from .config import RuntimeConfig


class InvokableModel(Protocol):
    """Minimal model interface needed by the reliable invoker."""

    def bind(self, **kwargs: Any) -> InvokableModel:
        """Return a model with invocation-specific options."""
        ...

    def invoke(self, messages: object) -> AIMessage:
        """Invoke the model."""
        ...


class RuntimeModelInvoker(Protocol):
    """Reliable invocation interface consumed by the graph."""

    def new_run_timing(self) -> tuple[str, str]:
        """Return serializable start and deadline timestamps."""
        ...

    def invoke(
        self,
        model: InvokableModel,
        messages: Sequence[AnyMessage],
        ledger: UsageLedger,
        run_deadline: str | None = None,
    ) -> AIMessage:
        """Invoke a bound model and update the supplied ledger."""
        ...


class RunDeadlineExceeded(RuntimeError):
    """The total run deadline has elapsed."""


class ProviderFailure(RuntimeError):
    """A normalized provider failure safe for state and traces."""

    def __init__(
        self,
        *,
        retryable: bool,
        status_code: int | None = None,
    ) -> None:
        detail = (
            f"HTTP {status_code}"
            if status_code is not None
            else "transport or request error"
        )
        super().__init__(f"provider request failed: {detail}")
        self.retryable = retryable
        self.status_code = status_code


def estimate_message_tokens(messages: Sequence[AnyMessage]) -> int:
    """Conservatively estimate input tokens without a provider tokenizer."""
    characters = sum(len(str(message.content)) for message in messages)
    return max(1, math.ceil(characters / 3))


def classify_provider_error(
    error: Exception,
) -> tuple[bool, int | None]:
    """Return retryability and status without exposing provider content."""
    if isinstance(
        error,
        (
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            httpx.TimeoutException,
            httpx.TransportError,
            openai.APIConnectionError,
            openai.APITimeoutError,
        ),
    ):
        return True, None

    status_code = getattr(error, "status_code", None)
    if status_code is None:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)

    if status_code in {429, 500, 502, 503, 504}:
        return True, status_code
    return False, status_code


class ReliableModelInvoker:
    """Invoke one run's model under explicit reliability policy."""

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self.config = config
        self.clock = clock
        self.sleeper = sleeper
        self.jitter = jitter

    def new_run_timing(self) -> tuple[str, str]:
        """Create persistent timing values for a new run."""
        started_at = self.clock()
        deadline = started_at + self.config.timeouts.run_seconds
        return (
            datetime.fromtimestamp(started_at, UTC).isoformat(),
            datetime.fromtimestamp(deadline, UTC).isoformat(),
        )

    def _deadline_timestamp(self, run_deadline: str | None) -> float:
        if run_deadline is None:
            _, run_deadline = self.new_run_timing()
        return datetime.fromisoformat(run_deadline).timestamp()

    def _check_deadline(self, deadline: float) -> None:
        if self.clock() >= deadline:
            raise RunDeadlineExceeded("run deadline exceeded")

    def _output_parameter(self) -> str:
        if self.config.model.provider == "openai":
            return "max_completion_tokens"
        return "max_tokens"

    def invoke(
        self,
        model: InvokableModel,
        messages: Sequence[AnyMessage],
        ledger: UsageLedger,
        run_deadline: str | None = None,
    ) -> AIMessage:
        """Invoke with preflight budgets and bounded transient retries."""
        estimated_input_tokens = estimate_message_tokens(messages)
        deadline = self._deadline_timestamp(run_deadline)

        for retry_index in range(self.config.retry.max_retries + 1):
            self._check_deadline(deadline)
            reservation = preflight(
                ledger,
                self.config.budget,
                self.config.pricing,
                estimated_input_tokens,
                self.config.model.max_output_tokens,
            )

            try:
                remaining_run_time = deadline - self.clock()
                if remaining_run_time <= 0:
                    raise RunDeadlineExceeded("run deadline exceeded")
                attempt_model = model.bind(
                    **{
                        self._output_parameter(): (
                            reservation.output_token_limit
                        ),
                        "timeout": min(
                            self.config.timeouts.call_seconds,
                            remaining_run_time,
                        ),
                    }
                )
                response = attempt_model.invoke(messages)
            except RunDeadlineExceeded:
                raise
            except Exception as error:
                if self.clock() >= deadline:
                    raise RunDeadlineExceeded(
                        "provider attempt exceeded run deadline"
                    ) from None
                retryable, status_code = classify_provider_error(error)
                if (
                    not retryable
                    or retry_index >= self.config.retry.max_retries
                ):
                    raise ProviderFailure(
                        retryable=retryable,
                        status_code=status_code,
                    ) from None

                delay = (
                    self.config.retry.base_delay_seconds
                    * (2**retry_index)
                    + self.jitter()
                    * self.config.retry.base_delay_seconds
                )
                if self.clock() + delay >= deadline:
                    raise RunDeadlineExceeded(
                        "retry backoff would exceed run deadline"
                    ) from None
                self.sleeper(delay)
                continue

            if not isinstance(response, AIMessage):
                raise TypeError("provider did not return an AIMessage")
            record_response(
                ledger,
                reservation,
                response,
                self.config.pricing,
            )
            self._check_deadline(deadline)
            return response

        raise AssertionError("retry loop exhausted without returning")
