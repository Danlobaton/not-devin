"""Usage accounting and optional per-run budget enforcement."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from langchain_core.messages import AIMessage

from .config import BudgetConfig, ModelPricing

BudgetTerminalReason = Literal[
    "model_call_budget",
    "token_budget",
    "cost_budget",
]


class BudgetExceeded(RuntimeError):
    """A configured run budget has been exhausted."""

    def __init__(
        self,
        terminal_reason: BudgetTerminalReason,
        message: str,
    ) -> None:
        super().__init__(message)
        self.terminal_reason = terminal_reason


@dataclass
class UsageLedger:
    """Serializable cumulative model usage for one run."""

    attempts: int = 0
    successful_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reserved_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def to_state(self) -> dict[str, int | float]:
        """Convert the ledger to graph-state-compatible primitives."""
        return asdict(self)

    @classmethod
    def from_state(
        cls,
        state: dict[str, int | float] | None,
    ) -> UsageLedger:
        """Restore a ledger from graph state."""
        return cls(**(state or {}))


@dataclass(frozen=True)
class UsageReservation:
    """Conservative usage reserved before one provider attempt."""

    estimated_input_tokens: int
    output_token_limit: int
    estimated_cost_usd: float


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    pricing: ModelPricing | None,
) -> float:
    """Estimate provider cost from token counts."""
    if pricing is None:
        return 0.0
    return (
        input_tokens * pricing.input_per_million
        + output_tokens * pricing.output_per_million
    ) / 1_000_000


def preflight(
    ledger: UsageLedger,
    budget: BudgetConfig,
    pricing: ModelPricing | None,
    estimated_input_tokens: int,
    max_output_tokens: int,
) -> UsageReservation:
    """Reserve one attempt and return its affordable output-token limit."""
    if (
        budget.max_model_calls is not None
        and ledger.attempts >= budget.max_model_calls
    ):
        raise BudgetExceeded(
            "model_call_budget",
            "model call budget exhausted",
        )

    output_limit = max_output_tokens
    if budget.max_total_tokens is not None:
        remaining_tokens = (
            budget.max_total_tokens
            - ledger.total_tokens
            - ledger.reserved_tokens
            - estimated_input_tokens
        )
        if remaining_tokens <= 0:
            raise BudgetExceeded(
                "token_budget",
                "token budget exhausted",
            )
        output_limit = min(output_limit, remaining_tokens)

    if budget.max_cost_usd is not None:
        if pricing is None:
            raise ValueError("pricing is required for cost enforcement")
        remaining_cost = budget.max_cost_usd - ledger.estimated_cost_usd
        input_cost = estimate_cost(
            estimated_input_tokens,
            0,
            pricing,
        )
        affordable_output_cost = remaining_cost - input_cost
        if affordable_output_cost <= 0:
            raise BudgetExceeded(
                "cost_budget",
                "cost budget exhausted",
            )
        affordable_output_tokens = int(
            affordable_output_cost
            * 1_000_000
            / pricing.output_per_million
            + 1e-9
        )
        if affordable_output_tokens <= 0:
            raise BudgetExceeded(
                "cost_budget",
                "cost budget exhausted",
            )
        output_limit = min(output_limit, affordable_output_tokens)

    reserved_cost = estimate_cost(
        estimated_input_tokens,
        output_limit,
        pricing,
    )
    ledger.attempts += 1
    ledger.reserved_tokens += estimated_input_tokens + output_limit
    ledger.estimated_cost_usd += reserved_cost
    return UsageReservation(
        estimated_input_tokens=estimated_input_tokens,
        output_token_limit=output_limit,
        estimated_cost_usd=reserved_cost,
    )


def record_response(
    ledger: UsageLedger,
    reservation: UsageReservation,
    response: AIMessage,
    pricing: ModelPricing | None,
) -> None:
    """Reconcile a successful response against its reservation."""
    usage = response.usage_metadata
    if usage is None:
        input_tokens = reservation.estimated_input_tokens
        output_tokens = reservation.output_token_limit
        total_tokens = input_tokens + output_tokens
    else:
        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]
        total_tokens = usage["total_tokens"]

    actual_cost = estimate_cost(input_tokens, output_tokens, pricing)
    ledger.successful_calls += 1
    ledger.input_tokens += input_tokens
    ledger.output_tokens += output_tokens
    ledger.total_tokens += total_tokens
    ledger.reserved_tokens = max(
        0,
        (
            ledger.reserved_tokens
            - reservation.estimated_input_tokens
            - reservation.output_token_limit
        ),
    )
    ledger.estimated_cost_usd = max(
        0.0,
        (
            ledger.estimated_cost_usd
            - reservation.estimated_cost_usd
            + actual_cost
        ),
    )
