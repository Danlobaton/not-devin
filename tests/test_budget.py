import pytest
from langchain_core.messages import AIMessage

from not_devin.runtime.budget import (
    BudgetExceeded,
    UsageLedger,
    preflight,
    record_response,
)
from not_devin.runtime.config import BudgetConfig, ModelPricing

PRICING = ModelPricing(
    provider="openai",
    model="gpt",
    input_per_million=1.0,
    output_per_million=2.0,
)


def test_unbounded_budget_preserves_output_limit() -> None:
    ledger = UsageLedger()

    reservation = preflight(
        ledger,
        BudgetConfig(),
        None,
        estimated_input_tokens=100,
        max_output_tokens=1000,
    )

    assert reservation.output_token_limit == 1000
    assert ledger.attempts == 1
    assert ledger.estimated_cost_usd == 0


def test_model_call_budget_counts_failed_attempts() -> None:
    ledger = UsageLedger()
    budget = BudgetConfig(max_model_calls=1)
    preflight(ledger, budget, None, 100, 100)

    with pytest.raises(BudgetExceeded) as captured:
        preflight(ledger, budget, None, 100, 100)

    assert captured.value.terminal_reason == "model_call_budget"


def test_token_budget_reduces_output_limit() -> None:
    ledger = UsageLedger(total_tokens=800)

    reservation = preflight(
        ledger,
        BudgetConfig(max_total_tokens=1000),
        None,
        estimated_input_tokens=100,
        max_output_tokens=500,
    )

    assert reservation.output_token_limit == 100


def test_token_budget_stops_when_input_uses_remaining_tokens() -> None:
    with pytest.raises(BudgetExceeded) as captured:
        preflight(
            UsageLedger(total_tokens=950),
            BudgetConfig(max_total_tokens=1000),
            None,
            estimated_input_tokens=50,
            max_output_tokens=100,
        )

    assert captured.value.terminal_reason == "token_budget"


def test_cost_budget_reduces_output_limit() -> None:
    reservation = preflight(
        UsageLedger(),
        BudgetConfig(max_cost_usd=0.001),
        PRICING,
        estimated_input_tokens=200,
        max_output_tokens=1000,
    )

    assert reservation.output_token_limit == 400
    assert reservation.estimated_cost_usd == pytest.approx(0.001)


def test_failed_attempt_keeps_conservative_cost_reservation() -> None:
    ledger = UsageLedger()

    preflight(
        ledger,
        BudgetConfig(max_cost_usd=1),
        PRICING,
        estimated_input_tokens=100,
        max_output_tokens=100,
    )

    assert ledger.estimated_cost_usd == pytest.approx(0.0003)
    assert ledger.successful_calls == 0
    assert ledger.reserved_tokens == 200


def test_success_reconciles_reservation_with_actual_usage() -> None:
    ledger = UsageLedger()
    reservation = preflight(
        ledger,
        BudgetConfig(max_cost_usd=1),
        PRICING,
        estimated_input_tokens=100,
        max_output_tokens=100,
    )
    response = AIMessage(
        content="done",
        usage_metadata={
            "input_tokens": 50,
            "output_tokens": 20,
            "total_tokens": 70,
        },
    )

    record_response(ledger, reservation, response, PRICING)

    assert ledger.successful_calls == 1
    assert ledger.input_tokens == 50
    assert ledger.output_tokens == 20
    assert ledger.total_tokens == 70
    assert ledger.reserved_tokens == 0
    assert ledger.estimated_cost_usd == pytest.approx(0.00009)
