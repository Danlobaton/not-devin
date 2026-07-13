from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from not_devin.runtime.budget import BudgetExceeded, UsageLedger
from not_devin.runtime.config import RuntimeConfig, load_runtime_config
from not_devin.runtime.invoker import (
    ProviderFailure,
    ReliableModelInvoker,
    RunDeadlineExceeded,
)


class StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__("secret response body")
        self.status_code = status_code


class ScriptedModel:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = iter(outcomes)
        self.bound_limits: list[dict[str, Any]] = []

    def bind(self, **kwargs: Any) -> "ScriptedModel":
        self.bound_limits.append(kwargs)
        return self

    def invoke(self, messages: object) -> AIMessage:
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        assert isinstance(outcome, AIMessage)
        return outcome


class FakeClock:
    def __init__(self, value: float = 0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def runtime_config(
    tmp_path: Path,
    extra: str = "",
    run_seconds: float = 30,
) -> RuntimeConfig:
    path = tmp_path / "not-devin.toml"
    path.write_text(
        f"""
[model]
provider = "openai"
name = "gpt"
max_output_tokens = 100

[timeouts]
call_seconds = 10
run_seconds = {run_seconds}

[retry]
max_retries = 2
base_delay_seconds = 1

{extra}
""",
        encoding="utf-8",
    )
    return load_runtime_config(path)


def response() -> AIMessage:
    return AIMessage(
        content="done",
        usage_metadata={
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        },
    )


def make_invoker(
    config: RuntimeConfig,
    clock: FakeClock | None = None,
    sleeps: list[float] | None = None,
) -> ReliableModelInvoker:
    recorded_sleeps = sleeps if sleeps is not None else []
    return ReliableModelInvoker(
        config,
        clock=clock or FakeClock(),
        sleeper=recorded_sleeps.append,
        jitter=lambda: 0,
    )


def test_retries_transient_status_twice_with_backoff(
    tmp_path: Path,
) -> None:
    sleeps: list[float] = []
    model = ScriptedModel([StatusError(429), StatusError(503), response()])
    ledger = UsageLedger()

    result = make_invoker(
        runtime_config(tmp_path),
        sleeps=sleeps,
    ).invoke(model, [HumanMessage(content="hello")], ledger)

    assert result.content == "done"
    assert ledger.attempts == 3
    assert ledger.successful_calls == 1
    assert sleeps == [1, 2]


def test_does_not_retry_non_transient_status(tmp_path: Path) -> None:
    sleeps: list[float] = []

    with pytest.raises(ProviderFailure) as captured:
        make_invoker(
            runtime_config(tmp_path),
            sleeps=sleeps,
        ).invoke(
            ScriptedModel([StatusError(400)]),
            [HumanMessage(content="hello")],
            UsageLedger(),
        )

    assert captured.value.retryable is False
    assert "secret response body" not in str(captured.value)
    assert captured.value.__cause__ is None
    assert sleeps == []


def test_stops_when_run_deadline_has_passed(tmp_path: Path) -> None:
    clock = FakeClock()
    invoker = make_invoker(runtime_config(tmp_path), clock=clock)
    _, deadline = invoker.new_run_timing()
    clock.value = 31

    with pytest.raises(RunDeadlineExceeded):
        invoker.invoke(
            ScriptedModel([response()]),
            [HumanMessage(content="hello")],
            UsageLedger(),
            deadline,
        )


def test_skips_backoff_that_crosses_deadline(tmp_path: Path) -> None:
    clock = FakeClock()
    sleeps: list[float] = []
    config = runtime_config(tmp_path, run_seconds=0.5)

    with pytest.raises(RunDeadlineExceeded):
        make_invoker(config, clock, sleeps).invoke(
            ScriptedModel([StatusError(503)]),
            [HumanMessage(content="hello")],
            UsageLedger(),
        )

    assert sleeps == []


def test_passes_reduced_output_limit_to_provider(tmp_path: Path) -> None:
    config = runtime_config(
        tmp_path,
        "[budget]\nmax_total_tokens = 20\n",
    )
    model = ScriptedModel([response()])

    make_invoker(config).invoke(
        model,
        [HumanMessage(content="hello")],
        UsageLedger(),
    )

    assert model.bound_limits == [
        {"max_completion_tokens": 18, "timeout": 10}
    ]


def test_maps_attempt_budget_to_budget_exception(tmp_path: Path) -> None:
    config = runtime_config(
        tmp_path,
        "[budget]\nmax_model_calls = 1\n",
    )
    ledger = UsageLedger(attempts=1)

    with pytest.raises(BudgetExceeded) as captured:
        make_invoker(config).invoke(
            ScriptedModel([response()]),
            [HumanMessage(content="hello")],
            ledger,
        )

    assert captured.value.terminal_reason == "model_call_budget"


def test_failed_retry_retains_conservative_cost_reservation(
    tmp_path: Path,
) -> None:
    config = runtime_config(
        tmp_path,
        """
[budget]
max_cost_usd = 1

[pricing]
provider = "openai"
model = "gpt"
input_per_million = 1
output_per_million = 2
""",
    )
    ledger = UsageLedger()

    make_invoker(config).invoke(
        ScriptedModel([StatusError(503), response()]),
        [HumanMessage(content="hello")],
        ledger,
    )

    assert ledger.attempts == 2
    assert ledger.estimated_cost_usd == pytest.approx(0.000222)


def test_failed_retries_reserve_token_budget(tmp_path: Path) -> None:
    config = runtime_config(
        tmp_path,
        "[budget]\nmax_total_tokens = 150\n",
    )
    ledger = UsageLedger()

    with pytest.raises(BudgetExceeded) as captured:
        make_invoker(config).invoke(
            ScriptedModel(
                [StatusError(503), StatusError(503), response()]
            ),
            [HumanMessage(content="hello")],
            ledger,
        )

    assert captured.value.terminal_reason == "token_budget"
    assert ledger.attempts == 2
    assert ledger.reserved_tokens == 150


def test_attempt_failure_after_deadline_is_timeout(tmp_path: Path) -> None:
    clock = FakeClock()

    class DeadlineFailureModel:
        def bind(self, **kwargs: Any) -> "DeadlineFailureModel":
            assert kwargs["timeout"] == 10
            return self

        def invoke(self, messages: object) -> AIMessage:
            clock.value = 31
            raise StatusError(503)

    with pytest.raises(RunDeadlineExceeded) as captured:
        make_invoker(runtime_config(tmp_path), clock=clock).invoke(
            DeadlineFailureModel(),
            [HumanMessage(content="hello")],
            UsageLedger(),
        )

    assert captured.value.__cause__ is None
