"""Typed runtime configuration loaded from TOML."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

ProviderName = Literal["openai", "anthropic"]


class ConfigError(ValueError):
    """Invalid runtime configuration."""


@dataclass(frozen=True)
class ProviderConfig:
    provider: ProviderName
    name: str
    max_output_tokens: int = 4096
    temperature: float = 0


@dataclass(frozen=True)
class TimeoutConfig:
    call_seconds: float = 120
    run_seconds: float = 1800


@dataclass(frozen=True)
class RetryConfig:
    max_retries: int = 2
    base_delay_seconds: float = 1.0


@dataclass(frozen=True)
class BudgetConfig:
    max_model_calls: int | None = None
    max_total_tokens: int | None = None
    max_cost_usd: float | None = None


@dataclass(frozen=True)
class ModelPricing:
    provider: ProviderName
    model: str
    input_per_million: float
    output_per_million: float


@dataclass(frozen=True)
class RuntimeConfig:
    model: ProviderConfig
    timeouts: TimeoutConfig
    retry: RetryConfig
    budget: BudgetConfig
    pricing: ModelPricing | None = None


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a table")
    return value


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{name} must be positive")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConfigError(f"{name} must be non-negative")
    return value


def _positive_float(value: Any, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ConfigError(f"{name} must be positive")
    return float(value)


def _optional_positive_int(
    section: dict[str, Any],
    name: str,
) -> int | None:
    value = section.get(name)
    return None if value is None else _positive_int(value, name)


def _optional_positive_float(
    section: dict[str, Any],
    name: str,
) -> float | None:
    value = section.get(name)
    return None if value is None else _positive_float(value, name)


def load_runtime_config(path: Path) -> RuntimeConfig:
    """Load and validate runtime configuration from TOML."""
    with path.open("rb") as config_file:
        data = tomllib.load(config_file)

    if "model" not in data:
        raise ConfigError("model section is required")

    model_data = _section(data, "model")
    provider_value = model_data.get("provider")
    if provider_value not in {"openai", "anthropic"}:
        raise ConfigError(f"unsupported provider: {provider_value}")
    provider = cast(ProviderName, provider_value)

    model_name = model_data.get("name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ConfigError("model name is required")

    max_output_tokens = _positive_int(
        model_data.get("max_output_tokens", 4096),
        "max_output_tokens",
    )
    temperature_value = model_data.get("temperature", 0)
    if (
        not isinstance(temperature_value, (int, float))
        or isinstance(temperature_value, bool)
        or temperature_value < 0
    ):
        raise ConfigError("temperature must be non-negative")

    timeout_data = _section(data, "timeouts")
    retry_data = _section(data, "retry")
    budget_data = _section(data, "budget")

    model = ProviderConfig(
        provider=provider,
        name=model_name,
        max_output_tokens=max_output_tokens,
        temperature=float(temperature_value),
    )
    timeouts = TimeoutConfig(
        call_seconds=_positive_float(
            timeout_data.get("call_seconds", 120),
            "call_seconds",
        ),
        run_seconds=_positive_float(
            timeout_data.get("run_seconds", 1800),
            "run_seconds",
        ),
    )
    retry = RetryConfig(
        max_retries=_nonnegative_int(
            retry_data.get("max_retries", 2),
            "max_retries",
        ),
        base_delay_seconds=_positive_float(
            retry_data.get("base_delay_seconds", 1.0),
            "base_delay_seconds",
        ),
    )
    budget = BudgetConfig(
        max_model_calls=_optional_positive_int(
            budget_data,
            "max_model_calls",
        ),
        max_total_tokens=_optional_positive_int(
            budget_data,
            "max_total_tokens",
        ),
        max_cost_usd=_optional_positive_float(
            budget_data,
            "max_cost_usd",
        ),
    )

    pricing = None
    if "pricing" in data:
        pricing_data = _section(data, "pricing")
        pricing_provider = pricing_data.get("provider")
        pricing_model = pricing_data.get("model")
        if pricing_provider not in {"openai", "anthropic"}:
            raise ConfigError("pricing provider is invalid")
        if not isinstance(pricing_model, str) or not pricing_model:
            raise ConfigError("pricing model is required")
        pricing = ModelPricing(
            provider=cast(ProviderName, pricing_provider),
            model=pricing_model,
            input_per_million=_positive_float(
                pricing_data.get("input_per_million"),
                "input_per_million",
            ),
            output_per_million=_positive_float(
                pricing_data.get("output_per_million"),
                "output_per_million",
            ),
        )
        if pricing.provider != provider or pricing.model != model.name:
            raise ConfigError("pricing must match the configured provider/model")

    if budget.max_cost_usd is not None and pricing is None:
        raise ConfigError("pricing is required when max_cost_usd is configured")

    return RuntimeConfig(
        model=model,
        timeouts=timeouts,
        retry=retry,
        budget=budget,
        pricing=pricing,
    )
