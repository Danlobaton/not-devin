from pathlib import Path

import pytest

from not_devin.runtime.config import ConfigError, load_runtime_config


def write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "not-devin.toml"
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_defaults_with_unbounded_budgets(tmp_path: Path) -> None:
    config = load_runtime_config(
        write_config(
            tmp_path,
            """
[model]
provider = "openai"
name = "gpt-5-nano"
""",
        )
    )

    assert config.model.max_output_tokens == 4096
    assert config.model.temperature == 0
    assert config.timeouts.call_seconds == 120
    assert config.timeouts.run_seconds == 1800
    assert config.retry.max_retries == 2
    assert config.retry.base_delay_seconds == 1.0
    assert config.budget.max_model_calls is None
    assert config.budget.max_total_tokens is None
    assert config.budget.max_cost_usd is None


def test_loads_anthropic_with_pricing_and_budgets(tmp_path: Path) -> None:
    config = load_runtime_config(
        write_config(
            tmp_path,
            """
[model]
provider = "anthropic"
name = "claude-sonnet-4-6"
max_output_tokens = 8192

[budget]
max_model_calls = 20
max_total_tokens = 100000
max_cost_usd = 2.0

[pricing]
provider = "anthropic"
model = "claude-sonnet-4-6"
input_per_million = 3.0
output_per_million = 15.0
""",
        )
    )

    assert config.model.provider == "anthropic"
    assert config.budget.max_cost_usd == 2.0
    assert config.pricing is not None
    assert config.pricing.output_per_million == 15.0


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "model section is required"),
        (
            '[model]\nprovider = "other"\nname = "model"\n',
            "unsupported provider",
        ),
        (
            '[model]\nprovider = "openai"\nname = ""\n',
            "model name is required",
        ),
        (
            (
                '[model]\nprovider = "openai"\nname = "gpt"\n'
                "[timeouts]\ncall_seconds = 0\n"
            ),
            "call_seconds must be positive",
        ),
        (
            (
                '[model]\nprovider = "openai"\nname = "gpt"\n'
                "[budget]\nmax_cost_usd = 1.0\n"
            ),
            "pricing is required",
        ),
    ],
)
def test_rejects_invalid_configuration(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    with pytest.raises(ConfigError, match=message):
        load_runtime_config(write_config(tmp_path, content))


def test_rejects_pricing_for_different_model(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="pricing must match"):
        load_runtime_config(
            write_config(
                tmp_path,
                """
[model]
provider = "openai"
name = "gpt-a"

[budget]
max_cost_usd = 1.0

[pricing]
provider = "openai"
model = "gpt-b"
input_per_million = 1.0
output_per_million = 2.0
""",
            )
        )
