from pathlib import Path
from typing import Any

import pytest

from not_devin.runtime.config import load_runtime_config
from not_devin.runtime.providers import create_chat_model


def config_for(tmp_path: Path, provider: str, name: str):
    path = tmp_path / "not-devin.toml"
    path.write_text(
        f"""
[model]
provider = "{provider}"
name = "{name}"
max_output_tokens = 2048
temperature = 0.2

[timeouts]
call_seconds = 45
""",
        encoding="utf-8",
    )
    return load_runtime_config(path)


@pytest.mark.parametrize(
    ("provider", "name", "constructor_name", "token_parameter"),
    [
        ("openai", "gpt-5-nano", "ChatOpenAI", "max_completion_tokens"),
        (
            "anthropic",
            "claude-sonnet-4-6",
            "ChatAnthropic",
            "max_tokens",
        ),
    ],
)
def test_creates_provider_with_shared_reliability_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    name: str,
    constructor_name: str,
    token_parameter: str,
) -> None:
    captured: dict[str, Any] = {}
    sentinel = object()

    def fake_constructor(**kwargs: Any) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        f"not_devin.runtime.providers.{constructor_name}",
        fake_constructor,
    )

    result = create_chat_model(config_for(tmp_path, provider, name))

    assert result is sentinel
    assert captured["model"] == name
    assert captured["temperature"] == 0.2
    assert captured["timeout"] == 45
    assert captured["max_retries"] == 0
    assert captured[token_parameter] == 2048
