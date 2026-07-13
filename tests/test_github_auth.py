from pathlib import Path

import httpx
import pytest

from not_devin.github.auth import GitHubAppTokenProvider
from not_devin.github.errors import GitHubServiceError


def test_exchanges_and_reuses_installation_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            201,
            json={
                "token": "installation-secret",
                "expires_at": "2099-01-01T00:00:00Z",
            },
        )

    monkeypatch.setattr(
        "not_devin.github.auth.jwt.encode",
        lambda *args, **kwargs: "signed-app-jwt",
    )
    client = httpx.Client(
        base_url="https://api.github.test",
        transport=httpx.MockTransport(handler),
    )
    provider = GitHubAppTokenProvider(
        app_id=1,
        installation_id=2,
        private_key="private-key",
        client=client,
    )

    assert provider.get_token() == "installation-secret"
    assert provider.get_token() == "installation-secret"
    assert len(requests) == 1
    assert requests[0].url.path == "/app/installations/2/access_tokens"
    assert requests[0].headers["authorization"] == "Bearer signed-app-jwt"


def test_from_env_reads_private_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_path = tmp_path / "app.pem"
    key_path.write_text("private-key", encoding="utf-8")
    monkeypatch.setenv("GITHUB_APP_ID", "11")
    monkeypatch.setenv("GITHUB_APP_INSTALLATION_ID", "22")
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY_PATH", str(key_path))

    provider = GitHubAppTokenProvider.from_env()

    assert provider.app_id == 11
    assert provider.installation_id == 22
    assert provider.private_key == "private-key"


def test_from_env_rejects_missing_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_APP_ID", raising=False)

    with pytest.raises(
        GitHubServiceError,
        match="GITHUB_APP_ID is required",
    ):
        GitHubAppTokenProvider.from_env()


def test_token_exchange_error_does_not_expose_response_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "not_devin.github.auth.jwt.encode",
        lambda *args, **kwargs: "signed-app-jwt",
    )
    client = httpx.Client(
        base_url="https://api.github.test",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                401,
                text="installation-secret",
                request=request,
            )
        ),
    )
    provider = GitHubAppTokenProvider(1, 2, "private-key", client)

    with pytest.raises(GitHubServiceError) as captured:
        provider.get_token()

    assert "HTTP 401" in str(captured.value)
    assert "installation-secret" not in str(captured.value)
