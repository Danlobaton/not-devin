"""GitHub App installation-token authentication."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

import httpx
import jwt

from .errors import GitHubServiceError, raise_for_github_error

GITHUB_API_URL = "https://api.github.com"
GITHUB_API_VERSION = "2022-11-28"


class TokenProvider(Protocol):
    """Provides a valid GitHub access token."""

    def get_token(self) -> str:
        """Return a token suitable for a GitHub API request."""
        ...


def require_env(name: str) -> str:
    """Read a required environment variable."""
    value = os.environ.get(name)
    if not value:
        raise GitHubServiceError(f"{name} is required")
    return value


class GitHubAppTokenProvider:
    """Create and cache installation tokens for a GitHub App."""

    def __init__(
        self,
        app_id: int,
        installation_id: int,
        private_key: str,
        client: httpx.Client | None = None,
    ) -> None:
        self.app_id = app_id
        self.installation_id = installation_id
        self.private_key = private_key
        self.client = client or httpx.Client(
            base_url=GITHUB_API_URL,
            timeout=httpx.Timeout(10.0),
        )
        self._token: str | None = None
        self._expires_at = datetime.min.replace(tzinfo=UTC)

    @classmethod
    def from_env(cls) -> GitHubAppTokenProvider:
        """Construct a provider from GitHub App environment variables."""
        app_id = int(require_env("GITHUB_APP_ID"))
        installation_id = int(
            require_env("GITHUB_APP_INSTALLATION_ID")
        )
        key_path = Path(require_env("GITHUB_APP_PRIVATE_KEY_PATH"))
        private_key = key_path.read_text(encoding="utf-8")
        return cls(app_id, installation_id, private_key)

    def get_token(self) -> str:
        """Return a cached token or exchange a new app JWT."""
        now = datetime.now(UTC)
        if (
            self._token is not None
            and self._expires_at > now + timedelta(minutes=1)
        ):
            return self._token

        app_jwt = jwt.encode(
            {
                "iat": int((now - timedelta(seconds=60)).timestamp()),
                "exp": int((now + timedelta(minutes=9)).timestamp()),
                "iss": str(self.app_id),
            },
            self.private_key,
            algorithm="RS256",
        )

        try:
            response = self.client.post(
                (
                    f"/app/installations/{self.installation_id}"
                    "/access_tokens"
                ),
                headers={
                    "Accept": "application/vnd.github+json",
                    "Authorization": f"Bearer {app_jwt}",
                    "X-GitHub-Api-Version": GITHUB_API_VERSION,
                },
            )
        except httpx.RequestError as error:
            raise GitHubServiceError(
                "create installation token failed: network error"
            ) from error

        raise_for_github_error(response, "create installation token")
        payload = response.json()
        self._token = payload["token"]
        self._expires_at = datetime.fromisoformat(
            payload["expires_at"].replace("Z", "+00:00")
        )
        return self._token
