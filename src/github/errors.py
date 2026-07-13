"""Sanitized errors shared by GitHub adapters."""

import httpx


class GitHubServiceError(RuntimeError):
    """A safe-to-report GitHub operation failure."""


def raise_for_github_error(
    response: httpx.Response,
    operation: str,
) -> None:
    """Raise a sanitized error for a failed GitHub response."""
    if response.is_success:
        return

    message = f"{operation} failed with HTTP {response.status_code}"
    if (
        response.status_code == 403
        and response.headers.get("x-ratelimit-remaining") == "0"
    ):
        reset = response.headers.get("x-ratelimit-reset")
        if reset:
            message = f"{message}; rate limit resets at {reset}"

    raise GitHubServiceError(message)
