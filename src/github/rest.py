"""Repository-scoped GitHub REST API adapter."""

from __future__ import annotations

from typing import Any

import httpx

from .auth import (
    GITHUB_API_URL,
    GITHUB_API_VERSION,
    TokenProvider,
    require_env,
)
from .errors import GitHubServiceError, raise_for_github_error


class GitHubRestService:
    """Perform approved operations against one GitHub repository."""

    def __init__(
        self,
        owner: str,
        repo: str,
        token_provider: TokenProvider,
        client: httpx.Client | None = None,
    ) -> None:
        self.owner = owner
        self.repo = repo
        self.token_provider = token_provider
        self.client = client or httpx.Client(
            base_url=GITHUB_API_URL,
            timeout=httpx.Timeout(10.0),
        )

    @classmethod
    def from_env(
        cls,
        token_provider: TokenProvider,
    ) -> GitHubRestService:
        """Construct a service from ``GITHUB_REPOSITORY=owner/repo``."""
        repository = require_env("GITHUB_REPOSITORY")
        parts = repository.split("/")
        if len(parts) != 2 or not all(parts):
            raise GitHubServiceError(
                "GITHUB_REPOSITORY must have the form owner/repo"
            )
        return cls(parts[0], parts[1], token_provider)

    @property
    def repository_path(self) -> str:
        """Return the URL path prefix for the configured repository."""
        return f"/repos/{self.owner}/{self.repo}"

    def _request(
        self,
        method: str,
        path: str,
        operation: str,
        **kwargs: Any,
    ) -> httpx.Response:
        try:
            response = self.client.request(
                method,
                path,
                headers={
                    "Accept": "application/vnd.github+json",
                    "Authorization": (
                        f"Bearer {self.token_provider.get_token()}"
                    ),
                    "X-GitHub-Api-Version": GITHUB_API_VERSION,
                },
                **kwargs,
            )
        except httpx.RequestError as error:
            raise GitHubServiceError(
                f"{operation} failed: network error"
            ) from error

        raise_for_github_error(response, operation)
        return response

    def get_issue_context(self, issue_number: int) -> dict[str, object]:
        """Return an issue and every issue comment oldest-first."""
        issue = self._request(
            "GET",
            f"{self.repository_path}/issues/{issue_number}",
            "get issue",
        ).json()

        comments: list[dict[str, object]] = []
        page = 1
        while True:
            payload = self._request(
                "GET",
                f"{self.repository_path}/issues/{issue_number}/comments",
                "list issue comments",
                params={
                    "per_page": 100,
                    "page": page,
                    "sort": "created",
                    "direction": "asc",
                },
            ).json()
            comments.extend(
                {
                    "id": comment["id"],
                    "author": (comment.get("user") or {}).get("login"),
                    "body": comment.get("body") or "",
                    "created_at": comment["created_at"],
                    "url": comment["html_url"],
                }
                for comment in payload
            )
            if len(payload) < 100:
                break
            page += 1

        return {
            "number": issue["number"],
            "title": issue["title"],
            "body": issue.get("body") or "",
            "state": issue["state"],
            "url": issue["html_url"],
            "author": (issue.get("user") or {}).get("login"),
            "comments": comments,
        }

    def get_ci_status(self, head_sha: str) -> dict[str, object]:
        """Return normalized check runs and their aggregate status."""
        checks: list[dict[str, object]] = []
        page = 1
        while True:
            payload = self._request(
                "GET",
                f"{self.repository_path}/commits/{head_sha}/check-runs",
                "list check runs",
                params={"per_page": 100, "page": page},
            ).json()
            page_checks = payload.get("check_runs", [])
            checks.extend(
                {
                    "name": check["name"],
                    "status": check["status"],
                    "conclusion": check.get("conclusion"),
                    "url": check.get("details_url"),
                }
                for check in page_checks
            )
            if len(page_checks) < 100:
                break
            page += 1

        if not checks:
            status = "no_checks"
        elif any(check["status"] != "completed" for check in checks):
            status = "pending"
        elif any(
            check["conclusion"] not in {"success", "neutral", "skipped"}
            for check in checks
        ):
            status = "failure"
        else:
            status = "success"

        return {
            "head_sha": head_sha,
            "status": status,
            "checks": checks,
        }

    def reply_to_issue(
        self,
        issue_number: int,
        body: str,
    ) -> dict[str, object]:
        """Create an issue comment."""
        comment = self._request(
            "POST",
            f"{self.repository_path}/issues/{issue_number}/comments",
            "reply to issue",
            json={"body": body},
        ).json()
        return {
            "id": comment["id"],
            "url": comment["html_url"],
        }

    def open_pull_request(
        self,
        title: str,
        body: str,
        head: str,
        base: str,
        draft: bool,
    ) -> dict[str, object]:
        """Create a pull request from an existing remote branch."""
        pull_request = self._request(
            "POST",
            f"{self.repository_path}/pulls",
            "open pull request",
            json={
                "title": title,
                "body": body,
                "head": head,
                "base": base,
                "draft": draft,
            },
        ).json()
        return {
            "number": pull_request["number"],
            "url": pull_request["html_url"],
            "draft": pull_request["draft"],
        }
