"""Provider-neutral interface for repository-scoped GitHub operations."""

from typing import Protocol


class GitHubService(Protocol):
    """Operations exposed to the agent's model-facing GitHub tools."""

    def get_issue_context(self, issue_number: int) -> dict[str, object]:
        """Return an issue and its chronological comment thread."""
        ...

    def get_ci_status(self, head_sha: str) -> dict[str, object]:
        """Return check results for a remote commit SHA."""
        ...

    def reply_to_issue(
        self,
        issue_number: int,
        body: str,
    ) -> dict[str, object]:
        """Create an issue comment."""
        ...

    def open_pull_request(
        self,
        title: str,
        body: str,
        head: str,
        base: str,
        draft: bool,
    ) -> dict[str, object]:
        """Open a pull request from an existing remote branch."""
        ...
