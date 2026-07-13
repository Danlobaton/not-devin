"""Model-facing GitHub tools."""

from langchain_core.tools import BaseTool, tool

from ..github import GitHubService


def build_github_tools(service: GitHubService) -> list[BaseTool]:
    """Build repository-scoped tools backed by a GitHub service."""

    @tool
    def get_issue_context(issue_number: int) -> dict[str, object]:
        """Get an issue and its complete chronological comment thread.

        Args:
            issue_number: Repository-local issue number.
        """
        return service.get_issue_context(issue_number)

    @tool
    def get_ci_status(head_sha: str) -> dict[str, object]:
        """Get check results for a remote commit SHA.

        Args:
            head_sha: Full commit SHA already present on GitHub.
        """
        return service.get_ci_status(head_sha)

    @tool
    def reply_to_issue(
        issue_number: int,
        body: str,
    ) -> dict[str, object]:
        """Post a Markdown reply to an issue.

        Args:
            issue_number: Repository-local issue number.
            body: Markdown comment body.
        """
        return service.reply_to_issue(issue_number, body)

    @tool
    def open_pull_request(
        title: str,
        body: str,
        head: str,
        base: str,
        draft: bool = True,
    ) -> dict[str, object]:
        """Open a pull request from an existing remote branch.

        Args:
            title: Pull request title.
            body: Pull request Markdown description.
            head: Existing remote branch containing the changes.
            base: Branch that should receive the changes.
            draft: Whether to create the pull request as a draft.
        """
        return service.open_pull_request(title, body, head, base, draft)

    return [
        get_issue_context,
        get_ci_status,
        reply_to_issue,
        open_pull_request,
    ]
