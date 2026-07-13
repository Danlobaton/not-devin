from typing import Any

from not_devin.tools.github import build_github_tools


class FakeGitHubService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def get_issue_context(self, issue_number: int) -> dict[str, object]:
        self.calls.append(("get_issue_context", (issue_number,)))
        return {"number": issue_number, "title": "Bug", "comments": []}

    def get_ci_status(self, head_sha: str) -> dict[str, object]:
        self.calls.append(("get_ci_status", (head_sha,)))
        return {"head_sha": head_sha, "status": "success", "checks": []}

    def reply_to_issue(self, issue_number: int, body: str) -> dict[str, object]:
        self.calls.append(("reply_to_issue", (issue_number, body)))
        return {"id": 10, "url": "https://example.test/comment/10"}

    def open_pull_request(
        self,
        title: str,
        body: str,
        head: str,
        base: str,
        draft: bool,
    ) -> dict[str, object]:
        self.calls.append(
            ("open_pull_request", (title, body, head, base, draft))
        )
        return {
            "number": 2,
            "url": "https://example.test/pull/2",
            "draft": draft,
        }


def tools_by_name(service: FakeGitHubService) -> dict[str, Any]:
    return {tool.name: tool for tool in build_github_tools(service)}


def test_get_issue_context_tool() -> None:
    service = FakeGitHubService()
    result = tools_by_name(service)["get_issue_context"].invoke(
        {"issue_number": 7}
    )

    assert result == {"number": 7, "title": "Bug", "comments": []}
    assert service.calls == [("get_issue_context", (7,))]


def test_get_ci_status_tool() -> None:
    service = FakeGitHubService()
    result = tools_by_name(service)["get_ci_status"].invoke(
        {"head_sha": "abc123"}
    )

    assert result["status"] == "success"
    assert service.calls == [("get_ci_status", ("abc123",))]


def test_reply_to_issue_tool() -> None:
    service = FakeGitHubService()
    result = tools_by_name(service)["reply_to_issue"].invoke(
        {"issue_number": 7, "body": "Implemented in #2"}
    )

    assert result["id"] == 10
    assert service.calls == [
        ("reply_to_issue", (7, "Implemented in #2"))
    ]


def test_open_pull_request_tool_defaults_to_draft() -> None:
    service = FakeGitHubService()
    result = tools_by_name(service)["open_pull_request"].invoke(
        {
            "title": "Fix bug",
            "body": "Closes #7",
            "head": "not-devin/issue-7",
            "base": "main",
        }
    )

    assert result["number"] == 2
    assert service.calls == [
        (
            "open_pull_request",
            ("Fix bug", "Closes #7", "not-devin/issue-7", "main", True),
        )
    ]
