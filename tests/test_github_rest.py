from collections.abc import Callable

import httpx
import pytest

from not_devin.github.errors import GitHubServiceError
from not_devin.github.rest import GitHubRestService


class StaticTokenProvider:
    def get_token(self) -> str:
        return "installation-token"


def make_service(
    handler: Callable[[httpx.Request], httpx.Response],
) -> GitHubRestService:
    client = httpx.Client(
        base_url="https://api.github.test",
        transport=httpx.MockTransport(handler),
    )
    return GitHubRestService(
        "octo",
        "repo",
        StaticTokenProvider(),
        client,
    )


def test_get_issue_context_paginates_comments() -> None:
    comment_pages: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repos/octo/repo/issues/7":
            return httpx.Response(
                200,
                json={
                    "number": 7,
                    "title": "Bug",
                    "body": "Please fix",
                    "state": "open",
                    "html_url": "https://github.test/octo/repo/issues/7",
                    "user": {"login": "alice"},
                },
            )

        page = int(request.url.params["page"])
        comment_pages.append(page)
        count = 100 if page == 1 else 1
        return httpx.Response(
            200,
            json=[
                {
                    "id": (page - 1) * 100 + index,
                    "body": f"comment-{page}-{index}",
                    "created_at": "2026-01-01T00:00:00Z",
                    "html_url": "https://github.test/comment",
                    "user": {"login": "bob"},
                }
                for index in range(count)
            ],
        )

    result = make_service(handler).get_issue_context(7)

    assert result["title"] == "Bug"
    assert result["author"] == "alice"
    assert len(result["comments"]) == 101
    assert comment_pages == [1, 2]


@pytest.mark.parametrize(
    ("checks", "expected"),
    [
        ([], "no_checks"),
        (
            [
                {
                    "name": "tests",
                    "status": "in_progress",
                    "conclusion": None,
                    "details_url": "https://github.test/check",
                }
            ],
            "pending",
        ),
        (
            [
                {
                    "name": "tests",
                    "status": "completed",
                    "conclusion": "failure",
                    "details_url": "https://github.test/check",
                }
            ],
            "failure",
        ),
        (
            [
                {
                    "name": "tests",
                    "status": "completed",
                    "conclusion": "success",
                    "details_url": "https://github.test/check",
                }
            ],
            "success",
        ),
    ],
)
def test_get_ci_status_aggregates_checks(
    checks: list[dict[str, object]],
    expected: str,
) -> None:
    service = make_service(
        lambda request: httpx.Response(
            200,
            json={"total_count": len(checks), "check_runs": checks},
        )
    )

    result = service.get_ci_status("abc123")

    assert result["head_sha"] == "abc123"
    assert result["status"] == expected
    assert len(result["checks"]) == len(checks)


def test_reply_to_issue_posts_comment() -> None:
    bodies: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.read())
        return httpx.Response(
            201,
            json={
                "id": 10,
                "html_url": "https://github.test/comment/10",
            },
        )

    result = make_service(handler).reply_to_issue(7, "Implemented")

    assert result == {
        "id": 10,
        "url": "https://github.test/comment/10",
    }
    assert bodies == [b'{"body":"Implemented"}']


def test_open_pull_request_posts_existing_head_branch() -> None:
    payloads: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(request.read())
        return httpx.Response(
            201,
            json={
                "number": 2,
                "html_url": "https://github.test/pull/2",
                "draft": True,
            },
        )

    result = make_service(handler).open_pull_request(
        "Fix bug",
        "Closes #7",
        "not-devin/issue-7",
        "main",
        True,
    )

    assert result["url"] == "https://github.test/pull/2"
    assert payloads == [
        (
            b'{"title":"Fix bug","body":"Closes #7",'
            b'"head":"not-devin/issue-7","base":"main","draft":true}'
        )
    ]


def test_rate_limit_error_includes_reset_without_body() -> None:
    service = make_service(
        lambda request: httpx.Response(
            403,
            text="installation-token",
            headers={
                "x-ratelimit-remaining": "0",
                "x-ratelimit-reset": "12345",
            },
            request=request,
        )
    )

    with pytest.raises(GitHubServiceError) as captured:
        service.get_issue_context(7)

    assert "HTTP 403" in str(captured.value)
    assert "12345" in str(captured.value)
    assert "installation-token" not in str(captured.value)


def test_network_error_is_normalized() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("installation-token", request=request)

    with pytest.raises(
        GitHubServiceError,
        match="network error",
    ):
        make_service(handler).get_ci_status("abc123")
