#!/usr/bin/env bash
# PostToolUse hook: after a real pytest invocation succeeds, commit any
# pending *tracked* changes as one "unit of work verified" commit.
#
# Never stages untracked files blindly (that's how a stray .env or key
# file gets committed) and only fires on an actual pytest invocation,
# not any command that happens to contain the substring "pytest".
set -u

input=$(cat)

decision=$(printf '%s' "$input" | jq -r '
  if ((.tool_input.command // "")
      | test("(^|&&\\s*|;\\s*)(pytest|python -m pytest|python3 -m pytest|uv run pytest|poetry run pytest)(\\s|$)"))
     and (.tool_response.success == true)
  then "commit" else "skip" end
')

[ "$decision" = "commit" ] || exit 0

if [ -z "${CLAUDE_PROJECT_DIR:-}" ]; then
  echo "auto-commit hook: CLAUDE_PROJECT_DIR not set, skipping" >&2
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR" || exit 0
git rev-parse --git-dir >/dev/null 2>&1 || exit 0

# Stage everything except common secret-file patterns, so new source
# files are still picked up but stray credentials are not.
git add -A -- . ':!*.env' ':!*.env.*' ':!*.pem' ':!*.key' ':!*id_rsa*' ':!*credentials*'

staged=$(git diff --cached --name-only)

# Defense in depth: verify nothing secret-shaped made it into the stage
# (e.g. a nested path or naming variant the pathspec excludes missed).
if printf '%s\n' "$staged" | grep -qiE '(^|/)\.env(\.|$)|\.pem$|\.key$|id_rsa|credentials'; then
  echo "auto-commit hook: refusing to commit, staged changes include a secret-like path" >&2
  git reset -- . >/dev/null 2>&1
  exit 0
fi

if [ -n "$staged" ]; then
  git commit -m "Automated commit: unit of work verified (pytest passing)" --quiet
fi

exit 0
