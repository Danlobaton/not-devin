from __future__ import annotations

import argparse
from pathlib import Path


def cmd_run(workspace: Path, task: str) -> int:
    """Run the agent on a workspace + task. Not implemented yet."""
    _ = (workspace, task)
    print("run: not implemented yet")
    return 1


def cmd_inspect(run_id: str) -> int:
    """Inspect a persisted run trace. Not implemented yet."""
    _ = run_id
    print("inspect: not implemented yet")
    return 1


def cmd_resume(run_id: str) -> int:
    """Resume a run from its event log. Not implemented yet."""
    _ = run_id
    print("resume: not implemented yet")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="not-devin")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the agent on a workspace")
    run_p.add_argument("--workspace", type=Path, required=True)
    run_p.add_argument("--task", type=str, required=True)

    inspect_p = sub.add_parser("inspect", help="Inspect a run trace")
    inspect_p.add_argument("run_id")

    resume_p = sub.add_parser("resume", help="Resume a run from its event log")
    resume_p.add_argument("run_id")

    args = parser.parse_args(argv)

    if args.command == "run":
        return cmd_run(args.workspace, args.task)
    if args.command == "inspect":
        return cmd_inspect(args.run_id)
    if args.command == "resume":
        return cmd_resume(args.run_id)

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
