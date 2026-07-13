"""not-devin command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

from .event_log.reader import read_events
from .event_log.render import render_timeline
from .event_log.writer import DEFAULT_ROOT
from .runner import resume_agent, run_agent

DEFAULT_MAX_ITERATIONS = 25


def _default_model() -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    load_dotenv()
    return ChatOpenAI(model="gpt-4o-mini")


def cmd_run(
    workspace: Path,
    task: str,
    model: BaseChatModel | None = None,
    root: Path = DEFAULT_ROOT,
) -> int:
    """Run the agent on a workspace + task, logging every event."""
    run_id, final_state = run_agent(
        model or _default_model(),
        task=task,
        workspace=str(workspace),
        max_iterations=DEFAULT_MAX_ITERATIONS,
        root=root,
    )
    print(f"run_id: {run_id}")
    print(f"terminal_reason: {final_state.get('terminal_reason')}")
    return 0 if final_state.get("terminal_reason") == "success" else 1


def cmd_inspect(run_id: str, root: Path = DEFAULT_ROOT) -> int:
    """Inspect a persisted run's event log."""
    events = read_events(Path(root) / run_id / "events.jsonl")
    if not events:
        print(f"no events found for run_id: {run_id}")
        return 1
    print(render_timeline(events))
    return 0


def cmd_resume(
    run_id: str,
    model: BaseChatModel | None = None,
    root: Path = DEFAULT_ROOT,
) -> int:
    """Resume a run from its event log."""
    final_state = resume_agent(run_id, model or _default_model(), root=root)
    print(f"terminal_reason: {final_state.get('terminal_reason')}")
    return 0 if final_state.get("terminal_reason") == "success" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="not-devin")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the agent on a workspace")
    run_p.add_argument("--workspace", type=Path, required=True)
    run_p.add_argument("--task", type=str, required=True)

    inspect_p = sub.add_parser("inspect", help="Inspect a run's event log")
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
