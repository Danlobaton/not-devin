"""Runtime verification of claimed success via an operator-supplied command."""

from __future__ import annotations

import subprocess
from collections.abc import Callable

from langchain_core.messages import HumanMessage

from .state import AgentState

VERIFY_TIMEOUT_SECONDS = 300
MAX_VERIFY_OUTPUT_CHARS = 10_000


def build_verify_node() -> Callable[[AgentState], dict]:
    """Build a graph node that verifies the model's claim of success.

    With no ``verify_command`` in state the node declares success
    immediately (preserving pre-verify behavior). Otherwise it runs the
    command in the workspace: exit 0 is success; anything else feeds the
    tail-truncated output back to the agent as a HumanMessage so the model
    can iterate, bounded by the run's ``max_iterations``.

    Returns:
        A node callable producing a partial AgentState update.
    """

    def verify_node(state: AgentState) -> dict:
        command = state.get("verify_command")
        if not command:
            return {"terminal_reason": "success"}

        try:
            result = subprocess.run(
                command,
                # Operator-supplied CLI config (npm-script trust level),
                # never model-derived — shell semantics are the use case.
                shell=True,
                cwd=state["workspace"],
                capture_output=True,
                text=True,
                timeout=VERIFY_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return {
                "messages": [
                    HumanMessage(
                        content=(
                            "Verification command timed out after "
                            f"{VERIFY_TIMEOUT_SECONDS} seconds: {command}"
                        )
                    )
                ]
            }

        if result.returncode == 0:
            return {"terminal_reason": "success"}

        output = (result.stdout + result.stderr)[-MAX_VERIFY_OUTPUT_CHARS:]
        return {
            "messages": [
                HumanMessage(
                    content=(
                        f"Verification failed (exit {result.returncode}).\n"
                        f"Command: {command}\n"
                        f"Output (tail):\n{output}"
                    )
                )
            ]
        }

    return verify_node
