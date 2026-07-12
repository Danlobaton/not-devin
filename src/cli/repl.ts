import type { AgentRuntime } from "../core/runtime.ts";
import type { ApprovalInterruptPayload, RunResult } from "../core/types.ts";

export interface ReplIO {
  readonly askTask: () => Promise<string | null>;
  readonly askApproval: (payload: ApprovalInterruptPayload) => Promise<boolean>;
  readonly print: (text: string) => void;
}

const EXIT_COMMANDS = new Set(["exit", "quit"]);

/** Drives one interactive session on a single thread until the user exits. */
export async function runRepl(
  runtime: AgentRuntime,
  threadId: string,
  model: string,
  io: ReplIO,
): Promise<void> {
  for (;;) {
    const task = await io.askTask();
    if (task === null || EXIT_COMMANDS.has(task.trim().toLowerCase())) {
      return;
    }

    let result = await runtime.runTurn({ threadId, input: task, model });
    while (result.interrupted && result.interruptPayload !== undefined) {
      const approved = await io.askApproval(result.interruptPayload);
      result = await runtime.resume({ threadId, approved, model });
    }

    io.print(formatOutcome(result));
  }
}

function formatOutcome(result: RunResult): string {
  if (result.outcome === null) {
    return "(no outcome)";
  }
  return `[${result.outcome.reason}] ${result.outcome.detail}`;
}
