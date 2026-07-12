import { randomUUID } from "node:crypto";
import { parseArgs } from "node:util";
import { confirm, input } from "@inquirer/prompts";
import { resolveModel } from "../core/model-registry.ts";
import { AgentRuntime } from "../core/runtime.ts";
import { formatTrace } from "./inspect.ts";
import { runRepl, type ReplIO } from "./repl.ts";

const DEFAULT_BUDGETS = {
  maxIterations: 50,
  maxToolCalls: 50,
  maxWallClockMs: 10 * 60 * 1000,
};

async function main(): Promise<void> {
  const { positionals, values } = parseArgs({
    args: process.argv.slice(2),
    allowPositionals: true,
    options: {
      model: { type: "string" },
      workspace: { type: "string" },
      "runs-dir": { type: "string" },
    },
  });

  const runsDir = values["runs-dir"] ?? ".not-devin/runs";

  if (positionals[0] === "inspect") {
    const threadId = positionals[1];
    if (threadId === undefined) {
      console.error("Usage: not-devin inspect <threadId> [--runs-dir <path>]");
      process.exitCode = 1;
      return;
    }
    console.log(await formatTrace(threadId, runsDir));
    return;
  }

  if (values.model === undefined) {
    console.error(
      "Usage: not-devin --model <provider:model> [--workspace <path>] [--runs-dir <path>]",
    );
    process.exitCode = 1;
    return;
  }

  const runtime = new AgentRuntime(
    {
      model: values.model,
      budgets: DEFAULT_BUDGETS,
      workspaceRoot: values.workspace ?? process.cwd(),
      approvalMode: "policy",
      runsDir,
    },
    [],
    resolveModel,
  );

  const io: ReplIO = {
    askTask: async () => {
      try {
        return await input({ message: "task>" });
      } catch (error) {
        if (isExitPromptError(error)) {
          return null;
        }
        throw error;
      }
    },
    askApproval: async (payload) => {
      try {
        return await confirm({
          message: `Approve ${payload.toolName}(${JSON.stringify(payload.args)})?`,
          default: false,
        });
      } catch (error) {
        if (isExitPromptError(error)) {
          return false;
        }
        throw error;
      }
    },
    print: (text: string) => console.log(text),
  };

  await runRepl(runtime, randomUUID(), values.model, io);
}

function isExitPromptError(error: unknown): boolean {
  return error instanceof Error && error.name === "ExitPromptError";
}

main().catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});
