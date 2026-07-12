import { AIMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { FakeToolCallingModel } from "langchain";
import { describe, expect, it, vi } from "vitest";
import { z } from "zod";
import { AgentRuntime } from "../core/runtime.ts";
import type { ApprovalInterruptPayload, RegisteredTool } from "../core/types.ts";
import { runRepl, type ReplIO } from "./repl.ts";

const defaultBudgets = {
  maxIterations: 8,
  maxToolCalls: 8,
  maxWallClockMs: 30_000,
};

function makeRuntime(
  modelResolver: (modelId: string) => Promise<BaseChatModel>,
  tools: RegisteredTool[] = [],
): AgentRuntime {
  return new AgentRuntime(
    {
      model: "openai:test-model",
      budgets: defaultBudgets,
      workspaceRoot: "/tmp/workspace",
      approvalMode: "policy",
    },
    tools,
    modelResolver,
  );
}

function makeIO(options: {
  tasks?: (string | null)[];
  approvals?: boolean[];
} = {}): ReplIO & { printed: string[] } {
  const printed: string[] = [];
  const tasks = options.tasks ?? [null];
  const approvals = options.approvals ?? [];
  let taskIndex = 0;
  let approvalIndex = 0;
  return {
    printed,
    askTask: async () => tasks[taskIndex++] ?? null,
    askApproval: async () => approvals[approvalIndex++] ?? true,
    print: (text: string) => printed.push(text),
  };
}

describe("runRepl", () => {
  it("exits without running a turn when the first input is an exit command", async () => {
    const model = new FakeToolCallingModel({
      toolCalls: [[]],
      structuredResponse: new AIMessage("unused"),
    });
    const runtime = makeRuntime(async () => model);
    const runTurnSpy = vi.spyOn(runtime, "runTurn");
    const io = makeIO({ tasks: ["exit"] });

    await runRepl(runtime, "thread-exit", "openai:test-model", io);

    expect(runTurnSpy).not.toHaveBeenCalled();
    expect(io.printed).toEqual([]);
  });

  it("runs a task and prints its outcome when no approval is needed", async () => {
    const model = new FakeToolCallingModel({
      toolCalls: [[], []],
      structuredResponse: new AIMessage("Done."),
    });
    const runtime = makeRuntime(async () => model);
    const io = makeIO({ tasks: ["say hello", null] });

    await runRepl(runtime, "thread-success", "openai:test-model", io);

    expect(io.printed).toHaveLength(1);
    expect(io.printed[0]).toContain("success");
  });

  it("asks for approval with the pending call and resumes once approved", async () => {
    const protectedTool = tool(
      async ({ text }: { text: string }) => `protected:${text}`,
      {
        name: "protected_echo",
        description: "Protected echo",
        schema: z.object({ text: z.string() }),
      },
    );
    const model = new FakeToolCallingModel({
      toolCalls: [[{ id: "call_1", name: "protected_echo", args: { text: "secret" } }], []],
      structuredResponse: new AIMessage("Approved run complete."),
    });
    const runtime = makeRuntime(async () => model, [
      { tool: protectedTool, requiresApproval: true },
    ]);

    const printed: string[] = [];
    const approvalCalls: ApprovalInterruptPayload[] = [];
    const tasks: (string | null)[] = ["use protected tool", null];
    let taskIndex = 0;
    const io: ReplIO = {
      askTask: async () => tasks[taskIndex++] ?? null,
      askApproval: async (payload) => {
        approvalCalls.push(payload);
        return true;
      },
      print: (text: string) => printed.push(text),
    };

    await runRepl(runtime, "thread-approval", "openai:test-model", io);

    expect(approvalCalls).toHaveLength(1);
    expect(approvalCalls[0]?.toolName).toBe("protected_echo");
    expect(printed).toHaveLength(1);
    expect(printed[0]).toContain("success");
  });

  it("prints approval_denied and does not retry the tool when approval is refused", async () => {
    const protectedTool = tool(
      async ({ text }: { text: string }) => `protected:${text}`,
      {
        name: "protected_echo",
        description: "Protected echo",
        schema: z.object({ text: z.string() }),
      },
    );
    const model = new FakeToolCallingModel({
      toolCalls: [[{ id: "call_1", name: "protected_echo", args: { text: "secret" } }]],
      structuredResponse: new AIMessage("Should not finish"),
    });
    const runtime = makeRuntime(async () => model, [
      { tool: protectedTool, requiresApproval: true },
    ]);
    const io = makeIO({ tasks: ["use protected tool", null], approvals: [false] });

    await runRepl(runtime, "thread-denied", "openai:test-model", io);

    expect(io.printed).toHaveLength(1);
    expect(io.printed[0]).toContain("approval_denied");
  });
});
