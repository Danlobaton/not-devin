import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { tool } from "@langchain/core/tools";
import { AIMessage } from "@langchain/core/messages";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { z } from "zod";
import { FakeToolCallingModel } from "langchain";
import { AgentRuntime } from "./runtime.ts";
import type { RegisteredTool } from "./types.ts";
import { UnsupportedModelError } from "./model-registry.ts";
import { validateToolCall } from "./tool-validation.ts";

const defaultBudgets = {
  maxIterations: 8,
  maxToolCalls: 8,
  maxWallClockMs: 30_000,
};

const echoTool = tool(
  async ({ text }: { text: string }) => `echo:${text}`,
  {
    name: "echo",
    description: "Echo text back",
    schema: z.object({ text: z.string() }),
  },
);

const protectedEchoTool = tool(
  async ({ text }: { text: string }) => `protected:${text}`,
  {
    name: "protected_echo",
    description: "Protected echo",
    schema: z.object({ text: z.string() }),
  },
);

const workspaceTool = tool(
  async (_input, config) => {
    const workspaceRoot = config?.configurable?.workspaceRoot;
    return `workspace:${workspaceRoot ?? "missing"}`;
  },
  {
    name: "workspace_probe",
    description: "Reads workspace root from runtime context",
    schema: z.object({}),
  },
);

const runLikeTool = tool(
  async ({ command }: { command: string }) => `ran:${command}`,
  {
    name: "run_like_tool",
    description: "Tool whose approval depends on the command argument",
    schema: z.object({ command: z.string() }),
  },
);

const requiresApprovalUnlessSafe = (call: { args: Record<string, unknown> }) =>
  call.args.command !== "safe";

function makeRuntime(
  modelResolver: (modelId: string) => Promise<BaseChatModel>,
  tools: RegisteredTool[],
  options?: {
    approvalMode?: "policy" | "skip_all";
    budgets?: typeof defaultBudgets;
    runsDir?: string;
    workspaceRoot?: string;
  },
): AgentRuntime {
  return new AgentRuntime(
    {
      model: "openai:test-model",
      budgets: options?.budgets ?? defaultBudgets,
      workspaceRoot: options?.workspaceRoot ?? "/tmp/workspace",
      approvalMode: options?.approvalMode ?? "policy",
      ...(options?.runsDir === undefined ? {} : { runsDir: options.runsDir }),
    },
    tools,
    modelResolver,
  );
}

describe("validateToolCall", () => {
  it("rejects malformed schema arguments before execution", () => {
    const result = validateToolCall(
      { id: "call_1", name: "echo", args: { text: 123 }, type: "tool_call" },
      { tool: echoTool, requiresApproval: false },
    );

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.reason).toContain("failed schema validation");
    }
  });
});

describe("AgentRuntime", () => {
  let runsDir: string;

  beforeEach(async () => {
    runsDir = await mkdtemp(join(tmpdir(), "not-devin-runs-"));
  });

  afterEach(async () => {
    await rm(runsDir, { recursive: true, force: true });
  });

  it("terminates with success when the model responds without tool calls", async () => {
    const model = new FakeToolCallingModel({
      toolCalls: [[], []],
      structuredResponse: new AIMessage("Done."),
    });
    const runtime = makeRuntime(async () => model, [{ tool: echoTool, requiresApproval: false }]);

    const result = await runtime.runTurn({
      threadId: "success-thread",
      input: "Say done",
      model: "openai:test-model",
    });

    expect(result.interrupted).toBe(false);
    expect(result.outcome?.reason).toBe("success");
    expect(result.events.some((event) => event.type === "terminal")).toBe(true);
  });

  it("executes multiple tool calls sequentially in one model turn", async () => {
    const executionOrder: string[] = [];
    const countingEcho = tool(
      async ({ text }: { text: string }) => {
        executionOrder.push(text);
        return `echo:${text}`;
      },
      {
        name: "echo",
        description: "Echo text back",
        schema: z.object({ text: z.string() }),
      },
    );
    const model = new FakeToolCallingModel({
      toolCalls: [
        [
          { id: "call_1", name: "echo", args: { text: "first" } },
          { id: "call_2", name: "echo", args: { text: "second" } },
        ],
        [],
      ],
      structuredResponse: new AIMessage("Finished."),
    });
    const runtime = makeRuntime(async () => model, [
      { tool: countingEcho, requiresApproval: false },
    ]);

    const result = await runtime.runTurn({
      threadId: "sequential-thread",
      input: "Echo twice",
      model: "openai:test-model",
    });

    expect(result.outcome?.reason).toBe("success");
    expect(executionOrder).toEqual(["first", "second"]);
    expect(result.events.filter((event) => event.type === "tool_call")).toHaveLength(2);
  });

  it("switches models on the same thread across turns", async () => {
    const openaiModel = new FakeToolCallingModel({
      toolCalls: [[], []],
      structuredResponse: new AIMessage("from-openai"),
    });
    const anthropicModel = new FakeToolCallingModel({
      toolCalls: [[], []],
      structuredResponse: new AIMessage("from-anthropic"),
    });
    const seenModels: string[] = [];
    const runtime = makeRuntime(async (modelId) => {
      seenModels.push(modelId);
      return modelId === "openai:test-model" ? openaiModel : anthropicModel;
    }, [{ tool: echoTool, requiresApproval: false }]);

    await runtime.runTurn({
      threadId: "same-thread",
      input: "first",
      model: "openai:test-model",
    });
    await runtime.runTurn({
      threadId: "same-thread",
      input: "second",
      model: "anthropic:test-model",
    });

    expect(seenModels).toEqual(["openai:test-model", "anthropic:test-model"]);
  });

  it("interrupts for approval and resumes when approved", async () => {
    const model = new FakeToolCallingModel({
      toolCalls: [
        [{ id: "call_1", name: "protected_echo", args: { text: "secret" } }],
        [],
      ],
      structuredResponse: new AIMessage("Approved run complete."),
    });
    const runtime = makeRuntime(async () => model, [
      { tool: protectedEchoTool, requiresApproval: true },
    ]);

    const paused = await runtime.runTurn({
      threadId: "approval-thread",
      input: "Use protected tool",
      model: "openai:test-model",
    });

    expect(paused.interrupted).toBe(true);
    expect(paused.interruptPayload?.toolName).toBe("protected_echo");

    const resumed = await runtime.resume({
      threadId: "approval-thread",
      approved: true,
      model: "openai:test-model",
    });

    expect(resumed.interrupted).toBe(false);
    expect(resumed.outcome?.reason).toBe("success");
  });

  it("terminates with approval_denied when resume rejects the tool", async () => {
    const model = new FakeToolCallingModel({
      toolCalls: [[{ id: "call_1", name: "protected_echo", args: { text: "secret" } }]],
      structuredResponse: new AIMessage("Should not finish"),
    });
    const runtime = makeRuntime(async () => model, [
      { tool: protectedEchoTool, requiresApproval: true },
    ]);

    await runtime.runTurn({
      threadId: "deny-thread",
      input: "Use protected tool",
      model: "openai:test-model",
    });

    const denied = await runtime.resume({
      threadId: "deny-thread",
      approved: false,
      model: "openai:test-model",
    });

    expect(denied.interrupted).toBe(false);
    expect(denied.outcome?.reason).toBe("approval_denied");
  });

  it("does not pause when a per-call approval predicate returns false for the call's args", async () => {
    const model = new FakeToolCallingModel({
      toolCalls: [[{ id: "call_1", name: "run_like_tool", args: { command: "safe" } }], []],
      structuredResponse: new AIMessage("Done."),
    });
    const runtime = makeRuntime(async () => model, [
      { tool: runLikeTool, requiresApproval: requiresApprovalUnlessSafe },
    ]);

    const result = await runtime.runTurn({
      threadId: "predicate-safe-thread",
      input: "Run something safe",
      model: "openai:test-model",
    });

    expect(result.interrupted).toBe(false);
    expect(result.outcome?.reason).toBe("success");
  });

  it("pauses for approval when a per-call approval predicate returns true for the call's args", async () => {
    const model = new FakeToolCallingModel({
      toolCalls: [[{ id: "call_1", name: "run_like_tool", args: { command: "rm -rf /" } }]],
      structuredResponse: new AIMessage("Should not finish"),
    });
    const runtime = makeRuntime(async () => model, [
      { tool: runLikeTool, requiresApproval: requiresApprovalUnlessSafe },
    ]);

    const result = await runtime.runTurn({
      threadId: "predicate-unsafe-thread",
      input: "Run something unsafe",
      model: "openai:test-model",
    });

    expect(result.interrupted).toBe(true);
    expect(result.interruptPayload?.toolName).toBe("run_like_tool");
  });

  it("auto-approves protected tools when skipPermissionsCheck is enabled", async () => {
    const model = new FakeToolCallingModel({
      toolCalls: [
        [{ id: "call_1", name: "protected_echo", args: { text: "secret" } }],
        [],
      ],
      structuredResponse: new AIMessage("Bypassed."),
    });
    const runtime = makeRuntime(async () => model, [
      { tool: protectedEchoTool, requiresApproval: true },
    ]);

    const result = await runtime.runTurn({
      threadId: "bypass-thread",
      input: "Use protected tool",
      model: "openai:test-model",
      skipPermissionsCheck: true,
    });

    expect(result.interrupted).toBe(false);
    expect(result.outcome?.reason).toBe("success");
    expect(
      result.events.some(
        (event) =>
          event.type === "approval_decision" &&
          event.approved &&
          event.reason === "bypassed via --skip-permissions-check",
      ),
    ).toBe(true);
  });

  it("terminates with invalid_tool_call for unknown tools", async () => {
    const model = new FakeToolCallingModel({
      toolCalls: [[{ id: "call_1", name: "missing_tool", args: {} }]],
      structuredResponse: new AIMessage("Nope"),
    });
    const runtime = makeRuntime(async () => model, [{ tool: echoTool, requiresApproval: false }]);

    const result = await runtime.runTurn({
      threadId: "unknown-tool-thread",
      input: "Call missing tool",
      model: "openai:test-model",
    });

    expect(result.outcome?.reason).toBe("invalid_tool_call");
  });

  it("terminates with invalid_tool_call for schema-invalid arguments", async () => {
    const model = new FakeToolCallingModel({
      toolCalls: [[{ id: "call_1", name: "echo", args: { text: 123 } }]],
      structuredResponse: new AIMessage("Nope"),
    });
    const runtime = makeRuntime(async () => model, [{ tool: echoTool, requiresApproval: false }]);

    const result = await runtime.runTurn({
      threadId: "invalid-args-thread",
      input: "Call with bad args",
      model: "openai:test-model",
    });

    expect(result.outcome?.reason).toBe("invalid_tool_call");
  });

  it("terminates with iteration_limit when tool-call budget is exceeded", async () => {
    const model = new FakeToolCallingModel({
      toolCalls: [
        [{ id: "call_1", name: "echo", args: { text: "one" } }],
        [{ id: "call_2", name: "echo", args: { text: "two" } }],
      ],
      structuredResponse: new AIMessage("loop"),
    });
    const runtime = makeRuntime(
      async () => model,
      [{ tool: echoTool, requiresApproval: false }],
      { budgets: { maxIterations: 10, maxToolCalls: 1, maxWallClockMs: 30_000 } },
    );

    const result = await runtime.runTurn({
      threadId: "budget-thread",
      input: "Loop tools",
      model: "openai:test-model",
    });

    expect(result.outcome?.reason).toBe("iteration_limit");
  });

  it("terminates with repeated_tool_calls when the model repeats an identical request", async () => {
    const model = new FakeToolCallingModel({
      toolCalls: [
        [{ id: "call_1", name: "echo", args: { text: "same" } }],
        [{ id: "call_2", name: "echo", args: { text: "same" } }],
      ],
      structuredResponse: new AIMessage("loop"),
    });
    const runtime = makeRuntime(async () => model, [{ tool: echoTool, requiresApproval: false }]);

    const result = await runtime.runTurn({
      threadId: "repeat-thread",
      input: "Repeat tool",
      model: "openai:test-model",
    });

    expect(result.outcome?.reason).toBe("repeated_tool_calls");
  });

  it("terminates with timeout when the wall-clock budget is exceeded", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-01-01T00:00:00.000Z"));

    const model = new FakeToolCallingModel({
      toolCalls: [
        [{ id: "call_1", name: "echo", args: { text: "one" } }],
        [],
      ],
      structuredResponse: new AIMessage("done"),
    });
    const runtime = makeRuntime(
      async () => model,
      [{ tool: echoTool, requiresApproval: false }],
      { budgets: { maxIterations: 8, maxToolCalls: 8, maxWallClockMs: 1_000 } },
    );

    const first = await runtime.runTurn({
      threadId: "timeout-thread",
      input: "Start",
      model: "openai:test-model",
    });
    expect(first.interrupted).toBe(false);

    vi.setSystemTime(new Date("2026-01-01T00:00:02.000Z"));
    const second = await runtime.runTurn({
      threadId: "timeout-thread",
      input: "Continue after timeout",
      model: "openai:test-model",
    });

    expect(second.outcome?.reason).toBe("timeout");
    vi.useRealTimers();
  });

  it("passes workspaceRoot to tools through invoke config", async () => {
    const model = new FakeToolCallingModel({
      toolCalls: [
        [{ id: "call_1", name: "workspace_probe", args: {} }],
        [],
      ],
      structuredResponse: new AIMessage("done"),
    });
    const runtime = makeRuntime(
      async () => model,
      [{ tool: workspaceTool, requiresApproval: false }],
      { workspaceRoot: "/safe/workspace" },
    );

    const result = await runtime.runTurn({
      threadId: "workspace-thread",
      input: "Probe workspace",
      model: "openai:test-model",
    });

    const toolResult = result.events.find((event) => event.type === "tool_result");
    expect(toolResult?.type === "tool_result" && toolResult.payload).toBe(
      "workspace:/safe/workspace",
    );
  });

  it("persists only new events to JSONL", async () => {
    const model = new FakeToolCallingModel({
      toolCalls: [[], []],
      structuredResponse: new AIMessage("Done."),
    });
    const runtime = makeRuntime(
      async () => model,
      [{ tool: echoTool, requiresApproval: false }],
      { runsDir },
    );

    await runtime.runTurn({
      threadId: "persist-thread",
      input: "First",
      model: "openai:test-model",
    });
    await runtime.runTurn({
      threadId: "persist-thread",
      input: "Second",
      model: "openai:test-model",
    });

    const log = await readFile(join(runsDir, "persist-thread.jsonl"), "utf8");
    const lines = log.trim().split("\n");
    expect(lines.length).toBeGreaterThan(2);
    expect(lines.filter((line) => line.includes('"type":"user_task"'))).toHaveLength(2);
  });

  it("terminates with provider_failure when the model resolver throws", async () => {
    const runtime = makeRuntime(async () => {
      throw new Error("provider down");
    }, [{ tool: echoTool, requiresApproval: false }]);

    const result = await runtime.runTurn({
      threadId: "provider-thread",
      input: "Fail",
      model: "openai:test-model",
    });

    expect(result.outcome?.reason).toBe("provider_failure");
  });

  it("caches resolved models across turns", async () => {
    let resolveCount = 0;
    const model = new FakeToolCallingModel({
      toolCalls: [[], []],
      structuredResponse: new AIMessage("Done."),
    });
    const runtime = makeRuntime(async () => {
      resolveCount += 1;
      return model;
    }, [{ tool: echoTool, requiresApproval: false }]);

    await runtime.runTurn({
      threadId: "cache-thread",
      input: "One",
      model: "openai:test-model",
    });
    await runtime.runTurn({
      threadId: "cache-thread-2",
      input: "Two",
      model: "openai:test-model",
    });

    expect(resolveCount).toBe(1);
  });

  it("terminates with iteration_limit when graph recursion is exhausted", async () => {
    const model = new FakeToolCallingModel({
      toolCalls: [
        [{ id: "call_1", name: "echo", args: { text: "loop" } }],
        [{ id: "call_2", name: "echo", args: { text: "loop" } }],
        [{ id: "call_3", name: "echo", args: { text: "loop" } }],
      ],
      structuredResponse: new AIMessage("loop"),
    });
    const runtime = makeRuntime(
      async () => model,
      [{ tool: echoTool, requiresApproval: false }],
      { budgets: { maxIterations: 3, maxToolCalls: 20, maxWallClockMs: 30_000 } },
    );

    const result = await runtime.runTurn({
      threadId: "recursion-thread",
      input: "Loop until recursion limit",
      model: "openai:test-model",
    });

    expect(result.outcome?.reason).toBe("iteration_limit");
  });
});

describe("resolveModel", () => {
  it("rejects unsupported model identifiers", async () => {
    const { resolveModel } = await import("./model-registry.ts");
    await expect(resolveModel("example-model")).rejects.toBeInstanceOf(UnsupportedModelError);
  });
});
