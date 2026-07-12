import {
  AIMessage,
  HumanMessage,
  ToolMessage,
  type BaseMessage,
} from "@langchain/core/messages";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import {
  Annotation,
  Command,
  END,
  INTERRUPT,
  MemorySaver,
  START,
  StateGraph,
  getConfig,
  interrupt,
  isInterrupted,
  messagesStateReducer,
  GraphRecursionError,
} from "@langchain/langgraph";
import { appendRunEvents } from "./audit/event-log.ts";
import { createAuditEvent, terminalEvent } from "./audit/events.ts";
import {
  getFinishReason,
  getLatestToolCalls,
  getModelUsage,
  outcomeForModelResponse,
} from "./message-helpers.ts";
import { UnsupportedModelError, resolveModel, type ModelResolver } from "./model-registry.ts";
import { toolCallSignature, validateToolCall } from "./tool-validation.ts";
import type {
  ApprovalInterruptPayload,
  AuditEvent,
  RegisteredTool,
  ResumeParams,
  RunConfig,
  RunOutcome,
  RunResult,
  RunTurnParams,
  TimestampedAuditEventPayload,
} from "./types.ts";

/** Per-invocation context threaded through the graph via LangGraph's `configurable`. */
interface GraphConfigurable {
  thread_id: string;
  model: string;
  skipPermissionsCheck: boolean;
}

/** Shape of the LangGraph state channel — see `AgentState` for reducers. */
interface AgentGraphState {
  messages: BaseMessage[];
  events: TimestampedAuditEventPayload[];
  toolCallCount: number;
  outcome: RunOutcome | null;
  startedAtMs: number;
  lastToolSignature: string | null;
}

const AgentState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: messagesStateReducer,
    default: () => [],
  }),
  events: Annotation<TimestampedAuditEventPayload[]>({
    reducer: (left, right) => left.concat(right),
    default: () => [],
  }),
  toolCallCount: Annotation<number>({
    reducer: (_, right) => right,
    default: () => 0,
  }),
  outcome: Annotation<RunOutcome | null>({
    reducer: (_, right) => right,
    default: () => null,
  }),
  startedAtMs: Annotation<number>({
    reducer: (left, right) => (right === 0 ? left : right),
    default: () => 0,
  }),
  lastToolSignature: Annotation<string | null>({
    reducer: (_, right) => right,
    default: () => null,
  }),
});

/** Reads and validates the `configurable` context LangGraph passes into every node. */
function readConfigurable(): GraphConfigurable {
  const config = getConfig();
  const configurable = config.configurable as Partial<GraphConfigurable> | undefined;
  if (
    configurable?.thread_id === undefined ||
    configurable.model === undefined ||
    configurable.skipPermissionsCheck === undefined
  ) {
    throw new Error("Graph invocation is missing required configurable context.");
  }
  return {
    thread_id: configurable.thread_id,
    model: configurable.model,
    skipPermissionsCheck: configurable.skipPermissionsCheck,
  };
}

/** Builds the `invalid_tool_call` terminal outcome for a structural/schema validation failure. */
function invalidToolOutcome(detail: string): RunOutcome {
  return { reason: "invalid_tool_call", detail };
}

/** Builds the LangGraph agent loop with model routing, approval interrupts, and sequential tools. */
export class AgentRuntime {
  private readonly graph;
  private readonly toolsByName: Map<string, RegisteredTool>;
  private readonly runConfig: RunConfig;
  private readonly resolveModelFn: ModelResolver;
  private readonly modelCache = new Map<string, BaseChatModel>();
  private readonly persistedEventCount = new Map<string, number>();

  /**
   * Graph node: sends the accumulated message history to the model, enforces the
   * wall-clock budget, and records `model_request`/`model_response` audit events.
   * Sets a terminal outcome when the budget is exceeded, the provider call fails,
   * or the model finishes without requesting further tool calls.
   */
  private readonly modelCall = async (state: AgentGraphState) => {
    if (state.outcome !== null) {
      return {};
    }

    const { model } = readConfigurable();
    const startedAtMs = state.startedAtMs === 0 ? Date.now() : state.startedAtMs;
    const elapsed = Date.now() - startedAtMs;
    if (elapsed > this.runConfig.budgets.maxWallClockMs) {
      const outcome: RunOutcome = {
        reason: "timeout",
        detail: `Exceeded wall-clock budget of ${this.runConfig.budgets.maxWallClockMs}ms`,
      };
      return {
        startedAtMs,
        outcome,
        events: [terminalEvent(outcome)],
      };
    }

    const events: TimestampedAuditEventPayload[] = [
      createAuditEvent({
        type: "model_request",
        model,
        messageCount: state.messages.length,
        toolCount: this.toolsByName.size,
      }),
    ];

    const modelStarted = Date.now();
    try {
      const chatModel = await this.getChatModel(model);
      const response = await chatModel.invoke(state.messages);
      const aiMessage = AIMessage.isInstance(response) ? response : new AIMessage(response);
      const toolCallCount = aiMessage.tool_calls?.length ?? 0;
      const finishReason = getFinishReason(aiMessage);

      events.push(
        createAuditEvent({
          type: "model_response",
          model,
          text: typeof aiMessage.content === "string" ? aiMessage.content : null,
          toolCallCount,
          finishReason,
          usage: getModelUsage(aiMessage),
          durationMs: Date.now() - modelStarted,
        }),
      );

      if (toolCallCount === 0) {
        const outcome = outcomeForModelResponse(finishReason, toolCallCount);
        if (outcome !== null) {
          events.push(terminalEvent(outcome));
          return {
            startedAtMs,
            messages: [aiMessage],
            outcome,
            events,
          };
        }
      }

      return {
        startedAtMs,
        messages: [aiMessage],
        events,
      };
    } catch (error) {
      const detail =
        error instanceof UnsupportedModelError
          ? error.message
          : String(error);
      const outcome: RunOutcome = {
        reason: "provider_failure",
        detail,
      };
      events.push(terminalEvent(outcome));
      return { startedAtMs, outcome, events };
    }
  };

  /**
   * Graph node: validates each pending tool call, then gates it behind approval
   * policy — auto-approving, bypassing, or pausing via `interrupt()` for a human
   * decision. Sets `approval_denied`/`invalid_tool_call` terminal outcomes as needed.
   */
  private readonly approval = async (state: AgentGraphState) => {
    if (state.outcome !== null) {
      return {};
    }

    const { skipPermissionsCheck } = readConfigurable();
    const toolCalls = getLatestToolCalls(state.messages);
    const events: TimestampedAuditEventPayload[] = [];

    for (const call of toolCalls) {
      const validation = validateToolCall(call, this.toolsByName.get(call.name));
      if (!validation.ok) {
        const outcome = invalidToolOutcome(validation.reason);
        events.push(terminalEvent(outcome));
        return { outcome, events };
      }

      const registered = this.toolsByName.get(call.name);
      if (registered === undefined) {
        const outcome = invalidToolOutcome(`Model requested unknown tool "${call.name}"`);
        events.push(terminalEvent(outcome));
        return { outcome, events };
      }

      const approvalPolicy =
        typeof registered.requiresApproval === "function"
          ? registered.requiresApproval(call)
          : registered.requiresApproval;
      const requiresApproval = this.runConfig.approvalMode === "policy" && approvalPolicy;

      let approved = true;
      let reason = "auto-approved";

      if (requiresApproval && !skipPermissionsCheck) {
        const payload: ApprovalInterruptPayload = {
          callId: validation.callId,
          toolName: call.name,
          args: validation.args,
        };
        const decision = interrupt(payload) as { approved: boolean; reason?: string };
        approved = decision.approved;
        reason = decision.reason ?? "human decision";
      } else if (skipPermissionsCheck && requiresApproval) {
        reason = "bypassed via --skip-permissions-check";
      } else if (!requiresApproval) {
        reason = "allowed by policy";
      }

      events.push(
        createAuditEvent({
          type: "approval_decision",
          callId: validation.callId,
          approved,
          reason,
        }),
      );

      if (!approved) {
        const outcome: RunOutcome = {
          reason: "approval_denied",
          detail: `Approval denied for tool "${call.name}"`,
        };
        events.push(terminalEvent(outcome));
        return { outcome, events };
      }
    }

    return { events };
  };

  /**
   * Graph node: executes approved tool calls sequentially against the tool
   * registry, enforcing the tool-call budget and the repeated-call guard.
   * Tool errors become `ToolMessage`s fed back to the model rather than
   * terminal outcomes — only budget/validation/repetition failures end the run.
   */
  private readonly executeTools = async (state: AgentGraphState) => {
    if (state.outcome !== null) {
      return {};
    }

    const toolCalls = getLatestToolCalls(state.messages);
    const toolMessages: ToolMessage[] = [];
    const events: TimestampedAuditEventPayload[] = [];
    let toolCallCount = state.toolCallCount;
    let lastToolSignature = state.lastToolSignature;

    for (const call of toolCalls) {
      if (toolCallCount >= this.runConfig.budgets.maxToolCalls) {
        const outcome: RunOutcome = {
          reason: "iteration_limit",
          detail: `Exceeded tool-call budget of ${this.runConfig.budgets.maxToolCalls}`,
        };
        events.push(terminalEvent(outcome));
        return { toolCallCount, lastToolSignature, outcome, events };
      }

      const validation = validateToolCall(call, this.toolsByName.get(call.name));
      if (!validation.ok) {
        const outcome = invalidToolOutcome(validation.reason);
        events.push(terminalEvent(outcome));
        return { toolCallCount, lastToolSignature, outcome, events };
      }

      const signature = toolCallSignature(call);
      if (lastToolSignature === signature) {
        const outcome: RunOutcome = {
          reason: "repeated_tool_calls",
          detail: `Model repeated identical tool call "${call.name}"`,
        };
        events.push(terminalEvent(outcome));
        return { toolCallCount, lastToolSignature, outcome, events };
      }

      const registered = this.toolsByName.get(call.name);
      if (registered === undefined) {
        const outcome = invalidToolOutcome(`Model requested unknown tool "${call.name}"`);
        events.push(terminalEvent(outcome));
        return { toolCallCount, lastToolSignature, outcome, events };
      }

      toolCallCount += 1;
      events.push(createAuditEvent({ type: "tool_call", call }));

      const started = Date.now();
      const { thread_id } = readConfigurable();
      try {
        const result = await registered.tool.invoke(validation.args, {
          configurable: {
            workspaceRoot: this.runConfig.workspaceRoot,
            thread_id,
          },
        });
        const durationMs = Date.now() - started;
        events.push(
          createAuditEvent({
            type: "tool_result",
            callId: validation.callId,
            status: "ok",
            payload: result,
            durationMs,
          }),
        );
        toolMessages.push(
          new ToolMessage({
            content: typeof result === "string" ? result : JSON.stringify(result),
            tool_call_id: validation.callId,
            name: call.name,
          }),
        );
      } catch (error) {
        const durationMs = Date.now() - started;
        events.push(
          createAuditEvent({
            type: "tool_result",
            callId: validation.callId,
            status: "error",
            payload: { error: String(error) },
            durationMs,
          }),
        );
        toolMessages.push(
          new ToolMessage({
            content: JSON.stringify({ error: String(error) }),
            tool_call_id: validation.callId,
            name: call.name,
          }),
        );
      }

      lastToolSignature = signature;
    }

    return {
      messages: toolMessages,
      toolCallCount,
      lastToolSignature,
      events,
    };
  };

  /** Router: goes to approval when the model requested tool calls, else ends the turn. */
  private readonly routeAfterModel = (state: AgentGraphState): "approval" | typeof END => {
    if (state.outcome !== null) {
      return END;
    }
    return getLatestToolCalls(state.messages).length > 0 ? "approval" : END;
  };

  /** Router: proceeds to tool execution once approval has been resolved (or ends on denial). */
  private readonly routeAfterApproval = (state: AgentGraphState): "executeTools" | typeof END => {
    if (state.outcome !== null) {
      return END;
    }
    return "executeTools";
  };

  /** Router: loops back to the model after tools run, unless a terminal outcome was set. */
  private readonly routeAfterTools = (state: AgentGraphState): "modelCall" | typeof END => {
    if (state.outcome !== null) {
      return END;
    }
    return "modelCall";
  };

  constructor(
    runConfig: RunConfig,
    registeredTools: readonly RegisteredTool[],
    resolveModelFn: ModelResolver = resolveModel,
  ) {
    this.runConfig = runConfig;
    this.resolveModelFn = resolveModelFn;
    this.toolsByName = new Map(registeredTools.map((entry) => [entry.tool.name, entry]));

    const workflow = new StateGraph(AgentState)
      .addNode("modelCall", this.modelCall)
      .addNode("approval", this.approval)
      .addNode("executeTools", this.executeTools)
      .addEdge(START, "modelCall")
      .addConditionalEdges("modelCall", this.routeAfterModel, ["approval", END])
      .addConditionalEdges("approval", this.routeAfterApproval, ["executeTools", END])
      .addConditionalEdges("executeTools", this.routeAfterTools, ["modelCall", END]);

    this.graph = workflow.compile({ checkpointer: new MemorySaver() });
  }

  /** Returns a cached, tool-bound chat model for the given model identifier. */
  private async getChatModel(modelId: string): Promise<BaseChatModel> {
    const cached = this.modelCache.get(modelId);
    if (cached !== undefined) {
      return cached;
    }

    const bound = await this.bindTools(await this.resolveModelFn(modelId));
    this.modelCache.set(modelId, bound);
    return bound;
  }

  /** Binds the runtime tool registry to a chat model for the current turn. */
  private async bindTools(model: BaseChatModel): Promise<BaseChatModel> {
    const tools = [...this.toolsByName.values()].map((entry) => entry.tool);
    if (tools.length === 0 || !("bindTools" in model)) {
      return model;
    }
    return (model as BaseChatModel & { bindTools: (tools: unknown[]) => BaseChatModel }).bindTools(
      tools,
    ) as BaseChatModel;
  }

  /** Runs one user turn, optionally pausing on approval interrupts. */
  async runTurn(params: RunTurnParams): Promise<RunResult> {
    const skipPermissionsCheck =
      params.skipPermissionsCheck ?? this.runConfig.approvalMode === "skip_all";
    const invokeConfig = this.buildInvokeConfig(params.threadId, params.model, skipPermissionsCheck);

    const initialEvents: TimestampedAuditEventPayload[] = [
      createAuditEvent({ type: "user_task", task: params.input }),
    ];

    try {
      const snapshot = await this.graph.getState(invokeConfig);
      const existingState = snapshot.values as AgentGraphState | undefined;
      const preserveStartedAt = (existingState?.startedAtMs ?? 0) > 0;

      const result = await this.graph.invoke(
        {
          messages: [new HumanMessage(params.input)],
          events: initialEvents,
          startedAtMs: preserveStartedAt ? 0 : Date.now(),
          outcome: null,
          lastToolSignature: null,
        },
        invokeConfig,
      );

      return this.toRunResult(result, invokeConfig);
    } catch (error) {
      if (error instanceof GraphRecursionError) {
        return this.recursionLimitResult(invokeConfig);
      }
      throw error;
    }
  }

  /** Resumes a paused turn after an approval interrupt. */
  async resume(params: ResumeParams): Promise<RunResult> {
    const skipPermissionsCheck =
      params.skipPermissionsCheck ?? this.runConfig.approvalMode === "skip_all";
    const invokeConfig = this.buildInvokeConfig(
      params.threadId,
      params.model ?? this.runConfig.model,
      skipPermissionsCheck,
    );

    try {
      const result = await this.graph.invoke(
        new Command({
          resume: {
            approved: params.approved,
            reason: params.approved ? "approved by user" : "denied by user",
          },
        }),
        invokeConfig,
      );

      return this.toRunResult(result, invokeConfig);
    } catch (error) {
      if (error instanceof GraphRecursionError) {
        return this.recursionLimitResult(invokeConfig);
      }
      throw error;
    }
  }

  /** Builds graph invocation config for a thread and model selection. */
  private buildInvokeConfig(
    threadId: string,
    model: string,
    skipPermissionsCheck: boolean,
  ): {
    configurable: GraphConfigurable;
    recursionLimit: number;
  } {
    return {
      configurable: {
        thread_id: threadId,
        model,
        skipPermissionsCheck,
      },
      recursionLimit: this.runConfig.budgets.maxIterations,
    };
  }

  /** Maps LangGraph recursion exhaustion to the runtime iteration_limit outcome. */
  private async recursionLimitResult(invokeConfig: {
    configurable: GraphConfigurable;
    recursionLimit: number;
  }): Promise<RunResult> {
    const snapshot = await this.graph.getState(invokeConfig);
    const state = snapshot.values as AgentGraphState;
    const outcome: RunOutcome = {
      reason: "iteration_limit",
      detail: `Reached max iterations (${invokeConfig.recursionLimit})`,
    };
    const terminal = terminalEvent(outcome);
    await this.graph.updateState(invokeConfig, {
      outcome,
      events: [...(state.events ?? []), terminal],
    });

    const events = this.stampEvents([...(state.events ?? []), terminal]);
    await this.persistNewEvents(invokeConfig.configurable.thread_id, events);
    return {
      events,
      outcome,
      interrupted: false,
    };
  }

  /** Converts graph state into the runtime's public result shape. */
  private async toRunResult(
    result: Record<string, unknown>,
    invokeConfig: {
      configurable: GraphConfigurable;
      recursionLimit: number;
    },
  ): Promise<RunResult> {
    const snapshot = await this.graph.getState(invokeConfig);
    const state = snapshot.values as AgentGraphState;
    const events = this.stampEvents(state.events ?? []);

    await this.persistNewEvents(invokeConfig.configurable.thread_id, events);

    if (isInterrupted(result)) {
      const interruptValue = result[INTERRUPT][0]?.value as ApprovalInterruptPayload | undefined;
      return {
        events,
        outcome: state.outcome,
        interrupted: true,
        interruptPayload: interruptValue,
      };
    }

    return {
      events,
      outcome: state.outcome,
      interrupted: false,
    };
  }

  /** Appends only events that have not yet been persisted for this thread. */
  private async persistNewEvents(threadId: string, events: readonly AuditEvent[]): Promise<void> {
    const runsDir = this.runConfig.runsDir;
    if (runsDir === undefined) {
      return;
    }

    const persisted = this.persistedEventCount.get(threadId) ?? 0;
    const newEvents = events.slice(persisted);
    if (newEvents.length === 0) {
      return;
    }

    await appendRunEvents(threadId, newEvents, runsDir);
    this.persistedEventCount.set(threadId, events.length);
  }

  /** Adds sequence numbers while preserving per-event timestamps. */
  private stampEvents(payloads: readonly TimestampedAuditEventPayload[]): AuditEvent[] {
    return payloads.map((payload, seq) => {
      const { emittedAt, ...rest } = payload;
      return {
        ...rest,
        seq,
        at: emittedAt,
      };
    });
  }
}
