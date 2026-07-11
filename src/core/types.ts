/**
 * Core contracts for the agent loop. Every subsystem (runtime, tools, model
 * adapters, persistence) is built against these types rather than against
 * each other directly.
 */

export type ToolResultStatus = "ok" | "error";

/**
 * A tool call as requested by the model. Arguments are kept as a raw,
 * unvalidated string — the model is an untrusted caller, and validation is a
 * distinct step the runtime performs before a Tool ever sees the call.
 */
export interface ToolCall {
  readonly id: string;
  readonly name: string;
  readonly rawArguments: string;
}

export interface ToolResult {
  readonly callId: string;
  readonly status: ToolResultStatus;
  readonly payload: unknown;
  readonly durationMs: number;
  readonly truncated: boolean;
  readonly artifactPath?: string;
}

export type ModelFinishReason = "stop" | "tool_calls" | "length" | "error";

export interface ModelUsage {
  readonly promptTokens: number;
  readonly completionTokens: number;
}

/**
 * A single model turn. `toolCalls` is a list — even though the runtime
 * currently executes tool calls one at a time — because a provider can
 * legitimately return more than one call in a single turn.
 */
export interface ModelResponse {
  readonly text: string | null;
  readonly toolCalls: readonly ToolCall[];
  readonly finishReason: ModelFinishReason;
  readonly usage: ModelUsage | null;
}

export interface ToolSchema {
  readonly name: string;
  readonly description: string;
  readonly parameters: Record<string, unknown>;
}

export type MessageRole = "user" | "assistant" | "tool";

export interface ModelMessage {
  readonly role: MessageRole;
  readonly content: string;
  readonly toolCallId?: string;
}

export interface ModelRequest {
  /** Model id for this turn. The client/router routes to the matching provider. */
  readonly model: string;
  readonly messages: readonly ModelMessage[];
  readonly tools: readonly ToolSchema[];
}

export type RunOutcomeReason =
  | "success"
  | "iteration_limit"
  | "timeout"
  | "repeated_tool_calls"
  | "approval_denied"
  | "invalid_tool_call"
  | "provider_failure";

export interface RunOutcome {
  readonly reason: RunOutcomeReason;
  readonly detail: string;
}

export interface Budgets {
  readonly maxIterations: number;
  readonly maxToolCalls: number;
  readonly maxWallClockMs: number;
}

/**
 * "policy" runs the normal risk-based approval check per call. "skip_all"
 * is the `--skip-permissions-check` escape hatch: every call is
 * auto-approved without a real risk decision, but an `approval_decision`
 * event is still emitted (with a reason noting the bypass) so the trace
 * stays honest about what happened, even though no one actually reviewed it.
 */
export type ApprovalMode = "policy" | "skip_all";

export interface RunConfig {
  /** Default model for the first turn; later turns may override via ModelRequest.model. */
  readonly model: string;
  readonly budgets: Budgets;
  readonly workspaceRoot: string;
  readonly approvalMode: ApprovalMode;
}

interface EventEnvelope {
  readonly seq: number;
  readonly at: string;
}

/**
 * The durable, append-only event log. This is the source of truth for a run
 * and what gets persisted to JSONL. It is deliberately richer than what gets
 * sent back to the model each turn — see `deriveMessages` for that
 * derivation.
 */
export type RunEvent = EventEnvelope &
  (
    | { readonly type: "user_task"; readonly task: string }
    | { readonly type: "model_request"; readonly request: ModelRequest }
    | { readonly type: "model_response"; readonly response: ModelResponse }
    | { readonly type: "tool_call"; readonly call: ToolCall }
    | { readonly type: "tool_result"; readonly result: ToolResult }
    | {
        readonly type: "approval_decision";
        readonly callId: string;
        readonly approved: boolean;
        readonly reason: string;
      }
    | { readonly type: "terminal"; readonly outcome: RunOutcome }
  );

export type RunEventPayload = Omit<RunEvent, keyof EventEnvelope>;

export type RunEventType = RunEvent["type"];
