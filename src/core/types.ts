/**
 * Runtime configuration, budgets, outcomes, and audit events.
 * Model messages and tool calls use LangChain-native types.
 */

import type { ToolCall } from "@langchain/core/messages";
import type { StructuredToolInterface } from "@langchain/core/tools";
import type { TimestampedAuditEventPayload } from "./audit/events.ts";

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
  /** Default model for the first turn; later turns may override per invocation. */
  readonly model: string;
  readonly budgets: Budgets;
  readonly workspaceRoot: string;
  readonly approvalMode: ApprovalMode;
  /** Directory for append-only JSONL run logs. Omit to disable persistence. */
  readonly runsDir?: string;
}

/** Tool plus policy metadata used by the runtime approval gate. */
export interface RegisteredTool {
  readonly tool: StructuredToolInterface;
  /** Fixed policy, or a predicate evaluated per call against that call's actual arguments. */
  readonly requiresApproval: boolean | ((call: ToolCall) => boolean);
}

interface EventEnvelope {
  readonly seq: number;
  readonly at: string;
}

export type ModelUsage = {
  readonly promptTokens: number;
  readonly completionTokens: number;
};

export type AuditEventPayload =
  | { readonly type: "user_task"; readonly task: string }
  | {
      readonly type: "model_request";
      readonly model: string;
      readonly messageCount: number;
      readonly toolCount: number;
    }
  | {
      readonly type: "model_response";
      readonly model: string;
      readonly text: string | null;
      readonly toolCallCount: number;
      readonly finishReason: string | null;
      readonly usage: ModelUsage | null;
      readonly durationMs: number;
    }
  | { readonly type: "tool_call"; readonly call: ToolCall }
  | {
      readonly type: "tool_result";
      readonly callId: string;
      readonly status: "ok" | "error";
      readonly payload: unknown;
      readonly durationMs: number;
    }
  | {
      readonly type: "approval_decision";
      readonly callId: string;
      readonly approved: boolean;
      readonly reason: string;
    }
  | { readonly type: "terminal"; readonly outcome: RunOutcome };

export type AuditEvent = EventEnvelope & AuditEventPayload;

export interface RunTurnParams {
  readonly threadId: string;
  readonly input: string;
  readonly model: string;
  readonly skipPermissionsCheck?: boolean;
}

export interface ResumeParams {
  readonly threadId: string;
  readonly approved: boolean;
  readonly model?: string;
  readonly skipPermissionsCheck?: boolean;
}

export interface RunResult {
  readonly events: readonly AuditEvent[];
  readonly outcome: RunOutcome | null;
  readonly interrupted: boolean;
  readonly interruptPayload?: ApprovalInterruptPayload | undefined;
}

export interface ApprovalInterruptPayload {
  readonly callId: string;
  readonly toolName: string;
  readonly args: Record<string, unknown>;
}

export type { TimestampedAuditEventPayload };
