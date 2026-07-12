/**
 * Helpers for reading data out of LangChain message objects — tool calls,
 * finish reasons, token usage. Kept separate from `types.ts` so that file
 * stays pure type/interface declarations.
 */

import { AIMessage, type BaseMessage, type ToolCall } from "@langchain/core/messages";
import type { ModelUsage, RunOutcome } from "./types.ts";

/** Extracts tool calls from the latest assistant message, if any. */
export function getLatestToolCalls(messages: readonly BaseMessage[]): ToolCall[] {
  const last = messages.at(-1);
  if (last === undefined || !AIMessage.isInstance(last)) {
    return [];
  }
  return last.tool_calls ?? [];
}

/** Reads model finish reason from an assistant message when available. */
export function getFinishReason(message: AIMessage): string | null {
  const metadata = message.response_metadata as Record<string, unknown> | undefined;
  const fromMetadata = metadata?.finish_reason;
  if (typeof fromMetadata === "string") {
    return fromMetadata;
  }
  const fromKwargs = message.additional_kwargs?.finish_reason;
  return typeof fromKwargs === "string" ? fromKwargs : null;
}

/** Reads token usage from an assistant message when available. */
export function getModelUsage(message: AIMessage): ModelUsage | null {
  const metadata = message.response_metadata as Record<string, unknown> | undefined;
  const usage = metadata?.usage;
  if (typeof usage !== "object" || usage === null) {
    return null;
  }
  const record = usage as Record<string, unknown>;
  const promptTokens = record.prompt_tokens ?? record.input_tokens;
  const completionTokens = record.completion_tokens ?? record.output_tokens;
  if (typeof promptTokens !== "number" || typeof completionTokens !== "number") {
    return null;
  }
  return { promptTokens, completionTokens };
}

/** Maps provider finish reasons to runtime terminal outcomes. */
export function outcomeForModelResponse(
  finishReason: string | null,
  toolCallCount: number,
): RunOutcome | null {
  if (toolCallCount > 0) {
    return null;
  }

  if (finishReason === "length") {
    return {
      reason: "iteration_limit",
      detail: "Model stopped because the response hit the length limit",
    };
  }

  if (finishReason === "error") {
    return {
      reason: "provider_failure",
      detail: "Model returned an error finish reason",
    };
  }

  return {
    reason: "success",
    detail: "Model responded without requesting further tool calls",
  };
}
