import type { ToolCall } from "@langchain/core/messages";
import { interopSafeParse, isInteropZodSchema } from "@langchain/core/utils/types";
import type { RegisteredTool } from "./types.ts";

export type ToolValidationResult =
  | { readonly ok: true; readonly callId: string; readonly args: Record<string, unknown> }
  | { readonly ok: false; readonly reason: string };

/** Structural and schema validation before a tool is approved or executed. */
export function validateToolCall(
  call: ToolCall,
  registered: RegisteredTool | undefined,
): ToolValidationResult {
  if (call.name.length === 0) {
    return { ok: false, reason: "Tool call is missing a name" };
  }

  if (call.id === undefined || call.id.length === 0) {
    return { ok: false, reason: `Tool call "${call.name}" is missing an id` };
  }

  if (registered === undefined) {
    return { ok: false, reason: `Model requested unknown tool "${call.name}"` };
  }

  if (call.args === undefined || typeof call.args !== "object" || call.args === null) {
    return { ok: false, reason: `Tool call "${call.name}" (${call.id}) has non-object arguments` };
  }

  const schema = registered.tool.schema;
  if (schema !== undefined && isInteropZodSchema(schema)) {
    const parsed = interopSafeParse(schema, call.args);
    if (!parsed.success) {
      return {
        ok: false,
        reason: `Tool call "${call.name}" (${call.id}) failed schema validation: ${JSON.stringify(parsed.error.issues)}`,
      };
    }
    return { ok: true, callId: call.id, args: parsed.data as Record<string, unknown> };
  }

  return { ok: true, callId: call.id, args: call.args as Record<string, unknown> };
}

/** Stable signature used to detect repeated identical tool requests. */
export function toolCallSignature(call: ToolCall): string {
  return `${call.name}:${JSON.stringify(call.args ?? {})}`;
}
