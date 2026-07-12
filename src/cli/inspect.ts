import { readFile } from "node:fs/promises";
import { join } from "node:path";
import type { AuditEvent } from "../core/types.ts";

/** Reads a thread's JSONL run log and formats it as one readable line per event. */
export async function formatTrace(threadId: string, runsDir: string): Promise<string> {
  const path = join(runsDir, `${threadId}.jsonl`);
  let raw: string;
  try {
    raw = await readFile(path, "utf8");
  } catch {
    throw new Error(`No trace found for thread "${threadId}" at ${path}`);
  }

  return raw
    .trim()
    .split("\n")
    .filter((line) => line.length > 0)
    .map((line) => formatEvent(JSON.parse(line) as AuditEvent))
    .join("\n");
}

function formatEvent(event: AuditEvent): string {
  const prefix = `[${event.seq}] ${event.at} ${event.type}:`;
  switch (event.type) {
    case "user_task":
      return `${prefix} ${event.task}`;
    case "model_request":
      return `${prefix} model=${event.model} messages=${event.messageCount} tools=${event.toolCount}`;
    case "model_response":
      return `${prefix} text=${event.text ?? "(none)"} toolCalls=${event.toolCallCount} finish=${event.finishReason ?? "(none)"}`;
    case "tool_call":
      return `${prefix} ${event.call.name}(${JSON.stringify(event.call.args)})`;
    case "tool_result":
      return `${prefix} ${event.status} in ${event.durationMs}ms`;
    case "approval_decision":
      return `${prefix} ${event.approved ? "approved" : "denied"} — ${event.reason}`;
    case "terminal":
      return `${prefix} ${event.outcome.reason} — ${event.outcome.detail}`;
  }
}
