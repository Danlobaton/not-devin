import { appendFile, mkdir } from "node:fs/promises";
import { join } from "node:path";
import type { AuditEvent } from "../types.ts";

/** Appends new audit events for a thread to a JSONL run log. */
export async function appendRunEvents(
  threadId: string,
  events: readonly AuditEvent[],
  runsDir: string,
): Promise<void> {
  if (events.length === 0) {
    return;
  }

  await mkdir(runsDir, { recursive: true });
  const path = join(runsDir, `${threadId}.jsonl`);
  const lines = events.map((event) => JSON.stringify(event)).join("\n");
  await appendFile(path, `${lines}\n`, "utf8");
}
