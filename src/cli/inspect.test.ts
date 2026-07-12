import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { AuditEvent } from "../core/types.ts";
import { formatTrace } from "./inspect.ts";

describe("formatTrace", () => {
  let runsDir: string;

  beforeEach(async () => {
    runsDir = await mkdtemp(join(tmpdir(), "not-devin-inspect-"));
  });

  afterEach(async () => {
    await rm(runsDir, { recursive: true, force: true });
  });

  it("formats each event in a JSONL trace file as one readable line", async () => {
    const events: AuditEvent[] = [
      { seq: 0, at: "2026-07-11T00:00:00.000Z", type: "user_task", task: "fix the bug" },
      {
        seq: 1,
        at: "2026-07-11T00:00:01.000Z",
        type: "terminal",
        outcome: { reason: "success", detail: "Model responded without requesting further tool calls" },
      },
    ];
    const path = join(runsDir, "thread-1.jsonl");
    await writeFile(path, events.map((event) => JSON.stringify(event)).join("\n") + "\n", "utf8");

    const output = await formatTrace("thread-1", runsDir);
    const lines = output.split("\n");

    expect(lines).toHaveLength(2);
    expect(lines[0]).toContain("user_task");
    expect(lines[0]).toContain("fix the bug");
    expect(lines[1]).toContain("terminal");
    expect(lines[1]).toContain("success");
  });

  it("rejects with a clear error when the thread has no trace file", async () => {
    await expect(formatTrace("missing-thread", runsDir)).rejects.toThrow(/missing-thread/);
  });
});
