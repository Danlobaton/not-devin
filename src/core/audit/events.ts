import type { AuditEventPayload, RunOutcome } from "../types.ts";

export type TimestampedAuditEventPayload = AuditEventPayload & {
  readonly emittedAt: string;
};

/** Creates an audit payload stamped at creation time. */
export function createAuditEvent(payload: AuditEventPayload): TimestampedAuditEventPayload {
  return {
    ...payload,
    emittedAt: new Date().toISOString(),
  };
}

/** Creates a terminal audit payload for the given outcome. */
export function terminalEvent(outcome: RunOutcome): TimestampedAuditEventPayload {
  return createAuditEvent({ type: "terminal", outcome });
}
