"""Renders a run's events into a human-readable timeline."""

from __future__ import annotations


def render_timeline(events: list[dict]) -> str:
    """Render events as one line per event, in order.

    Args:
        events: List of event dictionaries, each with at least 'seq' and 'type' keys.

    Returns:
        A newline-separated string with one line per event, or empty string if no events.
    """
    lines = [f"[{event.get('seq', '?')}] {event.get('type', 'unknown')} {_describe(event)}".rstrip() for event in events]
    return "\n".join(lines)


def _describe(event: dict) -> str:
    """Generate a human-readable description of an event.

    Args:
        event: An event dictionary.

    Returns:
        A formatted string describing the event's key details.
    """
    event_type = event.get("type")
    if event_type == "run_started":
        described = f"task={event.get('task')!r} workspace={event.get('workspace')}"
        if event.get("verify_command"):
            described += f" verify_command={event['verify_command']!r}"
        return described
    if event_type == "llm_start":
        return f"{len(event.get('messages', []))} messages sent"
    if event_type in ("llm_end", "tool_end"):
        if event.get("error"):
            return f"error={event['error']!r} latency_ms={event.get('latency_ms')}"
        if event_type == "llm_end":
            return f"latency_ms={event.get('latency_ms')}"
        return f"result={event.get('result')!r} latency_ms={event.get('latency_ms')}"
    if event_type == "tool_start":
        return f"tool={event.get('tool_name')} args={event.get('args')}"
    if event_type == "state_delta":
        return f"node={event.get('node')}"
    if event_type == "run_finished":
        return f"terminal_reason={event.get('terminal_reason')} iterations={event.get('iteration_count')}"
    return ""
