"""Append-only audit log for every tool call decision."""
from datetime import datetime, timezone


class AuditLogger:
    def __init__(self):
        self.events = []

    def record(self, event_type: str, payload: dict):
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        self.events.append(event)
        return event

    def all_events(self):
        return self.events
