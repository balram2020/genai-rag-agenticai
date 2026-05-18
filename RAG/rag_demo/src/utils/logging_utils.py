from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Event:
    ts: str
    level: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)


class EventLogger:
    """
    A teaching-friendly event logger.

    - Keeps a structured list of events for showing in Streamlit.
    - Also prints compact logs to stdout for terminal visibility.
    """

    def __init__(self) -> None:
        self.events: List[Event] = []

    def _emit(self, level: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        data = data or {}
        ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        evt = Event(ts=ts, level=level, message=message, data=data)
        self.events.append(evt)
        if data:
            print(f"[{ts}] {level.upper():7s} {message} | {data}")
        else:
            print(f"[{ts}] {level.upper():7s} {message}")

    def info(self, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        self._emit("info", message, data)

    def warn(self, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        self._emit("warn", message, data)

    def error(self, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        self._emit("error", message, data)

    def clear(self) -> None:
        self.events.clear()

