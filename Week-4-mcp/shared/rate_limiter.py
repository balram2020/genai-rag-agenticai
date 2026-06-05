"""Simple per-tool rolling-window rate limiter."""
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, max_calls_per_minute: int = 3):
        self.max_calls_per_minute = max_calls_per_minute
        self.calls = defaultdict(deque)

    def check(self, tool_name: str):
        now = time.time()
        window = self.calls[tool_name]

        while window and now - window[0] > 60:
            window.popleft()

        if len(window) >= self.max_calls_per_minute:
            retry_after = int(60 - (now - window[0]))
            return False, retry_after

        window.append(now)
        return True, 0
