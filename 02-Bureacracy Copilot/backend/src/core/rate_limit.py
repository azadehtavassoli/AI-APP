"""Simple in-memory rate limiting utilities.

Provides a lightweight fixed-window limiter for API endpoints and tool calls.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import Deque, Dict


@dataclass(frozen=True)
class RateLimitConfig:
    """Configuration for a fixed-window rate limit."""

    max_requests: int
    window_seconds: int


class FixedWindowRateLimiter:
    """Process-local fixed-window limiter backed by in-memory state."""

    def __init__(self) -> None:
        """Initialize process-local event buckets and lock state.

        Args:
            None.

        Returns:
            None: Creates empty per-key queues for request timestamps.
        """
        self._events: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str, config: RateLimitConfig) -> bool:
        """Return True when a request is allowed under the configured window."""

        now = monotonic()
        cutoff = now - config.window_seconds

        with self._lock:
            bucket = self._events[key]
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()

            if len(bucket) >= config.max_requests:
                return False

            bucket.append(now)
            return True


limiter = FixedWindowRateLimiter()
