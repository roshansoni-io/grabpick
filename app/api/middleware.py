"""Request middleware: per-IP rate limiting."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Deque, Dict

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from ..config import settings
from ..utils.logger import logger

_PUBLIC_PATHS = ("/static", "/originals")


def _client_ip(request: Request) -> str:
    """Return the client IP for rate limiting.

    ``X-Forwarded-For`` is only trusted when a reverse proxy is configured
    (``GRABPICK_TRUST_PROXY=1``). Otherwise it is spoofable, so the raw
    peer address is used.
    """
    if settings.trust_proxy:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    if request.client is not None:
        return request.client.host
    return "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window rate limiter keyed by client IP.

    Static media mounts are excluded so image loading is not throttled.
    """

    def __init__(self, app, max_requests: int | None = None, window_seconds: int | None = None):
        super().__init__(app)
        self.max_requests = max_requests or settings.rate_limit_max
        self.window_seconds = window_seconds or settings.rate_limit_window
        self._requests: Dict[str, Deque[float]] = defaultdict(deque)
        self._prune_every = 10_000
        self._calls = 0

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        if request.url.path.startswith(_PUBLIC_PATHS):
            return await call_next(request)

        now = time.monotonic()
        ip = _client_ip(request)

        self._calls += 1
        if self._calls % self._prune_every == 0:
            self._prune(now)

        window_start = now - self.window_seconds
        timestamps = self._requests[ip]
        while timestamps and timestamps[0] < window_start:
            timestamps.popleft()

        if len(timestamps) >= self.max_requests:
            logger.warning("Rate limit exceeded for IP %s", ip)
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please try again later."},
                headers={"Retry-After": str(self.window_seconds)},
            )

        timestamps.append(now)
        return await call_next(request)

    def _prune(self, now: float) -> None:
        window_start = now - self.window_seconds
        expired = [ip for ip, timestamps in self._requests.items()
                   if not timestamps or timestamps[-1] < window_start]
        for ip in expired:
            del self._requests[ip]