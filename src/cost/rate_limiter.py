import asyncio
import logging
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple token-bucket rate limiter per provider."""

    def __init__(self, limits: dict[str, dict] | None = None):
        self.limits = limits or {
            "openai": {"rpm": 60, "tpm": 200_000},
            "anthropic": {"rpm": 40, "tpm": 100_000},
            "tradier": {"rpm": 120, "tpm": 0},
            "polygon": {"rpm": 5, "tpm": 0},
            "finnhub": {"rpm": 30, "tpm": 0},
        }
        self._request_timestamps: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def acquire(self, provider: str) -> None:
        """Wait until a request slot is available for the provider."""
        limit_cfg = self.limits.get(provider, {"rpm": 30})
        rpm = limit_cfg.get("rpm", 30)

        async with self._lock:
            now = time.monotonic()
            window = 60.0
            timestamps = self._request_timestamps[provider]

            # Prune old timestamps
            self._request_timestamps[provider] = [
                t for t in timestamps if now - t < window
            ]
            timestamps = self._request_timestamps[provider]

            if len(timestamps) >= rpm:
                wait_time = window - (now - timestamps[0])
                if wait_time > 0:
                    logger.info(
                        "Rate limit hit for %s. Waiting %.1fs.", provider, wait_time
                    )
                    await asyncio.sleep(wait_time)

            self._request_timestamps[provider].append(time.monotonic())

    def acquire_sync(self, provider: str) -> None:
        """Synchronous version for non-async contexts."""
        limit_cfg = self.limits.get(provider, {"rpm": 30})
        rpm = limit_cfg.get("rpm", 30)

        now = time.monotonic()
        window = 60.0
        timestamps = self._request_timestamps[provider]

        self._request_timestamps[provider] = [
            t for t in timestamps if now - t < window
        ]
        timestamps = self._request_timestamps[provider]

        if len(timestamps) >= rpm:
            wait_time = window - (now - timestamps[0])
            if wait_time > 0:
                logger.info(
                    "Rate limit hit for %s. Waiting %.1fs.", provider, wait_time
                )
                time.sleep(wait_time)

        self._request_timestamps[provider].append(time.monotonic())
