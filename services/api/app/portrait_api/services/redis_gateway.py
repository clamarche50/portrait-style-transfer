from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Protocol

from portrait_api.config import Settings
from redis.asyncio import Redis


class ProgressStore(Protocol):
    async def ping(self) -> bool: ...

    async def close(self) -> None: ...

    async def rate_limit(self, key: str, limit: int, window_seconds: int) -> tuple[bool, int]: ...

    async def set_progress(self, job_id: str, event: dict[str, Any]) -> None: ...

    async def get_progress(self, job_id: str) -> dict[str, Any] | None: ...

    async def request_cancel(self, job_id: str) -> None: ...

    async def clear_cancel(self, job_id: str) -> None: ...


_RATE_LIMIT_SCRIPT = """
local current = redis.call('INCR', KEYS[1])
if current == 1 then redis.call('EXPIRE', KEYS[1], ARGV[1]) end
local ttl = redis.call('TTL', KEYS[1])
return {current, ttl}
"""


class RedisGateway:
    def __init__(self, settings: Settings) -> None:
        self.client = Redis.from_url(settings.redis_url, decode_responses=True)
        self.progress_ttl = settings.redis_progress_ttl_seconds

    async def ping(self) -> bool:
        return bool(await self.client.ping())

    async def close(self) -> None:
        await self.client.aclose()

    async def rate_limit(self, key: str, limit: int, window_seconds: int) -> tuple[bool, int]:
        current, ttl = await self.client.eval(
            _RATE_LIMIT_SCRIPT,
            1,
            f"rate:{key}",
            window_seconds,
        )
        return int(current) <= limit, max(int(ttl), 0)

    async def set_progress(self, job_id: str, event: dict[str, Any]) -> None:
        payload = json.dumps(event, separators=(",", ":"), default=str)
        key = f"job:{job_id}:progress"
        async with self.client.pipeline(transaction=True) as pipeline:
            pipeline.set(key, payload, ex=self.progress_ttl)
            pipeline.publish(f"job:{job_id}:events", payload)
            await pipeline.execute()

    async def get_progress(self, job_id: str) -> dict[str, Any] | None:
        payload = await self.client.get(f"job:{job_id}:progress")
        return json.loads(payload) if payload else None

    async def request_cancel(self, job_id: str) -> None:
        await self.client.set(f"job:{job_id}:cancel", "1", ex=self.progress_ttl)

    async def clear_cancel(self, job_id: str) -> None:
        await self.client.delete(f"job:{job_id}:cancel")


class MemoryProgressStore:
    """Async in-memory Redis substitute used only by tests."""

    def __init__(self) -> None:
        self.progress: dict[str, dict[str, Any]] = {}
        self.cancelled: set[str] = set()
        self.rate_counters: dict[str, tuple[int, float]] = {}
        self._lock = asyncio.Lock()

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        return None

    async def rate_limit(self, key: str, limit: int, window_seconds: int) -> tuple[bool, int]:
        now = time.monotonic()
        async with self._lock:
            count, expires = self.rate_counters.get(key, (0, now + window_seconds))
            if expires <= now:
                count, expires = 0, now + window_seconds
            count += 1
            self.rate_counters[key] = (count, expires)
            return count <= limit, max(int(expires - now), 0)

    async def set_progress(self, job_id: str, event: dict[str, Any]) -> None:
        self.progress[job_id] = dict(event)

    async def get_progress(self, job_id: str) -> dict[str, Any] | None:
        event = self.progress.get(job_id)
        return dict(event) if event else None

    async def request_cancel(self, job_id: str) -> None:
        self.cancelled.add(job_id)

    async def clear_cancel(self, job_id: str) -> None:
        self.cancelled.discard(job_id)
