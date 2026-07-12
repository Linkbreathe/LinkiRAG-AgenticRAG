"""Coalesce identical concurrent async work without caching failures."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

T = TypeVar("T")


class AsyncSingleFlight:
    def __init__(self):
        self._guard = threading.Lock()
        self._flights: dict[tuple[int, str], asyncio.Future] = {}

    async def run(self, key: str, factory: Callable[[], Awaitable[T]]) -> tuple[T, bool]:
        loop = asyncio.get_running_loop()
        flight_key = (id(loop), key)
        with self._guard:
            future = self._flights.get(flight_key)
            leader = future is None
            if leader:
                future = loop.create_future()
                self._flights[flight_key] = future
        assert future is not None
        if not leader:
            return await asyncio.shield(future), True
        try:
            result = await factory()
            if not future.done():
                future.set_result(result)
            return result, False
        except BaseException as exc:
            if not future.done():
                if isinstance(exc, asyncio.CancelledError):
                    future.cancel()
                else:
                    future.set_exception(exc)
                    # Consume the exception when there are no followers so the
                    # event loop does not report an orphaned Future.
                    future.exception()
            raise
        finally:
            with self._guard:
                self._flights.pop(flight_key, None)


GLOBAL_SINGLE_FLIGHT = AsyncSingleFlight()
