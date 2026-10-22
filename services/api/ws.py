"""Live decision stream for `/ws/live` (A7).

Scoring runs in worker threads, subscribers live on the event loop: `publish` is
thread-safe and hands each message to every subscriber's loop. A slow client never
blocks scoring: its queue is bounded and drops the oldest message when full.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

Message = dict[str, Any]


class Broadcaster:
    def __init__(self, max_queue: int = 1_000) -> None:
        self.max_queue = max_queue
        self._subscribers: dict[asyncio.Queue[Message], asyncio.AbstractEventLoop] = {}
        self._lock = threading.Lock()

    def subscribe(self) -> asyncio.Queue[Message]:
        queue: asyncio.Queue[Message] = asyncio.Queue(self.max_queue)
        with self._lock:
            self._subscribers[queue] = asyncio.get_running_loop()
        return queue

    def unsubscribe(self, queue: asyncio.Queue[Message]) -> None:
        with self._lock:
            self._subscribers.pop(queue, None)

    @property
    def subscribers(self) -> int:
        return len(self._subscribers)

    def publish(self, message: Message) -> None:
        with self._lock:
            targets = list(self._subscribers.items())
        for queue, loop in targets:
            try:
                loop.call_soon_threadsafe(_offer, queue, message)
            except RuntimeError:  # loop closed: the client is gone
                self.unsubscribe(queue)


def _offer(queue: asyncio.Queue[Message], message: Message) -> None:
    if queue.full():
        queue.get_nowait()
    queue.put_nowait(message)
