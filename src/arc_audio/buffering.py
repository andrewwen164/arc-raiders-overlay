import threading
from collections import deque


class LatestQueue:
    """Bounded queue; overflow evicts oldest, never blocks an audio callback."""

    def __init__(self, capacity=4):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items = deque()
        self._condition = threading.Condition()
        self.dropped = 0

    def put(self, item):
        with self._condition:
            if len(self._items) >= self.capacity:
                self._items.popleft()
                self.dropped += 1
            self._items.append(item)
            self._condition.notify()

    def get(self, timeout=0.05):
        with self._condition:
            if not self._items:
                self._condition.wait(timeout)
            return self._items.popleft() if self._items else None

    def empty(self):
        with self._condition:
            return not self._items

    def discard(self):
        with self._condition:
            self.dropped += len(self._items)
            self._items.clear()
