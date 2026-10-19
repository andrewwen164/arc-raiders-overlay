import threading

import pytest

from arc_audio.buffering import LatestQueue


def test_bounded_buffer_evicts_oldest():
    queue = LatestQueue(3)
    for item in range(100):
        queue.put(item)
    assert queue.dropped == 97
    assert [queue.get(0) for _ in range(3)] == [97, 98, 99]
    assert queue.get(0) is None


def test_producer_wakes_waiter():
    queue = LatestQueue(2)
    result = []
    waiting = threading.Thread(target=lambda: result.append(queue.get(.5)))
    waiting.start()
    queue.put("fresh")
    waiting.join(1)
    assert result == ["fresh"]


def test_discard_and_invalid_capacity():
    with pytest.raises(ValueError):
        LatestQueue(0)
    queue = LatestQueue(1)
    queue.put(1)
    queue.discard()
    assert queue.empty()
    assert queue.dropped == 1
