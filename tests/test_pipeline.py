import threading
import time

import numpy as np

from arc_audio.events import AudioChunk
from arc_audio.pipeline import Session
from arc_audio.settings import Settings


def test_session_stop_closes_log_and_workers(tmp_path):
    log = tmp_path / "events.jsonl"
    session = Session("demo", Settings(), log)
    session.start()
    deadline = time.monotonic() + 2
    while session.source.ident is None and time.monotonic() < deadline:
        time.sleep(.01)
    session.stop()
    session.worker.join(3)
    assert session.done.is_set()
    assert not session.source.is_alive()
    assert not session.error
    assert log.exists()


def test_stale_chunks_are_discarded(tmp_path):
    session = Session("demo", Settings(), tmp_path / "events.jsonl")
    session.audio.put(AudioChunk(np.ones((2400, 2), np.float32), 48000, .05, time.monotonic() - 2))
    session.start()
    deadline = time.monotonic() + 2
    while session.stale == 0 and time.monotonic() < deadline:
        time.sleep(.01)
    session.stop()
    session.worker.join(3)
    assert session.stale == 1
    assert session.events.empty()
    assert not any(t.name in ("audio-source", "audio-analysis") and t.is_alive() for t in threading.enumerate())


def test_target_only_filters_generic_alerts_before_logging_and_delivery(tmp_path, monkeypatch):
    """Integration contract using explicit fixture events, not model accuracy."""
    import arc_audio.pipeline as pipeline
    from arc_audio.events import Detection, utc_now

    class FixtureDetector:
        def process(self, chunk):
            return [Detection(utc_now(), chunk.position, label, .9, "unknown", "test-fixture")
                    for label in ("transient", "sound activity", "footsteps", "reload", "gunshot")]

    class FixtureSource(threading.Thread):
        def __init__(self, mode, queue, stop, *_args):
            super().__init__(daemon=True)
            self.mode, self.queue, self.stop = mode, queue, stop
            self.finished = threading.Event()
            self.xruns, self.error = 0, ""

        def run(self):
            self.queue.put(AudioChunk(np.zeros((2400, 2), np.float32), 48000, .05, time.monotonic()))
            self.finished.set()

    monkeypatch.setattr(pipeline, "AudioSource", FixtureSource)
    monkeypatch.setattr(pipeline, "make_detector", lambda *_args: FixtureDetector())
    log = tmp_path / "filtered.jsonl"
    session = pipeline.Session("file", Settings(target_only=True), log)
    session.start()
    session.worker.join(2)
    assert session.done.is_set() and not session.error
    import json
    assert {json.loads(line)["label"] for line in log.read_text().splitlines()} == {"footsteps", "reload", "gunshot"}
    delivered = []
    while not session.events.empty():
        delivered.append(session.events.get(0).label)
    assert set(delivered) == {"footsteps", "reload", "gunshot"}
