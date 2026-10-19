import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .audio import AudioSource
from .buffering import LatestQueue
from .classifier import make_detector
from .dataset import TARGET_CLASSES


class Session:
    """Python workers + bounded mailboxes; Qt polls, never receives per-chunk signals."""

    def __init__(self, mode, settings, log_path: Path, path="", device=None):
        self.stop_event = threading.Event()
        self.audio = LatestQueue(4)
        self.events = LatestQueue(24)
        self.source = AudioSource(mode, self.audio, self.stop_event, path, device)
        self.settings = settings
        self.log_path = log_path
        self.worker = threading.Thread(target=self._analyze, name="audio-analysis", daemon=True)
        self.done = threading.Event()
        self.error = ""
        self.status = "Loading detector"
        self.levels = (0.0, 0.0)
        self.processing_ms = 0.0
        self.pipeline_ms = 0.0
        self.stale = 0

    def start(self):
        self.worker.start()

    def stop(self):
        self.stop_event.set()
        self.events.discard()

    @property
    def dropped(self):
        return self.audio.dropped + self.events.dropped + self.source.xruns + self.stale

    def _analyze(self):
        try:
            detector = make_detector(self.settings.backend, self.settings.sensitivity,
                                     self.settings.min_confidence)
            if self.stop_event.is_set():
                return
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("w", encoding="utf-8", buffering=1) as log:
                self.source.start()
                self.status = "Running"
                while not self.stop_event.is_set():
                    chunk = self.audio.get()
                    if chunk is None:
                        if self.source.finished.is_set():
                            if self.source.error:
                                raise RuntimeError(self.source.error)
                            break
                        continue
                    if time.monotonic() - chunk.available_at > 0.25:
                        self.stale += 1
                        continue
                    self.levels = tuple(float(v) for v in np.sqrt(np.mean(chunk.samples ** 2, axis=0)))
                    started = time.perf_counter()
                    detections = detector.process(chunk)
                    self.processing_ms = (time.perf_counter() - started) * 1000
                    self.pipeline_ms = max(0, (time.monotonic() - chunk.available_at) * 1000)
                    if self.stop_event.is_set():
                        break
                    for event in detections:
                        if self.settings.target_only and self.source.mode != "demo" and event.label not in TARGET_CLASSES:
                            continue
                        event = replace(event, processing_ms=self.processing_ms, pipeline_ms=self.pipeline_ms)
                        log.write(json.dumps(event.to_dict()) + "\n")
                        self.events.put(event)
                # Drain playback scheduling without blocking UI (audio may be ahead of DAC).
                if not self.stop_event.is_set():
                    self.stop_event.wait(0.25)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.stop_event.set()
            if self.source.ident is not None:
                self.source.join(timeout=3)
                if self.source.is_alive():
                    self.error = "Audio driver did not stop within 3 seconds. Close the app and reconnect the device."
            self.status = "Error" if self.error else "Stopped"
            self.done.set()
