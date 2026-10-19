from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np


@dataclass(frozen=True)
class AudioChunk:
    samples: np.ndarray  # float32, frames x 2; never saved by the application
    sample_rate: int
    position: float  # seconds, end of chunk on source timeline
    available_at: float  # monotonic clock; estimated audible end for playback


@dataclass(frozen=True)
class Detection:
    timestamp: str
    audio_time: float
    label: str
    confidence: float
    direction: str
    backend: str
    processing_ms: float = 0.0
    pipeline_ms: float = 0.0
    window_start: float | None = None
    window_end: float | None = None
    present_at: float = 0.0  # internal monotonic time, omitted from logs

    def to_dict(self):
        data = asdict(self)
        data.pop("present_at")
        return data


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
