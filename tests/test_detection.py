import numpy as np
import pytest

from arc_audio.audio import demo_samples
from arc_audio.detection import BaselineDetector, Debounce, stereo_direction
from arc_audio.events import AudioChunk


@pytest.mark.parametrize("gains,expected", [((1, .1), "left"), ((.1, 1), "right"),
                                           ((1, 1), "unknown"), ((0, 0), "unknown")])
def test_direction(gains, expected):
    tone = np.sin(np.arange(2400) * .17).astype(np.float32) * .1
    assert stereo_direction(tone[:, None] * gains) == expected


def test_antiphase_is_unknown_and_not_silence():
    tone = np.sin(np.arange(2400) * .17).astype(np.float32) * .1
    assert stereo_direction(np.column_stack((tone, -tone))) == "unknown"


def test_cooldown_is_independent_of_wall_clock():
    gate = Debounce(.5)
    assert gate.accept("x", 2)
    assert not gate.accept("x", 2.499)
    assert gate.accept("x", 2.5)
    assert gate.accept("y", 2.5)


def test_silence_never_alerts():
    detector = BaselineDetector()
    for i in range(100):
        assert detector.process(AudioChunk(np.zeros((2400, 2), np.float32), 48000, (i + 1) * .05, 0)) == []


def test_demo_has_both_classes_and_all_lanes_without_spam():
    detector = BaselineDetector()
    events = []
    for i in range(180):
        events += detector.process(AudioChunk(demo_samples(i * 2400, 2400), 48000, (i + 1) * .05, 0))
    assert {e.label for e in events} == {"sound activity", "transient"}
    assert {e.direction for e in events} == {"left", "right", "unknown"}
    assert all(b.audio_time - a.audio_time >= .65 - 1e-6 for a, b in zip(events, events[1:]))
    assert all(e.backend == "baseline" for e in events)


def test_confidence_can_suppress_everything():
    detector = BaselineDetector(min_confidence=1)
    for i in range(180):
        assert not detector.process(AudioChunk(demo_samples(i * 2400, 2400), 48000, (i + 1) * .05, 0))
