import wave

import numpy as np
import pytest

from arc_audio.audio import AudioSource, WavReader, decode_pcm, resample, write_demo
from arc_audio.buffering import LatestQueue


def test_pcm24_extremes():
    raw = bytes([0, 0, 128, 0, 0, 0, 255, 255, 127])
    assert decode_pcm(raw, 3) == pytest.approx([-1, 0, 1 - 1 / 8388608])


def test_resampling_preserves_time_and_rejects_out_of_band_tone():
    rate = 48000
    t = np.arange(rate) / rate
    low = resample(np.sin(2 * np.pi * 1000 * t), rate, 16000)
    high = resample(np.sin(2 * np.pi * 14000 * t), rate, 16000)
    assert len(low) == 16000
    assert np.sqrt(np.mean(low ** 2)) > .65
    assert np.sqrt(np.mean(high ** 2)) < .01


def test_stereo_wav_and_mono_error(tmp_path):
    path = tmp_path / "test.wav"
    write_demo(path, 1)
    reader = WavReader(path)
    try:
        assert reader.rate == 48000
        assert reader.read(100).shape == (100, 2)
    finally:
        reader.close()
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0" * 320)
    with pytest.raises(ValueError, match="two stereo channels"):
        WavReader(path)


def test_demo_worker_stop_is_prompt_and_queue_is_bounded():
    import threading
    import time
    stop = threading.Event()
    queue = LatestQueue(1)
    source = AudioSource("demo", queue, stop)
    source.start()
    time.sleep(.18)
    stop.set()
    source.join(1)
    assert source.finished.is_set() and not source.is_alive()
    assert queue.dropped >= 1
    assert not source.error
