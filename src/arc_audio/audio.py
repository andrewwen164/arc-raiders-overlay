"""Streaming PCM, synthetic fixtures, and isolated Windows PortAudio adapters."""

import sys
import threading
import time
import wave
from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly

from .events import AudioChunk

BLOCK_SECONDS = 0.05


def resample(samples, source_rate, target_rate):
    from math import gcd
    if source_rate == target_rate:
        return np.asarray(samples, dtype=np.float32)
    factor = gcd(int(source_rate), int(target_rate))
    return resample_poly(samples, target_rate // factor, source_rate // factor, axis=0).astype(np.float32)


def decode_pcm(raw: bytes, width: int) -> np.ndarray:
    if width == 1:
        return (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
    if width == 2:
        return np.frombuffer(raw, "<i2").astype(np.float32) / 32768
    if width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        n = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        return ((n ^ 0x800000) - 0x800000).astype(np.float32) / 8388608
    if width == 4:
        return (np.frombuffer(raw, "<i4").astype(np.float64) / 2147483648).astype(np.float32)
    raise ValueError("Use uncompressed stereo PCM WAV (8, 16, 24, or 32-bit integer).")


class WavReader:
    def __init__(self, path):
        try:
            self.file = wave.open(str(path), "rb")
        except (wave.Error, EOFError) as exc:
            raise ValueError("Use an uncompressed stereo PCM WAV; export float/compressed files as PCM16.") from exc
        if self.file.getnchannels() != 2 or not 8000 <= self.file.getframerate() <= 192000:
            self.file.close()
            raise ValueError("WAV must have two stereo channels and a sample rate from 8–192 kHz.")
        self.rate = self.file.getframerate()
        self.frames = self.file.getnframes()
        self.position_frames = 0

    def read(self, frames):
        samples = decode_pcm(self.file.readframes(frames), self.file.getsampwidth()).reshape(-1, 2)
        self.position_frames += len(samples)
        return samples

    def close(self):
        self.file.close()


def demo_samples(start: int, count: int, rate=48000):
    """Repeatable 9-second loop: left onset, center activity, right onset."""
    t = (np.arange(start, start + count) / rate) % 9
    out = np.zeros((count, 2), dtype=np.float32)
    for at, gains, length in [(1, (1, 0.15), 0.18), (3.5, (0.7, 0.7), 1.0),
                               (6.5, (0.15, 1), 0.18)]:
        local = t - at
        mask = (local >= 0) & (local < length)
        # Finite synthetic tone burst, no gameplay samples or classifier predictions.
        tone = 0.3 * np.sin(2 * np.pi * 850 * local[mask])
        tone *= np.minimum(local[mask] / 0.015, 1) * np.exp(-local[mask] / length)
        out[mask] += tone[:, None] * np.array(gains, dtype=np.float32)
    return out


def write_demo(path, seconds=12, rate=48000):
    with wave.open(str(path), "wb") as out:
        out.setparams((2, 2, rate, 0, "NONE", "not compressed"))
        for start in range(0, int(seconds * rate), rate):
            x = demo_samples(start, min(rate, int(seconds * rate) - start), rate)
            out.writeframes((x * 32767).astype("<i2").tobytes())


def portaudio():
    if sys.platform != "win32":
        raise RuntimeError("WASAPI capture/playback requires Windows. Demo and offline evaluation are portable.")
    try:
        import pyaudiowpatch as pa
    except ImportError as exc:
        raise RuntimeError("Install Windows dependencies with: python -m pip install -e .") from exc
    return pa


@dataclass(frozen=True)
class Device:
    index: int
    name: str
    rate: int
    channels: int


def loopback_devices():
    pa = portaudio()
    result = []
    with pa.PyAudio() as p:
        for d in p.get_loopback_device_info_generator():
            # Direction has no defensible meaning for unspecified surround channel maps.
            if int(d["maxInputChannels"]) != 2:
                continue
            rate = int(d["defaultSampleRate"])
            try:
                if p.is_format_supported(rate, input_device=d["index"], input_channels=2,
                                         input_format=pa.paFloat32):
                    result.append(Device(int(d["index"]), d["name"], rate, 2))
            except ValueError:
                continue
    return result


class AudioSource(threading.Thread):
    def __init__(self, mode, queue, stop, path="", device=None):
        super().__init__(name="audio-source", daemon=True)
        self.mode, self.queue, self.stop = mode, queue, stop
        self.path, self.device = path, device
        self.error = ""
        self.status = "Starting"
        self.finished = threading.Event()
        self.xruns = 0

    def run(self):
        try:
            if self.mode == "demo":
                self._demo()
            elif self.mode == "file":
                self._file()
            elif self.mode == "live":
                self._live()
            else:
                raise ValueError("Unknown audio source")
        except Exception as exc:
            self.error = (f"{type(exc).__name__}: {exc}. Stop, reconnect/select a stereo output device, "
                          "then Start again. For WAV errors, export stereo PCM16 at 48 kHz.")
            self.status = "Error"
        finally:
            self.finished.set()

    def _demo(self):
        self.status = "Demo • synthetic audio • no output playback"
        rate, frame = 48000, 0
        count = int(rate * BLOCK_SECONDS)
        origin = time.monotonic()
        while not self.stop.is_set():
            deadline = origin + (frame + count) / rate
            if self.stop.wait(max(0, deadline - time.monotonic())):
                break
            x = demo_samples(frame, count, rate)
            frame += count
            self.queue.put(AudioChunk(x, rate, frame / rate, deadline))

    def _file(self):
        pa = portaudio()
        reader = WavReader(self.path)
        callback_error = []
        try:
            with pa.PyAudio() as p:
                output = p.get_default_output_device_info()
                rate = int(output["defaultSampleRate"])
                output_position = 0
                # Source file chunks are resampled independently with polyphase antialiasing.
                # Prefer recordings at the output rate to avoid block-edge artifacts.
                def callback(_input, frame_count, info, flags):
                    nonlocal output_position
                    try:
                        if self.stop.is_set():
                            return (None, pa.paAbort)
                        if flags:
                            self.xruns += 1
                        # Cumulative rounding avoids clock drift at rates such as 11025 Hz.
                        output_position += frame_count
                        boundary = round(output_position * reader.rate / rate)
                        x = reader.read(boundary - reader.position_frames)
                        if len(x) == 0:
                            return (None, pa.paComplete)
                        y = resample(x, reader.rate, rate)
                        delay = max(0.0, info["output_buffer_dac_time"] - info["current_time"])
                        at = time.monotonic() + delay + len(x) / reader.rate
                        self.queue.put(AudioChunk(x, reader.rate, reader.position_frames / reader.rate, at))
                        padded = np.zeros((frame_count, 2), dtype=np.float32)
                        padded[:min(len(y), frame_count)] = y[:frame_count]
                        eof = reader.position_frames >= reader.frames
                        return (padded.tobytes(), pa.paComplete if eof else pa.paContinue)
                    except Exception as exc:
                        callback_error.append(exc)
                        return (None, pa.paAbort)

                with p.open(format=pa.paFloat32, channels=2, rate=rate, output=True,
                            frames_per_buffer=round(rate * BLOCK_SECONDS), stream_callback=callback) as stream:
                    self.status = f"Playing WAV • {reader.rate} Hz → {rate} Hz • default output"
                    while stream.is_active() and not self.stop.wait(0.02):
                        pass
                    if callback_error:
                        raise callback_error[0]
                    if not self.stop.is_set():
                        self.stop.wait(stream.get_output_latency() + BLOCK_SECONDS)
                self.status = "Playback finished"
        finally:
            reader.close()

    def _live(self):
        if self.device is None:
            raise ValueError("Choose a compatible stereo WASAPI output under Live device")
        pa = portaudio()
        position = 0
        callback_error = []
        last_callback = [time.monotonic()]
        with pa.PyAudio() as p:
            current = p.get_device_info_by_index(self.device.index)
            if current["name"] != self.device.name or not current.get("isLoopbackDevice"):
                raise RuntimeError("Audio device list changed. Refresh devices and reselect your output")

            def callback(raw, count, info, flags):
                nonlocal position
                try:
                    if self.stop.is_set():
                        return (None, pa.paAbort)
                    last_callback[0] = time.monotonic()
                    if flags:
                        self.xruns += 1
                    x = np.frombuffer(raw, np.float32).reshape(-1, 2).copy()
                    position += len(x)
                    # ADC timestamp may be unavailable (0) for loopback drivers.
                    age = info["current_time"] - info["input_buffer_adc_time"] - len(x) / self.device.rate
                    age = age if 0 <= age < 2 else 0
                    self.queue.put(AudioChunk(x, self.device.rate, position / self.device.rate,
                                              time.monotonic() - age))
                    return (None, pa.paContinue)
                except Exception as exc:
                    callback_error.append(exc)
                    return (None, pa.paAbort)

            with p.open(format=pa.paFloat32, channels=2, rate=self.device.rate, input=True,
                        input_device_index=self.device.index,
                        frames_per_buffer=round(self.device.rate * BLOCK_SECONDS),
                        stream_callback=callback) as stream:
                self.status = f"Live whole-device loopback • {self.device.rate} Hz • stereo"
                while not self.stop.wait(0.1):
                    if callback_error:
                        raise callback_error[0]
                    if not stream.is_active():
                        raise RuntimeError("WASAPI stream stopped; device may have disconnected")
                    # Silence can legitimately produce no loopback packets. Show a warning,
                    # don't invent samples or repeatedly reopen a disconnected device.
                    elapsed = time.monotonic() - last_callback[0]
                    self.status = ("No audio packets: output idle or device disconnected; play audio or refresh devices"
                                   if elapsed > 3 else f"Live whole-device loopback • {self.device.rate} Hz")
                    if elapsed > 15:
                        raise RuntimeError("No WASAPI packets for 15 seconds (idle or disconnected)")
