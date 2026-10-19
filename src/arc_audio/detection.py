"""Deliberately heuristic baseline: never names a game sound or a person."""

import math

import numpy as np

from .events import AudioChunk, Detection, utc_now


def stereo_direction(samples: np.ndarray) -> str:
    if samples.ndim != 2 or samples.shape[1] != 2 or len(samples) == 0:
        return "unknown"
    energy = np.mean(np.square(samples.astype(np.float64)), axis=0)
    if float(energy.max()) < 1e-8:
        return "unknown"
    balance = float((energy[1] - energy[0]) / (energy.sum() + 1e-12))
    if balance < -0.35:
        return "left"
    if balance > 0.35:
        return "right"
    return "unknown"  # center placement means ambiguous evidence, never "in front"


class Debounce:
    def __init__(self, cooldown=0.65):
        self.cooldown = cooldown
        self.last = {}

    def accept(self, key, at):
        if at - self.last.get(key, -math.inf) < self.cooldown:
            return False
        self.last[key] = at
        return True


class BaselineDetector:
    def __init__(self, sensitivity=1.0, min_confidence=0.6):
        self.sensitivity = sensitivity
        self.min_confidence = min_confidence
        self.noise = 0.002
        self.previous_rms = 0.0
        self.previous_spectrum = None
        self.debounce = Debounce()
        self.warmup = 0.0

    def process(self, chunk: AudioChunk) -> list[Detection]:
        x = np.asarray(chunk.samples, dtype=np.float32)
        if not len(x):
            return []
        # Energy across both channels avoids cancellation of anti-phase stereo.
        rms = float(np.sqrt(np.mean(x * x)))
        spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))[:, None], axis=0)).mean(axis=1)
        spectrum /= float(np.linalg.norm(spectrum) + 1e-12)
        flux = 0.0
        if self.previous_spectrum is not None and len(spectrum) == len(self.previous_spectrum):
            flux = float(np.linalg.norm(np.maximum(spectrum - self.previous_spectrum, 0)))
        rise = rms / max(self.previous_rms, self.noise, 1e-5)
        self.previous_rms, self.previous_spectrum = rms, spectrum
        duration = len(x) / chunk.sample_rate
        threshold = max(0.004, self.noise * 3.0) / self.sensitivity
        # Follow quiet background rapidly, loud sustained background slowly.
        alpha = 1 - math.exp(-duration / (1.0 if rms < threshold else 12.0))
        self.noise = max(1e-5, self.noise + alpha * (rms - self.noise))
        self.warmup += duration
        if self.warmup < 0.3 or rms < threshold:
            return []
        transient = rise > 1.7 and flux > 0.12
        confidence = float(np.clip(0.35 + 0.23 * math.log2(max(rms / threshold, 1))
                                   + (0.18 if transient else 0), 0, 0.99))
        if confidence < self.min_confidence:
            return []
        # One shared cooldown prevents the same onset being reported as both classes.
        if not self.debounce.accept("activity", chunk.position):
            return []
        return [Detection(utc_now(), chunk.position, "transient" if transient else "sound activity",
                          confidence, stereo_direction(x), "baseline", present_at=chunk.available_at)]
