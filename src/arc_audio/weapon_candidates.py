"""Fast local transient proposals for human gunfire review, not a classifier.

Scores rank abrupt, prominent sounds. Impacts, machinery, footsteps and interface
sounds can all qualify; neither ``label`` nor ``confidence`` establishes gunfire
or a weapon identity. No audio is written and no fitted weights are used.
"""

import math
from numbers import Real

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

DETECTOR = "heuristic_transient_v1"


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _frame_statistics(x, rate):
    """Keep scalar envelopes for the timeline, never its full spectrogram."""
    hop = max(1, round(rate * .005))
    width = max(8, round(rate * .020))
    nfft = 1 << (width - 1).bit_length()
    count = max(1, math.ceil(len(x) / hop))
    rms, peak, flux, high = [np.zeros(count, np.float64) for _ in range(4)]
    frequencies = np.fft.rfftfreq(nfft, 1 / rate)
    high_mask = frequencies >= min(2000, rate * .30)
    taper = np.hanning(width)
    previous = np.zeros(nfft // 2 + 1)
    # A fixed batch bounds FFT temporary memory even for hour-long recordings.
    for first in range(0, count, 256):
        batch = np.zeros((min(256, count - first), width), np.float32)
        for j in range(len(batch)):
            origin = (first + j) * hop - width // 2
            a, b = max(0, origin), min(len(x), origin + width)
            batch[j, a - origin:b - origin] = x[a:b]
        powers = np.square(batch, dtype=np.float64)
        rms[first:first + len(batch)] = np.sqrt(powers.mean(axis=1))
        peak[first:first + len(batch)] = np.abs(batch).max(axis=1)
        spectra = np.abs(np.fft.rfft(batch * taper, n=nfft, axis=1))
        for j, spectrum in enumerate(spectra):
            mass = float(spectrum.sum())
            flux[first + j] = float(np.maximum(spectrum - previous, 0).sum()) / max(
                mass, float(previous.sum()), 1e-12)
            high[first + j] = float(spectrum[high_mask].sum()) / max(mass, 1e-12)
            previous = spectrum
    return hop, width, rms, peak, flux, high


def _features(x, rate):
    """Small, JSON-safe acoustic summaries of the proposed event span."""
    width = max(8, round(rate * .025))
    hop = max(1, round(rate * .010))
    nfft = 1 << (width - 1).bit_length()
    frames = []
    for first in range(-width // 2, len(x), hop):
        frame = np.zeros(width, np.float64)
        a, b = max(0, first), min(len(x), first + width)
        frame[a - first:b - first] = x[a:b]
        frames.append(np.abs(np.fft.rfft(frame * np.hanning(width), n=nfft)) ** 2)
    spectrum = np.mean(frames, axis=0)
    frequencies = np.fft.rfftfreq(nfft, 1 / rate)
    mass = max(float(spectrum.sum()), 1e-20)
    centroid = float(spectrum @ frequencies / mass)
    bandwidth = float(np.sqrt(spectrum @ (frequencies - centroid) ** 2 / mass))
    rolloff_index = min(len(frequencies) - 1, int(np.searchsorted(np.cumsum(spectrum), .85 * mass)))
    mel_max = 2595 * np.log10(1 + (rate / 2) / 700)
    edges = 700 * (10 ** (np.linspace(0, mel_max, 18) / 2595) - 1)
    filters = np.stack([np.maximum(0, np.minimum((frequencies - a) / max(b - a, 1e-12),
                                                 (c - frequencies) / max(c - b, 1e-12)))
                        for a, b, c in zip(edges, edges[1:], edges[2:])])
    log_mel = np.log10(np.maximum(filters @ spectrum, 1e-12))
    return {"duration": float(len(x) / rate),
            "rms": float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))),
            "peak": float(np.abs(x).max()),
            "zcr": float(np.mean(np.signbit(x[1:]) != np.signbit(x[:-1]))) if len(x) > 1 else 0.0,
            "centroid": centroid, "bandwidth": bandwidth, "rolloff": float(frequencies[rolloff_index]),
            "log_mel_summary": [float(value) for value in log_mel]}


def detect_candidates(signal: np.ndarray, rate: int = 16000, *, start: float = 0,
                      end: float | None = None, pre: float = .35, post: float = 1.,
                      min_spacing: float = .07) -> list[dict]:
    """Propose possible gunfire onsets; all timestamps use the original timeline.

    ``start``/``end`` limit both analyzed audio and returned context/event spans.
    The 150 ms event spans are rough proposals, distinct from playback context.
    Nearby peaks can remain separate at 70 ms spacing and share a burst ID.
    Scores are deterministic heuristic rankings, not calibrated probabilities.
    The input must be finite mono audio; decoding/resampling stays with callers.
    """
    if isinstance(rate, bool) or not isinstance(rate, (int, np.integer)) or rate <= 0:
        raise ValueError("rate must be a positive integer sample rate")
    x = np.asarray(signal)
    if x.ndim != 1 or not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
        raise ValueError("signal must be finite real mono audio")
    # Validate in chunks rather than allocating a boolean array for a long clip.
    for first in range(0, len(x), 65536):
        if not np.isfinite(x[first:first + 65536]).all():
            raise ValueError("signal must be finite real mono audio")
    duration = len(x) / rate
    start = _number(start, "start")
    end = duration if end is None else _number(end, "end")
    pre, post = _number(pre, "pre"), _number(post, "post")
    min_spacing = _number(min_spacing, "min_spacing")
    if start < 0 or end < start or end > duration + 1e-9:
        raise ValueError("range requires 0 <= start <= end <= audio duration")
    if pre < 0 or post < 0 or min_spacing <= 0:
        raise ValueError("pre/post must be nonnegative and min_spacing must be positive")
    if end <= start or len(x) == 0:
        return []
    offset, stop = math.ceil(start * rate), min(len(x), math.floor(end * rate))
    x = x[offset:stop]
    if not len(x) or float(np.max(np.abs(x))) < 1e-7:
        return []
    hop, width, rms, peak, flux, high = _frame_statistics(x, rate)
    floor_size = max(3, round(rate / hop * .75) | 1)
    floor = np.maximum(median_filter(rms, size=floor_size, mode="reflect"), 1e-6)
    previous_rms = np.concatenate(([floor[0]], rms[:-1]))
    ratio = rms / floor
    rise = rms / np.maximum(previous_rms, 1e-6)
    energy = np.clip(np.log2(np.maximum(np.maximum(ratio, rise), 1e-12)) / 4, 0, 1)
    crest = np.clip((peak / np.maximum(rms, 1e-8) - 2) / 8, 0, 1)
    score = .50 * energy + .25 * flux + .15 * crest + .10 * high
    eligible = (rms >= 1e-5) & ((ratio >= 1.7) | (rise >= 1.8))
    score = np.where(eligible, score, 0)
    # Padding includes onsets at either edge; no fabricated context is returned.
    indices, _ = find_peaks(np.pad(score, (1, 1)), height=.34, prominence=.10,
                            distance=max(1, math.ceil(min_spacing * rate / hop - 1e-9)))
    indices = indices - 1
    proposals = []
    for index in indices:
        center = int(index * hop)
        a, b = max(0, center - width // 2), min(len(x), center + width // 2)
        if b <= a:
            continue
        # Recover a sample-level peak from the local analysis frame, not its
        # arbitrary FFT center. This remains a proposed onset, not a ground truth.
        onset = a + int(np.argmax(np.abs(x[a:b])))
        timestamp = (offset + onset) / rate
        candidate = {"source_timestamp": timestamp, "detection_score": float(score[index]),
                     "confidence": float(score[index]), "noise_ratio": float(ratio[index]),
                     "noise_rms": float(floor[index]), "sample": onset}
        if proposals and timestamp - proposals[-1]["source_timestamp"] < min_spacing - 1 / rate:
            if candidate["detection_score"] > proposals[-1]["detection_score"]:
                proposals[-1] = candidate
        else:
            proposals.append(candidate)
    rows = []
    onsets = np.array([candidate["source_timestamp"] for candidate in proposals])
    for index, candidate in enumerate(proposals):
        timestamp = candidate["source_timestamp"]
        event_end = min(end, timestamp + .15)
        if index + 1 < len(proposals):
            event_end = min(event_end, proposals[index + 1]["source_timestamp"])
        first = candidate["sample"]
        last = min(len(x), max(first + 1, round((event_end * rate) - offset)))
        if event_end <= timestamp or last <= first:
            continue
        samples = x[first:last]
        features = _features(samples, rate)
        clipping = float(np.mean(np.abs(samples) >= .999))
        # Local context can contain other shots/music: this is a quality hint,
        # never a clean/noisy ground-truth label or a reason to delete a row.
        background = x[max(0, first - round(.35 * rate)):max(0, first - round(.035 * rate))]
        background_rms = (float(np.sqrt(np.mean(np.square(background, dtype=np.float64))))
                          if len(background) else candidate["noise_rms"])
        background_level = max(candidate["noise_rms"], background_rms, 1e-8)
        snr_db = float(20 * math.log10(max(features["rms"], 1e-8) / background_level))
        distinctness = 1.0
        if len(background) and background_rms > 1e-7:
            background_mel = np.array(_features(background, rate)["log_mel_summary"])
            event_mel = np.array(features["log_mel_summary"])
            a, b = event_mel - event_mel.mean(), background_mel - background_mel.mean()
            norm = float(np.linalg.norm(a) * np.linalg.norm(b))
            distinctness = float(np.clip((1 - a @ b / norm) / 2, 0, 1)) if norm > 1e-12 else 0.0
        neighbors = int(np.searchsorted(onsets, timestamp + .35, side="right")
                        - np.searchsorted(onsets, timestamp - .35, side="left") - 1)
        isolation = 1 / (1 + .3 * neighbors)
        quality = np.clip(40 * candidate["detection_score"]
                          + 20 * np.clip(snr_db / 30, 0, 1) + 15 * distinctness
                          + 15 * (1 - min(1, clipping * 10)) + 10 * isolation, 0, 100)
        rows.append({"start": timestamp, "end": event_end, "source_timestamp": timestamp,
                     "preview_start": max(start, timestamp - pre),
                     "preview_end": min(end, max(event_end, timestamp + post)),
                     "label": "gunshot", "confidence": candidate["confidence"],
                     "detection_score": candidate["detection_score"], "quality_score": float(quality),
                     "quality_factors": {"version": 2, "kind": "heuristic_review_priority",
                                         "background_level": float(background_level), "snr_db": snr_db,
                                         "competing_transients": neighbors,
                                         "spectral_distinctness": distinctness, "clipping_fraction": clipping},
                     "detector": DETECTOR, "confidence_kind": "heuristic_score", "features": features})
    group = 0
    for index, row in enumerate(rows):
        before = index > 0 and row["source_timestamp"] - rows[index - 1]["source_timestamp"] <= .35
        after = index + 1 < len(rows) and rows[index + 1]["source_timestamp"] - row["source_timestamp"] <= .35
        if after and not before:
            group += 1
        if before or after:
            row["burst_group_id"] = f"burst-{group:06d}"
    return rows
