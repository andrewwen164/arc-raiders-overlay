"""Small locally trained multi-label logistic model; no pretrained guesses."""

import hashlib
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.signal import stft
from scipy.special import expit

from .audio import resample
from .dataset import CORE_CLASSES, TARGET_CLASSES, load_dataset, media_path, read_media_audio, reviewed_spans
from .detection import Debounce
from .events import Detection, utc_now
from .settings import data_dir

RATE = 16000
WINDOW = 1.0
HOP = 0.25
FEATURE_VERSION = 1


def default_model_path():
    return data_dir() / "models" / "arc-targets.npz"


def _mel_filters():
    def mel(hz):
        return 2595 * np.log10(1 + hz / 700)
    points = 700 * (10 ** (np.linspace(mel(60), mel(7600), 34) / 2595) - 1)
    frequencies = np.fft.rfftfreq(512, 1 / RATE)
    return np.stack([np.maximum(0, np.minimum((frequencies - a) / (b - a),
                                             (c - frequencies) / (c - b)))
                     for a, b, c in zip(points, points[1:], points[2:])]).astype(np.float32)


MEL_FILTERS = _mel_filters()


def features(signal):
    x = np.asarray(signal, np.float32)
    if x.shape != (RATE,) or not np.isfinite(x).all():
        raise ValueError("Local model features require one finite mono second at 16 kHz")
    _, _, spectrum = stft(x, RATE, nperseg=400, noverlap=240, nfft=512, boundary=None, padded=False)
    logmel = np.log(np.maximum(MEL_FILTERS @ np.abs(spectrum) ** 2, 1e-12))
    # Four temporal bins retain some reload/onset structure. Learned weights,
    # not these features alone, determine semantic labels.
    bins = np.array_split(logmel, 4, axis=1)
    pooled = [stat(part, axis=1) for part in bins for stat in (np.mean, np.std, np.max)]
    return np.concatenate(pooled).astype(np.float32)


def window_labels(start, end, clip, classes=CORE_CLASSES):
    if not any(r["start"] <= start + 1e-6 and end <= r["end"] + 1e-6
               for r in reviewed_spans(clip, classes)):
        return None
    labels = np.zeros(len(classes), np.float32)
    for interval in clip.get("intervals", []):
        if interval["label"] not in classes:
            continue
        overlap = min(end, interval["end"]) - max(start, interval["start"])
        if overlap <= 0:
            continue
        minimum = min(.1, (interval["end"] - interval["start"]) / 2)
        if overlap < minimum - 1e-6:
            return None  # Don't call a weak boundary overlap background.
        labels[classes.index(interval["label"])] = 1
    return labels


def fit_logistic(x, y, regularization=.03):
    """Class-balanced independent binary heads, deterministic L-BFGS optimization."""
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), .1)
    z = np.clip((x - mean) / scale, -12, 12).astype(np.float64)
    z = np.column_stack((z, np.ones(len(z))))
    weights = []
    for column in y.T:
        positive = float(column.sum())
        if positive == 0 or positive == len(column):
            raise ValueError("Each class needs positive and negative training examples")
        importance = np.where(column > 0, len(column) / (2 * positive),
                              len(column) / (2 * (len(column) - positive)))

        def objective(w):
            logits = z @ w
            loss = np.mean(importance * (np.logaddexp(0, logits) - column * logits))
            loss += regularization * np.sum(w[:-1] ** 2) / 2
            gradient = z.T @ (importance * (expit(logits) - column)) / len(z)
            gradient[:-1] += regularization * w[:-1]
            return loss, gradient

        fit = minimize(objective, np.zeros(z.shape[1]), jac=True, method="L-BFGS-B",
                       options={"maxiter": 250})
        if not fit.success:
            raise RuntimeError(f"Training failed to converge: {fit.message}")
        weights.append(fit.x)
    return mean, scale, np.stack(weights, axis=1)


def probabilities(x, mean, scale, weights):
    z = np.clip((np.atleast_2d(x) - mean) / scale, -12, 12)
    return expit(np.column_stack((z, np.ones(len(z)))) @ weights)


def binary_metrics(y, scores, threshold=.6, classes=CORE_CLASSES):
    result = {}
    for index, label in enumerate(classes):
        truth, predicted = y[:, index] > .5, scores[:, index] >= threshold
        tp = int(np.sum(truth & predicted))
        fp = int(np.sum(~truth & predicted))
        fn = int(np.sum(truth & ~predicted))
        result[label] = {"tp": tp, "fp": fp, "fn": fn,
                         "precision": tp / (tp + fp) if tp + fp else None,
                         "recall": tp / (tp + fn) if tp + fn else None}
    return result


def _presence_rows(clip, classes):
    # Import here so the offline model and quick-review producer share exactly
    # the same eligibility rules, without making inference load review widgets.
    from .presence_review import presence_training_rows
    return presence_training_rows(clip, classes)


def _media_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clip_training_windows(clip, audio, classes):
    """Combine legacy exhaustive intervals and exact human presence questions.

    A presence answer supplies precisely one existing second, never a sliding
    family of windows and never event onset/offset or another class's negatives.
    """
    targets, legacy_keys, presence_keys = {}, set(), set()
    for end_frame in range(RATE, len(audio) + 1, round(HOP * RATE)):
        end = end_frame / RATE
        target = window_labels(end - WINDOW, end, clip, classes)
        if target is not None:
            key = (end_frame - RATE, end_frame)
            targets[key] = target
            legacy_keys.add(key)
    spans = reviewed_spans(clip, classes)
    for row in _presence_rows(clip, classes):
        first, last = round(row["start"] * RATE), round(row["end"] * RATE)
        if (abs(first - row["start"] * RATE) > 1e-6
                or abs(last - row["end"] * RATE) > 1e-6 or last - first != RATE):
            raise ValueError(f"Presence review is not an exact sample-aligned second: {clip['file']}")
        if first < 0 or last > len(audio):
            raise ValueError(f"Presence review exceeds available audio: {clip['file']}. Recheck its source and answer.")
        key = first, last
        target = np.array([float(row["present"])], np.float32)
        fully_reviewed = any(span["start"] <= row["start"] + 1e-6
                             and row["end"] <= span["end"] + 1e-6 for span in spans)
        # 'Any gunfire?' differs from a legacy grid window's weak-boundary
        # exclusion. Even a small audible overlap contradicts a human No.
        legacy_present = any(interval["label"] == "gunshot"
                             and min(row["end"], interval["end"]) > max(row["start"], interval["start"])
                             for interval in clip.get("intervals", []))
        if ((not row["present"] and legacy_present)
                or (row["present"] and fully_reviewed and not legacy_present)
                or (key in targets and not np.array_equal(targets[key], target))):
            raise ValueError(f"Conflicting human labels in {clip['file']} at {row['start']:.4f}-{row['end']:.4f}s. "
                             "Reopen quick review and resolve its answer or the overlapping saved intervals before training.")
        targets[key] = target
        presence_keys.add(key)
    return targets, legacy_keys, presence_keys


def train_model(manifest, output=None, progress=print, classes=CORE_CLASSES):
    classes = tuple(classes)
    if not classes or len(set(classes)) != len(classes) or not set(classes).issubset(TARGET_CLASSES):
        raise ValueError("Choose one or more unique supported sound classes")
    manifest = Path(manifest)
    manifest_bytes = manifest.read_bytes()
    doc = load_dataset(manifest)
    eligible = [c for c in doc["clips"] if c.get("track_confirmed")
                and (reviewed_spans(c, classes) or _presence_rows(c, classes))]
    if not eligible:
        raise ValueError("No reviewed clips with confirmed audio tracks covering the selected classes. Run label-training and review the selected event types plus background first.")
    for split in ("train", "validation"):
        selected = [c for c in eligible if c["split"] == split]
        if not selected:
            raise ValueError(f"No reviewed {split} clips. Label separate recording groups for train and validation.")
        for label in classes:
            if not (any(r["label"] == label for c in selected for r in c.get("intervals", []))
                    or (label == "gunshot" and any(row["present"] for c in selected
                                                   for row in _presence_rows(c, classes)))):
                raise ValueError(f"No {label} labels in {split}. Every selected class needs verified examples in both splits.")
    # Do not load/use held-out test audio while fitting or selecting anything.
    arrays = {s: ([], []) for s in ("train", "validation")}
    annotation_counts = {s: {"presence_windows": 0, "interval_windows": 0, "legacy_interval_windows": 0}
                         for s in arrays}
    used = []
    for clip in eligible:
        if clip["split"] == "test":
            continue
        path = media_path(manifest, clip)
        if not path.is_file() or path.stat().st_size != clip["size"]:
            raise ValueError(f"Media changed or is missing: {clip['file']}. Recheck annotations before training.")
        presence = _presence_rows(clip, classes)
        stamp = path.stat()
        if presence and clip.get("presence_media") is not None:
            if clip["presence_media"] != {"size": stamp.st_size, "mtime_ns": stamp.st_mtime_ns}:
                raise ValueError(f"Media changed since quick review: {clip['file']}. Recheck its source before training.")
        if presence and clip.get("source_sha256") and _media_sha256(path) != clip["source_sha256"]:
            raise ValueError(f"Media SHA-256 changed: {clip['file']}. Recheck quick-review answers before training.")
        progress(f"Reading {clip['file']} / {clip['split']}")
        # Cap decode to the reviewed portion, not the entire collection.
        ends = [r["end"] for r in reviewed_spans(clip, classes)] + [r["end"] for r in presence]
        audio = read_media_audio(path, clip["audio_stream"], RATE,
                                 max_seconds=max(ends))
        if path.stat().st_size != stamp.st_size or path.stat().st_mtime_ns != stamp.st_mtime_ns:
            raise ValueError(f"Media changed during training: {clip['file']}. Close writers and recheck labels.")
        xs, ys = arrays[clip["split"]]
        windows, legacy_keys, presence_keys = _clip_training_windows(clip, audio, classes)
        for (first, last), target in sorted(windows.items()):
            xs.append(features(audio[first:last]))
            ys.append(target)
        row_counts = annotation_counts[clip["split"]]
        row_counts["presence_windows"] += len(presence_keys)
        row_counts["interval_windows"] += len(legacy_keys - presence_keys)
        row_counts["legacy_interval_windows"] += len(legacy_keys)
        used.append({"file": clip["file"], "group": clip["group"], "split": clip["split"],
                     "audio_stream": clip["audio_stream"], "presence_windows": len(presence_keys),
                     "source_sha256": clip.get("source_sha256")})
    counts = {}
    converted = {}
    for split, (xs, ys) in arrays.items():
        if not xs:
            raise ValueError(f"No complete reviewed 1-second windows in {split}")
        x, y = np.stack(xs), np.stack(ys)
        minimum = 20 if split == "train" else 5
        if int(np.sum(np.all(y == 0, axis=1))) < minimum:
            raise ValueError(f"{split} needs at least {minimum} reviewed pure-background windows with none of the selected targets")
        for index, label in enumerate(classes):
            pos, neg = int(y[:, index].sum()), int((1 - y[:, index]).sum())
            if min(pos, neg) < minimum:
                raise ValueError(f"{split}/{label}: {pos} positive, {neg} negative windows; need at least {minimum} of each. Label more audio including background.")
        counts[split] = {"windows": len(y), **annotation_counts[split],
                         "positive_windows": dict(zip(classes, y.sum(axis=0).astype(int).tolist()))}
        converted[split] = x, y
    progress(f"Fitting {len(classes)} independent logistic heads on log-mel features")
    started = time.perf_counter()
    mean, scale, weights = fit_logistic(*converted["train"])
    x_val, y_val = converted["validation"]
    has_presence = any(row["presence_windows"] for row in annotation_counts.values())
    has_intervals = any(row["legacy_interval_windows"] for row in annotation_counts.values())
    annotation_kind = ("window_presence_and_intervals" if has_presence and has_intervals
                       else "window_presence" if has_presence else "intervals")
    report = {"version": 1, "feature_version": FEATURE_VERSION, "classes": list(classes),
              "created_at": utc_now(),
              "rate": RATE, "window_seconds": WINDOW, "hop_seconds": HOP,
              "training_seconds": time.perf_counter() - started,
              "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
              "annotation_kind": annotation_kind,
              "clips_used": used, "counts": counts,
              "validation_window_metrics_at_0.6": binary_metrics(y_val, probabilities(x_val, mean, scale, weights), classes=classes),
              "limitations": "Experimental shallow model. Scores uncalibrated. Sampled-window metrics are not event recall or gameplay accuracy; test split untouched. Presence answers describe only their exact seconds, not event boundaries. Overlapping windows are not independent examples. Cannot identify enemy/teammate/self or weapon identity."}
    if manifest.read_bytes() != manifest_bytes:
        raise ValueError("Dataset changed during training. Close the labeler and rerun; no model was saved.")
    output = Path(output) if output else default_model_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, mean=mean, scale=scale, weights=weights,
                            metadata=np.array(json.dumps(report)))
    temporary.replace(output)
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output, report


class LocalDetector:
    def __init__(self, sensitivity=1.0, min_confidence=.6, path=None):
        path = Path(path) if path else default_model_path()
        if not path.exists():
            raise RuntimeError("No trained ARC model. Label training/dataset.json, then run: python -m arc_audio train-local training/dataset.json")
        with np.load(path, allow_pickle=False) as model:
            self.mean, self.scale, self.weights = (model[k].copy() for k in ("mean", "scale", "weights"))
            self.metadata = json.loads(str(model["metadata"]))
        self.classes = self.metadata.get("classes", [])
        if (self.metadata.get("feature_version") != FEATURE_VERSION
                or not isinstance(self.classes, list) or not self.classes
                or not all(label in TARGET_CLASSES for label in self.classes)
                or len(set(self.classes)) != len(self.classes)
                or self.mean.shape != (384,) or self.scale.shape != (384,)
                or self.weights.shape != (385, len(self.classes))
                or not all(np.isfinite(a).all() for a in (self.mean, self.scale, self.weights))
                or np.any(self.scale <= 0)):
            raise ValueError("Invalid local model format; retrain with this application version")
        self.min_confidence = min_confidence
        self.parts, self.frames = deque(), 0
        self.previous_end = None
        self.last_inference = -1e9
        self.debounce = Debounce(.8)

    def process(self, chunk):
        duration = len(chunk.samples) / chunk.sample_rate
        if self.previous_end is not None and abs(chunk.position - duration - self.previous_end) > .02:
            self.parts.clear()
            self.frames = 0
        self.previous_end = chunk.position
        x = resample(chunk.samples.mean(axis=1), chunk.sample_rate, RATE)
        self.parts.append(x)
        self.frames += len(x)
        while self.parts and self.frames - len(self.parts[0]) >= RATE:
            self.frames -= len(self.parts.popleft())
        if self.frames < RATE or chunk.position - self.last_inference < HOP - 1e-6:
            return []
        self.last_inference = chunk.position
        signal = np.concatenate(self.parts)[-RATE:]
        scores = probabilities(features(signal), self.mean, self.scale, self.weights)[0]
        events = []
        for label, score in zip(self.classes, scores):
            if score >= self.min_confidence and self.debounce.accept(label, chunk.position):
                events.append(Detection(utc_now(), chunk.position, label, float(score), "unknown", "local",
                                        window_start=chunk.position - WINDOW, window_end=chunk.position,
                                        present_at=chunk.available_at))
        return events
