"""Point alerts matched one-to-one to labelled intervals, with no class aliases."""

import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .audio import BLOCK_SECONDS, WavReader
from .classifier import make_detector
from .dataset import TARGET_CLASSES
from .events import AudioChunk

LABELS = {"sound activity", "transient", *TARGET_CLASSES}


def read_annotations(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("version") != 1 or not isinstance(document.get("intervals"), list):
        raise ValueError("Annotations need version: 1 and an intervals array")
    result = []
    for row in document["intervals"]:
        if not isinstance(row, dict):
            raise ValueError("Each annotation must be an object")
        start, end = row.get("start"), row.get("end")
        if (isinstance(start, bool) or isinstance(end, bool)
                or not isinstance(start, (int, float)) or not isinstance(end, (int, float))
                or not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end):
            raise ValueError("Each annotation requires finite 0 <= start < end in seconds")
        if row.get("label") not in LABELS or row.get("direction", "any") not in {"left", "right", "unknown", "any"}:
            raise ValueError("Unsupported annotation label or direction")
        result.append(dict(row))
    return result


def match_events(detections, annotations, tolerance=0.15):
    """Maximum cardinality bipartite matching; duplicate alerts are false positives."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and nonnegative")
    candidates = []
    for detection in detections:
        allowed = []
        for index, interval in enumerate(annotations):
            if detection["label"] != interval["label"]:
                continue
            if interval.get("direction", "any") not in ("any", detection["direction"]):
                continue
            if interval["start"] - tolerance <= detection["audio_time"] <= interval["end"] + tolerance:
                allowed.append(index)
        candidates.append(sorted(allowed, key=lambda i: abs(detection["audio_time"] - annotations[i]["start"])))
    # Iterative augmenting paths avoid recursion limits on long recordings.
    owners = {}
    for origin in range(len(detections)):
        stack, visited_d, visited_a, parents = [origin], {origin}, set(), {}
        free = None
        while stack and free is None:
            d = stack.pop()
            for a in candidates[d]:
                if a in visited_a:
                    continue
                visited_a.add(a)
                parents[a] = d
                if a not in owners:
                    free = a
                    break
                other = owners[a]
                if other not in visited_d:
                    visited_d.add(other)
                    stack.append(other)
        if free is not None:
            reverse = {d: a for a, d in owners.items()}
            while True:
                d = parents[free]
                previous = reverse.get(d)
                owners[free] = d
                if previous is None:
                    break
                free = previous
    tp = len(owners)
    fp, fn = len(detections) - tp, len(annotations) - tp
    delays = [max(0, detections[d]["audio_time"] - annotations[a]["start"]) * 1000 for a, d in owners.items()]
    return {"true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "matched_onset_delay_ms": summarize(delays)}


def summarize(values):
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {"count": len(values), "mean": float(np.mean(values)), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "max": float(np.max(values))}


def evaluate(wav, annotations=None, backend="baseline", sensitivity=1.0, min_confidence=0.6,
             output=None, tolerance=0.15):
    detector = make_detector(backend, sensitivity, min_confidence)
    reader = WavReader(wav)
    detections, timings = [], []
    output_file = None
    try:
        if output:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            output_file = path.open("w", encoding="utf-8")
        while True:
            samples = reader.read(round(reader.rate * BLOCK_SECONDS))
            if not len(samples):
                break
            chunk = AudioChunk(samples, reader.rate, reader.position_frames / reader.rate, time.monotonic())
            started = time.perf_counter()
            events = detector.process(chunk)
            elapsed = (time.perf_counter() - started) * 1000
            timings.append(elapsed)
            for event in events:
                data = replace(event, processing_ms=elapsed, pipeline_ms=elapsed).to_dict()
                detections.append(data)
                if output_file:
                    output_file.write(json.dumps(data) + "\n")
    finally:
        reader.close()
        if output_file:
            output_file.close()
    report = {"backend": backend, "duration_seconds": reader.frames / reader.rate,
              "detections": len(detections), "processing_ms_per_chunk": summarize(timings),
              "note": "Offline speed excludes capture, playback, queueing and UI. Scores are not calibrated accuracy."}
    if annotations:
        truth = read_annotations(annotations)
        report["overall"] = match_events(detections, truth, tolerance)
        report["per_class"] = {label: match_events([d for d in detections if d["label"] == label],
                                                   [a for a in truth if a["label"] == label], tolerance)
                               for label in sorted({a["label"] for a in truth} | {d["label"] for d in detections})}
        report["matching"] = f"Exact class, optional direction, alert point within interval ±{tolerance}s; one-to-one maximum matching"
    else:
        report["accuracy"] = "Not computed: no annotations supplied"
    return report
