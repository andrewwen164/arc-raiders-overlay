"""Event evaluation on reviewed video/audio regions and held-out groups."""

import json
import time
from dataclasses import replace

import numpy as np

from .dataset import load_dataset, media_path, read_media_audio, reviewed_spans
from .evaluation import match_events, summarize
from .events import AudioChunk
from .local_model import RATE, LocalDetector


def evaluate_dataset(manifest, split="test", model_path=None, min_confidence=.6, tolerance=.15):
    match_events([], [], tolerance)  # Validate tolerance before decoding media.
    doc = load_dataset(manifest)
    classes = LocalDetector(min_confidence=min_confidence, path=model_path).classes
    clips = [c for c in doc["clips"] if c["split"] == split and c.get("track_confirmed") and reviewed_spans(c, classes)]
    if not clips:
        raise ValueError(f"No confirmed reviewed {split} clips; label held-out data first")
    all_detections, all_truth, timings, clip_reports = [], [], [], []
    for clip in clips:
        detector = LocalDetector(min_confidence=min_confidence, path=model_path)
        used_groups = {c["group"] for c in detector.metadata.get("clips_used", [])}
        if split == "test" and clip["group"] in used_groups:
            raise ValueError("A test recording group was used while training/validation; choose truly held-out test clips")
        path = media_path(manifest, clip)
        if path.stat().st_size != clip["size"]:
            raise ValueError(f"Media changed since labeling: {clip['file']}")
        spans = reviewed_spans(clip, classes)
        audio = read_media_audio(path, clip["audio_stream"], RATE, max_seconds=max(r["end"] for r in spans))
        detections = []
        # Reset at each reviewed region; never stitch disconnected audio together.
        # Include the first partial hop only as genuine context, not fabricated padding.
        for span in spans:
            detector = LocalDetector(min_confidence=min_confidence, path=model_path)
            start = round(span["start"] * RATE)
            end = min(len(audio), round(span["end"] * RATE))
            for first in range(start, end, 800):
                x = audio[first:min(first + 800, end)]
                chunk = AudioChunk(np.column_stack((x, x)), RATE, (first + len(x)) / RATE, time.monotonic())
                now = time.perf_counter()
                events = detector.process(chunk)
                elapsed = (time.perf_counter() - now) * 1000
                timings.append(elapsed)
                detections.extend(replace(e, processing_ms=elapsed, pipeline_ms=elapsed).to_dict() for e in events)
        truth = [row for row in clip["intervals"] if row["label"] in classes
                 and any(span["start"] <= row["start"] and row["end"] <= span["end"] for span in spans)]
        # Match per clip so coincident timestamps in different files cannot match.
        report = match_events(detections, truth, tolerance)
        by_class = {label: match_events([d for d in detections if d["label"] == label],
                                        [a for a in truth if a["label"] == label], tolerance)
                    for label in classes}
        clip_reports.append({"file": clip["file"], "group": clip["group"], "metrics": report,
                             "per_class": by_class, "detections": detections,
                             "reviewed_seconds": sum(s["end"] - s["start"] for s in spans)})
        all_detections.extend(detections)
        all_truth.extend(truth)

    def combine(reports):
        tp = sum(r["true_positives"] for r in reports)
        fp = sum(r["false_positives"] for r in reports)
        fn = sum(r["false_negatives"] for r in reports)
        return {"true_positives": tp, "false_positives": fp, "false_negatives": fn,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None}

    return {"split": split, "classes": classes, "threshold": min_confidence, "tolerance": tolerance,
            "overall": combine([r["metrics"] for r in clip_reports]),
            "per_class": {label: combine([r["per_class"][label] for r in clip_reports]) for label in classes},
            "processing_ms_per_chunk": summarize(timings), "clips": clip_reports,
            "note": "Reviewed intervals only; matching per clip, exact class, point alert within interval ±tolerance. Source tags do not establish enemy identity."}


def write_report(path, report):
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
