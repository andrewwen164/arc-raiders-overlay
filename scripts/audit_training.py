"""Read-only training readiness audit; no inference, model fitting or raw exports."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from arc_audio.dataset import TARGET_CLASSES, load_dataset, media_path, read_media_audio, reviewed_spans
from arc_audio.events import utc_now
from arc_audio.local_model import HOP, RATE, WINDOW, window_labels


def audit(manifest, classes=("gunshot",)):
    if not classes or len(set(classes)) != len(classes) or not set(classes).issubset(TARGET_CLASSES):
        raise ValueError("Choose unique supported sound classes")
    manifest = Path(manifest)
    original = manifest.read_bytes()
    document = load_dataset(manifest)
    report = {"created_at": utc_now(), "manifest": str(manifest),
              "manifest_sha256": hashlib.sha256(original).hexdigest(), "classes": list(classes),
              "window_seconds": WINDOW, "hop_seconds": HOP, "clips": [], "blockers": [],
              "note": "Readiness counts, not accuracy or independent examples. Labels/review scopes unchanged. Test audio is not decoded or window-counted."}
    totals = {split: {"positive": dict.fromkeys(classes, 0), "negative": dict.fromkeys(classes, 0),
                      "pure_background": 0, "groups": set()} for split in ("train", "validation", "test")}
    for clip in document["clips"]:
        spans = reviewed_spans(clip, classes)
        labels = [row for row in clip["intervals"] if row["label"] in classes]
        full = sum(any(span["start"] <= row["start"] and row["end"] <= span["end"] for span in spans)
                   for row in labels)
        partial = sum(not any(span["start"] <= row["start"] and row["end"] <= span["end"] for span in spans)
                      and any(min(span["end"], row["end"]) > max(span["start"], row["start"]) for span in spans)
                      for row in labels)
        counts = {"positive": dict.fromkeys(classes, 0), "negative": dict.fromkeys(classes, 0), "pure_background": 0}
        if spans and clip.get("track_confirmed") and clip["split"] != "test":
            path = media_path(manifest, clip)
            if path.stat().st_size != clip["size"]:
                raise ValueError(f"Media changed since labeling: {clip['file']}")
            signal = read_media_audio(path, clip["audio_stream"], RATE, max_seconds=max(s["end"] for s in spans))
            for end_frame in range(RATE, len(signal) + 1, round(HOP * RATE)):
                end = end_frame / RATE
                targets = window_labels(end - WINDOW, end, clip, classes)
                if targets is None:
                    continue
                for index, label in enumerate(classes):
                    counts["positive"][label] += int(targets[index] > .5)
                    counts["negative"][label] += int(targets[index] <= .5)
                counts["pure_background"] += int(np.all(targets == 0))
            split = totals[clip["split"]]
            split["groups"].add(clip["group"])
            split["pure_background"] += counts["pure_background"]
            for label in classes:
                for kind in ("positive", "negative"):
                    split[kind][label] += counts[kind][label]
        report["clips"].append({"file": clip["file"], "split": clip["split"], "group": clip["group"],
                                "audio_stream": clip["audio_stream"], "track_confirmed": clip["track_confirmed"],
                                "duration": clip["duration"], "reviewed_spans": spans,
                                "reviewed_seconds": sum(s["end"] - s["start"] for s in spans),
                                "labels_fully_reviewed": full, "labels_partly_reviewed": partial,
                                "labels_outside_review": len(labels) - full - partial, "windows": counts})
    for split_name, minimum in (("train", 20), ("validation", 5)):
        split = totals[split_name]
        if not split["groups"]:
            report["blockers"].append(f"No confirmed reviewed {split_name} session")
        if split["pure_background"] < minimum:
            report["blockers"].append(f"{split_name}: {split['pure_background']} pure-background windows; need {minimum}")
        for label in classes:
            for kind in ("positive", "negative"):
                if split[kind][label] < minimum:
                    report["blockers"].append(f"{split_name}/{label}: {split[kind][label]} {kind} windows; need {minimum}")
    for split in totals.values():
        split["groups"] = sorted(split["groups"])
    changed = manifest.read_bytes() != original
    if changed:
        report["blockers"].append("Manifest changed during audit; rerun on the saved annotations")
    report.update(totals=totals, trainable=not report["blockers"], manifest_changed_during_audit=changed)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--classes", nargs="+", choices=TARGET_CLASSES, default=["gunshot"])
    parser.add_argument("--output", type=Path, default=Path("artifacts/pilot-readiness/report.json"))
    args = parser.parse_args()
    report = audit(args.manifest, tuple(args.classes))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
