"""Offline model proposals, stored separately from human annotations.

No suggestion or rejection marks a range reviewed. The trainer never reads this
sidecar. Only an explicit human acceptance copies an interval into dataset.json.
"""

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from .annotations import SOURCE_TAGS, invalidate_review, remove_interval, write_interval
from .dataset import (
    TARGET_CLASSES,
    load_dataset,
    media_path,
    read_media_audio,
    save_dataset,
    validate_span,
    validate_weapon,
)

RATE = 16000
AST_CLASSES = ("gunshot", "footsteps")


def suggestion_path(manifest):
    return Path(manifest).with_suffix(".suggestions.json")


def media_stamp(path):
    info = Path(path).stat()
    return {"size": info.st_size, "mtime_ns": info.st_mtime_ns}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_suggestions(manifest):
    path = suggestion_path(manifest)
    if not path.exists():
        return {"version": 1, "clips": []}
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("version") != 1 or not isinstance(doc.get("clips"), list):
        raise ValueError("Invalid suggestions sidecar; expected version 1 and clips")
    seen = set()
    for clip in doc["clips"]:
        if (not isinstance(clip, dict) or not isinstance(clip.get("file"), str)
                or not isinstance(clip.get("media"), dict)
                or not isinstance(clip.get("audio_stream"), int)
                or not isinstance(clip.get("suggestions"), list)):
            raise ValueError("Invalid suggestion clip metadata")
        for row in clip["suggestions"]:
            validate_span(row, clip.get("duration", 0))
            score = row.get("confidence")
            identity = row.get("id")
            if (not isinstance(identity, str) or not identity or identity in seen
                    or row.get("label") not in TARGET_CLASSES
                    or row.get("status") not in ("pending", "accepted", "rejected", "corrected")
                    or isinstance(score, bool) or not isinstance(score, (int, float))
                    or not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError("Invalid suggestion ID, label, status or model score")
            seen.add(identity)
            validate_weapon(row.get("proposed_weapon"))
            if "collection_source_id" in row:
                if row["label"] != "gunshot" or not isinstance(row["collection_source_id"], str):
                    raise ValueError("Weapon candidates require a gunshot label and source identity")
                for field, maximum in (("detection_score", 1), ("quality_score", 100)):
                    value = row.get(field)
                    if (isinstance(value, bool) or not isinstance(value, (int, float))
                            or not math.isfinite(value) or not 0 <= value <= maximum):
                        raise ValueError("Weapon candidate scores must be finite and within their documented range")
                validate_span({"start": row.get("preview_start"), "end": row.get("preview_end")}, clip["duration"])
                timestamp = row.get("source_timestamp")
                if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                        or not math.isfinite(timestamp)
                        or not row["preview_start"] <= timestamp < row["preview_end"]):
                    raise ValueError("Weapon candidate onset must lie within its source preview")
                possible = row.get("possible_weapons", [])
                if (not isinstance(possible, list) or any(not isinstance(weapon, str) for weapon in possible)
                        or len(set(possible)) != len(possible)):
                    raise ValueError("Possible weapons must be a list")
                for weapon in possible:
                    validate_weapon(weapon)
            if "last_edit" in row:
                edit = row["last_edit"]
                validate_span(edit, clip.get("duration", 0))
                if edit.get("label") not in TARGET_CLASSES or edit.get("source", "unknown") not in SOURCE_TAGS:
                    raise ValueError("Invalid saved human edit in suggestion")
                validate_weapon(edit.get("weapon"))
                if edit.get("weapon") is not None and edit["label"] != "gunshot":
                    raise ValueError("Weapon identity is supported only on saved gunshot edits")
    return doc


def current_suggestions(bundle, manifest, clip):
    """Hide proposals if the media or selected audio stream has changed."""
    entries = [entry for entry in bundle["clips"] if entry["file"] == clip["file"]
               and entry["audio_stream"] == clip["audio_stream"]]
    if not entries:
        return []
    try:
        stamp = media_stamp(media_path(manifest, clip))
    except FileNotFoundError:
        return []
    return [row for entry in entries if entry["media"] == stamp for row in entry["suggestions"]]


def accept_suggestion(clip, row, start, end, label, source="unknown", weapon=None):
    if row.get("proposal_kind") == "gunshot_absence":
        if not clip.get("track_confirmed"):
            raise ValueError("Listen to and confirm the game-audio track first")
        edit = {"start": start, "end": end, "label": "gunshot", "source": source,
                "review_kind": "gunshot_absence"}
        validate_span(edit, clip["duration"])
        if label != "gunshot" or weapon or source not in SOURCE_TAGS:
            raise ValueError("A background decision reviews absence of gunfire only; reject it and add an event for audible sounds")
        if any(interval["label"] == "gunshot" and max(start, interval["start"]) < min(end, interval["end"])
               for interval in clip["intervals"]):
            raise ValueError("This range overlaps a saved gunshot. Trim it or reject the background proposal; existing labels preserved")
        _clear_absence_review(clip, row)
        clip.setdefault("reviewed", []).append({"start": start, "end": end, "classes": ["gunshot"],
                                                "proposal_id": row["id"], "review_kind": "gunshot_absence"})
        row.update(status="accepted", last_edit=edit)
        return
    # Re-acceptance updates the linked human label, including after an interrupted
    # sidecar save. Other human labels are never removed or duplicated.
    index = next((i for i, old in enumerate(clip["intervals"]) if old.get("proposal_id") == row["id"]), None)
    if weapon is None:
        weapon = suggestion_edit(clip, row).get("weapon", row.get("proposed_weapon"))
    interval = write_interval(clip, start, end, label, source, index=index, proposal_id=row["id"], weapon=weapon)
    proposed = row.get("proposed_weapon")
    row["status"] = "corrected" if label == "gunshot" and interval.get("weapon") != proposed else "accepted"
    row["last_edit"] = dict(interval)
    # Deliberately do not add a reviewed range: misses still require inspection.


def change_suggestion_status(clip, row, status):
    """Undo or reverse any decision, removing only its linked human interval."""
    if status not in ("pending", "rejected"):
        raise ValueError("Use acceptance to save a label; otherwise choose pending or rejected")
    if row.get("proposal_kind") == "gunshot_absence":
        _clear_absence_review(clip, row)
    for interval in list(clip["intervals"]):
        if interval.get("proposal_id") == row["id"]:
            row["last_edit"] = dict(interval)
            remove_interval(clip, interval)
    row["status"] = status


def _clear_absence_review(clip, row):
    """Reverse a verified absence, including a dataset-first interrupted save.

    A later manual review merge can erase provenance. In that case invalidate
    only gunshot coverage within the accepted range, retaining other classes.
    """
    linked = any(review.get("proposal_id") == row["id"] for review in clip.get("reviewed", []))
    clip["reviewed"] = [review for review in clip.get("reviewed", []) if review.get("proposal_id") != row["id"]]
    previous = row.get("last_edit")
    if (not linked and row["status"] in ("accepted", "corrected") and previous
            and previous.get("review_kind") == "gunshot_absence"):
        invalidate_review(clip, previous)


def suggestion_edit(clip, row):
    result = next((interval for interval in clip["intervals"] if interval.get("proposal_id") == row["id"]),
                  row.get("last_edit"))
    if result is not None:
        return result
    return dict(row, **({"weapon": row["proposed_weapon"]} if row.get("proposed_weapon") else {}))


def window_spans(count, window, hop):
    """Full windows plus an end-aligned window; short clips get one padded input."""
    if count <= 0 or window <= 0 or not 0 < hop <= window:
        raise ValueError("Audio and window/hop sizes must be positive; hop cannot exceed window")
    last = max(0, count - window)
    starts = list(range(0, last + 1, hop))
    if starts[-1] != last:
        starts.append(last)
    return [(start, min(count, start + window)) for start in starts]


def scan_windows(signal, scorer, *, duration, window_seconds, hop_seconds, threshold, classes,
                 identity, provenance):
    rows, measurements = [], []
    spans = window_spans(len(signal), round(window_seconds * RATE), round(hop_seconds * RATE))
    for first, last in spans:
        started = time.perf_counter()
        scores = scorer(signal[first:last])
        elapsed = (time.perf_counter() - started) * 1000
        measurements.append(elapsed)
        for label in classes:
            score = float(scores[label])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Model returned an invalid score; no predictions were fabricated")
            if score < threshold:
                continue
            start, end = first / RATE, min(last / RATE, duration)
            if end <= start:
                continue
            token = json.dumps([identity, provenance, label, first, last], sort_keys=True)
            rows.append({"id": hashlib.sha256(token.encode()).hexdigest()[:32],
                         "start": start, "end": end, "label": label, "confidence": score,
                         "status": "pending", "processing_ms": elapsed,
                         "provenance": provenance})
    return rows, measurements


def generate_suggestions(manifest, *, model_path=None, threshold=.2, window_seconds=2.0,
                         hop_seconds=2.0, classes=("gunshot",), allow_unconfirmed=False,
                         progress=print):
    from .classifier import MODEL_ID, WINDOW_SECONDS, ASTDetector, model_dir

    if (not all(math.isfinite(x) for x in (threshold, window_seconds, hop_seconds))
            or not 0 < threshold <= 1 or not .5 <= window_seconds <= WINDOW_SECONDS
            or not .25 <= hop_seconds <= window_seconds):
        raise ValueError("Use threshold in (0,1], window 0.5..10.24 s and hop 0.25..window seconds")
    if not classes or len(set(classes)) != len(classes) or not set(classes).issubset(AST_CLASSES):
        raise ValueError("Pretrained suggestions support only gunshot and footsteps")
    manifest = Path(manifest)
    dataset = load_dataset(manifest)
    original = manifest.read_bytes()
    bundle = load_suggestions(manifest)
    original_sidecar = suggestion_path(manifest).read_bytes() if suggestion_path(manifest).exists() else None
    selected = [clip for clip in dataset["clips"] if clip["split"] != "test"
                and (clip.get("track_confirmed") or allow_unconfirmed)]
    if not selected:
        raise ValueError("No eligible clips: confirm game-audio tracks first. Test clips are always excluded.")
    location = Path(model_path) if model_path else model_dir()
    # Cached-only model loading: the generation command performs no downloads.
    detector = ASTDetector(path=location)
    provenance = {"backend": "ast", "model": MODEL_ID,
                  "weights_sha256": file_sha256(location / "model.safetensors"),
                  "config_sha256": file_sha256(location / "config.json"),
                  "features_sha256": file_sha256(location / "preprocessor_config.json"),
                  "window_seconds": window_seconds, "hop_seconds": hop_seconds,
                  "short_window_feature_padding": window_seconds < WINDOW_SECONDS}
    report = {"provenance": provenance, "threshold": threshold, "clips": [],
              "notice": "Uncalibrated window suggestions, not verified events or onset boundaries. "
                        "Short inputs are feature-padded; performance on ARC is unmeasured. "
                        "Review misses as well as proposals. Test clips excluded."}
    for clip in selected:
        progress(f"Scoring {clip['file']} / stream {clip['audio_stream']} "
                 f"({'confirmed' if clip.get('track_confirmed') else 'UNCONFIRMED provisional track'})")
        path = media_path(manifest, clip)
        stamp = media_stamp(path)
        if stamp["size"] != clip["size"]:
            raise ValueError(f"Media changed: {clip['file']}; recheck the inventory and labels")
        digest = file_sha256(path)
        signal = read_media_audio(path, clip["audio_stream"], RATE)
        identity = [clip["file"], digest, clip["audio_stream"]]
        rows, timings = scan_windows(signal, detector.score_window, duration=clip["duration"],
                                    window_seconds=window_seconds, hop_seconds=hop_seconds,
                                    threshold=threshold, classes=classes, identity=identity,
                                    provenance=provenance)
        if media_stamp(path) != stamp:
            raise ValueError(f"Media changed during analysis: {clip['file']}; proposals were not saved")
        previous = current_suggestions(bundle, manifest, clip)
        by_id = {row["id"]: row for row in previous}
        for row in rows:
            if row["id"] not in by_id:
                by_id[row["id"]] = row
        entry = {"file": clip["file"], "duration": clip["duration"], "audio_stream": clip["audio_stream"],
                 "media": stamp, "media_sha256": digest,
                 "track_confirmed_at_generation": bool(clip.get("track_confirmed")),
                 "suggestions": sorted(by_id.values(), key=lambda row: (row["start"], row["label"]))}
        bundle["clips"] = [c for c in bundle["clips"]
                           if (c["file"], c["audio_stream"]) != (clip["file"], clip["audio_stream"])] + [entry]
        report["clips"].append({"file": clip["file"], "audio_stream": clip["audio_stream"],
                                "windows_scored": len(timings), "qualifying_windows": len(rows),
                                "processing_mean_ms": float(np.mean(timings)),
                                "processing_max_ms": max(timings)})
        progress(f"  {len(rows)} qualifying class/window suggestions in {len(timings)} windows")
    # Never overwrite reviews made concurrently in another labeler/terminal.
    current_sidecar = suggestion_path(manifest).read_bytes() if suggestion_path(manifest).exists() else None
    if manifest.read_bytes() != original or current_sidecar != original_sidecar:
        raise ValueError("Labels or suggestions changed during analysis. Close the labeler and rerun; nothing overwritten.")
    bundle["last_run"] = report
    save_dataset(suggestion_path(manifest), bundle)
    return suggestion_path(manifest), report
