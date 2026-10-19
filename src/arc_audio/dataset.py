"""Local media inventory and human labels. Unreviewed audio is never negative data."""

import hashlib
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np

CORE_CLASSES = ("footsteps", "reload", "gunshot")
TARGET_CLASSES = CORE_CLASSES + ("landing", "snap_hook", "zipline", "climb_vault", "healing")
CLASS_NAMES = {"footsteps": "Footsteps", "reload": "Reload", "gunshot": "Gunfire",
               "landing": "Landing / fall impact", "snap_hook": "Snap Hook use", "zipline": "Zip-line use",
               "climb_vault": "Climb / vault", "healing": "Healing / consumables"}
MEDIA_SUFFIXES = {".mp4", ".mkv", ".mov", ".wav"}


def validate_weapon(value):
    """Optional human weapon identity; event taxonomy remains unchanged."""
    if value is not None and (not isinstance(value, str) or not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", value)
                              or len(value) > 80):
        raise ValueError("Weapon must be an optional lowercase identifier such as anvil or il_toro")


def require_av():
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("Install media tools: uv pip install --python .venv/Scripts/python.exe 'av>=16,<20'") from exc
    return av


def save_dataset(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="." + path.stem + "-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(document, indent=2))
        # Indexers/antivirus can briefly hold a Windows destination open. A
        # bounded retry retains atomic replacement and never truncates labels.
        for attempt in range(6):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(.04 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def media_path(manifest, clip):
    root = Path(manifest).resolve().parent
    path = (root / clip["file"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Dataset media must be inside the dataset folder")
    return path


def validate_span(row, duration):
    if not isinstance(row, dict):
        raise ValueError("Intervals must be objects with start/end seconds")
    for field in ("start", "end"):
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Interval timestamps must be finite numbers")
    if not 0 <= row["start"] < row["end"] <= duration + 0.05:
        raise ValueError("Intervals require 0 <= start < end <= clip duration")


def load_dataset(path):
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("version") != 1 or not isinstance(doc.get("clips"), list):
        raise ValueError("Dataset requires version 1 and a clips array")
    if "weapons" in doc:
        weapons = doc["weapons"]
        if not isinstance(weapons, list) or any(not isinstance(weapon, str) for weapon in weapons):
            raise ValueError("Dataset weapons must be a list of lowercase identifiers")
        if len(set(weapons)) != len(weapons):
            raise ValueError("Dataset weapon vocabulary contains duplicates")
        for weapon in weapons:
            validate_weapon(weapon)
    seen, groups, collection_groups = set(), {}, {}
    for clip in doc["clips"]:
        if not isinstance(clip, dict) or not isinstance(clip.get("file"), str):
            raise ValueError("Each clip requires a local media filename")
        resolved = media_path(path, clip)
        if resolved in seen:
            raise ValueError("Duplicate media entry in dataset")
        seen.add(resolved)
        duration = clip.get("duration")
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("Each clip requires a positive duration")
        if not isinstance(clip.get("track_confirmed", False), bool):
            raise ValueError("track_confirmed must be true or false")
        group = clip.get("group")
        split = clip.get("split")
        if not isinstance(group, str) or not group or split not in ("train", "validation", "test"):
            raise ValueError("Each clip needs a group and train/validation/test split")
        if group in groups and groups[group] != split:
            raise ValueError(f"Group {group} crosses splits; prevent recording/session leakage")
        groups[group] = split
        source_id = clip.get("collection_source_id")
        if source_id is not None:
            if not isinstance(source_id, str) or not source_id:
                raise ValueError("Collection source identity must be a nonempty string")
            if source_id in collection_groups and collection_groups[source_id] != (group, split):
                raise ValueError("A public source video crosses groups/splits; keep every variant together")
            collection_groups[source_id] = (group, split)
        streams = clip.get("audio_streams", [])
        if clip.get("audio_stream") not in [s["index"] for s in streams]:
            raise ValueError(f"Invalid audio stream for {clip['file']}")
        for row in clip.get("reviewed", []):
            validate_span(row, duration)
            classes = row.get("classes", list(CORE_CLASSES))
            if (not isinstance(classes, list) or not classes or len(set(classes)) != len(classes)
                    or not all(label in TARGET_CLASSES for label in classes)):
                raise ValueError("Reviewed ranges need a nonempty list of supported classes")
        for row in clip.get("intervals", []):
            validate_span(row, duration)
            validate_weapon(row.get("weapon"))
            if row.get("weapon") is not None and row.get("label") != "gunshot":
                raise ValueError("Weapon identity is supported only on gunshot annotations")
            if row.get("label") not in TARGET_CLASSES:
                raise ValueError(f"Unsupported sound label; allowed: {', '.join(TARGET_CLASSES)}")
            if row.get("source", "unknown") not in ("unknown", "self", "teammate", "other_raider"):
                raise ValueError("Invalid human source tag")
        from .presence_review import validate_presence_reviews
        validate_presence_reviews(clip)
    return doc


def prepare_dataset(folder):
    """Probe headers only; retain originals and existing human annotations."""
    av = require_av()
    folder = Path(folder).resolve()
    manifest = folder / "dataset.json"
    is_new = not manifest.exists()
    doc = load_dataset(manifest) if manifest.exists() else {"version": 1, "classes": list(TARGET_CLASSES), "clips": []}
    known = {clip["file"] for clip in doc["clips"]}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in MEDIA_SUFFIXES or path.name in known:
            continue
        with av.open(str(path)) as media:
            streams = [{"index": s.index, "channels": s.codec_context.layout.nb_channels,
                        "rate": s.codec_context.sample_rate, "title": s.metadata.get("title", ""),
                        "codec": s.codec_context.name,
                        "start": float(s.start_time * s.time_base) if s.start_time is not None else 0}
                       for s in media.streams.audio]
            if not streams:
                continue
            duration = float(media.duration / av.time_base) if media.duration else 0
            if duration <= 0:
                duration = max(float(s.duration * s.time_base) for s in media.streams.audio if s.duration)
        # All clips from one recording day stay together by default. Users should
        # merge groups further if a session spans days; filenames aren't labels.
        date = re.search(r"(20\d{6})\d{6}", path.stem)
        group = date.group(1) if date else path.stem
        split = next((c["split"] for c in doc["clips"] if c["group"] == group), "train")
        doc["clips"].append({"file": path.name, "duration": duration, "size": path.stat().st_size,
                             "group": group, "split": split, "audio_streams": streams,
                             "audio_stream": streams[0]["index"], "track_confirmed": False,
                             "reviewed": [], "intervals": []})
    if is_new:
        assign_group_splits(doc)
    doc["classes"] = list(TARGET_CLASSES)
    save_dataset(manifest, doc)
    return manifest, dataset_summary(doc)


def assign_group_splits(doc):
    """Deterministic group allocation with nonempty train/validation when possible."""
    groups = sorted({c["group"] for c in doc["clips"]},
                    key=lambda group: hashlib.sha256(group.encode()).hexdigest())
    assignments = dict.fromkeys(groups, "train")
    if len(groups) >= 2:
        holdout = max(1, round(len(groups) * .2))
        for group in groups[:holdout]:
            assignments[group] = "validation"
        if len(groups) >= 3:
            for group in groups[holdout:holdout * 2]:
                assignments[group] = "test"
    for clip in doc["clips"]:
        clip["split"] = assignments[clip["group"]]


def merge_spans(spans):
    merged = []
    for row in sorted(spans, key=lambda r: r["start"]):
        if merged and row["start"] <= merged[-1]["end"] + 1e-6:
            merged[-1]["end"] = max(merged[-1]["end"], row["end"])
        else:
            merged.append({"start": row["start"], "end": row["end"]})
    return merged


def reviewed_spans(clip, classes=CORE_CLASSES):
    """Coverage for a selected task; legacy reviews cover the original three only."""
    requested = set(classes)
    return merge_spans([row for row in clip.get("reviewed", [])
                        if requested.issubset(row.get("classes", CORE_CLASSES))])


def merge_reviewed(spans):
    """Merge only reviews with identical scope; preserve what was actually checked."""
    scopes, linked = {}, []
    for row in spans:
        scope = tuple(label for label in TARGET_CLASSES if label in row.get("classes", CORE_CLASSES))
        if row.get("proposal_id"):
            # Durable linkage makes undo safe even if the sidecar decision save
            # failed. Never merge it into unrelated manual coverage.
            linked.append(dict(row, classes=list(scope)))
            continue
        scopes.setdefault(scope, []).append(row)
    return [dict(row, classes=list(scope)) for scope, rows in scopes.items() for row in merge_spans(rows)] + linked


def dataset_summary(doc):
    from .presence_review import presence_summary
    return {"clips": len(doc["clips"]), "minutes": round(sum(c["duration"] for c in doc["clips"]) / 60, 2),
            "quick_review_windows": presence_summary(doc),
            "confirmed_tracks": sum(bool(c.get("track_confirmed")) for c in doc["clips"]),
            "reviewed_seconds": round(sum(r["end"] - r["start"] for c in doc["clips"]
                                          for r in merge_spans(c.get("reviewed", []))), 2),
            "intervals": {label: sum(r["label"] == label for c in doc["clips"] for r in c.get("intervals", []))
                          for label in TARGET_CLASSES},
            "clips_by_split": {s: sum(c["split"] == s for c in doc["clips"])
                               for s in ("train", "validation", "test")}}


def read_media_audio(path, stream_index, rate=16000, max_seconds=None):
    """Decode one selected track into RAM, preserving media-timeline timestamps.

    No audio extraction is written to disk. Timestamp gaps are filled with zeros;
    reviewed labels on a real gap must still be decided by the reviewer.
    """
    av = require_av()
    parts, ends, extent = [], [], 0
    limit = round(max_seconds * rate) if max_seconds is not None else None
    with av.open(str(path)) as media:
        stream = next((s for s in media.streams.audio if s.index == stream_index), None)
        if stream is None:
            raise ValueError("Selected audio track no longer exists")
        origin = float(media.start_time / av.time_base) if media.start_time else 0
        converter = av.AudioResampler(format="fltp", layout="mono", rate=rate)

        def append(frame):
            nonlocal extent
            values = frame.to_ndarray().reshape(-1).astype(np.float32)
            start = max(0, round((float(frame.pts * frame.time_base) - origin) * rate)) if frame.pts is not None else extent
            if limit is not None:
                values = values[:max(0, limit - start)]
            if len(values):
                parts.append(values)
                ends.append(start)
                extent = max(extent, start + len(values))

        for frame in media.decode(stream):
            for converted in converter.resample(frame):
                append(converted)
            if limit is not None and extent >= limit:
                break
        else:
            for converted in converter.resample(None):
                append(converted)
    if not parts:
        raise ValueError("Selected audio track decoded no samples")
    if extent > rate * 60 * 60:
        raise ValueError("Media timeline exceeds one hour; split the recording before training")
    signal = np.zeros(extent, np.float32)
    for start, values in zip(ends, parts):
        signal[start:start + len(values)] = values
    return signal
