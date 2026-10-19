"""Propose quiet-looking intervals for explicit human gunfire-absence review.

Distance from a transient proposal does not establish absence of gunfire. This
module never marks audio negative/reviewed and never changes human annotations.
"""

import hashlib
import json
import math
from numbers import Real
from pathlib import Path

from .dataset import load_dataset, media_path, save_dataset
from .suggestions import file_sha256, load_suggestions, media_stamp, suggestion_path
from .weapon_pipeline import collection_lock

NOTICE = ("Pending gunfire-absence proposals only. Unreviewed audio is not background; "
          "listen to each entire interval and explicitly confirm no gunfire before using it as a negative.")


def _bytes(path):
    return path.read_bytes() if path.exists() else None


def _merge(spans):
    result = []
    for start, end in sorted(spans):
        if result and start <= result[-1][1] + 1e-9:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def _ranges(rows, duration):
    result = []
    for row in rows:
        start = row.get("start")
        end = row.get("end")
        start = 0.0 if start is None else start
        end = duration if end is None else end
        if (not all(isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
                    for value in (start, end)) or not 0 <= start < end <= duration + .05):
            continue
        result.append((float(start), min(float(end), duration)))
    return _merge(result)


def _windows(ranges, excluded, seconds, anchors, budget):
    """Centered, disjoint tiling of each free gap; stable across review decisions."""
    for start, end in ranges:
        cursor = start
        gaps = []
        for a, b in excluded:
            if b <= cursor:
                continue
            if a >= end:
                break
            if a > cursor:
                gaps.append((cursor, min(a, end)))
            cursor = max(cursor, min(b, end))
        if cursor < end:
            gaps.append((cursor, end))
        for a, b in gaps:
            count = math.floor((b - a + 1e-9) / seconds)
            if count < 1:
                continue
            padding = max(0, (b - a - count * seconds) / 2)
            # Clearance inside one free gap is a concave tent (or monotonic
            # toward a boundary). Only the best nearby tiles can enter a
            # source's bounded top-N queue; do not enumerate hours of tiny tiles.
            left = max((last for _, last in anchors if last <= a), default=None)
            right = min((first for first, _ in anchors if first >= b), default=None)
            ideal = ((left + right) / 2 if left is not None and right is not None else
                     b if left is not None else a if right is not None else (a + b) / 2)
            nearest = max(0, min(count - 1, round((ideal - a - padding) / seconds - .5)))
            for i in range(max(0, nearest - budget), min(count, nearest + budget + 1)):
                first = a + padding + i * seconds
                yield first, min(b, first + seconds), start, end


def propose_background(manifest: Path | str, *, seconds: float = 2., margin: float = .75,
                       max_per_source: int = 8) -> dict:
    """Append pending absence-review proposals; never create negative truth.

    Existing collection proposals, including rejections, conservatively exclude
    their intervals/onsets plus ``margin``. Test clips, missing/stale media and
    clips without registered source ranges/current proposal fingerprints skip.
    Decisions and IDs survive reruns. No media decoding or raw exports occur.
    """
    if (not isinstance(seconds, Real) or isinstance(seconds, bool) or not math.isfinite(seconds)
            or seconds <= 0 or not isinstance(margin, Real) or isinstance(margin, bool)
            or not math.isfinite(margin) or margin < 0
            or not isinstance(max_per_source, int) or isinstance(max_per_source, bool) or max_per_source < 1):
        raise ValueError("Use positive finite seconds, nonnegative finite margin and positive integer max_per_source")
    manifest = Path(manifest).resolve()
    sidecar = suggestion_path(manifest)
    settings = {"algorithm": "proposal_gap_v1", "seconds": float(seconds), "margin": float(margin)}
    report = {"new_candidates": 0, "total": 0, "skipped": [], "notice": NOTICE}
    with collection_lock(manifest):
        revisions = (_bytes(manifest), _bytes(sidecar))
        document, bundle = load_dataset(manifest), load_suggestions(manifest)
        registered = {}
        for row in document.get("weapon_sources", []):
            if isinstance(row, dict) and isinstance(row.get("source_id"), str):
                registered.setdefault(row["source_id"], []).append(row)
        pools, stamps = {}, []
        for clip in document["clips"]:
            source_id = clip.get("collection_source_id")
            if not source_id:
                continue
            reason = None
            if clip["split"] == "test":
                reason = "reserved test source"
            elif source_id not in registered:
                reason = "no registered source ranges"
            entries = [entry for entry in bundle["clips"] if entry["file"] == clip["file"]
                       and entry["audio_stream"] == clip["audio_stream"]]
            if reason is None and len(entries) != 1:
                reason = "missing or ambiguous source proposal entry"
            if reason is not None:
                report["skipped"].append({"file": clip["file"], "reason": reason})
                continue
            entry = entries[0]
            path = media_path(manifest, clip)
            try:
                stamp = media_stamp(path)
                if stamp != entry["media"] or stamp["size"] != clip.get("size"):
                    raise ValueError("stale media stamp")
                digest = clip.get("source_sha256")
                if not digest or digest != entry.get("media_sha256") or digest != file_sha256(path):
                    raise ValueError("missing or stale media SHA-256")
            except (OSError, ValueError) as exc:
                report["skipped"].append({"file": clip["file"], "reason": str(exc)})
                continue
            ranges = _ranges(registered[source_id], clip["duration"])
            if not ranges:
                report["skipped"].append({"file": clip["file"], "reason": "no valid source ranges"})
                continue
            stamps.append((path, stamp))
            positive = [row for row in entry["suggestions"] if row["label"] == "gunshot"
                        and row.get("proposal_kind") != "gunshot_absence"]
            positive += [row for row in clip.get("intervals", []) if row["label"] == "gunshot"]
            excluded, anchors = [], []
            for row in positive:
                onset = row.get("source_timestamp", row["start"])
                first, last = min(row["start"], onset), max(row["end"], onset)
                excluded.append((max(0, first - margin), min(clip["duration"], last + margin)))
                anchors.append((first, last))
            excluded = _merge(excluded)
            contexts = registered[source_id]
            for first, last, range_start, range_end in _windows(
                    ranges, excluded, float(seconds), anchors, max_per_source):
                midpoint = (first + last) / 2
                distance = min((max(a - midpoint, midpoint - b, 0) for a, b in anchors),
                               default=min(midpoint - range_start, range_end - midpoint))
                context = next((row for row in contexts
                                if (row.get("start") or 0) <= midpoint
                                < (row.get("end") or clip["duration"])), contexts[0])
                token = [source_id, digest, clip["audio_stream"], round(first, 9), round(last, 9), settings]
                identity = hashlib.sha256(json.dumps(token, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:32]
                row = {"id": identity, "status": "pending", "start": first, "end": last,
                       "label": "gunshot", "proposal_kind": "gunshot_absence", "confidence": 0.,
                       "detection_score": 0., "quality_score": float(100 * distance / (distance + seconds)),
                       "source_timestamp": midpoint, "preview_start": first, "preview_end": last,
                       "proposed_weapon": None, "possible_weapons": [], "collection_source_id": source_id,
                       "source_file": clip["file"], "source_url": context.get("source_url", ""),
                       "source_title": context.get("metadata", {}).get("source_title", ""),
                       "duplicate_group_id": None, "duplicate_similarity": None,
                       "provenance": {**settings, "source_sha256": digest, "notice": NOTICE}}
                pools.setdefault(source_id, []).append((distance, clip["file"], first, entry, row))
        for pool in pools.values():
            pool.sort(key=lambda value: (-value[0], value[1], value[2]))
            for _, _, _, entry, row in pool[:max_per_source]:
                existing = next((old for old in entry["suggestions"] if old["id"] == row["id"]
                                 or (old.get("proposal_kind") == "gunshot_absence"
                                     and abs(old["start"] - row["start"]) < 1e-7
                                     and abs(old["end"] - row["end"]) < 1e-7)), None)
                if existing is None:
                    entry["suggestions"].append(row)
                    report["new_candidates"] += 1
        report["total"] = sum(row.get("proposal_kind") == "gunshot_absence"
                              for entry in bundle["clips"] for row in entry["suggestions"])
        if (_bytes(manifest), _bytes(sidecar)) != revisions:
            raise ValueError("Dataset or suggestions changed during background proposal generation; newer edits preserved")
        for path, stamp in stamps:
            if media_stamp(path) != stamp:
                raise ValueError("Media changed during background proposal generation; no proposals saved")
        if report["new_candidates"]:
            save_dataset(sidecar, bundle)
    return report
