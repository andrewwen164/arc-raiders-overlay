"""Flag possible acoustic duplicates without deleting or accepting proposals.

Centered log-mel cosine similarity is only a cheap review aid. Similar gunshots
can be separate real events; these groups never establish a copied recording,
weapon identity, recording independence or training/evaluation split safety.
"""

import hashlib
import json
import math
from numbers import Real

import numpy as np
from scipy.spatial import cKDTree

NEIGHBORS = 64


def _vector(row):
    features = row.get("features")
    if not isinstance(features, dict):
        return None
    values = features.get("log_mel_summary")
    if not isinstance(values, (list, tuple, np.ndarray)):
        return None
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if vector.shape != (16,) or not np.isfinite(vector).all():
        return None
    # Extreme malformed values should not overflow centering or normalization.
    scale = float(np.max(np.abs(vector)))
    if scale == 0:
        return None
    vector = vector / scale
    vector -= vector.mean()
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return None
    return vector / norm


def _timestamp(row):
    value = row.get("source_timestamp", row.get("start"))
    return (float(value) if isinstance(value, Real) and not isinstance(value, bool)
            and math.isfinite(value) else None)


def annotate_duplicates(entries: list[dict], threshold: float = .995) -> dict:
    """Mutate only duplicate annotations on rows with usable 16-band features.

    Rows from the same file/track must describe the same onset (within 20 ms).
    Separate shots in a burst are preserved, even when spectra are identical.
    Across files/tracks, high similarity is a possible-duplicate flag only.
    At most 64 nearest neighbors per row are examined, so the search can miss
    pairs in dense clusters. Tree storage is linear; queries use fixed batches.
    All IDs, decisions, human edits, ordering and every candidate are retained.
    """
    if (not isinstance(threshold, Real) or isinstance(threshold, bool)
            or not math.isfinite(threshold) or not 0 < threshold <= 1):
        raise ValueError("duplicate threshold must be finite and in (0, 1]")
    records, skipped = [], 0
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("suggestions"), list):
            continue
        origin = (str(entry.get("file", "")), str(entry.get("audio_stream", "")))
        for row in entry["suggestions"]:
            if not isinstance(row, dict):
                skipped += 1
                continue
            vector = _vector(row)
            if vector is None:
                skipped += 1
                continue
            timestamp = _timestamp(row)
            identity = json.dumps([*origin, str(row.get("id", "")), timestamp], separators=(",", ":"))
            records.append((identity, origin, timestamp, row, vector))
    # Stable ordering also makes nearest-neighbor tie behavior reproducible when
    # a caller changes the order of wrappers/candidates without changing IDs.
    records.sort(key=lambda record: record[0])
    report = {"algorithm": "centered_log_mel_cosine_knn", "threshold": float(threshold),
              "neighbors": NEIGHBORS, "eligible_candidates": len(records), "skipped_candidates": skipped,
              "comparisons": 0, "duplicate_groups": 0, "duplicate_candidates": 0,
              "possible_duplicates_only": True}
    for _, _, _, row, _ in records:
        row["duplicate_group_id"] = None
        row["duplicate_similarity"] = None
    if len(records) < 2:
        return report
    vectors = np.stack([record[4] for record in records])
    tree = cKDTree(vectors)
    parents = np.arange(len(records))
    similarities = np.full(len(records), -1.0)
    origins = [{record[1]: (record[2], record[2])} for record in records]

    def owner(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = int(parents[index])
        return index

    radius = math.sqrt(max(0, 2 - 2 * threshold)) + 1e-10
    for first in range(0, len(records), 256):
        _, neighbors = tree.query(vectors[first:first + 256], k=min(NEIGHBORS, len(records)),
                                  distance_upper_bound=radius, workers=1)
        for local, indices in enumerate(neighbors):
            i = first + local
            for raw in np.atleast_1d(indices):
                j = int(raw)
                if j == i or j >= len(records):
                    continue
                report["comparisons"] += 1
                if records[i][1] == records[j][1]:
                    a, b = records[i][2], records[j][2]
                    if a is None or b is None or abs(a - b) > .020000001:
                        continue
                similarity = float(np.clip(vectors[i] @ vectors[j], -1, 1))
                if similarity < threshold - 1e-12:
                    continue
                a, b = owner(i), owner(j)
                if a != b:
                    large, small = (origins[a], origins[b]) if len(origins[a]) >= len(origins[b]) else (
                        origins[b], origins[a])
                    conflict = False
                    for origin, (low, high) in small.items():
                        if origin in large:
                            other_low, other_high = large[origin]
                            if (low is None or other_low is None
                                    or max(high, other_high) - min(low, other_low) > .020000001):
                                conflict = True
                                break
                    if conflict:
                        continue  # A cross-file bridge cannot merge distinct shots in one recording.
                    for origin, bounds in small.items():
                        if origin in large:
                            old = large[origin]
                            large[origin] = (min(old[0], bounds[0]), max(old[1], bounds[1]))
                        else:
                            large[origin] = bounds
                    root, child = min(a, b), max(a, b)
                    parents[child] = root
                    origins[root], origins[child] = large, {}
                similarities[i] = max(similarities[i], similarity)
                similarities[j] = max(similarities[j], similarity)
    groups = {}
    for i in range(len(records)):
        groups.setdefault(owner(i), []).append(i)
    for indices in groups.values():
        if len(indices) < 2:
            continue
        identities = [records[i][0] for i in indices]
        digest = hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest()[:16]
        for i in indices:
            records[i][3]["duplicate_group_id"] = f"possible-duplicate-{digest}"
            records[i][3]["duplicate_similarity"] = float(similarities[i])
        report["duplicate_groups"] += 1
        report["duplicate_candidates"] += len(indices)
    return report
