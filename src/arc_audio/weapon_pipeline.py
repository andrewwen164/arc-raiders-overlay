"""Curated public/local sources -> pending proposals in the existing dataset.

Acquisition is explicit. No generated proposal is a training label. Checkpoints
are per video, and existing human annotations/decisions are never regenerated.
"""

import copy
import hashlib
import json
import logging
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from .dataset import TARGET_CLASSES, load_dataset, media_path, read_media_audio, save_dataset
from .media_import import _probe
from .suggestions import file_sha256, load_suggestions, media_stamp, suggestion_path

LOGGER = logging.getLogger(__name__)


def _bytes(path: Path) -> bytes | None:
    return path.read_bytes() if path.exists() else None


@contextmanager
def collection_lock(manifest: Path | str) -> Iterator[None]:
    path = Path(manifest).with_suffix(".collection.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        stream = path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise ValueError(f"Collection already running ({path}). If it crashed, close collectors and remove that lock.") from exc
    try:
        with stream:
            stream.write("Collection in progress. Close the labeler before collecting.\n")
            yield
    finally:
        path.unlink(missing_ok=True)


def _checkpoint(manifest, document, proposals, revisions):
    paths = (manifest, suggestion_path(manifest))
    if tuple(_bytes(path) for path in paths) != revisions:
        raise ValueError("Dataset or suggestions changed during collection. Close the labeler and rerun; newer labels preserved.")
    # Dataset first: a failed sidecar write can be retried without losing labels.
    save_dataset(manifest, document)
    save_dataset(paths[1], proposals)
    return tuple(_bytes(path) for path in paths)


def build_weapon_dataset(csv_path: Path | str, manifest: Path | str, *, storage: Path | str | None = None,
                         dry_run: bool = False, max_sources: int | None = None, source_url: str | None = None,
                         pre: float = .35, post: float = 1., min_spacing: float = .07,
                         ffmpeg_location: Path | str | None = None, progress: Callable[[str], None] = print) -> dict:
    from .weapon_candidates import detect_candidates
    from .weapon_sources import fetch_source, read_manifest

    if (not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                for value in (pre, post, min_spacing)) or pre < 0 or post < 0 or min_spacing <= 0):
        raise ValueError("Use finite nonnegative --pre/--post and positive --min-spacing seconds")
    manifest = Path(manifest).resolve()
    rows = read_manifest(csv_path)
    weapon_vocabulary = {row["weapon"] for row in rows}
    if source_url:
        rows = [row for row in rows if row["source_url"] == source_url]
        if not rows:
            raise ValueError("No manifest rows match --source-url")
    sources = {}
    for row in rows:
        sources.setdefault(row["source_id"], []).append(row)
    if max_sources is not None:
        if max_sources < 1:
            raise ValueError("--max-sources must be positive")
        sources = dict(list(sources.items())[:max_sources])
    storage = Path(storage).resolve() if storage else manifest.parent / "sources"
    if not storage.is_relative_to(manifest.parent):
        raise ValueError("Source storage must be inside the dataset folder")
    report = {"sources": len(sources), "manifest_rows": sum(map(len, sources.values())),
              "downloaded": 0, "cached": 0, "failed": [], "skipped_test": [],
              "new_candidates": 0, "candidates": 0, "storage": str(storage), "dry_run": dry_run}
    if dry_run:
        report["rows"] = [row for group in sources.values() for row in group]
        return report
    with collection_lock(manifest):
        document = load_dataset(manifest) if manifest.exists() else {
            "version": 1, "classes": list(TARGET_CLASSES), "clips": [], "review_classes": ["gunshot"]}
        proposals = load_suggestions(manifest)
        revisions = (_bytes(manifest), _bytes(suggestion_path(manifest)))
        registered = {row["id"]: row for row in document.get("weapon_sources", [])}
        for group in sources.values():
            for row in group:
                old = registered.get(row["id"], {})
                registered[row["id"]] = dict(old, **copy.deepcopy(row))
        document["weapon_sources"] = list(registered.values())
        document["weapons"] = sorted(set(document.get("weapons", [])) | weapon_vocabulary)
        document["weapon_manifest"] = {"path": str(Path(csv_path).resolve()), "sha256": file_sha256(csv_path)}
        revisions = _checkpoint(manifest, document, proposals, revisions)
        for index, (source_id, group) in enumerate(sources.items(), 1):
            progress(f"[{index}/{len(sources)}] {group[0]['source_url']}")
            clips = [clip for clip in document["clips"] if clip.get("collection_source_id") == source_id]
            if any(clip["split"] == "test" for clip in clips):
                report["skipped_test"].append(source_id)
                progress("  Reserved test source: not downloaded or proposed.")
                continue
            original_clips = copy.deepcopy(document["clips"])
            original_proposals = copy.deepcopy(proposals)
            original_counts = {key: report[key] for key in ("downloaded", "cached", "new_candidates", "candidates")}
            try:
                acquired = fetch_source(group[0], storage, ffmpeg_location=ffmpeg_location)
                path = Path(acquired["path"]).resolve()
                if not path.is_relative_to(manifest.parent) or not path.is_file():
                    raise ValueError("Acquired media must be a file inside the dataset folder")
                stamp = media_stamp(path)
                digest = file_sha256(path)
                clip = next((clip for clip in clips if media_path(manifest, clip) == path), None)
                if clips and clip is None:
                    raise ValueError("Source media path changed; preserve existing annotations and import a separate source explicitly")
                if clip and (clip["size"] != stamp["size"] or clip.get("source_sha256", digest) != digest):
                    raise ValueError("Source media changed since collection; annotations were preserved, inspect source cache")
                metadata = acquired["metadata"]
                if clip is None:
                    clip = {"file": path.relative_to(manifest.parent).as_posix(), **_probe(path),
                            "size": stamp["size"], "group": "public-source:" + source_id,
                            "split": "train", "track_confirmed": False, "intervals": [], "reviewed": [],
                            "collection_source_id": source_id, "source_metadata": metadata, "source_sha256": digest}
                    document["clips"].append(clip)
                report["cached" if acquired.get("cache_hit") else "downloaded"] += 1
                entry = next((entry for entry in proposals["clips"] if entry["file"] == clip["file"]
                              and entry["audio_stream"] == clip["audio_stream"]), None)
                if entry and entry["media"] != stamp:
                    raise ValueError("Source media fingerprint changed; stale proposals were retained for inspection")
                if entry is None:
                    entry = {"file": clip["file"], "duration": clip["duration"], "audio_stream": clip["audio_stream"],
                             "media": stamp, "media_sha256": digest, "suggestions": [], "collection_runs": []}
                    proposals["clips"].append(entry)
                settings = {"detector": "heuristic_transient_v1", "pipeline_revision": 3,
                            "pre": pre, "post": post, "min_spacing": min_spacing}
                run_key = hashlib.sha256(json.dumps([digest, group, settings, clip["audio_stream"]],
                                                    sort_keys=True).encode()).hexdigest()
                if run_key not in entry.setdefault("collection_runs", []):
                    audio = read_media_audio(path, clip["audio_stream"], 16000,
                                             max_seconds=max(row["end"] or clip["duration"] for row in group))
                    ranges = {}
                    for row in group:
                        ranges.setdefault((row["start"] or 0., row["end"] or clip["duration"]), []).append(row)
                    for (start, end), contexts in ranges.items():
                        if not 0 <= start < end <= clip["duration"] + .05:
                            raise ValueError(f"Manifest range {start:g}..{end:g} exceeds source duration {clip['duration']:g}")
                        audio_end = min(end, clip["duration"], len(audio) / 16000)
                        if start >= audio_end:
                            raise ValueError("Manifest range contains no decoded audio; check source timestamps")
                        candidates = detect_candidates(audio, 16000, start=start, end=audio_end,
                                                       pre=pre, post=post, min_spacing=min_spacing)
                        burst_starts = {}
                        for candidate in candidates:
                            if candidate.get("burst_group_id"):
                                burst_starts.setdefault(candidate["burst_group_id"], candidate["source_timestamp"])
                        for candidate in candidates:
                            # A single onset stays one proposal even with overlapping manifest rows.
                            onset = candidate["source_timestamp"]
                            if candidate.get("burst_group_id"):
                                first_onset = burst_starts[candidate["burst_group_id"]]
                                burst_token = f"{source_id}:{digest}:{clip['audio_stream']}:{round(first_onset * 1000)}"
                                candidate["burst_group_id"] = "burst-" + hashlib.sha256(burst_token.encode()).hexdigest()[:20]
                            existing = next((row for row in entry["suggestions"]
                                             if row.get("collection_source_id") == source_id
                                             and row.get("proposal_kind") != "gunshot_absence"
                                             and abs(row["source_timestamp"] - onset) < min_spacing / 2), None)
                            active_contexts = [row for row in group
                                               if (row["start"] or 0) <= onset < (row["end"] or clip["duration"])]
                            weapons = sorted({row["weapon"] for row in active_contexts})
                            proposed_weapon = weapons[0] if len(weapons) == 1 else None
                            context = contexts[0]
                            token = f"{source_id}:{digest}:{clip['audio_stream']}:{round(onset * 100)}"
                            candidate.update(id=hashlib.sha256(token.encode()).hexdigest()[:32], status="pending",
                                             proposed_weapon=proposed_weapon, possible_weapons=weapons,
                                             collection_source_id=source_id, source_file=clip["file"],
                                             manifest_refs=[row["id"] for row in active_contexts],
                                             source_url=context["source_url"],
                                             source_title=context["metadata"].get("source_title", metadata.get("title", "")),
                                             manifest_quality=context["metadata"].get("quality", ""),
                                             manifest_notes=context["metadata"].get("notes", ""),
                                             duplicate_group_id=None, duplicate_similarity=None,
                                             provenance={**settings, "source_sha256": digest,
                                                         "notice": "Heuristic transient; weapon proposed from source context only"})
                            if existing is None:
                                entry["suggestions"].append(candidate)
                                report["new_candidates"] += 1
                            elif existing["status"] == "pending":
                                candidate["id"] = existing["id"]
                                existing.update(candidate)
                    entry["collection_runs"].append(run_key)
                if media_stamp(path) != stamp:
                    raise ValueError("Source changed during detection; close the recording/downloader and rerun")
                entry["suggestions"].sort(key=lambda row: row["start"])
                report["candidates"] += len(entry["suggestions"])
                for row in group:
                    registered[row["id"]].update(status="downloaded", file=clip["file"], source_metadata=metadata)
                    registered[row["id"]].pop("error", None)
                progress(f"  {len(entry['suggestions'])} pending/reviewed candidates; source retained at {clip['file']}")
            except Exception as exc:
                # Acquisition/decoding errors isolate one video. Checkpoint conflicts below abort the run.
                error = {"source_id": source_id, "url": group[0]["source_url"], "error": str(exc)}
                document["clips"] = original_clips
                proposals = original_proposals
                report.update(original_counts)
                report["failed"].append(error)
                LOGGER.warning("source_failed %s", json.dumps(error))
                progress(f"  FAILED: {exc}")
                for row in group:
                    registered[row["id"]].update(status="failed", error=str(exc))
            revisions = _checkpoint(manifest, document, proposals, revisions)
        # Similarity is optional review metadata. It must not undo usable source
        # acquisition or turn proposals into labels when scoring fails.
        selected_entries = [entry for entry in proposals["clips"] if any(
            clip.get("collection_source_id") and clip["split"] != "test"
            and clip["file"] == entry["file"] and clip["audio_stream"] == entry["audio_stream"]
            for clip in document["clips"])]
        original_duplicate_fields = [(row, row.get("duplicate_group_id"), row.get("duplicate_similarity"))
                                     for entry in selected_entries for row in entry["suggestions"]]
        try:
            from .weapon_dedupe import annotate_duplicates
            report["deduplication"] = annotate_duplicates(selected_entries)
        except Exception as exc:
            for row, group_id, similarity in original_duplicate_fields:
                row.update(duplicate_group_id=group_id, duplicate_similarity=similarity)
            report["deduplication"] = {"error": str(exc), "notice": "Candidates retained; similarity scoring skipped"}
            LOGGER.warning("dedupe_failed %s", json.dumps(report["deduplication"]))
        document["weapon_collection_report"] = report
        _checkpoint(manifest, document, proposals, revisions)
    return report


def weapon_dataset_status(manifest: Path | str) -> dict:
    from .presence_review import presence_summary

    document = load_dataset(manifest)
    proposals = load_suggestions(manifest)
    sources = document.get("weapon_sources", [])
    discoveries = document.get("discovered_weapon_sources", [])
    report = {}
    human_weapons = {row["weapon"] for clip in document["clips"] for row in clip.get("intervals", []) if row.get("weapon")}
    for weapon in sorted(set(document.get("weapons", [])) | {row["weapon"] for row in sources}
                         | {row["weapon"] for row in discoveries} | human_weapons):
        selected = [row for row in sources if row["weapon"] == weapon]
        videos = {row["source_id"] for row in selected}
        unverified = {row["id"] for row in discoveries if row["weapon"] == weapon} - videos
        downloaded = {row["source_id"] for row in selected if row.get("status") == "downloaded"}
        channels = {row.get("source_metadata", {}).get("channel_id")
                    or row.get("source_metadata", {}).get("channel")
                    or row.get("source_metadata", {}).get("uploader") for row in selected}
        channels.discard(None)
        channels.discard("")
        candidates = [row for entry in proposals["clips"] for row in entry["suggestions"]
                      if (row.get("last_edit", {}).get("weapon") if row["status"] in ("accepted", "corrected")
                          else row.get("proposed_weapon")) == weapon]
        counts = {status: sum(row["status"] == status for row in candidates)
                  for status in ("pending", "accepted", "rejected", "corrected")}
        accepted = [interval for clip in document["clips"] for interval in clip.get("intervals", [])
                    if interval.get("weapon") == weapon and interval["label"] == "gunshot"]
        accepted_by_video = {}
        accepted_by_channel = {}
        accepted_channels = set()
        for clip in document["clips"]:
            count = sum(interval.get("weapon") == weapon for interval in clip.get("intervals", []))
            if count:
                key = clip.get("collection_source_id", clip["group"])
                accepted_by_video[key] = accepted_by_video.get(key, 0) + count
                meta = clip.get("source_metadata", {})
                channel = meta.get("channel_id") or meta.get("channel") or meta.get("uploader")
                if channel:
                    accepted_channels.add(channel)
                    accepted_by_channel[channel] = accepted_by_channel.get(channel, 0) + count
        warnings = []
        if len(accepted_by_video) < 3:
            warnings.append("Fewer than three reviewed source videos; source diversity is insufficient")
        if accepted and max(accepted_by_video.values(), default=0) / len(accepted) > .7:
            warnings.append("More than 70% of accepted shots come from one source video")
        if len(accepted_channels) < 2:
            warnings.append("Fewer than two known channels supply accepted shots; channel diversity unestablished")
        if accepted and max(accepted_by_channel.values(), default=0) / len(accepted) > .7:
            warnings.append("More than 70% of accepted shots come from one known channel")
        report[weapon] = {"discovered_sources": len(videos | unverified), "unverified_sources": len(unverified),
                          "downloaded_sources": len(downloaded),
                          "failed_sources": len({row["source_id"] for row in selected if row.get("status") == "failed"}),
                          **counts, "accepted_annotations": len(accepted),
                          "duplicate_groups": len({row["duplicate_group_id"] for row in candidates if row.get("duplicate_group_id")}),
                          "unique_videos": len(videos), "unique_channels": len(channels),
                          "accepted_unique_videos": len(accepted_by_video),
                          "accepted_unique_channels": len(accepted_channels), "warnings": warnings}
    ambiguous = sum(not row.get("proposed_weapon") for entry in proposals["clips"]
                    for row in entry["suggestions"] if row.get("collection_source_id")
                    and row.get("proposal_kind") != "gunshot_absence")
    background = {status: sum(row["status"] == status for entry in proposals["clips"] for row in entry["suggestions"]
                             if row.get("proposal_kind") == "gunshot_absence")
                  for status in ("pending", "accepted", "rejected")}
    return {"weapons": report, "ambiguous_candidates": ambiguous,
            "background_review": background,
            "quick_review_windows": presence_summary(document),
            "notice": "Heuristic proposals and source-context weapon names are not verified recognition. No classifier trained."}


def register_discoveries(manifest: Path | str, discovery: dict) -> None:
    """Explicit metadata registration only; these rows are never ingestion inputs."""
    manifest = Path(manifest).resolve()
    with collection_lock(manifest):
        original = manifest.read_bytes()
        document = load_dataset(manifest)
        known = {(row["weapon"], row["id"]): row for row in document.get("discovered_weapon_sources", [])}
        for source in discovery["sources"]:
            row = dict(source, verification="unverified")
            known[(row["weapon"], row["id"])] = row
        document["discovered_weapon_sources"] = list(known.values())
        if manifest.read_bytes() != original:
            raise ValueError("Dataset changed while registering discovery metadata; nothing overwritten")
        save_dataset(manifest, document)
