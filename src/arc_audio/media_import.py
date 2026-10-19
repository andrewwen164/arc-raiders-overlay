"""Import user-selected local media without replacing files or changing labels."""

import math
import os
import re
import shutil
import tempfile
from pathlib import Path

from .dataset import MEDIA_SUFFIXES, media_path, require_av


def _key(path):
    return os.path.normcase(str(Path(path).resolve()))


def _probe(path):
    av = require_av()
    with av.open(str(path)) as media:
        streams = [{"index": stream.index, "channels": stream.codec_context.layout.nb_channels,
                    "rate": stream.codec_context.sample_rate, "title": stream.metadata.get("title", ""),
                    "codec": stream.codec_context.name,
                    "start": float(stream.start_time * stream.time_base)
                    if stream.start_time is not None else 0}
                   for stream in media.streams.audio]
        if not streams:
            raise ValueError("No audio tracks found; select a recording containing audio")
        duration = float(media.duration / av.time_base) if media.duration else 0
        if duration <= 0:
            duration = max((float(stream.duration * stream.time_base)
                            for stream in media.streams.audio if stream.duration), default=0)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Cannot determine a positive recording duration")
    return {"duration": duration, "audio_streams": streams, "audio_stream": streams[0]["index"]}


def _safe_name(path):
    # Windows reserved names/characters must also be avoided when imported on Linux.
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", path.stem).strip(" .")[:100] or "recording"
    if re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])", stem.split(".")[0]):
        stem = "recording_" + stem
    return stem, path.suffix.lower()


def _copy_and_probe(source, folder):
    """Publish a fully copied/probed file atomically, refusing any existing name."""
    folder.mkdir(exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".import-", suffix=source.suffix, dir=folder)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as incoming:
            shutil.copyfileobj(incoming, output, length=1024 * 1024)
        metadata = _probe(temporary)
        size = temporary.stat().st_size
        stem, suffix = _safe_name(source)
        for index in range(1, 10001):
            destination = folder / f"{stem}{'' if index == 1 else '_' + str(index)}{suffix}"
            try:
                # Unlike rename on POSIX, link never overwrites an existing file.
                # Both paths are on the same volume; Windows NTFS supports this.
                os.link(temporary, destination)
                return destination, metadata, size
            except FileExistsError:
                continue
        raise ValueError("Too many files share this name; rename the selected recording")
    finally:
        temporary.unlink(missing_ok=True)


def import_media(manifest, document, paths, group):
    """Return added clip rows, skipped messages and errors; never mutate/save document.

    External selections are copied to ``imported/``. In-dataset selections are
    referenced in place. New rows have no labels or reviewed coverage and require
    human audio-track confirmation. A repeated external source is skipped only
    when its original path, byte size and modification timestamp still match.
    """
    result = {"added": [], "skipped": [], "errors": []}
    if (not isinstance(group, str) or not group.strip() or len(group.strip()) > 128
            or any(ord(character) < 32 for character in group)):
        result["errors"].append("Enter a session group (1–128 characters, without control characters)")
        return result
    group = group.strip()
    root = Path(manifest).resolve().parent
    imported = root / "imported"
    try:
        if not root.is_dir():
            raise ValueError("Dataset folder does not exist")
        if not imported.resolve().is_relative_to(root):
            raise ValueError("The imported folder must stay inside the dataset folder")
        known = {_key(media_path(manifest, clip)) for clip in document["clips"]}
        group_splits = {clip["split"] for clip in document["clips"] if clip["group"] == group}
        if len(group_splits) > 1 or not group_splits.issubset({"train", "validation", "test"}):
            raise ValueError("Existing session group crosses dataset splits; fix that conflict first")
        split = next(iter(group_splits), "train")
    except (KeyError, TypeError, ValueError, OSError) as exc:
        result["errors"].append(str(exc))
        return result

    sources_seen = set()
    for selection in paths:
        destination = None
        try:
            source = Path(selection).resolve(strict=True)
            source_key = _key(source)
            if not source.is_file() or source.suffix.lower() not in MEDIA_SUFFIXES:
                raise ValueError("Select an MP4, MKV, MOV or WAV recording")
            stat = source.stat()
            if source_key in known or source_key in sources_seen:
                result["skipped"].append(f"{source.name}: already in this dataset/import")
                continue
            previous = [clip for clip in document["clips"]
                        if clip.get("imported_from") and _key(clip["imported_from"]) == source_key]
            if previous:
                unchanged = any(clip.get("imported_source_size") == stat.st_size
                                and clip.get("imported_source_mtime_ns") == stat.st_mtime_ns
                                for clip in previous)
                if unchanged:
                    result["skipped"].append(f"{source.name}: unchanged source already imported")
                    continue
                raise ValueError("Source changed since its last import; rename it to import a distinct recording")
            external = not source.is_relative_to(root)
            if external:
                destination, metadata, size = _copy_and_probe(source, imported)
                after = source.stat()
                if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                    raise ValueError("Recording changed during import; finish recording and try again")
                path = destination
            else:
                path, metadata, size = source, _probe(source), stat.st_size
            clip = {"file": path.relative_to(root).as_posix(), **metadata, "size": size,
                    "group": group, "split": split, "track_confirmed": False,
                    "reviewed": [], "intervals": []}
            if external:
                clip.update(imported_from=str(source), imported_source_size=stat.st_size,
                            imported_source_mtime_ns=stat.st_mtime_ns)
            result["added"].append(clip)
            sources_seen.add(source_key)
            known.add(_key(path))
        except Exception as exc:
            # A bad selection should not prevent importing the remaining files.
            # Remove only the new, exclusively-created copy from this attempt.
            if destination is not None:
                destination.unlink(missing_ok=True)
            result["errors"].append(f"{selection}: {exc}")
    return result
