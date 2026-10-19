"""Curated source manifests and explicit, cached local/YouTube acquisition.

Source context is a proposal, never a human sound annotation. Media is kept once
per source, on its original timeline; candidate extraction applies row ranges.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .dataset import validate_weapon

MANIFEST_COLUMNS = {"weapon", "source_title", "source_url", "start", "end", "quality", "notes"}
_MEDIA_SUFFIXES = {".wav", ".flac", ".ogg", ".opus", ".mp3", ".aac", ".m4a",
                   ".webm", ".mkv", ".mp4", ".mov"}


class SourceFetchError(RuntimeError):
    """One source could not be acquired; batch callers should continue with others."""


def parse_timestamp(value: str | float | int | None) -> float | None:
    """Parse blank, seconds, MM:SS or HH:MM:SS; reject invalid/negative times."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise ValueError("Timestamp must be seconds, MM:SS or HH:MM:SS")
    pieces = str(value).strip().split(":")
    if not 1 <= len(pieces) <= 3 or any(not piece.strip() for piece in pieces):
        raise ValueError(f"Invalid timestamp: {value!r}")
    try:
        numbers = [float(piece) for piece in pieces]
    except ValueError as exc:
        raise ValueError(f"Invalid timestamp: {value!r}") from exc
    if any(not math.isfinite(number) or number < 0 for number in numbers):
        raise ValueError("Timestamps must be finite and nonnegative")
    if len(numbers) > 1:
        if any(number != int(number) for number in numbers[:-1]):
            raise ValueError("Only timestamp seconds may contain a decimal")
        if numbers[-1] >= 60 or (len(numbers) == 3 and numbers[1] >= 60):
            raise ValueError("Timestamp seconds and HH:MM minutes must be below 60")
    result = sum(number * 60 ** index for index, number in enumerate(reversed(numbers)))
    if not math.isfinite(result):
        raise ValueError("Timestamp exceeds the supported range")
    return result


def _local_path(value: str) -> Path | None:
    parsed = urlparse(value)
    if parsed.scheme == "file":
        if parsed.netloc and parsed.netloc.lower() != "localhost":
            return Path("//" + parsed.netloc + unquote(parsed.path))
        path = unquote(parsed.path)
        if os.name == "nt" and re.match(r"^/[A-Za-z]:", path):
            path = path[1:]
        return Path(path)
    if not parsed.scheme or re.match(r"^[A-Za-z]:[\\/]", value):
        return Path(value)
    return None


def source_key(url: str) -> str:
    """Canonical YouTube id or resolved local path, stable across URL timestamps."""
    local = _local_path(url)
    if local is not None:
        return os.path.normcase(str(local.resolve()))
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    video = None
    if host == "youtu.be":
        video = parsed.path.strip("/").split("/")[0]
    elif host in {"youtube.com", "youtube-nocookie.com"}:
        if parsed.path.rstrip("/") == "/watch":
            video = parse_qs(parsed.query).get("v", [None])[0]
        else:
            parts = parsed.path.strip("/").split("/")
            if len(parts) >= 2 and parts[0] in {"embed", "shorts", "live", "v"}:
                video = parts[1]
    if video and re.fullmatch(r"[A-Za-z0-9_-]{11}", video):
        return video
    if parsed.scheme not in {"https", "http"} or not parsed.hostname:
        raise ValueError("Source must be a local file or an HTTP(S) media/YouTube URL")
    # Other sites can be used explicitly through yt-dlp, but remain separate sources.
    return parsed._replace(fragment="", scheme=parsed.scheme.lower(), netloc=parsed.netloc.lower()).geturl()


def normalize_weapon(value: str) -> str:
    """Normalize a human/source weapon name without asserting acoustic identity."""
    weapon = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not weapon:
        raise ValueError("Every manifest row needs a weapon name")
    validate_weapon(weapon)
    return weapon


def read_manifest(csv_path: Path) -> list[dict]:
    """Read validated rows without changing the CSV or assigning human labels.

    Exact raw CSV cells are retained under ``metadata``. Reordering rows does not
    change identities. Multi-weapon overlapping ranges are explicitly ambiguous.
    """
    csv_path = Path(csv_path).resolve()
    result = []
    seen = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as incoming:
        reader = csv.DictReader(incoming)
        missing = MANIFEST_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Manifest is missing columns: " + ", ".join(sorted(missing)))
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Manifest has duplicate column names")
        for line, metadata in enumerate(reader, 2):
            if None in metadata or any(value is None for value in metadata.values()):
                raise ValueError(f"Manifest row {line}: wrong number of CSV cells")
            try:
                if metadata.get("verification", "").strip().lower() == "unverified":
                    raise ValueError("Discovered sources are unverified; check relevance/ranges and explicitly curate this CSV row first")
                weapon = normalize_weapon(metadata["weapon"])
                source_url = metadata["source_url"].strip()
                if not source_url:
                    raise ValueError("Source URL/file is blank")
                local = _local_path(source_url)
                local = ((csv_path.parent / local).resolve() if local is not None else None)
                identity = source_key(str(local) if local is not None else source_url)
                start, end = parse_timestamp(metadata["start"]), parse_timestamp(metadata["end"])
                if end is not None and end <= (start or 0):
                    raise ValueError("End must be greater than start")
            except ValueError as exc:
                raise ValueError(f"Manifest row {line}: {exc}") from exc
            digest = hashlib.sha256(json.dumps([identity, weapon, start, end],
                                              separators=(",", ":")).encode()).hexdigest()[:24]
            row = {"id": "manifest_" + digest, "source_id": identity, "source_url": source_url,
                   "weapon": weapon, "start": start, "end": end, "metadata": dict(metadata),
                   "manifest_path": str(csv_path), "possible_weapons": [weapon],
                   "ambiguous_weapon": False}
            if local is not None:
                row["local_path"] = str(local)
            if row["id"] in seen:
                # Keep alternate metadata instead of losing notes from duplicate rows.
                seen[row["id"]].setdefault("duplicate_manifest_metadata", []).append(dict(metadata))
                continue
            result.append(row)
            seen[row["id"]] = row
    by_source: dict[str, list[dict]] = {}
    for row in result:
        by_source.setdefault(row["source_id"], []).append(row)
    for rows in by_source.values():
        for row in rows:
            start, end = row["start"] or 0, row["end"] if row["end"] is not None else math.inf
            possible = {row["weapon"]}
            for other in rows:
                other_start = other["start"] or 0
                other_end = other["end"] if other["end"] is not None else math.inf
                if max(start, other_start) < min(end, other_end):
                    possible.add(other["weapon"])
            row["possible_weapons"] = sorted(possible)
            row["ambiguous_weapon"] = len(possible) > 1
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as incoming:
        for block in iter(lambda: incoming.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or path.is_symlink():
        raise SourceFetchError("Source cache must remain inside its explicit storage directory")
    return resolved


def _atomic_json(path: Path, document: dict) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=".source-", suffix=".json", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(document, output, indent=2, allow_nan=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _cached(folder: Path, identity: str, local: Path | None) -> dict | None:
    info_file = _inside(folder / "source.json", folder)
    if not info_file.exists():
        return None
    try:
        document = json.loads(info_file.read_text(encoding="utf-8"))
        if document.get("source_id") != identity:
            raise SourceFetchError("Source cache identity conflict")
        relative = Path(document["media_file"])
        if relative.is_absolute() or len(relative.parts) != 1:
            raise SourceFetchError("Source cache has an unsafe media path")
        path = _inside(folder / relative, folder)
        if (not path.is_file() or path.stat().st_size != document["media_size"]
                or _sha256(path) != document["media_sha256"]):
            return None
        if local is not None:
            if not local.is_file():
                raise SourceFetchError(f"Original local source no longer exists: {local}")
            if _sha256(local) != document.get("original_sha256"):
                return None
        return {"path": path, "metadata": document, "cache_hit": True}
    except PermissionError as exc:
        raise SourceFetchError(
            f"Cannot read cached source {folder}: access denied. Check its file permissions; "
            "the existing source was preserved and will not be downloaded again automatically"
        ) from exc
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _publish_media(incoming: Path, destination: Path, digest: str) -> None:
    """Exclusively copy into the cache, inheriting its directory permissions.

    Hard links preserve a temporary file's restrictive Windows permissions. An
    exclusive destination prevents replacement of existing media and gets the
    canonical directory's ACL instead. Metadata is published only after close.
    A concurrent reader may fail/retry while copying; it cannot use a partial
    copy as a valid fingerprinted cache entry.
    """
    try:
        output = destination.open("xb")
    except FileExistsError:
        if _sha256(destination) != digest:
            raise SourceFetchError("Existing source file conflicts with its content fingerprint") from None
        return
    try:
        with output, incoming.open("rb") as source:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        if _sha256(destination) != digest:
            raise SourceFetchError("Source changed during cache publication; retry acquisition")
    except BaseException:
        # Only the process that obtained exclusive creation owns this cleanup.
        destination.unlink(missing_ok=True)
        raise


def _download(source_url: str, temporary: Path, ffmpeg_location: str | Path | None) -> tuple[Path, dict]:
    try:
        import yt_dlp
    except ImportError as exc:
        raise SourceFetchError("Missing yt-dlp; install the collection extra: pip install -e '.[collection]'") from exc
    options = {"format": "bestaudio", "noplaylist": True, "outtmpl": str(temporary / "audio.%(ext)s"),
               "quiet": True, "no_warnings": False, "socket_timeout": 20, "retries": 2,
               "fragment_retries": 2, "extractor_retries": 2, "continuedl": True,
               "overwrites": False, "writethumbnail": False, "writeinfojson": False,
               "windowsfilenames": True}
    # yt-dlp enables only Deno by default. Also enable an already installed Node
    # explicitly; this does not install runtimes or fetch remote JS components.
    runtimes = {name: {"path": executable} for name in ("deno", "node")
                if (executable := shutil.which(name))}
    if runtimes:
        options["js_runtimes"] = runtimes
    if ffmpeg_location is not None:
        ffmpeg = Path(ffmpeg_location).expanduser().resolve()
        if not ffmpeg.exists():
            raise SourceFetchError(f"ffmpeg location does not exist: {ffmpeg}")
        options["ffmpeg_location"] = str(ffmpeg)
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(source_url, download=True)
            if not isinstance(info, dict) or info.get("_type") in {"playlist", "multi_video"}:
                raise SourceFetchError("Select an individual source video, not a playlist")
    except SourceFetchError:
        raise
    except Exception as exc:
        message = str(exc)
        if "ffmpeg" in message.lower():
            message += "; install ffmpeg or pass --ffmpeg-location with its executable/folder"
        raise SourceFetchError("Download failed: " + message) from exc
    media = [path for path in temporary.iterdir()
             if path.is_file() and path.suffix.lower() in _MEDIA_SUFFIXES]
    if len(media) != 1:
        raise SourceFetchError("Download did not produce one usable audio file; check yt-dlp/ffmpeg output")
    path = _inside(media[0], temporary.resolve())
    if path.stat().st_size == 0:
        raise SourceFetchError("Downloaded source audio is empty")
    # Copy only JSON-safe fields; the extractor's entire internal object can contain callables.
    metadata = {key: info.get(key) for key in ("id", "title", "webpage_url", "channel", "channel_id",
                                              "uploader", "uploader_id", "duration", "chapters")}
    metadata = json.loads(json.dumps(metadata, default=str, allow_nan=False))
    return path, metadata


def fetch_source(row: dict, storage: Path, *, ffmpeg_location: str | Path | None = None) -> dict:
    """Acquire a source once, returning ``path``, JSON-safe ``metadata``, cache_hit.

    Downloading happens only when this function is explicitly invoked. Source
    ranges do not create extra media copies. A changed local source creates a new
    content-addressed file, retaining any earlier canonical media/annotations.
    There is no cookie handling, authentication, restriction bypass or upload.
    """
    try:
        root = Path(storage).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        source_url = row["source_url"]
        local = Path(row["local_path"]).resolve() if row.get("local_path") else _local_path(source_url)
        local = local.resolve() if local is not None else None
        identity = source_key(str(local) if local is not None else source_url)
        if row.get("source_id", identity) != identity:
            raise SourceFetchError("Manifest source identity does not match its URL/file")
        folder = _inside(root / ("source_" + hashlib.sha256(identity.encode()).hexdigest()[:24]), root)
        folder.mkdir(exist_ok=True)
        cached = _cached(folder, identity, local)
        if cached:
            return cached
        with tempfile.TemporaryDirectory(prefix=".fetch-", dir=folder) as temporary_name:
            temporary = Path(temporary_name).resolve()
            if local is not None:
                if not local.is_file():
                    raise SourceFetchError(f"Local source does not exist: {local}")
                before = local.stat()
                if before.st_size <= 0:
                    raise SourceFetchError("Local source audio/media is empty")
                downloaded = temporary / ("audio" + local.suffix.lower())
                shutil.copyfile(local, downloaded)
                after = local.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise SourceFetchError("Local source changed while copying; retry when recording has finished")
                metadata = {"id": identity, "title": row.get("metadata", {}).get("source_title") or local.stem,
                            "channel": None, "uploader": None, "duration": None, "chapters": None}
            else:
                downloaded, metadata = _download(source_url, temporary, ffmpeg_location)
            downloaded = _inside(downloaded, temporary)
            digest = _sha256(downloaded)
            destination = _inside(folder / ("audio_" + digest[:24] + downloaded.suffix.lower()), folder)
            _publish_media(downloaded, destination, digest)
            metadata.update({"source_id": identity, "source_url": source_url,
                             "media_file": destination.name, "media_size": destination.stat().st_size,
                             "media_sha256": digest})
            if local is not None:
                metadata.update({"original_path": str(local), "original_sha256": digest,
                                 "original_size": before.st_size, "original_mtime_ns": before.st_mtime_ns})
            _atomic_json(_inside(folder / "source.json", folder), metadata)
            return {"path": destination, "metadata": metadata, "cache_hit": False}
    except SourceFetchError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SourceFetchError(f"Cannot acquire source: {exc}") from exc
