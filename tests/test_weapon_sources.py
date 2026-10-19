"""Curated manifests/cache acquisition; all media is synthetic or fake downloads."""

import csv
import hashlib
import json
import sys
import types
import wave
from pathlib import Path

import pytest

from arc_audio import weapon_sources
from arc_audio.weapon_sources import (
    SourceFetchError,
    fetch_source,
    parse_timestamp,
    read_manifest,
    source_key,
)


def manifest(path, rows):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["weapon", "source_title", "source_url", "start", "end",
                                                   "quality", "notes", "extra"])
        writer.writeheader()
        for row in rows:
            values = dict(weapon="Ferro", source_title="A synthetic source", source_url="sound.wav",
                          start="", end="", quality="", notes="", extra="")
            values.update(row)
            writer.writerow(values)
    return path


def audio(path, samples=100):
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        output.writeframes(b"\x01\x00" * samples)
    return path


def test_unverified_discovery_csv_is_not_implicitly_curated(tmp_path):
    path = tmp_path / "discovered.csv"
    path.write_text("weapon,source_title,source_url,start,end,quality,notes,verification\n"
                    "Anvil,Unverified,sound.wav,,,,,unverified\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unverified"):
        read_manifest(path)


@pytest.mark.parametrize(("value", "expected"), [(None, None), ("", None), ("  ", None), ("0", 0),
                                                ("01:02.5", 62.5), ("1:02:03.25", 3723.25),
                                                ("95:10", 5710), (1.25, 1.25)])
def test_timestamps(value, expected):
    assert parse_timestamp(value) == expected


@pytest.mark.parametrize("value", [True, "nan", "inf", "-1", "1:-1", "1:60", "1:60:00", "1.5:20",
                                         "1:2:3:4", "1::2", ":20", "text", "1e309"])
def test_invalid_timestamps(value):
    with pytest.raises(ValueError):
        parse_timestamp(value)


def test_youtube_canonical_identity():
    identity = "MlK0E83JfEE"
    assert source_key("https://www.youtube.com/watch?v=" + identity + "&t=30") == identity
    assert source_key("https://youtu.be/" + identity + "?t=90") == identity
    assert source_key("https://m.youtube.com/shorts/" + identity) == identity
    assert source_key("https://www.youtube-nocookie.com/embed/" + identity) == identity


def test_manifest_retains_fields_relative_paths_and_stable_ids(tmp_path):
    rows = [{"weapon": "Ferro", "notes": "  untouched notes  ", "extra": "future field", "start": "00:01"},
            {"weapon": "Stitcher", "start": "00:05", "end": "00:08"}]
    path = manifest(tmp_path / "sources.csv", rows)
    before = path.read_bytes()
    first = read_manifest(path)
    assert path.read_bytes() == before and read_manifest(path) == first
    assert first[0]["metadata"]["notes"] == "  untouched notes  "
    assert first[0]["metadata"]["extra"] == "future field"
    assert first[0]["local_path"] == str((tmp_path / "sound.wav").resolve())
    manifest(path, rows[::-1])
    assert {row["id"] for row in first} == {row["id"] for row in read_manifest(path)}
    assert first[0]["weapon"] == "ferro" and first[0]["start"] == 1


def test_multigun_showcase_without_timestamps_is_ambiguous(tmp_path):
    url = "https://www.youtube.com/watch?v=MlK0E83JfEE"
    rows = read_manifest(manifest(tmp_path / "sources.csv", [
        {"weapon": "Ferro", "source_url": url}, {"weapon": "Stitcher", "source_url": url},
        {"weapon": "Anvil", "source_url": url}]))
    assert len({row["source_id"] for row in rows}) == 1
    assert all(row["ambiguous_weapon"] for row in rows)
    assert all(row["possible_weapons"] == ["anvil", "ferro", "stitcher"] for row in rows)


def test_disjoint_ranges_and_overlapping_ranges(tmp_path):
    rows = read_manifest(manifest(tmp_path / "sources.csv", [
        {"weapon": "Ferro", "start": "0", "end": "5"},
        {"weapon": "Stitcher", "start": "5", "end": "10"},
        {"weapon": "Anvil", "start": "9", "end": "12"}]))
    assert rows[0]["ambiguous_weapon"] is False
    assert rows[1]["possible_weapons"] == ["anvil", "stitcher"]
    assert rows[2]["possible_weapons"] == ["anvil", "stitcher"]


def test_exact_duplicate_manifest_rows_not_imported_twice(tmp_path):
    rows = read_manifest(manifest(tmp_path / "sources.csv", [
        {"notes": "first"}, {"notes": "second"}]))
    assert len(rows) == 1
    assert rows[0]["duplicate_manifest_metadata"][0]["notes"] == "second"


@pytest.mark.parametrize("row", [{"end": "0"}, {"start": "5", "end": "2"}, {"weapon": "   "},
                                  {"source_url": ""}, {"start": "nan"}])
def test_manifest_rejects_invalid_rows(tmp_path, row):
    with pytest.raises(ValueError, match="Manifest row 2"):
        read_manifest(manifest(tmp_path / "sources.csv", [row]))


def test_local_acquisition_cached_and_changed_media_preserved(tmp_path):
    original = audio(tmp_path / "sound.wav")
    row = read_manifest(manifest(tmp_path / "sources.csv", [{}]))[0]
    first = fetch_source(row, tmp_path / "cache")
    assert first["cache_hit"] is False
    assert first["path"].read_bytes() == original.read_bytes()
    assert first["path"] != original
    assert first["metadata"]["original_path"] == str(original.resolve())
    assert json.loads(json.dumps(first["metadata"])) == first["metadata"]
    second = fetch_source(row, tmp_path / "cache")
    assert second["cache_hit"] is True and second["path"] == first["path"]
    audio(original, samples=200)
    third = fetch_source(row, tmp_path / "cache")
    assert third["cache_hit"] is False and third["path"] != first["path"]
    assert first["path"].exists() and third["path"].exists()
    assert not list((tmp_path / "cache").rglob("*.tmp"))


def test_same_basename_different_sources_cannot_collide(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir()
    two.mkdir()
    audio(one / "sound.wav")
    audio(two / "sound.wav", 150)
    first = read_manifest(manifest(one / "sources.csv", [{}]))[0]
    second = read_manifest(manifest(two / "sources.csv", [{}]))[0]
    assert fetch_source(first, tmp_path / "cache")["path"] != fetch_source(second, tmp_path / "cache")["path"]


def test_corrupted_cache_is_not_silently_used(tmp_path):
    original = audio(tmp_path / "sound.wav")
    row = read_manifest(manifest(tmp_path / "sources.csv", [{}]))[0]
    fetched = fetch_source(row, tmp_path / "cache")
    fetched["path"].write_bytes(b"corrupt")
    with pytest.raises(SourceFetchError, match="fingerprint"):
        fetch_source(row, tmp_path / "cache")
    assert original.read_bytes() != b"corrupt"


def test_bad_cache_media_path_rejected(tmp_path):
    audio(tmp_path / "sound.wav")
    row = read_manifest(manifest(tmp_path / "sources.csv", [{}]))[0]
    fetched = fetch_source(row, tmp_path / "cache")
    metadata_path = fetched["path"].parent / "source.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["media_file"] = "../outside.wav"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(SourceFetchError, match="unsafe"):
        fetch_source(row, tmp_path / "cache")


def test_downloader_failure_is_clean_and_can_retry(tmp_path, monkeypatch):
    row = read_manifest(manifest(tmp_path / "sources.csv", [{
        "source_url": "https://youtu.be/MlK0E83JfEE"}]))[0]

    class BrokenDownloader:
        def __init__(self, options):
            assert options["format"] == "bestaudio" and options["socket_timeout"] == 20

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def extract_info(self, url, download):
            raise RuntimeError("private video")

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=BrokenDownloader))
    for _ in range(2):
        with pytest.raises(SourceFetchError, match="private video"):
            fetch_source(row, tmp_path / "cache")
    assert not list((tmp_path / "cache").rglob("source.json"))
    assert not list((tmp_path / "cache").rglob(".fetch-*"))


def test_downloader_metadata_and_cache_only_once(tmp_path, monkeypatch):
    row = read_manifest(manifest(tmp_path / "sources.csv", [{
        "source_url": "https://youtu.be/MlK0E83JfEE"}]))[0]
    calls = []

    class Downloader:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def extract_info(self, url, download):
            calls.append(url)
            path = self.options["outtmpl"].replace("%(ext)s", "wav")
            audio(Path(path))
            return {"id": "MlK0E83JfEE", "title": "Synthetic", "duration": 1,
                    "channel": "test-channel", "uploader": "test-user",
                    "chapters": [{"title": "Ferro", "start_time": 0, "end_time": 1}]}

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=Downloader))
    first = fetch_source(row, tmp_path / "cache")
    second = fetch_source(row, tmp_path / "cache")
    assert first["metadata"]["channel"] == "test-channel"
    assert first["metadata"]["chapters"][0]["title"] == "Ferro"
    assert second["cache_hit"] and len(calls) == 1
    assert first["path"] == second["path"]


@pytest.mark.parametrize("installed", [{"node": "C:/tools/node.exe"},
                                        {"deno": "C:/tools/deno.exe", "node": "C:/tools/node.exe"}, {}])
def test_downloader_enables_existing_js_runtimes_only(tmp_path, monkeypatch, installed):
    row = read_manifest(manifest(tmp_path / "sources.csv", [{
        "source_url": "https://youtu.be/MlK0E83JfEE"}]))[0]
    monkeypatch.setattr(weapon_sources.shutil, "which", installed.get)

    class Downloader:
        def __init__(self, options):
            assert options.get("js_runtimes", {}) == {
                name: {"path": location} for name, location in installed.items()}
            assert not options.get("remote_components")
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def extract_info(self, url, download):
            audio(Path(self.options["outtmpl"].replace("%(ext)s", "wav")))
            return {"id": "MlK0E83JfEE", "title": "Synthetic"}

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=Downloader))
    assert fetch_source(row, tmp_path / "cache")["path"].exists()


def test_unreadable_cache_reports_access_error_without_redownload(tmp_path, monkeypatch):
    audio(tmp_path / "sound.wav")
    row = read_manifest(manifest(tmp_path / "sources.csv", [{}]))[0]
    fetched = fetch_source(row, tmp_path / "cache")
    original_hash = weapon_sources._sha256

    def blocked_hash(path):
        if path == fetched["path"]:
            raise PermissionError("synthetic cache permissions")
        return original_hash(path)

    def forbidden_download(*args, **kwargs):
        pytest.fail("An unreadable cache must not trigger another download")

    monkeypatch.setattr(weapon_sources, "_sha256", blocked_hash)
    monkeypatch.setattr(weapon_sources, "_download", forbidden_download)
    with pytest.raises(SourceFetchError, match="access denied"):
        fetch_source(row, tmp_path / "cache")
    assert fetched["path"].exists()


def test_exclusive_copy_preserves_existing_media_and_removes_owned_partial(tmp_path, monkeypatch):
    incoming = audio(tmp_path / "source.wav")
    destination = tmp_path / "published.wav"
    digest = hashlib.sha256(incoming.read_bytes()).hexdigest()
    destination.write_bytes(b"existing canonical media")
    with pytest.raises(SourceFetchError, match="fingerprint"):
        weapon_sources._publish_media(incoming, destination, digest)
    assert destination.read_bytes() == b"existing canonical media"
    destination.unlink()

    def interrupted_copy(source, output, length):
        output.write(source.read(10))
        raise OSError("synthetic copy failure")

    monkeypatch.setattr(weapon_sources.shutil, "copyfileobj", interrupted_copy)
    with pytest.raises(OSError, match="synthetic copy failure"):
        weapon_sources._publish_media(incoming, destination, digest)
    assert not destination.exists() and incoming.exists()
