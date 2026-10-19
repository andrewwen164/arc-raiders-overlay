"""Synthetic local-file import tests; no real recordings or manifests are touched."""

import copy
import wave

import pytest

from arc_audio.dataset import load_dataset, save_dataset
from arc_audio.media_import import import_media


def wav(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setparams((2, 2, 16000, 0, "NONE", "not compressed"))
        output.writeframes(b"\x00\x00\x00\x00" * 1600)
    return path


@pytest.fixture
def inventory(tmp_path):
    pytest.importorskip("av")
    folder = tmp_path / "dataset"
    folder.mkdir()
    return folder / "dataset.json", {"version": 1, "clips": []}


def test_external_copy_nonmutation_and_manifest_contract(tmp_path, inventory):
    manifest, doc = inventory
    save_dataset(manifest, doc)
    original_manifest = manifest.read_bytes()
    original_doc = copy.deepcopy(doc)
    source = wav(tmp_path / "outside" / "shot.wav")
    original_audio = source.read_bytes()
    result = import_media(manifest, doc, [source], "session-one")
    assert not result["errors"] and not result["skipped"]
    clip, = result["added"]
    assert clip["file"] == "imported/shot.wav"
    assert clip["group"] == "session-one" and clip["split"] == "train"
    assert clip["track_confirmed"] is False
    assert clip["reviewed"] == clip["intervals"] == []
    assert clip["duration"] == pytest.approx(.1)
    assert clip["audio_streams"][0]["channels"] == 2
    assert source.read_bytes() == original_audio
    assert (manifest.parent / clip["file"]).read_bytes() == original_audio
    assert doc == original_doc and manifest.read_bytes() == original_manifest
    save_dataset(manifest, dict(doc, clips=result["added"]))
    assert load_dataset(manifest)["clips"] == result["added"]


def test_existing_media_is_referenced_and_duplicates_skip(inventory):
    manifest, doc = inventory
    source = wav(manifest.parent / "custom.wav")
    result = import_media(manifest, doc, [source, source], "session-one")
    assert len(result["added"]) == len(result["skipped"]) == 1
    assert result["added"][0]["file"] == "custom.wav"
    assert not (manifest.parent / "imported").exists()
    doc["clips"].extend(result["added"])
    again = import_media(manifest, doc, [source], "session-two")
    assert not again["added"] and len(again["skipped"]) == 1


def test_colliding_names_preserve_files_and_session_split(tmp_path, inventory):
    manifest, doc = inventory
    old = wav(manifest.parent / "imported" / "clip.wav")
    doc["clips"] = import_media(manifest, doc, [old], "shared-session")["added"]
    doc["clips"][0]["split"] = "validation"
    doc["clips"][0]["intervals"] = [{"start": 0, "end": .05, "label": "gunshot"}]
    previous = copy.deepcopy(doc)
    sources = [wav(tmp_path / name / "clip.wav") for name in ("a", "b")]
    result = import_media(manifest, doc, sources, "shared-session")
    assert not result["errors"]
    assert [clip["file"] for clip in result["added"]] == ["imported/clip_2.wav", "imported/clip_3.wav"]
    assert all(clip["split"] == "validation" for clip in result["added"])
    assert doc == previous and old.exists()


def test_external_repeat_and_changed_source(tmp_path, inventory):
    manifest, doc = inventory
    source = wav(tmp_path / "outside.wav")
    doc["clips"].extend(import_media(manifest, doc, [source], "session")["added"])
    again = import_media(manifest, doc, [source], "session")
    assert not again["added"] and len(again["skipped"]) == 1
    with source.open("ab") as output:
        output.write(b"changed")
    changed = import_media(manifest, doc, [source], "session")
    assert not changed["added"] and "changed" in changed["errors"][0]
    assert len(list((manifest.parent / "imported").iterdir())) == 1


@pytest.mark.parametrize("group", ["", "   ", None, "bad\ngroup", "a" * 129])
def test_invalid_session_group_has_no_side_effects(tmp_path, inventory, group):
    manifest, doc = inventory
    source = wav(tmp_path / "outside.wav")
    result = import_media(manifest, doc, [source], group)
    assert result["errors"] and not result["added"]
    assert not list(manifest.parent.iterdir())


def test_invalid_media_cleans_temporary_copy_and_continues(tmp_path, inventory):
    manifest, doc = inventory
    invalid = tmp_path / "broken.mp4"
    invalid.write_bytes(b"not a media file")
    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("notes", encoding="utf-8")
    valid = wav(tmp_path / "valid.wav")
    result = import_media(manifest, doc, [invalid, unsupported, tmp_path / "missing.mp4", valid], "session")
    assert len(result["errors"]) == 3 and len(result["added"]) == 1
    assert [p.name for p in (manifest.parent / "imported").iterdir()] == ["valid.wav"]
    assert invalid.read_bytes() == b"not a media file"


def test_no_audio_media_is_rejected_and_cleaned(tmp_path, inventory, monkeypatch):
    from arc_audio import media_import

    manifest, doc = inventory
    source = tmp_path / "video.mp4"
    source.write_bytes(b"synthetic header stub")

    class Media:
        streams = type("Streams", (), {"audio": []})()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(media_import, "require_av", lambda: type("Av", (), {"open": lambda _: Media()}))
    result = import_media(manifest, doc, [source], "session")
    assert "No audio tracks" in result["errors"][0]
    assert not list((manifest.parent / "imported").iterdir())


def test_group_split_conflict_is_not_silently_reshuffled(inventory):
    manifest, doc = inventory
    for name in ("a.wav", "b.wav"):
        path = wav(manifest.parent / name)
        doc["clips"].extend(import_media(manifest, doc, [path], "session")["added"])
    doc["clips"][1]["split"] = "test"
    before = copy.deepcopy(doc)
    result = import_media(manifest, doc, [wav(manifest.parent / "c.wav")], "session")
    assert "crosses" in result["errors"][0] and not result["added"]
    assert doc == before


def test_escaping_dataset_media_path_is_rejected(tmp_path, inventory):
    manifest, doc = inventory
    source = wav(tmp_path / "outside.wav")
    doc["clips"] = [{"file": "../outside.wav", "group": "session", "split": "train"}]
    result = import_media(manifest, doc, [source], "session")
    assert "inside" in result["errors"][0] and not result["added"]
