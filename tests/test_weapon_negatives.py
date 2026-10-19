"""No decoding: synthetic metadata fixtures verify absence proposals stay unknown."""

import json
import os

import pytest

from arc_audio.dataset import load_dataset, save_dataset
from arc_audio.suggestions import file_sha256, load_suggestions, media_stamp, suggestion_path
from arc_audio.weapon_negatives import propose_background


def fixture(tmp_path):
    manifest = tmp_path / "dataset.json"
    media = tmp_path / "source.bin"
    media.write_bytes(b"Synthetic metadata-only fixture; not playable audio.")
    digest = file_sha256(media)
    clip = {"file": media.name, "duration": 20., "size": media.stat().st_size,
            "audio_stream": 1, "audio_streams": [{"index": 1}], "group": "source-1", "split": "train",
            "track_confirmed": True, "collection_source_id": "source-1", "source_sha256": digest,
            "intervals": [{"start": 9., "end": 9.2, "label": "gunshot", "source": "unknown"}],
            "reviewed": [{"start": 2., "end": 3., "classes": ["footsteps"]}]}
    source = {"id": "ref-1", "source_id": "source-1", "source_url": "https://example.com/video",
              "weapon": "ferro", "start": 1., "end": 19., "metadata": {"source_title": "Fixture"}}
    document = {"version": 1, "clips": [clip], "weapon_sources": [source]}
    positive = {"id": "positive", "start": 5., "end": 5.15, "source_timestamp": 5.,
                "label": "gunshot", "status": "rejected", "confidence": .8,
                "collection_source_id": "source-1", "detection_score": .8, "quality_score": 80,
                "preview_start": 4.65, "preview_end": 6., "possible_weapons": [], "proposed_weapon": None}
    bundle = {"version": 1, "clips": [{"file": media.name, "duration": 20., "audio_stream": 1,
                                          "media": media_stamp(media), "media_sha256": digest,
                                          "suggestions": [positive]}]}
    save_dataset(manifest, document)
    save_dataset(suggestion_path(manifest), bundle)
    return manifest, media


def absence_rows(manifest):
    return [row for entry in load_suggestions(manifest)["clips"] for row in entry["suggestions"]
            if row.get("proposal_kind") == "gunshot_absence"]


def test_only_pending_absence_proposals_with_ranges_and_conservative_peak_margins(tmp_path):
    manifest, _ = fixture(tmp_path)
    original_manifest = manifest.read_bytes()
    original_bundle = load_suggestions(manifest)
    report = propose_background(manifest)
    rows = absence_rows(manifest)
    assert report["new_candidates"] == report["total"] == len(rows) > 0
    assert len(rows) <= 8 and not report["skipped"]
    assert manifest.read_bytes() == original_manifest
    assert load_suggestions(manifest)["clips"][0]["suggestions"][0] == original_bundle["clips"][0]["suggestions"][0]
    for row in rows:
        assert row["status"] == "pending" and row["label"] == "gunshot"
        assert row["confidence"] == row["detection_score"] == 0
        assert row["proposed_weapon"] is None and row["possible_weapons"] == []
        assert 1 <= row["start"] < row["end"] <= 19
        assert row["end"] - row["start"] == pytest.approx(2)
        assert not (row["start"] < 5.15 + .75 and 5 - .75 < row["end"])
        assert not (row["start"] < 9.2 + .75 and 9 - .75 < row["end"])
        assert row["preview_start"] == row["start"] and row["preview_end"] == row["end"]
        assert row["source_timestamp"] == (row["start"] + row["end"]) / 2
        assert 0 <= row["quality_score"] <= 100
        assert row["source_title"] == "Fixture" and row["collection_source_id"] == "source-1"
        assert "not background" in row["provenance"]["notice"]
    ordered = sorted(rows, key=lambda row: row["start"])
    assert all(a["end"] <= b["start"] for a, b in zip(ordered, ordered[1:]))


def test_rerun_preserves_decisions_and_ids_without_rewriting(tmp_path):
    manifest, _ = fixture(tmp_path)
    propose_background(manifest)
    bundle = load_suggestions(manifest)
    selected = next(row for row in bundle["clips"][0]["suggestions"] if row.get("proposal_kind") == "gunshot_absence")
    selected["status"] = "accepted"
    selected["human_note"] = "I listened"
    save_dataset(suggestion_path(manifest), bundle)
    original = suggestion_path(manifest).read_bytes()
    report = propose_background(manifest)
    assert report["new_candidates"] == 0
    assert suggestion_path(manifest).read_bytes() == original


def test_heldout_source_and_unregistered_clips_are_skipped(tmp_path):
    manifest, _ = fixture(tmp_path)
    document = load_dataset(manifest)
    document["clips"][0]["split"] = "test"
    save_dataset(manifest, document)
    before = suggestion_path(manifest).read_bytes()
    assert propose_background(manifest)["new_candidates"] == 0
    assert suggestion_path(manifest).read_bytes() == before
    document["clips"][0]["split"] = "train"
    document["weapon_sources"] = []
    save_dataset(manifest, document)
    assert propose_background(manifest)["new_candidates"] == 0


@pytest.mark.parametrize("change", ["missing", "stamp", "hash", "entry", "range"])
def test_missing_stale_or_invalid_source_data_skips_safely(tmp_path, change):
    manifest, media = fixture(tmp_path)
    if change == "missing":
        media.unlink()
    elif change == "stamp":
        media.write_bytes(b"changed")
    elif change == "hash":
        old = media.stat()
        media.write_bytes(b"!" * old.st_size)
        os.utime(media, ns=(old.st_atime_ns, old.st_mtime_ns))
    elif change == "entry":
        save_dataset(suggestion_path(manifest), {"version": 1, "clips": []})
    else:
        document = load_dataset(manifest)
        document["weapon_sources"][0]["start"] = 25
        save_dataset(manifest, document)
    original = suggestion_path(manifest).read_bytes()
    report = propose_background(manifest)
    assert report["new_candidates"] == 0 and report["skipped"]
    assert suggestion_path(manifest).read_bytes() == original


def test_multiple_registered_ranges_are_respected_and_limit_is_per_source(tmp_path):
    manifest, _ = fixture(tmp_path)
    document = load_dataset(manifest)
    document["weapon_sources"][0].update(start=1, end=4)
    document["weapon_sources"].append(dict(document["weapon_sources"][0], id="ref-2", start=11, end=19))
    save_dataset(manifest, document)
    assert propose_background(manifest, seconds=1, max_per_source=3)["new_candidates"] == 3
    for row in absence_rows(manifest):
        assert 1 <= row["start"] < row["end"] <= 4 or 11 <= row["start"] < row["end"] <= 19


def test_short_windows_still_use_a_bounded_candidate_pool(tmp_path):
    manifest, _ = fixture(tmp_path)
    report = propose_background(manifest, seconds=.00001, max_per_source=2)
    assert report["new_candidates"] == 2
    assert all(row["end"] - row["start"] == pytest.approx(.00001) for row in absence_rows(manifest))


def test_concurrent_label_edits_preserved_and_no_sidecar_commit(tmp_path, monkeypatch):
    manifest, _ = fixture(tmp_path)
    original = suggestion_path(manifest).read_bytes()
    real_hash = file_sha256

    def concurrent(path):
        document = json.loads(manifest.read_text(encoding="utf-8"))
        document["external_edit"] = "preserved"
        save_dataset(manifest, document)
        return real_hash(path)

    monkeypatch.setattr("arc_audio.weapon_negatives.file_sha256", concurrent)
    with pytest.raises(ValueError, match="changed during"):
        propose_background(manifest)
    assert load_dataset(manifest)["external_edit"] == "preserved"
    assert suggestion_path(manifest).read_bytes() == original
    assert not manifest.with_suffix(".collection.lock").exists()


def test_existing_collection_lock_prevents_competing_writer(tmp_path):
    manifest, _ = fixture(tmp_path)
    lock = manifest.with_suffix(".collection.lock")
    lock.write_text("owned by another collector", encoding="utf-8")
    original = suggestion_path(manifest).read_bytes()
    with pytest.raises(ValueError, match="Collection already running"):
        propose_background(manifest)
    assert suggestion_path(manifest).read_bytes() == original and lock.exists()


@pytest.mark.parametrize("kwargs", [{"seconds": 0}, {"seconds": -1}, {"seconds": float("nan")},
                                    {"seconds": True}, {"margin": -1}, {"margin": float("inf")},
                                    {"max_per_source": 0}, {"max_per_source": 1.5}, {"max_per_source": True}])
def test_invalid_arguments_do_not_create_files(tmp_path, kwargs):
    manifest = tmp_path / "absent" / "dataset.json"
    with pytest.raises(ValueError, match="positive finite"):
        propose_background(manifest, **kwargs)
    assert not manifest.parent.exists()
