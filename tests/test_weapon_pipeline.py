import csv
import json
import wave

import numpy as np
import pytest

from arc_audio.dataset import load_dataset, save_dataset
from arc_audio.suggestions import (
    accept_suggestion,
    change_suggestion_status,
    load_suggestions,
    suggestion_path,
)
from arc_audio.weapon_pipeline import build_weapon_dataset, register_discoveries, weapon_dataset_status


def fixture_sources(tmp_path):
    rng = np.random.default_rng(7)
    audio = rng.normal(0, .001, 16000 * 4)
    for onset in (1., 1.14, 2.5):
        start = round(onset * 16000)
        audio[start:start + 640] += rng.normal(0, .3, 640) * np.exp(-np.arange(640) / 180)
    path = tmp_path / "source.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    csv_path = tmp_path / "manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["weapon", "source_title", "source_url", "start", "end", "quality", "notes"])
        writer.writerow(["Ferro", "Synthetic fixture", str(path), "0.5", "1.9", "clean", "not gameplay"])
    return csv_path, tmp_path / "collection" / "dataset.json"


def test_vertical_slice_range_idempotency_and_human_decisions(tmp_path):
    csv_path, manifest = fixture_sources(tmp_path)
    report = build_weapon_dataset(csv_path, manifest, progress=lambda _msg: None)
    assert not report["failed"] and report["new_candidates"] >= 2
    doc, bundle = load_dataset(manifest), load_suggestions(manifest)
    clip, rows = doc["clips"][0], bundle["clips"][0]["suggestions"]
    assert not clip["intervals"] and not clip["reviewed"] and not clip["track_confirmed"]
    assert all(.5 <= row["source_timestamp"] < 1.9 for row in rows)
    assert all(row["proposed_weapon"] == "ferro" and row["status"] == "pending" for row in rows)
    clip["track_confirmed"] = True
    accept_suggestion(clip, rows[0], rows[0]["start"], rows[0]["end"], "gunshot", weapon="anvil")
    change_suggestion_status(clip, rows[1], "rejected")
    save_dataset(manifest, doc)
    save_dataset(suggestion_path(manifest), bundle)
    again = build_weapon_dataset(csv_path, manifest, progress=lambda _msg: None)
    assert again["new_candidates"] == 0 and again["cached"] == 1
    restored = load_dataset(manifest)
    assert restored["clips"][0]["intervals"][0]["weapon"] == "anvil"
    assert not restored["clips"][0]["reviewed"]
    statuses = [row["status"] for row in load_suggestions(manifest)["clips"][0]["suggestions"]]
    assert statuses[:2] == ["corrected", "rejected"]
    assert len(statuses) == len(rows)
    status = weapon_dataset_status(manifest)
    assert status["weapons"]["ferro"]["downloaded_sources"] == 1
    assert status["weapons"]["anvil"]["corrected"] == 1
    assert status["weapons"]["ferro"]["corrected"] == 0


def test_dry_run_no_writes_and_failed_source_does_not_abort_batch(tmp_path):
    csv_path, manifest = fixture_sources(tmp_path)
    with csv_path.open("a", newline="", encoding="utf-8") as output:
        csv.writer(output).writerow(["Stitcher", "missing", str(tmp_path / "absent.wav"), "", "", "", ""])
    preview = build_weapon_dataset(csv_path, manifest, dry_run=True)
    assert preview["sources"] == 2 and not manifest.parent.exists()
    report = build_weapon_dataset(csv_path, manifest, progress=lambda _msg: None)
    assert len(report["failed"]) == 1 and report["new_candidates"] > 0
    doc = load_dataset(manifest)
    assert len(doc["clips"]) == 1 and len(doc["weapon_sources"]) == 2
    assert weapon_dataset_status(manifest)["weapons"]["stitcher"]["failed_sources"] == 1


def test_multiple_weapons_and_sources_coexist_without_touching_old_labels(tmp_path):
    import shutil

    csv_path, manifest = fixture_sources(tmp_path)
    second = tmp_path / "other-source.wav"
    shutil.copyfile(tmp_path / "source.wav", second)
    with csv_path.open("a", newline="", encoding="utf-8") as output:
        csv.writer(output).writerow(["Stitcher", "Second source", str(second), "2", "3.9", "clean", "fixture"])
    report = build_weapon_dataset(csv_path, manifest, progress=lambda _msg: None)
    assert not report["failed"] and report["downloaded"] == 2
    document = load_dataset(manifest)
    assert len(document["clips"]) == 2
    assert len({clip["group"] for clip in document["clips"]}) == 2
    assert {row["proposed_weapon"] for entry in load_suggestions(manifest)["clips"]
            for row in entry["suggestions"]} == {"ferro", "stitcher"}


def test_one_video_many_weapons_remains_one_ambiguous_candidate_set(tmp_path):
    csv_path, manifest = fixture_sources(tmp_path)
    row = list(csv.DictReader(csv_path.open(encoding="utf-8")))[0]
    with csv_path.open("a", newline="", encoding="utf-8") as output:
        csv.writer(output).writerow(["Stitcher", row["source_title"], row["source_url"], ".5", "1.9", "", ""])
    build_weapon_dataset(csv_path, manifest, progress=lambda _msg: None)
    doc = load_dataset(manifest)
    assert len(doc["clips"]) == 1
    rows = load_suggestions(manifest)["clips"][0]["suggestions"]
    assert all(row["proposed_weapon"] is None and row["possible_weapons"] == ["ferro", "stitcher"] for row in rows)
    assert all(len(row["manifest_refs"]) == 2 for row in rows)


def test_holdout_never_regenerated_and_source_variants_cannot_cross_splits(tmp_path, monkeypatch):
    csv_path, manifest = fixture_sources(tmp_path)
    build_weapon_dataset(csv_path, manifest, progress=lambda _msg: None)
    doc = load_dataset(manifest)
    doc["clips"][0]["split"] = "test"
    save_dataset(manifest, doc)
    monkeypatch.setattr("arc_audio.weapon_sources.fetch_source", lambda *a, **kw: pytest.fail("Downloaded test"))
    report = build_weapon_dataset(csv_path, manifest, progress=lambda _msg: None)
    assert len(report["skipped_test"]) == 1
    doc["clips"].append(dict(doc["clips"][0], file="other.wav", group="different", split="validation"))
    save_dataset(manifest, doc)
    with pytest.raises(ValueError, match="source video crosses"):
        load_dataset(manifest)


def test_old_dataset_and_optional_weapon_validation(tmp_path):
    manifest = tmp_path / "old.json"
    clip = {"file": "old.wav", "duration": 5, "group": "old", "split": "train",
            "audio_stream": 0, "audio_streams": [{"index": 0}], "intervals": [
                {"start": 1, "end": 2, "label": "gunshot"}], "reviewed": []}
    save_dataset(manifest, {"version": 1, "clips": [clip]})
    assert "weapon" not in load_dataset(manifest)["clips"][0]["intervals"][0]
    clip["intervals"][0]["weapon"] = "ferro"
    save_dataset(manifest, {"version": 1, "clips": [clip]})
    assert load_dataset(manifest)["clips"][0]["intervals"][0]["weapon"] == "ferro"
    clip["intervals"][0]["weapon"] = ["invalid"]
    save_dataset(manifest, {"version": 1, "clips": [clip]})
    with pytest.raises(ValueError, match="Weapon"):
        load_dataset(manifest)


def test_concurrent_label_edits_preserved(tmp_path):
    csv_path, manifest = fixture_sources(tmp_path)
    def progress(message):
        if message.startswith("  "):
            document = json.loads(manifest.read_text(encoding="utf-8"))
            document["external_edit"] = True
            save_dataset(manifest, document)
    with pytest.raises(ValueError, match="changed during collection"):
        build_weapon_dataset(csv_path, manifest, progress=progress)
    assert load_dataset(manifest)["external_edit"]


def test_malformed_parameters_and_weapon_do_not_write_or_download(tmp_path):
    csv_path, manifest = fixture_sources(tmp_path)
    with pytest.raises(ValueError, match="finite"):
        build_weapon_dataset(csv_path, manifest, pre=float("nan"))
    assert not manifest.parent.exists()
    text = csv_path.read_text(encoding="utf-8").replace("Ferro", "a" * 81)
    csv_path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="Weapon"):
        build_weapon_dataset(csv_path, manifest)
    assert not manifest.parent.exists()


def test_burst_ids_distinguish_multiple_ranges_and_sources(tmp_path):
    import shutil

    csv_path, manifest = fixture_sources(tmp_path)
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    source = tmp_path / "source.wav"
    other = tmp_path / "other.wav"
    shutil.copyfile(source, other)
    with csv_path.open("a", newline="", encoding="utf-8") as output:
        csv.writer(output).writerow(["Stitcher", "Other source", str(other), ".5", "1.9", "", ""])
    build_weapon_dataset(csv_path, manifest, progress=lambda _msg: None)
    entries = load_suggestions(manifest)["clips"]
    ids = [{row.get("burst_group_id") for row in entry["suggestions"]} - {None} for entry in entries]
    assert ids[0] and ids[1] and not ids[0].intersection(ids[1])
    assert rows


def test_discovery_registration_is_metadata_only_and_idempotent(tmp_path):
    csv_path, manifest = fixture_sources(tmp_path)
    build_weapon_dataset(csv_path, manifest, progress=lambda _msg: None)
    before = load_dataset(manifest)["clips"]
    sidecar = suggestion_path(manifest).read_bytes()
    report = {"sources": [{"id": "new-video-id", "weapon": "ferro", "source_url": "https://example.com/new",
                            "source_title": "Unverified", "verification": "unverified"}]}
    register_discoveries(manifest, report)
    register_discoveries(manifest, report)
    doc = load_dataset(manifest)
    assert doc["clips"] == before and len(doc["discovered_weapon_sources"]) == 1
    assert suggestion_path(manifest).read_bytes() == sidecar
    status = weapon_dataset_status(manifest)["weapons"]["ferro"]
    assert status["discovered_sources"] == 2 and status["unverified_sources"] == 1
    assert status["downloaded_sources"] == 1


def test_optional_dedupe_failure_retains_usable_pending_candidates(tmp_path, monkeypatch):
    csv_path, manifest = fixture_sources(tmp_path)
    def broken(_entries):
        raise RuntimeError("Synthetic similarity failure")
    monkeypatch.setattr("arc_audio.weapon_dedupe.annotate_duplicates", broken)
    report = build_weapon_dataset(csv_path, manifest, progress=lambda _msg: None)
    assert not report["failed"] and report["new_candidates"] > 0
    assert "Synthetic similarity failure" in report["deduplication"]["error"]
    assert not load_dataset(manifest)["clips"][0]["intervals"]
    assert all(row["status"] == "pending" for row in load_suggestions(manifest)["clips"][0]["suggestions"])


def test_positive_refresh_never_replaces_an_absence_proposal(tmp_path):
    csv_path, manifest = fixture_sources(tmp_path)
    build_weapon_dataset(csv_path, manifest, progress=lambda _msg: None)
    bundle = load_suggestions(manifest)
    rows = bundle["clips"][0]["suggestions"]
    absence = dict(rows[0], id="absence-fixture", proposal_kind="gunshot_absence", detection_score=0,
                   confidence=0, proposed_weapon=None, possible_weapons=[])
    rows.insert(0, absence)
    save_dataset(suggestion_path(manifest), bundle)
    build_weapon_dataset(csv_path, manifest, pre=.36, progress=lambda _msg: None)
    restored = load_suggestions(manifest)["clips"][0]["suggestions"]
    assert next(row for row in restored if row["id"] == "absence-fixture")["detection_score"] == 0
    assert len([row for row in restored if row.get("proposal_kind") == "gunshot_absence"]) == 1
