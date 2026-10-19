import copy
import json

import numpy as np
import pytest

from arc_audio.dataset import load_dataset, save_dataset
from arc_audio.local_model import window_labels
from arc_audio.suggestions import (
    RATE,
    accept_suggestion,
    change_suggestion_status,
    current_suggestions,
    generate_suggestions,
    load_suggestions,
    media_stamp,
    scan_windows,
    suggestion_edit,
    suggestion_path,
    window_spans,
)


def sample_row(**changes):
    return dict({"id": "example", "start": 1, "end": 2, "label": "gunshot",
                 "confidence": .7, "status": "pending"}, **changes)


def sample_clip(**changes):
    return dict({"file": "synthetic.wav", "duration": 4, "size": 4, "group": "a", "split": "train",
                 "audio_stream": 1, "audio_streams": [{"index": 1}, {"index": 2}],
                 "track_confirmed": False, "intervals": [], "reviewed": []}, **changes)


def test_unaccepted_and_rejected_proposals_are_never_training_truth():
    clip = sample_clip()
    for status in ("pending", "rejected"):
        clip["suggestions"] = [sample_row(status=status)]  # Even if attached to a manifest by mistake.
        assert window_labels(1, 2, clip, ("gunshot",)) is None
    clip["reviewed"] = [{"start": 0, "end": 4, "classes": ["gunshot"]}]
    assert window_labels(1, 2, clip, ("gunshot",)).tolist() == [0]


def test_acceptance_requires_confirmation_and_preserves_review_scope():
    clip, row = sample_clip(), sample_row()
    with pytest.raises(ValueError, match="confirm"):
        accept_suggestion(clip, row, 1.1, 1.8, "reload")
    assert not clip["intervals"] and row["status"] == "pending"
    clip["track_confirmed"] = True
    accept_suggestion(clip, row, 1.1, 1.8, "reload", "teammate")
    assert clip["intervals"] == [{"start": 1.1, "end": 1.8, "label": "reload", "source": "teammate",
                                  "proposal_id": "example"}]
    assert row["status"] == "accepted" and not clip["reviewed"]
    assert window_labels(1, 2, clip, ("reload",)) is None
    accept_suggestion(clip, row, 1.2, 1.9, "gunshot")
    assert len(clip["intervals"]) == 1 and clip["intervals"][0]["start"] == 1.2
    row["status"] = "pending"  # Simulate interrupted sidecar save after annotation save.
    accept_suggestion(clip, row, 1.1, 1.8, "reload")
    assert len(clip["intervals"]) == 1


def test_decisions_can_be_reversed_and_edits_survive_reacceptance(tmp_path):
    clip, row = sample_clip(track_confirmed=True), sample_row()
    unrelated = {"start": 3, "end": 3.5, "label": "footsteps"}
    clip["intervals"].append(unrelated)
    accept_suggestion(clip, row, 1.2, 1.8, "reload", "teammate")
    clip["reviewed"] = [{"start": 0, "end": 4, "classes": ["reload", "footsteps"]}]
    change_suggestion_status(clip, row, "rejected")
    assert clip["intervals"] == [unrelated] and row["status"] == "rejected"
    from arc_audio.dataset import reviewed_spans
    assert reviewed_spans(clip, ("reload",)) == [{"start": 0, "end": 1.2}, {"start": 1.8, "end": 4}]
    assert reviewed_spans(clip, ("footsteps",)) == [{"start": 0, "end": 4}]
    manifest = tmp_path / "dataset.json"
    save_dataset(suggestion_path(manifest), {"version": 1, "clips": [{"file": clip["file"],
                 "audio_stream": 1, "media": {}, "duration": 4, "suggestions": [row]}]})
    restored = load_suggestions(manifest)["clips"][0]["suggestions"][0]
    assert suggestion_edit(clip, restored)["source"] == "teammate"
    change_suggestion_status(clip, restored, "pending")
    edit = suggestion_edit(clip, restored)
    accept_suggestion(clip, restored, edit["start"], edit["end"], edit["label"], edit["source"])
    assert len(clip["intervals"]) == 2 and clip["intervals"][0]["label"] == "reload"
    change_suggestion_status(clip, restored, "pending")
    assert clip["intervals"] == [unrelated]


def test_stale_media_and_other_tracks_hide_suggestions(tmp_path):
    path = tmp_path / "synthetic.wav"
    path.write_bytes(b"tone")
    clip = sample_clip()
    row = sample_row()
    bundle = {"version": 1, "clips": [{"file": clip["file"], "audio_stream": 1,
                                       "media": media_stamp(path), "suggestions": [row]}]}
    manifest = tmp_path / "dataset.json"
    assert current_suggestions(bundle, manifest, clip) == [row]
    assert not current_suggestions(bundle, manifest, dict(clip, audio_stream=2))
    path.write_bytes(b"changed audio")
    assert not current_suggestions(bundle, manifest, clip)


def test_window_coverage_includes_short_clips_and_final_samples():
    assert window_spans(3, 4, 2) == [(0, 3)]
    assert window_spans(9, 4, 3) == [(0, 4), (3, 7), (5, 9)]
    with pytest.raises(ValueError):
        window_spans(9, 4, 5)


def test_scanning_uses_model_scores_not_filename_or_loudness():
    signal = np.zeros(RATE * 3, np.float32)
    rows, timings = scan_windows(signal, lambda _x: {"gunshot": .71, "footsteps": .02}, duration=3,
                                window_seconds=2, hop_seconds=1, threshold=.2,
                                classes=("gunshot", "footsteps"), identity="synthetic",
                                provenance={"model": "test-double"})
    assert len(rows) == len(timings) == 2
    assert all(row["label"] == "gunshot" and row["confidence"] == .71 for row in rows)
    assert [(row["start"], row["end"]) for row in rows] == [(0, 2), (1, 3)]
    assert all(row["status"] == "pending" for row in rows)
    with pytest.raises(ValueError, match="invalid score"):
        scan_windows(signal, lambda _x: {"gunshot": float("nan")}, duration=3, window_seconds=2,
                     hop_seconds=1, threshold=.2, classes=("gunshot",), identity="test", provenance={})


@pytest.mark.parametrize("changes", [{"confidence": float("nan")}, {"start": -1},
                                     {"status": "automatically_true"}, {"label": "enemy"}])
def test_sidecar_validation(tmp_path, changes):
    bundle = {"version": 1, "clips": [{"file": "synthetic.wav", "audio_stream": 1,
                                       "media": {}, "duration": 4,
                                       "suggestions": [sample_row(**changes)]}]}
    manifest = tmp_path / "dataset.json"
    save_dataset(suggestion_path(manifest), bundle)
    with pytest.raises(ValueError):
        load_suggestions(manifest)


def test_generator_preserves_human_data_decisions_and_excludes_test(tmp_path, monkeypatch):
    from arc_audio import classifier, suggestions

    # This explicit test double validates orchestration, not sound recognition.
    class Scorer:
        def __init__(self, **_kwargs):
            pass

        def score_window(self, _signal):
            return {"gunshot": .8, "footsteps": .1}

    monkeypatch.setattr(classifier, "ASTDetector", Scorer)
    monkeypatch.setattr(suggestions, "read_media_audio", lambda *_args: np.zeros(4 * RATE, np.float32))
    model = tmp_path / "model"
    model.mkdir()
    for name in ("model.safetensors", "config.json", "preprocessor_config.json"):
        (model / name).write_text("test double, not weights")
    clips = [sample_clip(track_confirmed=True), sample_clip(file="test.wav", split="test", group="b"),
             sample_clip(file="unconfirmed.wav", group="c")]
    for clip in clips:
        (tmp_path / clip["file"]).write_bytes(b"tone")
    manifest = tmp_path / "dataset.json"
    save_dataset(manifest, {"version": 1, "clips": clips})
    original = manifest.read_bytes()
    path, report = generate_suggestions(manifest, model_path=model, progress=lambda _s: None)
    assert manifest.read_bytes() == original
    assert [row["file"] for row in report["clips"]] == ["synthetic.wav"]
    bundle = load_suggestions(manifest)
    row = bundle["clips"][0]["suggestions"][0]
    row["status"] = "rejected"
    save_dataset(path, bundle)
    generate_suggestions(manifest, model_path=model, progress=lambda _s: None)
    assert load_suggestions(manifest)["clips"][0]["suggestions"][0]["status"] == "rejected"
    assert load_dataset(manifest)["clips"][0]["intervals"] == []
    _, report = generate_suggestions(manifest, model_path=model, allow_unconfirmed=True, progress=lambda _s: None)
    assert {row["file"] for row in report["clips"]} == {"synthetic.wav", "unconfirmed.wav"}
    assert manifest.read_bytes() == original


def test_generator_never_overwrites_concurrent_human_edits(tmp_path, monkeypatch):
    from arc_audio import classifier, suggestions
    manifest = tmp_path / "dataset.json"
    (tmp_path / "synthetic.wav").write_bytes(b"tone")
    model = tmp_path / "model"
    model.mkdir()
    for name in ("model.safetensors", "config.json", "preprocessor_config.json"):
        (model / name).write_text("test fixture")
    doc = {"version": 1, "clips": [sample_clip(track_confirmed=True)]}
    save_dataset(manifest, doc)

    class ConcurrentScorer:
        def __init__(self, **_kwargs):
            pass

        def score_window(self, _signal):
            edited = copy.deepcopy(doc)
            edited["human_note"] = "concurrent edit"
            save_dataset(manifest, edited)
            return {"gunshot": .8}

    monkeypatch.setattr(classifier, "ASTDetector", ConcurrentScorer)
    monkeypatch.setattr(suggestions, "read_media_audio", lambda *_args: np.zeros(RATE, np.float32))
    with pytest.raises(ValueError, match="changed during analysis"):
        generate_suggestions(manifest, model_path=model, progress=lambda _s: None)
    assert json.loads(manifest.read_text())["human_note"] == "concurrent edit"
    assert not suggestion_path(manifest).exists()
