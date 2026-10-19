import json
import time
import wave

import numpy as np
import pytest

from arc_audio.dataset import (
    CORE_CLASSES,
    TARGET_CLASSES,
    assign_group_splits,
    load_dataset,
    merge_reviewed,
    prepare_dataset,
    reviewed_spans,
    save_dataset,
)
from arc_audio.events import AudioChunk
from arc_audio.local_model import (
    RATE,
    LocalDetector,
    features,
    fit_logistic,
    probabilities,
    train_model,
    window_labels,
)


def test_unknown_audio_never_becomes_a_negative():
    clip = {"reviewed": [{"start": 2, "end": 5}],
            "intervals": [{"start": 3, "end": 3.2, "label": "reload"}]}
    assert window_labels(0, 1, clip) is None
    assert window_labels(1.5, 2.5, clip) is None
    assert np.array_equal(window_labels(2, 3, clip), [0, 0, 0])
    assert np.array_equal(window_labels(2.5, 3.5, clip), [0, 1, 0])
    assert window_labels(3.19, 4.19, clip) is None  # weak boundary overlap excluded


def test_group_splits_cannot_leak():
    doc = {"clips": [{"group": group} for group in ["session-a", "session-a", "session-b", "session-c"]]}
    assign_group_splits(doc)
    assert doc["clips"][0]["split"] == doc["clips"][1]["split"]
    assert {c["split"] for c in doc["clips"]} == {"train", "validation", "test"}


def test_logmel_feature_contract_and_real_optimizer():
    t = np.arange(RATE) / RATE
    examples = [features(.2 * np.sin(2 * np.pi * hz * t)) for hz in (300, 300, 1300, 1300, 4000, 4000)]
    x = np.stack(examples)
    y = np.repeat(np.eye(3), 2, axis=0)
    assert x.shape == (6, 384) and np.isfinite(x).all()
    mean, scale, weights = fit_logistic(x, y)
    scores = probabilities(x, mean, scale, weights)
    assert np.array_equal(scores.argmax(axis=1), y.argmax(axis=1))
    with pytest.raises(ValueError):
        features(np.zeros(10))


def test_missing_model_does_not_invent_predictions(tmp_path):
    with pytest.raises(RuntimeError, match="train-local"):
        LocalDetector(path=tmp_path / "missing.npz")


def write_fixture(path, phase):
    t = np.arange(RATE * 32) / RATE
    audio = np.zeros_like(t)
    for start, end, hz in [(8, 16, 300), (16, 24, 1300), (24, 32, 4000)]:
        mask = (t >= start) & (t < end)
        audio[mask] = .2 * np.sin(2 * np.pi * hz * t[mask] + phase)
    with wave.open(str(path), "wb") as wav:
        wav.setparams((2, 2, RATE, 0, "NONE", "not compressed"))
        wav.writeframes((np.column_stack((audio, audio)) * 32767).astype("<i2").tobytes())


def test_training_roundtrip_and_runtime_contract_on_synthetic_only(tmp_path):
    pytest.importorskip("av")
    # These tones test learning/serialization only, not semantic gameplay accuracy.
    for name, phase in [("train.wav", 0), ("val.wav", .3), ("test.wav", .8)]:
        write_fixture(tmp_path / name, phase)
    manifest, _ = prepare_dataset(tmp_path)
    doc = load_dataset(manifest)
    for clip in doc["clips"]:
        clip["split"] = {"train.wav": "train", "val.wav": "validation", "test.wav": "test"}[clip["file"]]
        clip["track_confirmed"] = True
        clip["reviewed"] = [{"start": 0, "end": 32}]
        clip["intervals"] = [{"start": 8 + i * 8, "end": 16 + i * 8, "label": label}
                             for i, label in enumerate(CORE_CLASSES)]
    save_dataset(manifest, doc)
    path, report = train_model(manifest, tmp_path / "trained.npz", progress=lambda _text: None)
    assert {c["split"] for c in report["clips_used"]} == {"train", "validation"}
    assert not any(c["file"] == "test.wav" for c in report["clips_used"])
    detector = LocalDetector(min_confidence=0, path=path)
    events = []
    for i in range(20):
        x = np.sin(np.arange(800) * .3).astype(np.float32) * .1
        new = detector.process(AudioChunk(np.column_stack((x, x)), RATE, (i + 1) * .05, time.monotonic()))
        if i < 19:
            assert not new
        events.extend(new)
    assert {e.label for e in events} == set(CORE_CLASSES)
    assert all(e.direction == "unknown" and e.backend == "local" for e in events)
    # A timeline gap clears warm-up; past samples are not reused as continuous audio.
    assert not detector.process(AudioChunk(np.zeros((800, 2), np.float32), RATE, 4, time.monotonic()))
    from arc_audio.training_evaluation import evaluate_dataset
    evaluated = evaluate_dataset(manifest, model_path=path)
    assert evaluated["split"] == "test" and len(evaluated["clips"]) == 1
    assert set(evaluated["per_class"]) == set(CORE_CLASSES)


def test_unreviewed_manifest_refuses_training_and_split_leakage(tmp_path):
    pytest.importorskip("av")
    write_fixture(tmp_path / "empty.wav", 0)
    manifest, _ = prepare_dataset(tmp_path)
    with pytest.raises(ValueError, match="No reviewed clips"):
        train_model(manifest, tmp_path / "must-not-exist.npz")
    assert not (tmp_path / "must-not-exist.npz").exists()
    doc = load_dataset(manifest)
    duplicate = dict(doc["clips"][0], file="other.wav", split="validation")
    doc["clips"].append(duplicate)
    save_dataset(manifest, doc)
    with pytest.raises(ValueError, match="crosses splits"):
        load_dataset(manifest)


def test_manifest_rejects_media_path_escape(tmp_path):
    doc = {"version": 1, "clips": [{"file": "../outside.wav"}]}
    manifest = tmp_path / "dataset.json"
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="inside the dataset"):
        load_dataset(manifest)


def test_legacy_reviews_never_become_negatives_for_new_classes():
    clip = {"reviewed": [{"start": 0, "end": 10}], "intervals": []}
    assert np.array_equal(window_labels(0, 1, clip), [0, 0, 0])
    assert window_labels(0, 1, clip, ("snap_hook",)) is None
    assert window_labels(0, 1, clip, TARGET_CLASSES) is None
    clip["reviewed"].append({"start": 2, "end": 6, "classes": list(TARGET_CLASSES)})
    assert np.array_equal(window_labels(2, 3, clip, ("snap_hook", "landing")), [0, 0])
    assert window_labels(0, 1, clip, ("snap_hook", "landing")) is None


def test_review_merging_preserves_class_scope():
    combined = merge_reviewed([{"start": 0, "end": 2}, {"start": 1, "end": 4},
                              {"start": 2, "end": 5, "classes": ["snap_hook"]}])
    assert len(combined) == 2
    clip = {"reviewed": combined}
    assert reviewed_spans(clip) == [{"start": 0, "end": 4}]
    assert reviewed_spans(clip, ("snap_hook",)) == [{"start": 2, "end": 5}]


def test_optional_single_class_training_and_runtime_use_only_selected_head(tmp_path):
    pytest.importorskip("av")
    for name in ("train.wav", "val.wav", "test.wav"):
        write_fixture(tmp_path / name, .1)
    manifest, _ = prepare_dataset(tmp_path)
    doc = load_dataset(manifest)
    for clip in doc["clips"]:
        clip["split"] = {"train.wav": "train", "val.wav": "validation", "test.wav": "test"}[clip["file"]]
        clip["track_confirmed"] = True
        clip["reviewed"] = [{"start": 0, "end": 32, "classes": ["snap_hook"]}]
        # Synthetic tone label is a pipeline fixture, not a Snap Hook recording.
        clip["intervals"] = [{"start": 8, "end": 16, "label": "snap_hook"}]
    save_dataset(manifest, doc)
    path, report = train_model(manifest, tmp_path / "one-class.npz", progress=lambda _text: None,
                               classes=("snap_hook",))
    detector = LocalDetector(path=path, min_confidence=0)
    assert detector.classes == ["snap_hook"] and report["classes"] == ["snap_hook"]
    x = np.zeros((RATE, 2), np.float32)
    events = detector.process(AudioChunk(x, RATE, 1, time.monotonic()))
    assert [e.label for e in events] == ["snap_hook"]
    from arc_audio.training_evaluation import evaluate_dataset
    evaluated = evaluate_dataset(manifest, model_path=path)
    assert set(evaluated["per_class"]) == {"snap_hook"}


def test_requested_unreviewed_class_refuses_fit_even_when_core_review_exists(tmp_path):
    pytest.importorskip("av")
    write_fixture(tmp_path / "train.wav", 0)
    manifest, _ = prepare_dataset(tmp_path)
    doc = load_dataset(manifest)
    doc["clips"][0].update(track_confirmed=True, reviewed=[{"start": 0, "end": 32}])
    save_dataset(manifest, doc)
    with pytest.raises(ValueError, match="covering the selected classes"):
        train_model(manifest, tmp_path / "forbidden.npz", classes=("landing",))
    assert not (tmp_path / "forbidden.npz").exists()
