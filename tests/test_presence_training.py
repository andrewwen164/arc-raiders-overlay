"""Synthetic-only checks of exact human window truth, not game recognition."""

import hashlib
import wave

import numpy as np
import pytest

from arc_audio.dataset import load_dataset, save_dataset
from arc_audio.local_model import RATE, LocalDetector, _clip_training_windows, train_model


def _clip(tmp_path, name, split, answers, *, confirmed=True):
    path = tmp_path / f"{name}.wav"
    t = np.arange(RATE) / RATE
    signal = np.concatenate([.15 * np.sin(2 * np.pi * (1800 if answer else 210) * t)
                             for answer in answers]).astype(np.float32)
    with wave.open(str(path), "wb") as wav:
        wav.setparams((2, 2, RATE, 0, "NONE", "not compressed"))
        wav.writeframes((np.column_stack((signal, signal)) * 32767).astype("<i2").tobytes())
    return {"file": path.name, "duration": len(answers), "size": path.stat().st_size,
            "group": name, "split": split, "audio_stream": 0,
            "audio_streams": [{"index": 0, "channels": 2, "rate": RATE}],
            "track_confirmed": confirmed, "reviewed": [], "intervals": [],
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "presence_reviews": [{"id": f"{name}-{i}", "start": float(i), "end": float(i + 1),
                                  "label": "gunshot", "present": bool(answer),
                                  "audio_stream": 0, "proposal_ids": [f"proposal-{name}-{i}"]}
                                 for i, answer in enumerate(answers)]}


def _manifest(tmp_path, *, confirmed=True):
    manifest = tmp_path / "dataset.json"
    clips = [_clip(tmp_path, "train", "train", [True] * 20 + [False] * 20, confirmed=confirmed),
             _clip(tmp_path, "validation", "validation", [True] * 5 + [False] * 5, confirmed=confirmed),
             _clip(tmp_path, "held-out", "test", [True, False], confirmed=confirmed)]
    save_dataset(manifest, {"version": 1, "clips": clips})
    return manifest, clips


def test_exact_presence_fits_synthetic_model_without_sliding_or_test_decode(tmp_path, monkeypatch):
    pytest.importorskip("av")
    manifest, clips = _manifest(tmp_path)
    # Fitting must not even hash the final test media.
    clips[-1]["source_sha256"] = "0" * 64
    (tmp_path / clips[-1]["file"]).unlink()
    save_dataset(manifest, {"version": 1, "clips": clips})
    import arc_audio.local_model as module
    original = module.read_media_audio
    calls = []

    def decode(path, stream, rate, max_seconds=None):
        calls.append((path.name, max_seconds))
        assert path.name != "held-out.wav"
        return original(path, stream, rate, max_seconds=max_seconds)

    monkeypatch.setattr(module, "read_media_audio", decode)
    output, report = train_model(manifest, tmp_path / "experiment.npz", classes=("gunshot",), progress=lambda _: None)
    assert LocalDetector(path=output).classes == ["gunshot"]
    assert calls == [("train.wav", 40), ("validation.wav", 10)]
    assert report["annotation_kind"] == "window_presence"
    assert report["counts"]["train"]["windows"] == 40
    assert report["counts"]["train"]["presence_windows"] == 40
    assert report["counts"]["train"]["positive_windows"] == {"gunshot": 20}
    assert report["counts"]["validation"]["windows"] == 10
    assert "event recall" in report["limitations"]
    assert {row["split"] for row in report["clips_used"]} == {"train", "validation"}


@pytest.mark.parametrize("classes", [("footsteps",), ("gunshot", "reload")])
def test_presence_never_makes_other_classes_positive_or_negative(tmp_path, classes):
    manifest, _ = _manifest(tmp_path)
    with pytest.raises(ValueError, match="No reviewed clips"):
        train_model(manifest, tmp_path / "forbidden.npz", classes=classes)
    assert not (tmp_path / "forbidden.npz").exists()


def test_unconfirmed_track_or_unsure_have_no_training_truth(tmp_path):
    manifest, clips = _manifest(tmp_path, confirmed=False)
    with pytest.raises(ValueError, match="No reviewed clips"):
        train_model(manifest, tmp_path / "forbidden.npz", classes=("gunshot",))
    for clip in clips:
        clip["track_confirmed"] = True
        # Unsure is represented by no presence record, not False.
        clip["presence_reviews"] = []
    save_dataset(manifest, {"version": 1, "clips": clips})
    with pytest.raises(ValueError, match="No reviewed clips"):
        train_model(manifest, tmp_path / "forbidden.npz", classes=("gunshot",))


def test_presence_hash_mismatch_refuses_before_decode(tmp_path, monkeypatch):
    manifest, clips = _manifest(tmp_path)
    clips[0]["source_sha256"] = "0" * 64
    save_dataset(manifest, {"version": 1, "clips": clips})
    monkeypatch.setattr("arc_audio.local_model.read_media_audio", lambda *args, **kwargs: pytest.fail("must not decode changed media"))
    with pytest.raises(ValueError, match="SHA-256 changed"):
        train_model(manifest, tmp_path / "forbidden.npz", classes=("gunshot",))
    assert not (tmp_path / "forbidden.npz").exists()


def test_local_source_without_hash_still_checks_quick_review_media_identity(tmp_path, monkeypatch):
    manifest, clips = _manifest(tmp_path)
    clips[0].pop("source_sha256")
    path = tmp_path / clips[0]["file"]
    stamp = path.stat()
    clips[0]["presence_media"] = {"size": stamp.st_size, "mtime_ns": stamp.st_mtime_ns - 1000}
    save_dataset(manifest, {"version": 1, "clips": clips})
    monkeypatch.setattr("arc_audio.local_model.read_media_audio", lambda *a, **kw: pytest.fail("stale media decoded"))
    with pytest.raises(ValueError, match="Media changed since quick review"):
        train_model(manifest, tmp_path / "forbidden.npz", classes=("gunshot",))
    assert not (tmp_path / "forbidden.npz").exists()


def test_presence_exact_duplicates_count_once_and_legacy_same_window_dedupes(tmp_path):
    clip = _clip(tmp_path, "duplicate", "train", [True, False])
    clip["presence_reviews"].append(dict(clip["presence_reviews"][0], id="second-question-same-second"))
    with pytest.raises(ValueError, match="unique"):
        _clip_training_windows(clip, np.zeros(2 * RATE, np.float32), ("gunshot",))
    clip["presence_reviews"].pop()
    clip["reviewed"] = [{"start": 0, "end": 2, "classes": ["gunshot"]}]
    clip["intervals"] = [{"start": 0, "end": 1, "label": "gunshot"}]
    windows, legacy, presence = _clip_training_windows(clip, np.zeros(2 * RATE, np.float32), ("gunshot",))
    assert len(presence) == 2
    assert len(legacy) == 5
    assert len(windows) == 5
    assert windows[(0, RATE)].tolist() == [1]
    assert windows[(RATE, 2 * RATE)].tolist() == [0]


@pytest.mark.parametrize("present, interval", [(True, None), (False, (.95, 1.1))])
def test_presence_conflicting_exhaustive_legacy_truth_refuses(tmp_path, present, interval):
    clip = _clip(tmp_path, "conflict", "train", [present, False])
    clip["presence_reviews"] = clip["presence_reviews"][:1]
    clip["reviewed"] = [{"start": 0, "end": 2, "classes": ["gunshot"]}]
    clip["intervals"] = ([{"start": interval[0], "end": interval[1], "label": "gunshot"}]
                         if interval else [])
    with pytest.raises(ValueError, match="Conflicting human labels"):
        _clip_training_windows(clip, np.zeros(2 * RATE, np.float32), ("gunshot",))


def test_two_human_answers_for_same_second_must_agree(tmp_path):
    clip = _clip(tmp_path, "conflict", "train", [True])
    clip["presence_reviews"].append(dict(clip["presence_reviews"][0], id="other-question", present=False))
    with pytest.raises(ValueError, match="unique"):
        _clip_training_windows(clip, np.zeros(RATE, np.float32), ("gunshot",))


def test_partial_legacy_review_does_not_invent_exhaustive_truth(tmp_path):
    clip = _clip(tmp_path, "partial", "train", [True])
    clip["reviewed"] = [{"start": 0, "end": .5, "classes": ["gunshot"]}]
    windows, legacy, presence = _clip_training_windows(clip, np.zeros(RATE, np.float32), ("gunshot",))
    assert legacy == set()
    assert len(presence) == len(windows) == 1


def test_no_cannot_contradict_saved_gunfire_even_without_exhaustive_review(tmp_path):
    clip = _clip(tmp_path, "partial", "train", [False])
    clip["intervals"] = [{"start": .3, "end": .5, "label": "gunshot"}]
    with pytest.raises(ValueError, match="Conflicting human labels"):
        _clip_training_windows(clip, np.zeros(RATE, np.float32), ("gunshot",))


def test_truncated_decode_does_not_pad_a_human_second(tmp_path, monkeypatch):
    manifest, _ = _manifest(tmp_path)
    monkeypatch.setattr("arc_audio.local_model.read_media_audio", lambda *args, **kwargs: np.zeros(RATE // 2, np.float32))
    with pytest.raises(ValueError, match="exceeds available audio"):
        train_model(manifest, tmp_path / "forbidden.npz", classes=("gunshot",))


def test_presence_requires_independent_validation_and_existing_minimums(tmp_path):
    manifest, clips = _manifest(tmp_path)
    clips[1]["presence_reviews"] = []
    save_dataset(manifest, {"version": 1, "clips": clips})
    with pytest.raises(ValueError, match="No reviewed validation clips"):
        train_model(manifest, tmp_path / "forbidden.npz", classes=("gunshot",))
    clips[1]["presence_reviews"] = _clip(tmp_path, "val-fewer", "validation", [True] * 4 + [False] * 5)["presence_reviews"]
    save_dataset(manifest, {"version": 1, "clips": clips})
    with pytest.raises(ValueError, match="need at least 5 of each"):
        train_model(manifest, tmp_path / "forbidden.npz", classes=("gunshot",), progress=lambda _: None)


def test_manifest_edit_during_fit_prevents_artifact(tmp_path, monkeypatch):
    manifest, _ = _manifest(tmp_path)
    import arc_audio.local_model as module
    original = module.fit_logistic

    def fit(*args, **kwargs):
        doc = load_dataset(manifest)
        doc["reviewer_note"] = "concurrent edit"
        save_dataset(manifest, doc)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "fit_logistic", fit)
    with pytest.raises(ValueError, match="Dataset changed during training"):
        train_model(manifest, tmp_path / "forbidden.npz", classes=("gunshot",), progress=lambda _: None)
    assert not (tmp_path / "forbidden.npz").exists()
