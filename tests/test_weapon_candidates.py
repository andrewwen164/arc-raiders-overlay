"""Synthetic fixtures test proposal mechanics, never gunfire recognition accuracy."""

import json

import numpy as np
import pytest

from arc_audio.weapon_candidates import DETECTOR, detect_candidates


def impulses(rate=16000):
    x = np.random.default_rng(42).normal(0, .001, rate * 2).astype(np.float32)
    for at, level in ((.5, .8), (.57, .7), (.64, .9)):
        x[round(at * rate)] = level
    return x


@pytest.mark.parametrize("rate", [8000, 16000, 48000])
def test_rapid_peaks_stay_separate_and_share_a_burst(rate):
    rows = detect_candidates(impulses(rate), rate)
    assert [row["source_timestamp"] for row in rows] == pytest.approx([.5, .57, .64], abs=1 / rate)
    assert len({row["burst_group_id"] for row in rows}) == 1
    assert all(row["detector"] == DETECTOR and row["confidence_kind"] == "heuristic_score" for row in rows)
    assert all(row["label"] == "gunshot" for row in rows)  # A proposed review label only.
    assert all(0 < row["end"] - row["start"] <= .15 + 1e-9 for row in rows)
    assert all(a["end"] <= b["start"] for a, b in zip(rows, rows[1:]))


def test_context_and_events_respect_requested_absolute_range():
    rows = detect_candidates(impulses(), start=.53, end=.63, pre=10, post=10)
    assert len(rows) == 1
    assert rows[0]["source_timestamp"] == pytest.approx(.57)
    assert rows[0]["preview_start"] == .53
    assert rows[0]["preview_end"] == rows[0]["end"] == .63
    assert rows[0]["start"] == rows[0]["source_timestamp"]


def test_range_excludes_outside_impulses_even_from_context():
    assert detect_candidates(impulses(), start=.7, end=1.5) == []


def test_silence_quiet_noise_and_empty_have_no_events():
    assert detect_candidates(np.zeros(32000, np.float32)) == []
    assert detect_candidates(np.empty(0, np.float32)) == []
    assert detect_candidates(np.random.default_rng(42).normal(0, .001, 32000)) == []
    assert detect_candidates(impulses(), start=1, end=1) == []


def test_minimum_spacing_can_reduce_rapid_proposals():
    rows = detect_candidates(impulses(), min_spacing=.1)
    assert len(rows) == 2
    assert rows[1]["source_timestamp"] - rows[0]["source_timestamp"] >= .1


def test_output_is_deterministic_finite_json_and_features_describe_events():
    x = impulses()
    rows = detect_candidates(x)
    assert rows == detect_candidates(x.copy())
    json.dumps(rows, allow_nan=False)
    for row in rows:
        assert 0 <= row["confidence"] == row["detection_score"] <= 1
        assert 0 <= row["quality_score"] <= 100
        feature = row["features"]
        assert feature["duration"] == pytest.approx(row["end"] - row["start"], abs=1 / 16000)
        assert 0 < feature["rms"] <= feature["peak"]
        assert feature["peak"] > .6
        assert 0 <= feature["zcr"] <= 1
        assert 0 <= feature["centroid"] <= 8000
        assert 0 <= feature["bandwidth"] <= 8000
        assert 0 <= feature["rolloff"] <= 8000
        assert len(feature["log_mel_summary"]) == 16


def test_a_single_event_has_no_burst_id_and_zero_context_is_valid():
    x = np.zeros(16000, np.float32)
    x[8000] = .5
    rows = detect_candidates(x, pre=0, post=0)
    assert len(rows) == 1
    assert "burst_group_id" not in rows[0]
    assert rows[0]["preview_start"] == rows[0]["start"]
    assert rows[0]["preview_end"] == rows[0]["end"]


@pytest.mark.parametrize("kwargs", [
    {"rate": 0}, {"rate": True}, {"rate": 16000.5}, {"start": -1}, {"start": float("nan")},
    {"end": float("inf")}, {"end": 3}, {"start": 1, "end": .5}, {"pre": -1},
    {"post": -1}, {"min_spacing": 0}, {"min_spacing": float("nan")}, {"post": True},
])
def test_invalid_parameters_raise_actionable_errors(kwargs):
    with pytest.raises(ValueError):
        detect_candidates(impulses(), **kwargs)


@pytest.mark.parametrize("signal", [np.zeros((20, 2)), np.array([0, np.nan]),
                                    np.array([0, np.inf]), np.array([1j]), np.array(["bad"])])
def test_invalid_signal_rejected(signal):
    with pytest.raises(ValueError, match="finite real mono"):
        detect_candidates(signal)


def test_quality_factors_are_finite_explicit_heuristics():
    rows = detect_candidates(impulses())
    for row in rows:
        factors = row["quality_factors"]
        assert factors["kind"] == "heuristic_review_priority"
        assert factors["version"] == 2
        assert factors["background_level"] > 0
        assert np.isfinite(factors["snr_db"])
        assert factors["competing_transients"] == 2
        assert 0 <= factors["spectral_distinctness"] <= 1
        assert 0 <= factors["clipping_fraction"] <= 1
    json.dumps(rows, allow_nan=False)


def test_local_background_and_clipping_reduce_quality_without_removing_candidate():
    clean = np.zeros(32000, np.float32)
    clean[16000:16020] = .8
    mixed = np.random.default_rng(7).normal(0, .04, 32000).astype(np.float32)
    mixed[16000:16020] = 1
    clean_row = min(detect_candidates(clean), key=lambda row: abs(row["start"] - 1))
    mixed_row = min(detect_candidates(mixed), key=lambda row: abs(row["start"] - 1))
    assert clean_row["start"] == mixed_row["start"] == 1
    assert clean_row["quality_score"] > mixed_row["quality_score"]
    assert clean_row["quality_factors"]["snr_db"] > mixed_row["quality_factors"]["snr_db"]
    assert mixed_row["quality_factors"]["clipping_fraction"] > 0
