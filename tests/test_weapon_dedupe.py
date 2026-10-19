"""Acoustic grouping mechanics, not evidence that real recordings are duplicates."""

from copy import deepcopy

import numpy as np
import pytest

from arc_audio.weapon_dedupe import NEIGHBORS, annotate_duplicates


def row(identity, vector=None, at=1):
    return {"id": identity, "source_timestamp": at, "start": at, "end": at + .15,
            "features": {"log_mel_summary": list(np.arange(16) if vector is None else vector)},
            "status": "accepted", "last_edit": {"label": "gunshot", "source": "teammate"}}


def entry(file, *rows):
    return {"file": file, "audio_stream": 1, "suggestions": list(rows)}


def test_feature_copies_group_across_files_but_unrelated_do_not():
    a, b, unrelated = row("a"), row("b", np.arange(16) * 3 + 8), row("other", -np.arange(16))
    entries = [entry("a.mp4", a, unrelated), entry("b.mp4", b)]
    report = annotate_duplicates(entries)
    assert report["duplicate_groups"] == 1
    assert report["duplicate_candidates"] == 2
    assert a["duplicate_group_id"] == b["duplicate_group_id"]
    assert a["duplicate_similarity"] == pytest.approx(1)
    assert unrelated["duplicate_group_id"] is None
    assert unrelated["duplicate_similarity"] is None


def test_same_burst_distinct_shots_are_never_collapsed():
    a, b, repeated = row("a", at=1), row("b", at=1.07), row("again", at=3)
    for item in (a, b):
        item["burst_group_id"] = "burst-1"
    report = annotate_duplicates([entry("a.mp4", a, b, repeated)])
    assert report["duplicate_groups"] == 0
    assert all(item["duplicate_group_id"] is None for item in (a, b, repeated))


def test_same_file_nearly_same_onset_can_be_flagged():
    a, b = row("a", at=1), row("b", at=1.02)
    report = annotate_duplicates([entry("a.mp4", a, b)])
    assert report["duplicate_candidates"] == 2


def test_cross_file_bridge_cannot_group_separate_shots_in_same_recording():
    a, b, bridge = row("a", at=1), row("b", at=1.07), row("bridge", at=10)
    a["burst_group_id"] = b["burst_group_id"] = "burst-1"
    report = annotate_duplicates([entry("a.mp4", a, b), entry("b.mp4", bridge)])
    assert report["duplicate_candidates"] == 2
    assert a["duplicate_group_id"] is None or b["duplicate_group_id"] is None
    assert a["duplicate_group_id"] != b["duplicate_group_id"]


def test_status_ids_edits_all_candidates_and_wrapper_fields_are_preserved():
    entries = [entry("a.mp4", row("a")), entry("b.mp4", row("b"))]
    entries[1]["suggestions"][0]["status"] = "rejected"
    original = deepcopy(entries)
    annotate_duplicates(entries)
    stripped = deepcopy(entries)
    for clip in stripped:
        for item in clip["suggestions"]:
            del item["duplicate_group_id"], item["duplicate_similarity"]
    assert stripped == original


def test_rerun_and_reordered_entries_keep_same_group_identity():
    entries = [entry("a.mp4", row("a")), entry("b.mp4", row("b")), entry("c.mp4", row("c"))]
    report = annotate_duplicates(entries)
    original = deepcopy(entries)
    assert annotate_duplicates(entries) == report
    assert entries == original
    annotate_duplicates(list(reversed(entries)))
    assert entries == original


def test_malformed_missing_constant_and_nonfinite_features_are_skipped():
    malformed = [{"id": "missing"}, row("short", [1]), row("nan", [np.nan] * 16),
                 row("constant", [2] * 16), row("text", ["x"] * 16), {"features": None},
                 row("nested", [[1] * 16]), row("inf", [np.inf] * 16)]
    entries = [entry("bad.mp4", *malformed)]
    report = annotate_duplicates(entries)
    assert report["eligible_candidates"] == 0
    assert report["skipped_candidates"] == len(malformed)
    assert all("duplicate_group_id" not in item for item in malformed)


def test_singleton_stale_duplicate_markers_are_cleared_without_removing_row():
    a = row("a")
    a.update(duplicate_group_id="old", duplicate_similarity=.999)
    entries = [entry("one.mp4", a)]
    assert annotate_duplicates(entries)["duplicate_groups"] == 0
    assert a["duplicate_group_id"] is a["duplicate_similarity"] is None
    assert len(entries[0]["suggestions"]) == 1


def test_neighbor_queries_have_linear_comparison_budget():
    rng = np.random.default_rng(82)
    entries = [entry(f"{i}.mp4", row(str(i), rng.normal(size=16))) for i in range(400)]
    report = annotate_duplicates(entries, threshold=.1)
    assert report["comparisons"] <= len(entries) * NEIGHBORS


@pytest.mark.parametrize("threshold", [0, -1, 1.1, float("nan"), float("inf"), True, "bad"])
def test_invalid_threshold_is_rejected_before_mutation(threshold):
    entries = [entry("a.mp4", row("a"))]
    original = deepcopy(entries)
    with pytest.raises(ValueError, match="threshold"):
        annotate_duplicates(entries, threshold)
    assert entries == original


def test_no_entries_is_valid_and_extreme_finite_vectors_do_not_overflow():
    assert annotate_duplicates([])["eligible_candidates"] == 0
    rows = [row("large-a", [float(i - 8) * 1e306 for i in range(16)]),
            row("large-b", np.arange(16) * 1e306)]
    report = annotate_duplicates([entry("a.mp4", rows[0]), entry("b.mp4", rows[1])])
    assert report["duplicate_groups"] == 1
