import copy

import pytest

from arc_audio.dataset import merge_reviewed, reviewed_spans
from arc_audio.suggestions import accept_suggestion, change_suggestion_status


def fixture():
    clip = {"duration": 10, "track_confirmed": True, "intervals": [], "reviewed": []}
    row = {"id": "negative-fixture", "start": 2, "end": 4, "label": "gunshot",
           "status": "pending", "proposal_kind": "gunshot_absence", "confidence": 0}
    return clip, row


def test_background_acceptance_is_class_scoped_review_not_positive_interval():
    clip, row = fixture()
    accept_suggestion(clip, row, 2, 4, "gunshot")
    assert row["status"] == "accepted" and not clip["intervals"]
    assert reviewed_spans(clip, ("gunshot",)) == [{"start": 2, "end": 4}]
    assert not reviewed_spans(clip, ("footsteps",))
    accept_suggestion(clip, row, 2.5, 4.5, "gunshot")
    assert len(clip["reviewed"]) == 1
    assert reviewed_spans(clip, ("gunshot",)) == [{"start": 2.5, "end": 4.5}]
    change_suggestion_status(clip, row, "rejected")
    assert not clip["intervals"] and not clip["reviewed"]


def test_unconfirmed_and_conflicting_background_does_not_change_truth():
    clip, row = fixture()
    clip["track_confirmed"] = False
    with pytest.raises(ValueError, match="confirm"):
        accept_suggestion(clip, row, 2, 4, "gunshot")
    clip["track_confirmed"] = True
    clip["intervals"].append({"start": 3, "end": 3.2, "label": "gunshot"})
    before = copy.deepcopy((clip, row))
    with pytest.raises(ValueError, match="overlaps"):
        accept_suggestion(clip, row, 2, 4, "gunshot")
    assert (clip, row) == before
    with pytest.raises(ValueError, match="absence of gunfire only"):
        accept_suggestion(clip, row, 2, 4, "footsteps")
    assert (clip, row) == before


def test_undo_background_preserves_independent_manual_reviews():
    clip, row = fixture()
    manual = {"start": 0, "end": 10, "classes": ["gunshot", "footsteps"]}
    clip["reviewed"] = [copy.deepcopy(manual)]
    accept_suggestion(clip, row, 2, 4, "gunshot")
    change_suggestion_status(clip, row, "pending")
    assert clip["reviewed"] == [manual]
    # Current merges preserve linkage and independent manual coverage.
    accept_suggestion(clip, row, 2, 4, "gunshot")
    clip["reviewed"] = merge_reviewed(clip["reviewed"])
    change_suggestion_status(clip, row, "rejected")
    assert len(clip["reviewed"]) == 1
    assert reviewed_spans(clip, ("gunshot", "footsteps")) == [{"start": 0, "end": 10}]
    # Legacy/external metadata loss still takes the conservative fallback.
    accept_suggestion(clip, row, 2, 4, "gunshot")
    clip["reviewed"] = [{key: value for key, value in review.items() if key not in ("proposal_id", "review_kind")}
                        for review in clip["reviewed"]]
    clip["reviewed"] = merge_reviewed(clip["reviewed"])
    change_suggestion_status(clip, row, "rejected")
    assert reviewed_spans(clip, ("footsteps",)) == [{"start": 0, "end": 10}]
    assert reviewed_spans(clip, ("gunshot",)) == [{"start": 0, "end": 2}, {"start": 4, "end": 10}]


def test_interrupted_sidecar_save_can_remove_only_linked_absence_review():
    clip, row = fixture()
    pending_on_disk = copy.deepcopy(row)
    accept_suggestion(clip, row, 2.5, 4.5, "gunshot")
    change_suggestion_status(clip, pending_on_disk, "rejected")
    assert not clip["reviewed"] and not clip["intervals"]


def test_partial_save_then_unrelated_review_merge_does_not_leave_ghost_negative():
    clip, row = fixture()
    pending_on_disk = copy.deepcopy(row)
    accept_suggestion(clip, row, 2.5, 4.5, "gunshot")
    clip["reviewed"] = merge_reviewed(clip["reviewed"] + [{"start": 0, "end": 1, "classes": ["footsteps"]}])
    change_suggestion_status(clip, pending_on_disk, "rejected")
    assert not reviewed_spans(clip, ("gunshot",))
    assert reviewed_spans(clip, ("footsteps",)) == [{"start": 0, "end": 1}]


def test_partial_save_then_positive_edit_splits_linked_absence_durably():
    from arc_audio.annotations import write_interval

    clip, row = fixture()
    pending_on_disk = copy.deepcopy(row)
    accept_suggestion(clip, row, 2, 4, "gunshot")
    write_interval(clip, 2.8, 3.2, "gunshot")
    assert len(clip["reviewed"]) == 2
    change_suggestion_status(clip, pending_on_disk, "rejected")
    assert not reviewed_spans(clip, ("gunshot",))
    assert len(clip["intervals"]) == 1
