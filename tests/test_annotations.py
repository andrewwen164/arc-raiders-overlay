import pytest

from arc_audio.annotations import remove_interval, write_interval
from arc_audio.dataset import reviewed_spans
from arc_audio.local_model import window_labels


def test_correcting_class_and_boundaries_invalidates_only_affected_reviews():
    clip = {"duration": 10, "track_confirmed": True,
            "intervals": [{"start": 2, "end": 4, "label": "gunshot", "proposal_id": "id"}],
            "reviewed": [{"start": 0, "end": 10, "classes": ["gunshot", "reload", "footsteps"]}]}
    changed = write_interval(clip, 5, 6, "reload", "teammate", index=0)
    assert changed["proposal_id"] == "id"
    assert reviewed_spans(clip, ("footsteps",)) == [{"start": 0, "end": 10}]
    assert reviewed_spans(clip, ("gunshot",)) == [{"start": 0, "end": 2}, {"start": 4, "end": 10}]
    assert reviewed_spans(clip, ("reload",)) == [{"start": 0, "end": 5}, {"start": 6, "end": 10}]
    assert window_labels(2, 3, clip, ("gunshot",)) is None
    assert window_labels(0, 1, clip, ("footsteps",)).tolist() == [0]


def test_source_only_edit_preserves_fully_reviewed_coverage():
    clip = {"duration": 10, "track_confirmed": True,
            "intervals": [{"start": 2, "end": 4, "label": "gunshot"}],
            "reviewed": [{"start": 0, "end": 10, "classes": ["gunshot"]}]}
    write_interval(clip, 2, 4, "gunshot", "teammate", index=0)
    assert reviewed_spans(clip, ("gunshot",)) == [{"start": 0, "end": 10}]


def test_manual_add_and_delete_do_not_create_unchecked_negative_labels():
    clip = {"duration": 10, "track_confirmed": True, "intervals": [],
            "reviewed": [{"start": 0, "end": 10}]}  # Legacy core-only scope.
    interval = write_interval(clip, 2, 4, "gunshot")
    remove_interval(clip, interval)
    assert not clip["intervals"]
    assert window_labels(2, 3, clip, ("gunshot",)) is None
    assert reviewed_spans(clip, ("reload",)) == [{"start": 0, "end": 10}]
    assert reviewed_spans(clip, ("snap_hook",)) == []


def test_invalid_or_unconfirmed_edit_preserves_existing_labels():
    clip = {"duration": 4, "track_confirmed": True, "intervals": [], "reviewed": []}
    for values in [(2, 1, "gunshot"), (0, 1, "enemy")]:
        with pytest.raises(ValueError):
            write_interval(clip, *values)
        assert not clip["intervals"]
    clip["track_confirmed"] = False
    with pytest.raises(ValueError, match="confirm"):
        write_interval(clip, 0, 1, "gunshot")
