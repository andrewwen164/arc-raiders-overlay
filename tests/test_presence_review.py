import copy

import pytest

from arc_audio.dataset import load_dataset, save_dataset
from arc_audio.presence_review import (
    answer_presence,
    bind_presence_media,
    presence_answer,
    presence_summary,
    presence_training_rows,
    review_window,
    set_review_split,
    undo_presence,
    validate_presence_reviews,
)


def sample():
    clip = {"file": "synthetic.wav", "duration": 5., "audio_stream": 1, "audio_streams": [{"index": 1}],
            "group": "session", "split": "train", "track_confirmed": True,
            "intervals": [{"start": 4, "end": 4.5, "label": "footsteps"}], "reviewed": []}
    row = {"id": "fixture", "start": 2., "end": 2.15, "source_timestamp": 2., "label": "gunshot",
           "status": "pending", "proposed_weapon": "ferro"}
    return clip, row


def test_questions_record_exact_window_presence_without_event_times_or_neighbor_truth():
    clip, row = sample()
    old_events = copy.deepcopy(clip["intervals"])
    answer_presence(clip, row, "yes")
    review = clip["presence_reviews"][0]
    assert review["end"] - review["start"] == 1
    assert review["present"] is True and "weapon" not in review
    assert clip["intervals"] == old_events and not clip["reviewed"]
    assert row["status"] == "pending"
    answer_presence(clip, row, "no")
    assert len(clip["presence_reviews"]) == 1 and not clip["presence_reviews"][0]["present"]
    assert len(presence_training_rows(clip, ("gunshot",))) == 1
    assert not presence_training_rows(clip, ("footsteps", "gunshot"))
    answer_presence(clip, row, "unsure")
    assert not clip["presence_reviews"] and presence_answer(clip, row) == "unsure"
    assert clip["intervals"] == old_events and not clip["reviewed"]
    undo_presence(clip, row)
    assert presence_answer(clip, row) is None


def test_optional_weapon_is_a_human_positive_choice_only():
    clip, row = sample()
    answer_presence(clip, row, "yes", weapon="stitcher")
    assert clip["presence_reviews"][0]["weapon"] == "stitcher"
    original = copy.deepcopy((clip, row))
    with pytest.raises(ValueError, match="Only a Yes"):
        answer_presence(clip, row, "no", weapon="stitcher")
    assert (clip, row) == original


def test_identical_audio_questions_share_one_durable_answer_and_undo():
    clip, row = sample()
    other = dict(row, id="second-candidate")
    answer_presence(clip, row, "yes")
    assert presence_answer(clip, other) == "yes"
    answer_presence(clip, other, "no")
    assert presence_answer(clip, row) == "no" and len(clip["presence_reviews"]) == 1
    assert set(clip["presence_reviews"][0]["proposal_ids"]) == {row["id"], other["id"]}
    # The informational sidecar may still say Yes after a failed sidecar save.
    assert row["quick_review"]["answer"] == "yes"
    undo_presence(clip, other)
    assert presence_answer(clip, row) is None


def test_pending_sidecar_reconciles_with_dataset_after_partial_save():
    clip, row = sample()
    old_pending = copy.deepcopy(row)
    answer_presence(clip, row, "yes")
    assert presence_answer(clip, old_pending) == "yes"
    undo_presence(clip, old_pending)
    assert not clip["presence_reviews"]


def test_unsure_and_undo_are_durable_even_when_sidecar_write_fails():
    clip, row = sample()
    pending_sidecar = copy.deepcopy(row)
    answer_presence(clip, row, "unsure")
    assert presence_answer(clip, pending_sidecar) == "unsure"
    stale_unsure_sidecar = copy.deepcopy(row)
    undo_presence(clip, row)
    assert presence_answer(clip, stale_unsure_sidecar) is None
    validate_presence_reviews(clip)
    assert not presence_training_rows(clip, ("gunshot",))


def test_media_binding_refuses_replaced_source_without_changing_answers(tmp_path):
    clip, row = sample()
    path = tmp_path / "fixture.wav"
    path.write_bytes(b"original fixture")
    bind_presence_media(clip, path)
    answer_presence(clip, row, "no")
    before = copy.deepcopy(clip)
    path.write_bytes(b"modified fixture with different bytes")
    with pytest.raises(ValueError, match="Media changed"):
        bind_presence_media(clip, path)
    assert clip == before


@pytest.mark.parametrize("onset", [0, .01, 4.999])
def test_edge_questions_are_still_exact_complete_seconds(onset):
    clip, row = sample()
    window = review_window(clip, dict(row, source_timestamp=onset))
    assert 0 <= window["start"] < window["end"] <= clip["duration"]
    assert round(window["end"] * 16000) - round(window["start"] * 16000) == 16000


def test_confirmation_and_manual_truth_conflicts_do_not_destroy_existing_labels():
    clip, row = sample()
    clip["track_confirmed"] = False
    with pytest.raises(ValueError, match="game-audio"):
        answer_presence(clip, row, "yes")
    clip["track_confirmed"] = True
    clip["intervals"].append({"start": 2, "end": 2.15, "label": "gunshot"})
    original = copy.deepcopy((clip, row))
    with pytest.raises(ValueError, match="conflicts"):
        answer_presence(clip, row, "no")
    assert (clip, row) == original
    clip["intervals"].pop()
    clip["reviewed"] = [{"start": 0, "end": 5, "classes": ["gunshot"]}]
    with pytest.raises(ValueError, match="conflicts"):
        answer_presence(clip, row, "yes")
    answer_presence(clip, row, "unsure")
    assert clip["reviewed"] == [{"start": 0, "end": 5, "classes": ["gunshot"]}]


@pytest.mark.parametrize("change", [{"present": 1}, {"end": 2.4}, {"audio_stream": 2},
                                    {"label": "footsteps"}, {"proposal_ids": []},
                                    {"present": False, "weapon": "ferro"}])
def test_invalid_presence_data_is_rejected_on_dataset_load(tmp_path, change):
    clip, row = sample()
    answer_presence(clip, row, "yes")
    clip["presence_reviews"][0].update(change)
    path = tmp_path / "dataset.json"
    save_dataset(path, {"version": 1, "clips": [clip]})
    with pytest.raises(ValueError):
        load_dataset(path)


def test_duplicate_window_rows_are_rejected():
    clip, row = sample()
    answer_presence(clip, row, "yes")
    clip["presence_reviews"].append(dict(clip["presence_reviews"][0], id="different-id"))
    with pytest.raises(ValueError, match="unique"):
        validate_presence_reviews(clip)


def test_explicit_role_choice_moves_whole_group_but_never_test():
    clip, row = sample()
    answer_presence(clip, row, "no")
    sibling = dict(clip, file="sibling.wav", presence_reviews=[])
    heldout = dict(clip, file="test.wav", group="heldout", split="test", presence_reviews=[])
    document = {"clips": [clip, sibling, heldout]}
    set_review_split(document, clip, "validation")
    assert clip["split"] == sibling["split"] == "validation" and heldout["split"] == "test"
    assert presence_summary(document)["validation"] == {"yes": 0, "no": 1}
    with pytest.raises(ValueError, match="reserved test"):
        set_review_split(document, heldout, "train")
