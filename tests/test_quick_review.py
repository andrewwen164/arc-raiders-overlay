"""Disposable Qt review checks: human answers, exact windows and safe persistence."""

import copy
import wave

import numpy as np
import pytest

from arc_audio.dataset import load_dataset, save_dataset
from arc_audio.suggestions import media_stamp, suggestion_path


@pytest.fixture
def quick_window(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from arc_audio.quick_review import QuickReviewWindow

    media = tmp_path / "synthetic.wav"
    with wave.open(str(media), "wb") as stream:
        stream.setparams((2, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(np.zeros((16000 * 4, 2), "<i2").tobytes())
    clip = {"file": media.name, "duration": 4, "size": media.stat().st_size, "group": "synthetic",
            "split": "train", "track_confirmed": False, "audio_stream": 0,
            "audio_streams": [{"index": 0, "channels": 2}], "reviewed": [],
            "intervals": [{"start": 3, "end": 3.2, "label": "footsteps", "source": "unknown"}]}
    rows = [{"id": f"question-{i}", "start": start, "end": start + .15, "label": "gunshot",
             "confidence": .9 - i * .1, "status": "pending", "proposed_weapon": "ferro"}
            for i, start in enumerate((.3, 1.3, 2.3))]
    rows.append({"id": "legacy", "start": 3, "end": 3.2, "label": "gunshot", "confidence": .8,
                 "status": "accepted"})
    rows.append({"id": "background", "start": 2.8, "end": 3.9, "label": "gunshot", "confidence": .1,
                 "status": "pending", "proposal_kind": "gunshot_absence"})
    manifest = tmp_path / "dataset.json"
    save_dataset(manifest, {"version": 1, "weapons": ["ferro", "stitcher"], "clips": [clip]})
    save_dataset(suggestion_path(manifest), {"version": 1, "clips": [{"file": clip["file"], "duration": 4,
                  "audio_stream": 0, "media": media_stamp(media), "suggestions": rows}]})
    app = QApplication.instance() or QApplication([])
    window = QuickReviewWindow(manifest)
    window.output.setVolume(0)
    window.auto_advance.setChecked(False)
    window.player.stop()
    window.boundary_timer.stop()
    yield window
    window.close()
    window.deleteLater()
    app.processEvents()


def listened(window):
    window.heard = True
    window.update_controls()


def confirmed(window):
    listened(window)
    window.source_confirm_yes()


def test_questions_require_listening_and_explicit_game_track_confirmation(quick_window):
    w = quick_window
    assert not w.yes_button.isEnabled()
    w.save_answer("no")
    assert not w.clip.get("presence_reviews")
    w.source_confirm_yes()
    assert not w.clip["track_confirmed"]
    listened(w)
    assert w.confirm_button.isEnabled() and not w.no_button.isEnabled()
    w.source_confirm_yes()
    assert w.clip["track_confirmed"] and w.yes_button.isEnabled()
    assert load_dataset(w.manifest)["clips"][0]["track_confirmed"]


def test_yes_no_unsure_and_undo_never_write_event_intervals(quick_window):
    from arc_audio.presence_review import presence_answer

    w = quick_window
    original = copy.deepcopy(w.clip["intervals"])
    confirmed(w)
    w.save_answer("yes")
    assert presence_answer(w.clip, w.row) == "yes"
    assert w.clip["presence_reviews"][0].get("weapon") is None  # Source hint never becomes truth.
    w.save_answer("no")
    assert presence_answer(w.clip, w.row) == "no"
    w.save_answer("unsure")
    assert presence_answer(w.clip, w.row) == "unsure"
    assert not w.clip["presence_reviews"]
    w.undo_answer()
    assert presence_answer(w.clip, w.row) is None
    assert w.row["status"] == "pending"
    assert w.clip["intervals"] == original and not w.clip["reviewed"]


def test_optional_weapon_requires_human_multiple_choice(quick_window):
    w = quick_window
    confirmed(w)
    w.weapons.setCurrentIndex(w.weapons.findData("stitcher"))
    w.save_answer("yes")
    assert w.clip["presence_reviews"][0]["weapon"] == "stitcher"
    w.save_answer("no")  # No ignores the now-irrelevant optional weapon control.
    assert not w.clip["presence_reviews"][0]["present"]
    assert w.clip["presence_reviews"][0].get("weapon") is None


def test_previous_restores_answer_and_allows_reanswer(quick_window):
    from arc_audio.presence_review import presence_answer

    w = quick_window
    confirmed(w)
    w.auto_advance.setChecked(True)
    clip, row = w.clip, w.row
    w.save_answer("yes")
    assert w.row is not row
    w.previous()
    assert w.row is row and presence_answer(clip, row) == "yes"
    assert not w.no_button.isEnabled()  # Re-listen before changing truth.
    listened(w)
    w.auto_advance.setChecked(False)
    w.save_answer("no")
    assert presence_answer(clip, row) == "no"


def test_save_failure_restores_human_answers_and_candidate_metadata(quick_window, monkeypatch):
    import arc_audio.quick_review as module

    w = quick_window
    confirmed(w)
    before_clip, before_row = copy.deepcopy(w.clip), copy.deepcopy(w.row)
    monkeypatch.setattr(module, "save_dataset", lambda *_: (_ for _ in ()).throw(OSError("fixture disk full")))
    w.save_answer("yes")
    assert w.clip == before_clip and w.row == before_row
    assert "Not saved" in w.status.text()


def test_stale_window_cannot_overwrite_newer_dataset(quick_window):
    w = quick_window
    confirmed(w)
    newer = load_dataset(w.manifest)
    newer["newer_writer"] = True
    save_dataset(w.manifest, newer)
    w.save_answer("no")
    assert load_dataset(w.manifest)["newer_writer"]
    assert not w.clip.get("presence_reviews")
    assert "changed in another window" in w.status.text()


def test_background_is_a_question_and_legacy_decisions_are_excluded(quick_window):
    w = quick_window
    assert {row["id"] for _, row in w.queue} == {"question-0", "question-1", "question-2", "background"}
    w.select(len(w.queue) - 1)
    assert w.row["id"] == "background"
    assert "Any gunfire" in w.question.text()
    assert w.window_span["end"] - w.window_span["start"] == 1
    assert not w.clip.get("presence_reviews")


def test_role_choice_changes_group_only_on_explicit_activation(quick_window):
    w = quick_window
    assert w.clip["split"] == "train"
    w.role.setCurrentIndex(1)  # Programmatic display updates are not user decisions.
    assert w.clip["split"] == "train"
    w.change_role(1)
    assert w.clip["split"] == load_dataset(w.manifest)["clips"][0]["split"] == "validation"


def test_playback_failure_blocks_truth_even_after_previous_listen(quick_window):
    w = quick_window
    confirmed(w)
    w.playback_error(None, "fixture output disconnected")
    assert not w.no_button.isEnabled()
    w.save_answer("no")
    assert not w.clip.get("presence_reviews")


def test_changed_source_cannot_receive_a_human_answer(quick_window):
    w = quick_window
    confirmed(w)
    media = w.manifest.parent / w.clip["file"]
    media.write_bytes(media.read_bytes() + b"fixture changed")
    w.save_answer("yes")
    assert not w.clip.get("presence_reviews")
    assert "source audio or selected track changed" in w.status.text()


def test_weapon_choice_focus_disables_answer_and_navigation_shortcuts(quick_window):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    w = quick_window
    w.show()
    QApplication.processEvents()
    confirmed(w)
    w.weapons.setFocus()
    QApplication.processEvents()
    listened(w)
    assert all(not shortcut.isEnabled() for shortcut in w.shortcuts)
    QTest.keyClick(w.weapons, Qt.Key.Key_N)
    QTest.keyClick(w.weapons, Qt.Key.Key_Y)
    assert not w.clip.get("presence_reviews")
    w.update_shortcuts(None, w.yes_button)
    assert all(shortcut.isEnabled() for shortcut in w.shortcuts)


def test_seek_to_old_end_position_cannot_count_as_listened(quick_window):
    w = quick_window
    w.play_requested = True
    w.play_started_at = None
    w.sample_finished()
    assert not w.heard and not w.yes_button.isEnabled()
    assert "not played from its start" in w.status.text()


def test_dataset_answer_and_undo_survive_sidecar_save_failure(quick_window, monkeypatch):
    import arc_audio.quick_review as module
    from arc_audio.presence_review import presence_answer

    w = quick_window
    confirmed(w)
    w.save_answer("yes")
    actual_save = module.save_dataset

    def failed_sidecar(path, document):
        if path == suggestion_path(w.manifest):
            raise OSError("fixture candidate sidecar unavailable")
        return actual_save(path, document)

    monkeypatch.setattr(module, "save_dataset", failed_sidecar)
    w.save_answer("unsure")
    stored = load_dataset(w.manifest)["clips"][0]
    assert presence_answer(stored, w.row) == "unsure"
    assert "Human answer saved" in w.status.text()
    w.undo_answer()
    stored = load_dataset(w.manifest)["clips"][0]
    assert presence_answer(stored, w.row) is None
    reopened = module.QuickReviewWindow(w.manifest)
    assert presence_answer(reopened.clip, reopened.row) is None
    reopened.close()


def test_test_sources_are_never_queued(quick_window):
    w = quick_window
    w.queue = []
    w.clip["split"] = "test"
    w._build_queue()
    assert not w.queue
