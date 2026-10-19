"""Qt regression checks with synthetic local media and disposable manifests."""

import time
import wave

import numpy as np
import pytest

from arc_audio.dataset import save_dataset


@pytest.fixture
def label_window(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from arc_audio.labeler import LabelWindow

    tmp_path = tmp_path / "dataset"
    tmp_path.mkdir()
    media = tmp_path / "tone.wav"
    with wave.open(str(media), "wb") as output:
        output.setparams((2, 2, 16000, 0, "NONE", "not compressed"))
        output.writeframes(np.zeros((16000 * 10, 2), "<i2").tobytes())
    clip = {"file": media.name, "duration": 10, "size": media.stat().st_size, "group": "fixture",
            "split": "train", "track_confirmed": True, "audio_stream": 0,
            "audio_streams": [{"index": 0, "channels": 2}], "reviewed": [],
            "intervals": [{"start": 1, "end": 2, "label": "gunshot", "source": "unknown"},
                          {"start": 3, "end": 4, "label": "footsteps", "source": "unknown"}]}
    manifest = tmp_path / "dataset.json"
    save_dataset(manifest, {"version": 1, "clips": [clip]})
    app = QApplication.instance() or QApplication([])
    window = LabelWindow(manifest)
    actual_save = window.save
    monkeypatch.setattr(window, "preview_interval", lambda _row: None)
    yield window
    window.save = actual_save
    window.close()
    window.deleteLater()
    app.processEvents()


def test_failed_sorted_edit_rolls_back_and_cannot_target_a_different_row(label_window, monkeypatch):
    window = label_window
    window.table.selectRow(0)
    window.edit_selected_interval()
    window.start.setValue(5)
    window.end.setValue(6)
    actual_save = window.save
    monkeypatch.setattr(window, "save", lambda: False)
    window.save_interval_changes()
    assert [(row["start"], row["label"]) for row in window.clip["intervals"]] == [(1, "gunshot"), (3, "footsteps")]
    assert window.editing_interval_index is None
    monkeypatch.setattr(window, "save", actual_save)
    window.start.setValue(7)
    window.end.setValue(8)
    window.save_interval_changes()  # No stale editor index remains.
    assert window.clip["intervals"][1]["label"] == "footsteps"
    window.table.selectRow(0)
    window.edit_selected_interval()
    window.start.setValue(5)
    window.end.setValue(6)
    window.save_interval_changes()
    assert [(row["start"], row["label"]) for row in window.clip["intervals"]] == [(3, "footsteps"), (5, "gunshot")]


def test_failed_delete_restores_table_and_annotations(label_window, monkeypatch):
    window = label_window
    window.table.selectRow(0)
    monkeypatch.setattr(window, "save", lambda: False)
    window.delete_rows()
    assert len(window.clip["intervals"]) == window.table.rowCount() == 2
    assert window.row_keys == [("intervals", 0), ("intervals", 1)]


def test_manual_interval_add_is_visible_without_ai_suggestions(label_window):
    window = label_window
    window.review_tabs.setCurrentIndex(1)
    window.start.setValue(5)
    window.end.setValue(6)
    window.labels.setCurrentIndex(window.labels.findData("reload"))
    window.add_interval()
    assert window.clip["intervals"][-1]["label"] == "reload"
    assert window.review_tabs.currentIndex() == 0
    assert window.table.rowCount() == 3


def test_sidecar_save_failure_clears_sorted_editor_and_keeps_saved_labels(label_window, monkeypatch):
    from arc_audio.dataset import load_dataset
    from arc_audio.suggestions import media_stamp
    window = label_window
    window.clip["intervals"][0]["proposal_id"] = "linked"
    window.proposals = {"version": 1, "clips": [{"file": "tone.wav", "duration": 10, "audio_stream": 0,
                       "media": media_stamp(window.manifest.parent / "tone.wav"), "suggestions": [
                           {"id": "linked", "start": 1, "end": 2, "label": "gunshot",
                            "confidence": .8, "status": "accepted"}]}]}
    assert window.save()
    window.save_proposals()
    window.refresh_table()
    window.table.selectRow(0)
    window.edit_selected_interval()
    window.start.setValue(5)
    window.end.setValue(6)

    def fail():
        raise OSError("simulated sidecar write failure")

    monkeypatch.setattr(window, "save_proposals", fail)
    window.save_interval_changes()
    assert "Sound labels were saved" in window.status.text()
    assert window.editing_interval_index is None
    assert [(r["start"], r["label"]) for r in load_dataset(window.manifest)["clips"][0]["intervals"]] == [(3, "footsteps"), (5, "gunshot")]
    window.start.setValue(7)
    window.end.setValue(8)
    window.save_interval_changes()
    assert window.clip["intervals"][0]["label"] == "footsteps"


def test_background_import_updates_dropdown_and_preserves_labels(label_window, tmp_path):
    import shutil

    from PySide6.QtWidgets import QApplication
    window = label_window
    external = tmp_path / "new-recording.wav"
    shutil.copyfile(window.manifest.parent / "tone.wav", external)
    window.start_media_import([str(external)], "new-independent-session")
    deadline = time.monotonic() + 10
    while window.import_thread is not None and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(.005)
    assert window.import_thread is None, "Import worker did not shut down"
    assert window.clips.count() == 2
    added = window.document["clips"][1]
    assert window.clips.currentIndex() == 1
    assert (window.manifest.parent / added["file"]).is_file()
    assert added["group"] == "new-independent-session" and added["split"] == "train"
    assert not added["track_confirmed"] and not added["intervals"] and not added["reviewed"]
    assert len(window.document["clips"][0]["intervals"]) == 2


def seed_proposals(window, status="pending"):
    from arc_audio.suggestions import media_stamp
    window.proposals = {"version": 1, "clips": [{"file": "tone.wav", "duration": 10, "audio_stream": 0,
                       "media": media_stamp(window.manifest.parent / "tone.wav"), "suggestions": [
                           {"id": "first", "start": 5, "end": 6, "label": "gunshot",
                            "confidence": .8, "status": status},
                           {"id": "second", "start": 7, "end": 8, "label": "reload",
                            "confidence": .7, "status": "pending"}]}]}
    window.save_proposals()
    window.refresh_suggestions()


@pytest.mark.parametrize("status", ["pending", "rejected", "accepted"])
def test_single_selection_then_accept_button_works_without_review_step(label_window, status):
    from arc_audio.dataset import load_dataset
    from arc_audio.suggestions import load_suggestions
    window = label_window
    seed_proposals(window, status)
    window.suggestion_table.selectRow(0)  # Same path as a single row click, no Review call.
    assert window.start.value() == 5 and window.end.value() == 6
    assert window.labels.currentData() == "gunshot"
    assert window.accept_button.isEnabled()
    window.accept_button.click()
    assert load_suggestions(window.manifest)["clips"][0]["suggestions"][0]["status"] == "accepted"
    linked = [r for r in load_dataset(window.manifest)["clips"][0]["intervals"] if r.get("proposal_id") == "first"]
    assert len(linked) == 1 and linked[0]["start"] == 5 and linked[0]["end"] == 6
    assert window.suggestion_table.item(0, 4).text() == "accepted"


def test_accept_explains_missing_track_confirmation_and_enables_after_check(label_window):
    window = label_window
    window.confirmed.setChecked(False)
    seed_proposals(window)
    window.suggestion_table.selectRow(0)
    assert not window.accept_button.isEnabled()
    assert "tick 'I listened and confirmed this is game audio'" in window.accept_hint.text()
    window.accept_button.click()
    assert len(window.clip["intervals"]) == 2 and window.suggestion_rows[0]["status"] == "pending"
    window.confirmed.setChecked(True)
    assert window.accept_button.isEnabled()
    window.accept_button.click()
    assert len(window.clip["intervals"]) == 3 and window.suggestion_rows[0]["status"] == "accepted"


def test_switching_suggestion_rows_cannot_accept_the_previous_rows_fields(label_window):
    window = label_window
    seed_proposals(window)
    window.suggestion_table.selectRow(0)
    window.start.setValue(5.2)
    window.suggestion_table.selectRow(1)
    assert (window.start.value(), window.end.value(), window.labels.currentData()) == (7, 8, "reload")
    window.accept_button.click()
    assert window.suggestion_rows[0]["status"] == "pending"
    assert window.suggestion_rows[1]["status"] == "accepted"
    assert window.clip["intervals"][-1]["proposal_id"] == "second"


def test_clicking_another_column_in_same_row_preserves_draft_edits(label_window):
    window = label_window
    seed_proposals(window)
    window.suggestion_table.setCurrentCell(0, 0)
    window.start.setValue(5.2)
    window.suggestion_table.setCurrentCell(0, 3)
    assert window.start.value() == 5.2
    window.accept_button.click()
    assert window.clip["intervals"][-1]["start"] == 5.2


def test_whole_clip_review_uses_clip_bounds_and_selected_classes_without_changing_labels(label_window, monkeypatch):
    import copy

    from PySide6.QtWidgets import QMessageBox

    from arc_audio.dataset import reviewed_spans
    window = label_window
    for label, checkbox in window.review_checks.items():
        checkbox.setChecked(label == "gunshot")
    window.start.setValue(3)
    window.end.setValue(3.5)  # Last event editor bounds must not constrain whole-clip review.
    labels = copy.deepcopy(window.clip["intervals"])
    prompts = []

    def confirm(_parent, _title, message):
        prompts.append(message)
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", confirm)
    window.whole_review_button.click()
    assert "0.00 to 10.00 seconds (whole clip)" in prompts[0]
    assert reviewed_spans(window.clip, ("gunshot",)) == [{"start": 0, "end": 10}]
    assert reviewed_spans(window.clip, ("footsteps",)) == []
    assert window.clip["intervals"] == labels
    assert (window.start.value(), window.end.value()) == (3, 3.5)
    assert "10.00 / 10.00 seconds" in window.review_coverage.text()


def test_cancelled_whole_clip_review_does_not_invent_coverage(label_window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "question", lambda *_args: QMessageBox.StandardButton.No)
    label_window.whole_review_button.click()
    assert not label_window.clip["reviewed"]


def test_coverage_readout_exposes_labels_outside_reviewed_ranges(label_window):
    window = label_window
    for label, checkbox in window.review_checks.items():
        checkbox.setChecked(label == "gunshot")
    window.clip["reviewed"] = [{"start": 3, "end": 4, "classes": ["gunshot"]}]
    window.refresh_table()
    assert "1.00 / 10.00 seconds" in window.review_coverage.text()
    assert "0/1 sound labels fully inside" in window.review_coverage.text()
