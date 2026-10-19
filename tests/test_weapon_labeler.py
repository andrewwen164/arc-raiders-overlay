"""Weapon review stays in the existing labeler and creates only human labels."""

import copy
import shutil
import wave

import pytest

from arc_audio.dataset import load_dataset, save_dataset
from arc_audio.suggestions import load_suggestions, media_stamp, suggestion_path


@pytest.fixture
def weapon_window(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from arc_audio.labeler import LabelWindow

    media = tmp_path / "source.wav"
    with wave.open(str(media), "wb") as output:
        output.setparams((2, 2, 16000, 0, "NONE", "not compressed"))
        output.writeframes(bytes(16000 * 4 * 4))
    manifest = tmp_path / "dataset.json"
    clip = {"file": media.name, "duration": 4, "size": media.stat().st_size,
            "group": "source-video-1", "split": "train", "track_confirmed": True,
            "audio_stream": 0, "audio_streams": [{"index": 0, "channels": 2}],
            "intervals": [], "reviewed": []}
    second_media = tmp_path / "z-source.wav"
    shutil.copyfile(media, second_media)
    second_clip = dict(copy.deepcopy(clip), file=second_media.name, group="source-video-2")
    save_dataset(manifest, {"version": 1, "weapons": ["anvil", "ferro", "stitcher"], "clips": [clip, second_clip]})
    proposal = {"id": "shot-1", "start": 1, "end": 1.15, "preview_start": .65,
                "preview_end": 2, "label": "gunshot", "proposed_weapon": "ferro",
                "source_timestamp": 143.1, "source_url": "https://example.com/source",
                "source_title": "Weapon showcase", "manifest_quality": "clean",
                "manifest_notes": "Firing demonstration", "confidence": .8,
                "detection_score": .8, "quality_score": 75, "duplicate_group_id": "dup-1",
                "status": "pending"}
    save_dataset(suggestion_path(manifest), {"version": 1, "clips": [
        {"file": media.name, "duration": 4, "audio_stream": 0, "media": media_stamp(media),
         "suggestions": [proposal]},
        {"file": second_media.name, "duration": 4, "audio_stream": 0, "media": media_stamp(second_media),
         "suggestions": [dict(proposal, id="shot-2", proposed_weapon="anvil", quality_score=30,
                              detection_score=.4, duplicate_group_id="dup-2")]}]})
    app = QApplication.instance() or QApplication([])
    window = LabelWindow(manifest)
    monkeypatch.setattr(window, "preview_interval", lambda _row: None)
    yield window
    window.close()
    window.deleteLater()
    app.processEvents()


def test_manifest_weapon_and_source_context_are_visible(weapon_window):
    window = weapon_window
    window.suggestion_table.selectRow(0)
    assert window.weapons.currentData() == "ferro"
    assert window.weapons.findData("stitcher") >= 0
    for value in ("Weapon showcase", "143.10s", "https://example.com/source", "Firing demonstration", "75", "dup-1"):
        assert value in window.source_details.text()


def test_accept_weapon_then_correct_and_reject_is_reversible(weapon_window):
    window = weapon_window
    window.suggestion_table.selectRow(0)
    window.accept_button.click()
    assert window.clip["intervals"][0]["weapon"] == "ferro"
    assert not window.clip["reviewed"]
    window.suggestion_table.selectRow(0)
    window.weapons.setCurrentIndex(window.weapons.findData("stitcher"))
    window.save_weapon_button.click()
    saved = load_dataset(window.manifest)["clips"][0]
    assert len(saved["intervals"]) == 1 and saved["intervals"][0]["weapon"] == "stitcher"
    assert load_suggestions(window.manifest)["clips"][0]["suggestions"][0]["status"] == "corrected"
    assert not saved["reviewed"]
    window.suggestion_table.selectRow(0)
    window.reject_selected_suggestion()
    assert not window.clip["intervals"]
    window.suggestion_table.selectRow(0)
    window.accept_button.click()
    assert window.clip["intervals"][0]["weapon"] == "stitcher"


def test_unknown_can_clear_weapon_without_becoming_background(weapon_window):
    window = weapon_window
    window.suggestion_table.selectRow(0)
    window.weapons.setCurrentIndex(0)
    window.accept_button.click()
    assert window.clip["intervals"][0]["label"] == "gunshot"
    assert not window.clip["intervals"][0].get("weapon")
    assert not window.clip["reviewed"]


def test_weapon_editing_preserves_human_choice_and_track_gate(weapon_window):
    window = weapon_window
    window.suggestion_table.selectRow(0)
    window.confirmed.setChecked(False)
    assert not window.save_weapon_button.isEnabled()
    window.confirmed.setChecked(True)
    window.accept_button.click()
    window.table.selectRow(0)
    window.edit_selected_interval()
    assert window.weapons.currentData() == "ferro"
    window.end.setValue(1.25)
    window.save_interval_changes()
    assert window.clip["intervals"][0]["weapon"] == "ferro"
    assert window.clip["intervals"][0]["end"] == 1.25


def test_global_navigation_respects_weapon_source_and_quality_filters_without_saving(weapon_window):
    window = weapon_window
    before = window.manifest.read_bytes(), suggestion_path(window.manifest).read_bytes()
    window.autoplay.setChecked(False)
    window.suggestion_table.selectRow(0)
    window.navigate_candidate(1)
    assert window.clip["file"] == "z-source.wav" and window.active_suggestion_id == "shot-2"
    window.weapon_filter.setCurrentIndex(window.weapon_filter.findData("ferro"))
    assert window.clip["file"] == "source.wav" and window.active_suggestion_id == "shot-1"
    window.weapon_filter.setCurrentIndex(0)
    window.source_filter.setCurrentIndex(window.source_filter.findData("z-source.wav"))
    assert window.active_suggestion_id == "shot-2"
    window.source_filter.setCurrentIndex(0)
    window.minimum_quality.setValue(50)
    assert window.active_suggestion_id == "shot-1" and len(window.candidate_queue()) == 1
    window.minimum_detection.setValue(.9)
    assert not window.candidate_queue() and window.suggestion_table.rowCount() == 0
    assert before == (window.manifest.read_bytes(), suggestion_path(window.manifest).read_bytes())


def test_auto_advance_crosses_sources_and_counts_saved_decisions(weapon_window):
    window = weapon_window
    proposal = window.proposals["clips"][0]["suggestions"][0]
    window.proposals["clips"][0]["suggestions"].append(dict(proposal, id="skip-rejected", start=2,
                                                          end=2.15, status="rejected"))
    window.save_proposals()
    window.refresh_suggestions()
    window.auto_advance.setChecked(True)
    window.autoplay.setChecked(False)
    window.status_filter.setCurrentIndex(window.status_filter.findData("pending"))
    window.suggestion_table.selectRow(0)
    window.accept_button.click()
    assert window.active_suggestion_id == "shot-2" and window.clip["file"] == "z-source.wav"
    window.reject_selected_suggestion()
    assert not window.candidate_queue()
    assert "1 accepted" in window.review_progress.text() and "2 rejected" in window.review_progress.text()
    saved = load_dataset(window.manifest)
    assert len(saved["clips"][0]["intervals"]) == 1 and not saved["clips"][1]["intervals"]
    assert all(not clip["reviewed"] for clip in saved["clips"])


def test_sort_and_duplicate_filters_choose_expected_candidate(weapon_window):
    window = weapon_window
    window.sort_candidates.setCurrentIndex(window.sort_candidates.findData("quality"))
    assert [row["id"] for _clip, row in window.candidate_queue()] == ["shot-1", "shot-2"]
    window.duplicate_filter.setCurrentIndex(window.duplicate_filter.findData("dup-2"))
    assert window.active_suggestion_id == "shot-2"
    assert "1 match filters" in window.review_progress.text()


def test_preview_uses_explicit_candidate_context_without_extra_padding(weapon_window, monkeypatch):
    from types import SimpleNamespace

    from PySide6.QtMultimedia import QMediaPlayer

    from arc_audio.labeler import LabelWindow

    window = weapon_window
    positions = []
    player = window.player
    monkeypatch.setattr(window, "player", SimpleNamespace(setPosition=positions.append, play=lambda: None,
                        mediaStatus=lambda: QMediaPlayer.MediaStatus.LoadedMedia))
    LabelWindow.preview_interval(window, {"start": 1, "end": 1.15, "preview_start": .65, "preview_end": 2})
    assert positions == [650] and window.preview_end == 2000
    LabelWindow.preview_interval(window, {"start": 1, "end": 1.15})
    assert positions[-1] == 500 and window.preview_end == 1650
    monkeypatch.setattr(window, "player", player)


def test_cross_source_autoplay_waits_for_media_before_seeking(weapon_window, monkeypatch):
    from types import SimpleNamespace

    from PySide6.QtMultimedia import QMediaPlayer

    from arc_audio.labeler import LabelWindow

    window = weapon_window
    calls = []
    player = window.player
    monkeypatch.setattr(window, "player", SimpleNamespace(
        setPosition=lambda position: calls.append(("seek", position)), play=lambda: calls.append(("play",)),
        setActiveAudioTrack=lambda track: calls.append(("track", track)),
        hasVideo=lambda: False,
        mediaStatus=lambda: QMediaPlayer.MediaStatus.LoadingMedia))
    LabelWindow.preview_interval(window, {"start": 1, "end": 1.1, "preview_start": .65, "preview_end": 2})
    assert not calls and window.preview_seek_pending == 650
    window.media_ready(QMediaPlayer.MediaStatus.LoadedMedia)
    assert calls == [("track", 0), ("seek", 650), ("play",)]
    assert window.preview_seek_pending is None
    monkeypatch.setattr(window, "player", player)


def test_shortcuts_work_in_review_table_but_not_while_typing(weapon_window, monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    window = weapon_window
    calls = []
    monkeypatch.setattr(window, "accept_selected_suggestion", lambda: calls.append("accept"))
    monkeypatch.setattr(window, "reject_selected_suggestion", lambda: calls.append("reject"))
    window.show()
    window.activateWindow()
    window.suggestion_table.setFocus()
    QApplication.processEvents()
    QTest.keyClick(window.suggestion_table, Qt.Key.Key_A)
    QTest.keyClick(window.suggestion_table, Qt.Key.Key_R)
    assert calls == ["accept", "reject"]
    window.start.setFocus()
    QTest.keyClick(window.start, Qt.Key.Key_A)
    window.weapons.setFocus()
    QTest.keyClick(window.weapons, Qt.Key.Key_R)
    assert calls == ["accept", "reject"]


def test_hidden_accepted_proposal_is_updated_when_saved_interval_is_corrected(weapon_window):
    window = weapon_window
    window.suggestion_table.selectRow(0)
    window.accept_button.click()
    window.status_filter.setCurrentIndex(window.status_filter.findData("pending"))
    window.clips.setCurrentIndex(0)
    assert not window.suggestion_rows
    window.table.selectRow(0)
    window.edit_selected_interval()
    window.weapons.setCurrentIndex(window.weapons.findData("stitcher"))
    window.save_interval_changes()
    proposal = load_suggestions(window.manifest)["clips"][0]["suggestions"][0]
    assert proposal["status"] == "corrected" and proposal["last_edit"]["weapon"] == "stitcher"


def test_hidden_accepted_proposal_is_reopened_when_saved_interval_is_deleted(weapon_window):
    window = weapon_window
    window.suggestion_table.selectRow(0)
    window.accept_button.click()
    window.status_filter.setCurrentIndex(window.status_filter.findData("pending"))
    window.clips.setCurrentIndex(0)
    assert not window.suggestion_rows
    window.table.selectRow(0)
    window.delete_rows()
    proposal = load_suggestions(window.manifest)["clips"][0]["suggestions"][0]
    assert proposal["status"] == "pending" and proposal["last_edit"]["weapon"] == "ferro"
    assert not load_dataset(window.manifest)["clips"][0]["intervals"]


def test_failed_scoped_decision_restores_current_source_and_preserves_unrelated_proposals(weapon_window, monkeypatch):
    window = weapon_window
    unrelated = window.proposals["clips"][1]
    original = copy.deepcopy(window.proposals)
    window.suggestion_table.selectRow(0)
    monkeypatch.setattr(window, "save", lambda: False)
    window.accept_button.click()
    assert window.proposals == original and not window.clip["intervals"]
    assert window.proposals["clips"][1] is unrelated


def test_audio_only_source_does_not_reopen_previous_video_pane(weapon_window, monkeypatch):
    from types import SimpleNamespace

    from PySide6.QtMultimedia import QMediaPlayer

    window = weapon_window
    player = window.player
    monkeypatch.setattr(window, "player", SimpleNamespace(hasVideo=lambda: True, setActiveAudioTrack=lambda _index: None))
    window.clip["collection_source_id"] = "remote-audio-source"
    window.media_ready(QMediaPlayer.MediaStatus.LoadedMedia)
    assert window.video.isHidden()
    window.clip["has_video"] = True
    window.media_ready(QMediaPlayer.MediaStatus.LoadedMedia)
    assert not window.video.isHidden()
    window.clip["has_video"] = False
    window.media_ready(QMediaPlayer.MediaStatus.LoadedMedia)
    assert window.video.isHidden()
    monkeypatch.setattr(window, "player", player)


def seed_background(window):
    proposal = window.proposals["clips"][0]["suggestions"][0]
    proposal.update(proposal_kind="gunshot_absence", proposed_weapon=None, possible_weapons=[],
                    confidence=0, detection_score=0, start=2, end=3, preview_start=2, preview_end=3,
                    source_timestamp=2)
    window.save_proposals()
    window.refresh_suggestions()
    window.suggestion_table.selectRow(0)


def test_background_accept_cancel_is_safe_and_defaults_to_no(weapon_window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    window = weapon_window
    seed_background(window)
    assert "Background / no gunfire" in window.suggestion_table.item(0, 0).text()
    assert "ENTIRE" in window.source_details.text().upper()
    assert not window.save_weapon_button.isEnabled() and not window.correct_weapon_button.isEnabled()
    prompts = []
    def cancel(*args):
        prompts.append(args)
        return QMessageBox.StandardButton.No
    monkeypatch.setattr(QMessageBox, "question", cancel)
    before = window.manifest.read_bytes(), suggestion_path(window.manifest).read_bytes()
    window.accept_button.click()
    assert prompts[0][-1] == QMessageBox.StandardButton.No
    assert before == (window.manifest.read_bytes(), suggestion_path(window.manifest).read_bytes())
    assert not window.clip["intervals"] and not window.clip["reviewed"]


def test_confirmed_background_reviews_gunfire_only_and_reverses_cleanly(weapon_window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from arc_audio.dataset import reviewed_spans

    window = weapon_window
    seed_background(window)
    window.clip["reviewed"] = [{"start": 0, "end": 4, "classes": ["footsteps"]}]
    window.save()
    monkeypatch.setattr(QMessageBox, "question", lambda *_args: QMessageBox.StandardButton.Yes)
    window.accept_button.click()
    assert not window.clip["intervals"]
    assert reviewed_spans(window.clip, ("gunshot",)) == [{"start": 2, "end": 3}]
    assert reviewed_spans(window.clip, ("reload",)) == []
    window.suggestion_table.selectRow(0)
    window.reopen_selected_suggestion()
    assert reviewed_spans(window.clip, ("gunshot",)) == []
    assert reviewed_spans(window.clip, ("footsteps",)) == [{"start": 0, "end": 4}]


def test_background_cannot_overwrite_existing_gunfire(weapon_window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    window = weapon_window
    seed_background(window)
    window.clip["intervals"].append({"start": 2.3, "end": 2.5, "label": "gunshot", "source": "unknown"})
    window.save()
    original = copy.deepcopy(window.clip["intervals"])
    monkeypatch.setattr(QMessageBox, "question", lambda *_args: QMessageBox.StandardButton.Yes)
    window.accept_button.click()
    assert window.clip["intervals"] == original and not window.clip["reviewed"]
    assert "overlaps a saved gunshot" in window.status.text()


def test_deleting_background_review_reopens_hidden_proposal(weapon_window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    window = weapon_window
    seed_background(window)
    monkeypatch.setattr(QMessageBox, "question", lambda *_args: QMessageBox.StandardButton.Yes)
    window.accept_button.click()
    window.status_filter.setCurrentIndex(window.status_filter.findData("pending"))
    window.clips.setCurrentIndex(0)
    assert not window.suggestion_rows
    window.table.selectRow(0)
    window.delete_rows()
    assert not window.clip["reviewed"]
    assert load_suggestions(window.manifest)["clips"][0]["suggestions"][0]["status"] == "pending"
