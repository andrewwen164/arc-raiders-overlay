"""Exercise review UI with explicit fixtures on a disposable pilot manifest."""

import copy
import json
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from arc_audio.dataset import load_dataset, save_dataset
from arc_audio.labeler import LabelWindow
from arc_audio.suggestions import media_stamp, suggestion_path

ROOT = Path(__file__).resolve().parents[1]


def main():
    original = ROOT / "1st test" / "dataset.json"
    before = original.read_bytes()
    scratch = original.with_name(".labeler-smoke.json")
    sidecar = suggestion_path(scratch)
    if scratch.exists() or sidecar.exists():
        raise RuntimeError("Inspect existing scratch smoke files before rerunning")
    doc = copy.deepcopy(load_dataset(original))
    clip = doc["clips"][0]
    doc["clips"] = [clip]
    clip.update(intervals=[], reviewed=[], track_confirmed=False)
    save_dataset(scratch, doc)
    # These are synthetic UI test proposals, not actual model predictions.
    rows = [{"id": f"ui-fixture-{index}", "start": start, "end": end, "label": label,
             "confidence": .7, "status": "pending", "provenance": {"model": "UI test fixture"}}
            for index, (start, end, label) in enumerate([(1, 3, "gunshot"), (4, 5, "reload")])]
    save_dataset(sidecar, {"version": 1, "clips": [{"file": clip["file"], "duration": clip["duration"],
                                                  "audio_stream": clip["audio_stream"],
                                                  "media": media_stamp(original.parent / clip["file"]),
                                                  "suggestions": rows}]})
    app = QApplication([])
    window = LabelWindow(scratch)
    window.output.setVolume(0)
    report = {"errors": [], "checks": {}}
    frames = [0]
    window.video.videoSink().videoFrameChanged.connect(
        lambda frame: frames.__setitem__(0, frames[0] + int(frame.isValid())))
    window.show()

    def begin():
        try:
            assert window.suggestion_table.rowCount() == 2
            assert [label for label, checkbox in window.review_checks.items() if checkbox.isChecked()] == ["gunshot"]
            report["checks"]["pilot_defaults_to_gunfire_review_only"] = True
            window.suggestion_table.selectRow(0)
            assert not window.accept_button.isEnabled()
            assert "tick 'I listened and confirmed this is game audio'" in window.accept_hint.text()
            window.review_suggestion()
            window.accept_selected_suggestion()
            assert not window.clip["intervals"]
            assert window.suggestion_rows[0]["status"] == "pending"
            report["checks"]["unconfirmed_acceptance_blocked"] = True
        except Exception as exc:
            report["errors"].append(repr(exc))

    def finish():
        try:
            report["checks"].update(position_ms=window.player.position(), video_frames=frames[0])
            assert frames[0] > 0 and window.player.position() > 1000
            window.player.pause()
            window.confirmed.setChecked(True)
            window.start.setValue(1.2)
            window.end.setValue(2.4)
            window.sources.setCurrentText("teammate")
            assert window.accept_button.isEnabled()
            window.accept_button.click()
            assert window.clip["intervals"][0]["start"] == 1.2
            assert window.clip["intervals"][0]["end"] == 2.4
            assert window.clip["intervals"][0]["source"] == "teammate"
            assert not window.clip["reviewed"]
            assert window.suggestion_table.item(0, 4).text() == "accepted"
            # Revisit an accepted row and update the linked interval, not duplicate it.
            window.suggestion_table.selectRow(0)
            assert window.start.value() == 1.2
            window.start.setValue(1.3)
            window.end.setValue(2.5)
            window.accept_button.click()
            assert len(window.clip["intervals"]) == 1
            assert window.clip["intervals"][0]["start"] == 1.3
            assert window.suggestion_table.item(0, 1).text() == "1.30"
            # Rejection of an accepted row removes only its linked label.
            window.suggestion_table.selectRow(0)
            window.reject_selected_suggestion()
            assert not window.clip["intervals"]
            window.suggestion_table.selectRow(0)
            assert window.start.value() == 1.3
            window.accept_button.click()
            report["checks"]["single_row_selection_then_accept_button"] = True
            window.suggestion_table.selectRow(0)
            window.reopen_selected_suggestion()
            assert not window.clip["intervals"]
            assert window.suggestion_table.item(0, 4).text() == "pending"
            window.suggestion_table.selectRow(0)
            window.review_suggestion()
            window.accept_selected_suggestion()
            report["checks"]["accepted_and_rejected_decisions_reversible"] = True
            # Manual intervals do not require any AI candidate.
            window.start.setValue(8)
            window.end.setValue(8.5)
            window.labels.setCurrentIndex(window.labels.findData("landing"))
            window.add_interval()
            assert window.review_tabs.currentIndex() == 0
            window.table.selectRow(1)
            window.edit_selected_interval()
            window.start.setValue(8.1)
            window.end.setValue(8.6)
            window.labels.setCurrentIndex(window.labels.findData("footsteps"))
            window.save_interval_changes()
            assert window.clip["intervals"][1]["label"] == "footsteps"
            window.table.selectRow(1)
            window.delete_rows()
            assert len(window.clip["intervals"]) == 1
            report["checks"]["manual_interval_add_edit_delete"] = True
            window.review_tabs.setCurrentIndex(1)
            window.suggestion_table.selectRow(1)
            window.reject_selected_suggestion()
            assert window.suggestion_table.item(1, 4).text() == "rejected"
            assert len(window.clip["intervals"]) == 1
            # A concurrent generator/reviewer must not lose its decisions.
            previous_sidecar = sidecar.read_bytes()
            changed = json.loads(previous_sidecar)
            changed["external_review_note"] = "concurrent smoke fixture"
            save_dataset(sidecar, changed)
            try:
                window.save_proposals()
            except ValueError:
                report["checks"]["concurrent_proposal_save_blocked"] = True
            else:
                raise AssertionError("Concurrent sidecar edits must not be overwritten")
            saved = json.loads(sidecar.read_text())
            assert [row["status"] for row in saved["clips"][0]["suggestions"]] == ["accepted", "rejected"]
            report["checks"]["edited_accept_reject_saved_without_review_scope"] = True
            assert not window.player.errorString(), window.player.errorString()
            (ROOT / "artifacts").mkdir(exist_ok=True)
            window.grab().save(str(ROOT / "artifacts" / "suggestions-review.png"))
        except Exception as exc:
            report["errors"].append(repr(exc))
        finally:
            window.close()
            assert original.read_bytes() == before
            report["checks"]["real_manifest_untouched"] = True
            (ROOT / "artifacts" / "suggestions-smoke.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2))
            scratch.unlink()
            sidecar.unlink()
            app.quit()

    QTimer.singleShot(500, begin)
    QTimer.singleShot(2500, finish)
    app.exec()
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
