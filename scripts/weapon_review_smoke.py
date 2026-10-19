"""Muted real-source playback + disposable review decisions, never training truth."""

import argparse
import copy
import json
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import QApplication

from arc_audio.dataset import load_dataset, save_dataset
from arc_audio.labeler import LabelWindow
from arc_audio.suggestions import load_suggestions, suggestion_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    original = args.manifest.resolve()
    artifacts = Path(__file__).resolve().parents[1] / "artifacts" / "weapon-review-smoke"
    artifacts.mkdir(parents=True, exist_ok=True)
    before = original.read_bytes(), suggestion_path(original).read_bytes()
    scratch = original.with_name(".weapon-review-smoke.json")
    if scratch.exists() or suggestion_path(scratch).exists():
        raise RuntimeError("Inspect previous scratch files before retrying")
    doc = copy.deepcopy(load_dataset(original))
    doc["weapons"] = sorted(set(doc.get("weapons", [])) | {"ferro"})
    bundle = copy.deepcopy(load_suggestions(original))
    for entry in bundle["clips"]:
        entry["suggestions"] = [row for row in entry["suggestions"] if row.get("proposal_kind") != "gunshot_absence"]
    eligible = [clip for clip in doc["clips"] if any(
        entry["file"] == clip["file"] and len(entry["suggestions"]) >= 3
        and entry["suggestions"][0].get("proposed_weapon")
        for entry in bundle["clips"])]
    doc["clips"] = [min(eligible, key=lambda clip: clip["duration"])]
    clip = doc["clips"][0]
    clip.update(intervals=[], reviewed=[], track_confirmed=False)
    bundle["clips"] = [entry for entry in bundle["clips"] if entry["file"] == clip["file"]]
    rows = bundle["clips"][0]["suggestions"]
    if len(rows) < 3:
        raise ValueError("Smoke requires three proposals")
    for row in rows:
        row["status"] = "pending"
        row.pop("last_edit", None)
    save_dataset(scratch, doc)
    save_dataset(suggestion_path(scratch), bundle)
    app = QApplication([])
    window = LabelWindow(scratch)
    window.output.setVolume(0)
    report = {"errors": [], "fixture_only": True}
    window.player.errorOccurred.connect(lambda _code, text: report["errors"].append(text))
    window.show()

    def start():
        window.suggestion_table.selectRow(0)
        report["confirmation_gate"] = not window.accept_button.isEnabled()
        window.review_suggestion()

    def finish():
        try:
            report["position_ms"] = window.player.position()
            report["duration_ms"] = window.player.duration()
            assert window.player.mediaStatus() != QMediaPlayer.MediaStatus.InvalidMedia
            assert window.player.position() > 0 and window.player.duration() > 0
            window.player.pause()
            window.confirmed.setChecked(True)
            window.suggestion_table.selectRow(0)
            window.accept_button.click()
            assert len(window.clip["intervals"]) == 1
            assert window.clip["intervals"][0]["weapon"] == rows[0]["proposed_weapon"]
            window.suggestion_table.selectRow(1)
            window.reject_selected_suggestion()
            window.suggestion_table.selectRow(2)
            window.weapons.setCurrentIndex(window.weapons.findData("ferro"))
            window.save_weapon_button.click()
            annotations = load_dataset(scratch)["clips"][0]["intervals"]
            assert len(annotations) == 2 and any(row.get("weapon") == "ferro" for row in annotations)
            decisions = [row["status"] for row in load_suggestions(scratch)["clips"][0]["suggestions"]]
            assert decisions[:3] == ["accepted", "rejected", "corrected"]
            assert not window.clip["reviewed"]
            report["accept_reject_correct"] = True
            window.suggestion_table.selectRow(2)
            window.grab().save(str(artifacts / "review-smoke.png"))
        except Exception as exc:
            report["errors"].append(repr(exc))
        finally:
            window.close()

    QTimer.singleShot(400, start)
    QTimer.singleShot(2600, finish)
    QTimer.singleShot(12000, app.quit)
    app.exec()
    report["original_unchanged"] = before == (original.read_bytes(), suggestion_path(original).read_bytes())
    scratch.unlink()
    suggestion_path(scratch).unlink()
    save_dataset(artifacts / "review-smoke-report.json", report)
    print(json.dumps(report, indent=2))
    return int(bool(report["errors"]) or not report["original_unchanged"])


if __name__ == "__main__":
    raise SystemExit(main())
