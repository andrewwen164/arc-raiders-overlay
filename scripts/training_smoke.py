"""Windows/local-media labeler smoke. Does not annotate the real dataset."""

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
os.environ["LOCALAPPDATA"] = str(ARTIFACTS / "training-smoke-data")


def main():
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from arc_audio.dataset import load_dataset, save_dataset
    from arc_audio.labeler import LabelWindow

    manifest = ROOT / "training" / "dataset.json"
    document = load_dataset(manifest)
    if not document["clips"]:
        raise RuntimeError("Prepare a dataset before running this smoke")
    # A disposable manifest beside the originals resolves their relative paths.
    scratch = manifest.with_name(".labeler-smoke.json")
    if scratch.exists():
        raise RuntimeError("A previous labeler smoke manifest exists; inspect it before retrying")
    copy = json.loads(json.dumps(document))
    copy["clips"] = [copy["clips"][0]]
    copy["clips"][0].update(track_confirmed=False, reviewed=[], intervals=[])
    save_dataset(scratch, copy)
    report = {"checks": {}, "errors": []}
    app = QApplication([])
    window = LabelWindow(scratch)
    window.output.setVolume(.1)
    frames = [0]
    window.video.videoSink().videoFrameChanged.connect(lambda frame: frames.__setitem__(0, frames[0] + int(frame.isValid())))
    window.show()
    started = time.monotonic()

    def begin():
        window.player.play()

    def change_track():
        try:
            report["checks"]["position_advanced_ms"] = window.player.position()
            report["checks"]["video_frames"] = frames[0]
            assert window.player.position() > 100
            assert frames[0] > 0
            if window.tracks.count() > 1:
                window.tracks.setCurrentIndex(1)
                report["checks"]["track_switch_requested"] = True
        except Exception as exc:
            report["errors"].append(str(exc))

    def finish():
        try:
            window.player.pause()
            report["checks"]["player_error"] = window.player.errorString()
            assert not window.player.errorString()
            if window.tracks.count() > 1:
                report["checks"]["active_audio_track"] = window.player.activeAudioTrack()
                assert window.player.activeAudioTrack() == 1
            # Test annotation CRUD on the scratch copy only, not human labels.
            window.confirmed.setChecked(True)
            window.start.setValue(1)
            window.end.setValue(2)
            window.labels.setCurrentIndex(window.labels.findData("reload"))
            window.add_interval()
            assert window.clip["intervals"] == [{"start": 1.0, "end": 2.0, "label": "reload", "source": "unknown"}]
            window.table.selectRow(0)
            window.delete_rows()
            assert not window.clip["intervals"]
            report["checks"]["scratch_annotation_add_delete"] = True
        except Exception as exc:
            report["errors"].append(str(exc))
        finally:
            window.close()
            report["elapsed_seconds"] = time.monotonic() - started
            ARTIFACTS.mkdir(exist_ok=True)
            (ARTIFACTS / "training-smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2))
            # Exact generated file, never a recursive operation or original media.
            scratch.unlink()
            app.quit()

    QTimer.singleShot(500, begin)
    QTimer.singleShot(3000, change_track)
    QTimer.singleShot(4500, finish)
    app.exec()
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
