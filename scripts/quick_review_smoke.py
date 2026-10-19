"""Muted native quick-review launch; read-only against the selected real dataset."""

import argparse
import json
from pathlib import Path

from PySide6.QtCore import QTimer

from arc_audio import labeler
from arc_audio.suggestions import suggestion_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, nargs="?", default=Path("training/dataset.json"))
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    sidecar = suggestion_path(manifest)
    before = manifest.read_bytes(), sidecar.read_bytes() if sidecar.exists() else None
    artifacts = Path(__file__).resolve().parents[1] / "artifacts" / "quick-review-live-smoke"
    artifacts.mkdir(parents=True, exist_ok=True)
    report = {}
    original_loop = labeler.run_event_loop

    def observe(app, request_close):
        window = request_close.__self__
        window.output.setVolume(0)
        report.update(window=type(window).__name__, questions=len(window.queue))

        def inspect():
            window.grab().save(str(artifacts / "preview.png"))
            report.update(heard=window.heard, playback_failed=window.playback_failed,
                          position_ms=window.player.position(), status=window.status.text())

        QTimer.singleShot(4000, inspect)
        return original_loop(app, request_close)

    labeler.run_event_loop = observe
    try:
        report["exit_code"] = labeler.run_labeler(manifest, 5)
    finally:
        labeler.run_event_loop = original_loop
    report["originals_unchanged"] = before == (
        manifest.read_bytes(), sidecar.read_bytes() if sidecar.exists() else None)
    (artifacts / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return int(report["window"] != "QuickReviewWindow" or not report["originals_unchanged"]
               or report.get("playback_failed", True) or report["exit_code"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
