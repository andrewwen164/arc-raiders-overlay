"""Windows integration smoke, synthetic audio only; no game required or accessed.

Writes scratch data inside artifacts, not the user's real application settings.
Plays quiet synthetic tone bursts on the default output for nine seconds and
explicitly selects whole-device loopback. No captured samples are saved.
"""

import ctypes
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
os.environ["LOCALAPPDATA"] = str(ARTIFACTS / "windows-smoke-data")


def main():
    if sys.platform != "win32":
        print("SKIP: Windows integration checks require Windows")
        return 0
    from ctypes import wintypes

    from PySide6.QtCore import QRect, QTimer
    from PySide6.QtWidgets import QApplication

    from arc_audio.audio import loopback_devices, write_demo
    from arc_audio.pipeline import Session
    from arc_audio.settings import Settings
    from arc_audio.ui import ControlWindow
    from arc_audio.windows import HOTKEY_ID, WM_HOTKEY

    app = QApplication([])
    window = ControlWindow()
    window.show()
    api = ctypes.WinDLL("user32", use_last_error=True)
    api.GetForegroundWindow.restype = wintypes.HWND
    api.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
    api.GetWindowLongPtrW.restype = ctypes.c_ssize_t
    api.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    api.PostThreadMessageW.restype = wintypes.BOOL
    kernel = ctypes.WinDLL("kernel32")
    kernel.GetCurrentThreadId.restype = wintypes.DWORD
    report = {"platform": sys.platform, "python": sys.version, "checks": {}, "errors": [],
              "versions": {name: importlib.metadata.version(name) for name in
                           ("PySide6", "numpy", "scipy", "PyAudioWPatch")}}
    checks = report["checks"]
    captured = []
    stages = {}

    def record_error(exc):
        report["errors"].append(f"{type(exc).__name__}: {exc}")

    def inspect_native():
        try:
            foreground = api.GetForegroundWindow()
            window.overlay.hide()
            window.overlay.show()
            app.processEvents()
            checks["show_overlay_preserves_foreground"] = foreground == api.GetForegroundWindow()
            style = api.GetWindowLongPtrW(int(window.overlay.winId()), -20)
            checks["native_styles"] = {name: bool(style & flag) for name, flag in
                                       [("layered", 0x80000), ("transparent", 0x20),
                                        ("noactivate", 0x8000000), ("toolwindow", 0x80)]}
            checks["global_hotkey_registered"] = window.hotkey.registered
            checks["hotkey_message"] = window.hotkey.message
            screens = app.screens()
            checks["monitors"] = [{"name": s.name(), "x": s.geometry().x(), "y": s.geometry().y(),
                                   "width": s.geometry().width(), "height": s.geometry().height(),
                                   "scale": s.devicePixelRatio()} for s in screens]
            for screen in screens:
                window.overlay.target(screen)
                app.processEvents()
                assert window.overlay.geometry() == screen.geometry()
            checks["all_available_monitor_geometries"] = True
            # Verify no accidental unsigned coordinate conversion even without a physical
            # negative-origin monitor. This is geometry only, not a visual DPI test.
            window.overlay.setGeometry(QRect(-500, -200, 450, 300))
            checks["negative_coordinate_geometry"] = window.overlay.geometry().x() == -500
            window.monitor_changed()
            window.start_analysis()
        except Exception as exc:
            record_error(exc)

    def dispatch_hotkey():
        try:
            stages["demo_session"] = window.session
            if window.hotkey.registered:
                assert api.PostThreadMessageW(kernel.GetCurrentThreadId(), WM_HOTKEY, HOTKEY_ID, 0)
            else:
                checks["hotkey_dispatch"] = "SKIP: registration unavailable"
                window.emergency_stop()
            QTimer.singleShot(100, verify_stop)
        except Exception as exc:
            record_error(exc)

    def capture_overlay():
        try:
            checks["demo_indicators_visible"] = bool(window.overlay.indicators)
            assert checks["demo_indicators_visible"], "Synthetic cue was not delivered to the overlay"
            window.overlay.grab().save(str(ARTIFACTS / "overlay.png"))
        except Exception as exc:
            record_error(exc)

    def verify_stop():
        try:
            assert not window.overlay.isVisible()
            assert stages["demo_session"].stop_event.is_set()
            checks["hotkey_dispatch"] = True if window.hotkey.registered else checks["hotkey_dispatch"]
            checks["emergency_hides_overlay"] = True
        except Exception as exc:
            record_error(exc)

    def start_file():
        try:
            assert window.session is None, "Demo did not stop"
            devices = loopback_devices()
            if devices:
                # Prefer the default output's exact loopback analogue.
                import pyaudiowpatch as pa
                with pa.PyAudio() as p:
                    default = p.get_default_wasapi_loopback()
                device = next((d for d in devices if d.name == default["name"]), devices[0])
                capture = Session("live", Settings(target_only=False), ARTIFACTS / "loopback-events.jsonl", device=device)
                stages["capture"] = capture
                capture.start()
                checks["live_device"] = device.name
            else:
                checks["live_capture"] = "SKIP: no compatible stereo output"
            path = ARTIFACTS / "smoke-playback.wav"
            write_demo(path, seconds=9)
            window.target_only.setChecked(False)
            window.mode.setCurrentIndex(1)
            window.file_path = str(path)
            window.start_analysis()
            stages["file_session"] = window.session
        except Exception as exc:
            record_error(exc)

    def observe():
        session = stages.get("capture")
        if session:
            captured.append(max(session.levels))

    def finish():
        try:
            file_session = stages.get("file_session")
            if file_session:
                checks["file_playback_error"] = file_session.error
                checks["file_playback_finished"] = file_session.done.is_set()
                checks["file_event_count"] = len(file_session.log_path.read_text().splitlines())
                assert not file_session.error, file_session.error
                assert file_session.done.is_set(), "Playback did not reach EOF"
                assert checks["file_event_count"] >= 3
            else:
                raise AssertionError("File session did not start")
            capture = stages.get("capture")
            if capture:
                capture.stop()
                capture.worker.join(3)
                checks["live_capture_error"] = capture.error
                checks["live_capture_shutdown"] = capture.done.is_set()
                checks["live_max_rms"] = max(captured, default=0)
                checks["live_capture"] = not capture.error and checks["live_max_rms"] > .001
                assert checks["live_capture"], capture.error or "No synthetic playback captured"
                assert not capture.source.is_alive()
            assert checks.get("show_overlay_preserves_foreground")
            assert all(checks.get("native_styles", {}).values())
            # Render own window to an artifact for optional human inspection.
            window.grab().save(str(ARTIFACTS / "controls.png"))
        except Exception as exc:
            record_error(exc)
        finally:
            for key in ("capture", "file_session", "demo_session"):
                session = stages.get(key)
                if session:
                    session.stop()
            window.close()
            report["elapsed_seconds"] = time.monotonic() - started
            (ARTIFACTS / "windows-smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2))
            QTimer.singleShot(100, app.quit)

    timer = QTimer()
    timer.timeout.connect(observe)
    timer.start(50)
    started = time.monotonic()
    QTimer.singleShot(200, inspect_native)
    QTimer.singleShot(1500, capture_overlay)
    QTimer.singleShot(1800, dispatch_hotkey)
    QTimer.singleShot(2500, start_file)
    QTimer.singleShot(13500, finish)
    app.exec()
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
