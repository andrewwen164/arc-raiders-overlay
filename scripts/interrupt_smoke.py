"""Send real Windows Ctrl+C events to hidden, isolated test consoles only.

Never attaches to or signals a user's terminal. Child apps use disposable
settings, synthetic audio and scratch manifests; no real recordings are changed.
"""

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def send_to_owned_console(pid):
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.AttachConsole.argtypes = [wintypes.DWORD]
    kernel.AttachConsole.restype = wintypes.BOOL
    kernel.SetConsoleCtrlHandler.argtypes = [ctypes.c_void_p, wintypes.BOOL]
    kernel.SetConsoleCtrlHandler.restype = wintypes.BOOL
    kernel.GenerateConsoleCtrlEvent.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.GenerateConsoleCtrlEvent.restype = wintypes.BOOL
    # This function runs in a separate controller helper, never the parent tool.
    kernel.FreeConsole()
    if not kernel.AttachConsole(pid):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not kernel.SetConsoleCtrlHandler(None, True):
            raise ctypes.WinError(ctypes.get_last_error())
        # Ctrl+C cannot target a process-group ID. The entire *isolated test*
        # console contains only the owned child and this ignore-enabled helper.
        if not kernel.GenerateConsoleCtrlEvent(0, 0):
            raise ctypes.WinError(ctypes.get_last_error())
        time.sleep(.2)
    finally:
        kernel.FreeConsole()


def child(mode, folder):
    import signal

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from arc_audio import labeler, ui
    from arc_audio.dataset import TARGET_CLASSES, load_dataset, save_dataset
    from arc_audio.qt_interrupts import ConsoleInterrupts

    # Simulate a launcher-inherited ignore flag. Production code must clear it.
    ctypes.windll.kernel32.SetConsoleCtrlHandler(None, True)
    folder = Path(folder)
    os.environ["LOCALAPPDATA"] = str(folder / "appdata")
    manifest = None
    if mode.startswith("labeler"):
        media = folder / "data" / "tone.wav"
        media.parent.mkdir()
        with wave.open(str(media), "wb") as output:
            output.setparams((2, 2, 16000, 0, "NONE", "not compressed"))
            output.writeframes(bytes(16000 * 4))
        manifest = media.parent / "dataset.json"
        save_dataset(manifest, {"version": 1, "classes": list(TARGET_CLASSES), "clips": [
            {"file": media.name, "duration": 1, "size": media.stat().st_size, "group": "synthetic",
             "split": "train", "audio_stream": 0, "audio_streams": [{"index": 0, "channels": 2}],
             "track_confirmed": True, "intervals": [{"start": .1, "end": .3, "label": "gunshot"}],
             "reviewed": []}]})
        module, launch = labeler, lambda: labeler.run_labeler(manifest)
    else:
        module, launch = ui, ui.run_ui
    actual_loop = module.run_event_loop

    def observe_loop(app, request_close):
        window = request_close.__self__
        if mode == "control-demo":
            window.backend.setCurrentIndex(0)
            window.start_analysis()
        elif mode == "control-dialog":
            QTimer.singleShot(10, window.choose_file)
        elif mode == "labeler-import":
            from arc_audio import media_import
            actual_import = media_import.import_media

            def slower_import(*args):
                time.sleep(.8)  # Deterministically interrupt during an active copy/probe worker.
                return actual_import(*args)

            media_import.import_media = slower_import
            incoming = folder / "incoming.wav"
            shutil.copyfile(manifest.parent / "tone.wav", incoming)
            window.start_media_import([str(incoming)], "synthetic-second-session")

        ready = QTimer()
        ready.setInterval(25)

        def report_ready():
            handler = signal.getsignal(signal.SIGINT)
            if not isinstance(getattr(handler, "__self__", None), ConsoleInterrupts):
                return
            if mode == "control-demo" and (window.session is None or window.session.source.ident is None):
                return
            if mode == "control-dialog" and QApplication.activeModalWidget() is None:
                return
            (folder / "ready.json").write_text(json.dumps({"mode": mode, "handler_installed": True}))
            ready.stop()

        ready.timeout.connect(report_ready)
        ready.start()
        result = actual_loop(app, request_close)
        ready.stop()
        report = {"mode": mode, "exit_code": result,
                  "audio_workers_alive": [t.name for t in threading.enumerate()
                                          if t.name in ("audio-source", "audio-analysis") and t.is_alive()]}
        if manifest:
            doc = load_dataset(manifest)
            report.update(labels_preserved=doc["clips"][0]["intervals"] == [
                {"start": .1, "end": .3, "label": "gunshot"}],
                import_thread_finished=window.import_thread is None, clips=len(doc["clips"]))
        else:
            report.update(overlay_hidden=not window.overlay.isVisible(), tray_hidden=not window.tray.isVisible())
        (folder / "result.json").write_text(json.dumps(report, indent=2))
        return result

    module.run_event_loop = observe_loop
    return launch()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--send", type=int)
    parser.add_argument("--child")
    parser.add_argument("--folder", type=Path)
    parser.add_argument("--powershell-only", action="store_true")
    args = parser.parse_args()
    if sys.platform != "win32":
        print("SKIP: isolated console smoke requires Windows")
        return 0
    if args.send is not None:
        send_to_owned_console(args.send)
        return 0
    if args.child:
        return child(args.child, args.folder)
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    report = []
    with tempfile.TemporaryDirectory(prefix="interrupt-smoke-", dir=artifacts) as scratch:
        modes = () if args.powershell_only else ("control-idle", "control-demo", "control-dialog", "labeler-idle", "labeler-import")
        for mode in modes:
            folder = Path(scratch) / mode
            folder.mkdir()
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = 0  # Hide the test console; no user terminal is reused.
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--child", mode,
                                        "--folder", str(folder)], cwd=ROOT, startupinfo=startup,
                                       creationflags=subprocess.CREATE_NEW_CONSOLE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                deadline = time.monotonic() + 15
                while not (folder / "ready.json").exists() and time.monotonic() < deadline:
                    if process.poll() is not None:
                        break
                    time.sleep(.025)
                if not (folder / "ready.json").exists():
                    raise RuntimeError(f"{mode}: child did not become ready")
                started = time.monotonic()
                sender = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--send", str(process.pid)],
                                        cwd=ROOT, creationflags=subprocess.CREATE_NO_WINDOW,
                                        capture_output=True, text=True, timeout=5)
                if sender.returncode:
                    raise RuntimeError(sender.stderr or sender.stdout)
                output, errors = process.communicate(timeout=10)
                result = json.loads((folder / "result.json").read_text())
                result["shutdown_seconds"] = time.monotonic() - started
                assert process.returncode == result["exit_code"] == 130, (mode, output, errors)
                assert not result["audio_workers_alive"]
                assert result.get("labels_preserved", True) and result.get("import_thread_finished", True)
                assert result.get("overlay_hidden", True) and result.get("tray_hidden", True)
                assert mode != "labeler-import" or result["clips"] == 2
                assert "KeyboardInterrupt" not in errors and "QThread: Destroyed" not in errors
                report.append(result)
                print(f"PASS {mode}: real Ctrl+C, exit 130, {result['shutdown_seconds']:.2f}s", flush=True)
            finally:
                if process.poll() is None:
                    process.terminate()  # Only the owned disposable test child, on failure.
                    process.communicate(timeout=5)
        # Exercise the actual launch.ps1 inside an interactive PowerShell child.
        # Its hidden console is isolated exactly like the app cases above.
        folder = Path(scratch) / "powershell-launcher"
        folder.mkdir()
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
        environment = dict(os.environ, LOCALAPPDATA=str(folder / "appdata"))
        process = subprocess.Popen(["powershell.exe", "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-NoExit", "-Command",
                                    "& './launch.ps1'"], cwd=ROOT, env=environment, startupinfo=startup,
                                   creationflags=subprocess.CREATE_NEW_CONSOLE,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 15
            settings = folder / "appdata" / "ArcAudioCompanion" / "settings.json"
            while not settings.exists() and time.monotonic() < deadline:
                if process.poll() is not None:
                    break
                time.sleep(.025)
            if not settings.exists():
                subprocess.run(["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                               capture_output=True, text=True, timeout=5)
                output, errors = process.communicate(timeout=5)
                raise RuntimeError(f"PowerShell launcher did not open the control window: {output} / {errors}")
            time.sleep(.2)  # Allow constructor completion / interrupt handler installation.
            sender = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--send", str(process.pid)],
                                    cwd=ROOT, creationflags=subprocess.CREATE_NO_WINDOW,
                                    capture_output=True, text=True, timeout=5)
            if sender.returncode:
                raise RuntimeError(sender.stderr or sender.stdout)
            # A new command in this same PowerShell proves the terminal is reusable.
            output, errors = process.communicate("Write-Output 'ARC_TERMINAL_REUSABLE'\nexit\n", timeout=12)
            assert any(line.strip() == "ARC_TERMINAL_REUSABLE" for line in output.splitlines()), (output, errors)
            assert "KeyboardInterrupt" not in errors, errors
            report.append({"mode": "powershell-launcher", "terminal_reusable": True,
                           "powershell_exit_code": process.returncode})
            print("PASS powershell-launcher: same terminal accepted a new command after Ctrl+C", flush=True)
        finally:
            if process.poll() is None:
                subprocess.run(["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                               capture_output=True, text=True, timeout=5)
                process.communicate(timeout=5)
    output_name = "powershell-interrupt-smoke.json" if args.powershell_only else "interrupt-smoke.json"
    (artifacts / output_name).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
