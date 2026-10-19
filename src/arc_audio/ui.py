import sys
import threading
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import QProcess, QTimer, Signal
from PySide6.QtGui import QAction, QCloseEvent, QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStyle,
    QSystemTrayIcon,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .audio import loopback_devices
from .overlay import Overlay
from .pipeline import Session
from .qt_interrupts import run_event_loop
from .settings import data_dir, load_settings, save_settings
from .windows import EmergencyHotkey


class ControlWindow(QMainWindow):
    devices_ready = Signal(object, str)

    def __init__(self):
        super().__init__()
        self.settings, warning = load_settings()
        self.session = None
        self.pending = []
        self.last_overlay_error = ""
        self.quit_pending = False
        self.device_scan = None
        self.setWindowTitle("ARC Audio Companion • local audio lab")
        self.resize(700, 790)
        self.overlay = Overlay(self.settings)
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #121b25; color: #e6edf3; }
            QLabel { background: transparent; }
            QPushButton, QComboBox, QDoubleSpinBox { background: #243546; padding: 7px; border-radius: 5px; }
            QPushButton:disabled { color: #7e8b97; }
            QPushButton:hover { background: #36536b; }
            QTextEdit { background: #0d141d; border: 1px solid #314254; }
            QProgressBar { border: 1px solid #314254; border-radius: 3px; text-align: center; }
            QProgressBar::chunk { background: #39b89b; }
        """)
        root = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(root)
        self.setCentralWidget(scroll)
        layout = QVBoxLayout(root)
        title = QLabel("ARC / AUDIO COMPANION")
        title.setStyleSheet("font-size: 23px; font-weight: 700; color: #65e6ca; padding: 8px 0;")
        layout.addWidget(title)
        subtitle = QLabel("LOCAL AUDIO LAB  ·  BASELINE IS HEURISTIC  ·  NO GAME CONNECTION")
        layout.addWidget(subtitle)
        notice = QLabel("Live-match permission has not been established. This app is not approved by Embark, "
                        "and ban safety is not guaranteed. Whole-device loopback includes chat, music, and other apps.")
        notice.setWordWrap(True)
        notice.setStyleSheet("color: #ffcf75; padding: 8px 0;")
        layout.addWidget(notice)
        form = QFormLayout()
        layout.addLayout(form)
        self.mode = QComboBox()
        for label, value in [("Demo (synthetic, silent)", "demo"), ("Recorded stereo WAV (with playback)", "file"),
                             ("Live WASAPI loopback (explicit opt-in)", "live")]:
            self.mode.addItem(label, value)
        form.addRow("Input", self.mode)
        self.backend = QComboBox()
        self.backend.addItem("A · Baseline — sound activity / transient", "baseline")
        self.backend.addItem("B · Optional AST — cached model, 10.24 s clips", "ast")
        self.backend.addItem("C · Local ARC model — selected player-activity sounds", "local")
        self.backend.setCurrentIndex(max(0, self.backend.findData(self.settings.backend)))
        form.addRow("Detection", self.backend)
        self.target_only = QCheckBox("Only trained player-activity sound classes (recorded / live input)")
        self.target_only.setChecked(self.settings.target_only)
        self.target_only.setToolTip("Excludes generic activity/transients. Baseline remains available in demo for testing.")
        form.addRow(self.target_only)
        self.label_button = QPushButton("Label training videos…")
        self.label_button.clicked.connect(self.open_labeler)
        form.addRow("Local training", self.label_button)
        self.file_button = QPushButton("Load stereo WAV…")
        self.file_button.clicked.connect(self.choose_file)
        self.file_label = QLabel("No recording loaded")
        self.file_label.setWordWrap(True)
        self.file_path = ""
        file_row = QHBoxLayout()
        file_row.addWidget(self.file_button)
        file_row.addWidget(self.file_label, 1)
        form.addRow("Recording", file_row)
        self.devices = QComboBox()
        self.refresh = QPushButton("Refresh")
        self.refresh.clicked.connect(self.refresh_devices)
        device_row = QHBoxLayout()
        device_row.addWidget(self.devices, 1)
        device_row.addWidget(self.refresh)
        form.addRow("Live device", device_row)
        self.live_opt_in = QCheckBox("I understand that permission for live-match use has not been established")
        form.addRow(self.live_opt_in)
        self.monitors = QComboBox()
        form.addRow("Overlay monitor", self.monitors)
        self.monitors.currentIndexChanged.connect(self.monitor_changed)
        self.screen_connections = {}
        self.refresh_monitors()
        QGuiApplication.instance().screenAdded.connect(lambda _screen: self.refresh_monitors())
        QGuiApplication.instance().screenRemoved.connect(lambda _screen: self.refresh_monitors())
        self.spins = {}
        for key, label, low, high, step in [
            ("sensitivity", "Sensitivity (baseline)", .25, 3, .05),
            ("min_confidence", "Minimum confidence / score", 0, 1, .05),
            ("opacity", "Overlay opacity", .1, 1, .05),
            ("lifetime", "Indicator lifetime (seconds)", .3, 5, .1),
            ("ui_scale", "UI / indicator scale", .6, 2, .1),
        ]:
            spin = QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setSingleStep(step)
            spin.setValue(getattr(self.settings, key))
            spin.valueChanged.connect(self.options_changed)
            self.spins[key] = spin
            form.addRow(label, spin)
        self.visible = QCheckBox("Show overlay (click-through; does not accept focus)")
        self.visible.setChecked(True)
        self.visible.toggled.connect(self.toggle_overlay)
        layout.addWidget(self.visible)
        buttons = QHBoxLayout()
        self.start_button = QPushButton("Start analysis")
        self.stop_button = QPushButton("STOP / hide overlay")
        self.start_button.clicked.connect(self.start_analysis)
        self.stop_button.clicked.connect(self.emergency_stop)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        layout.addLayout(buttons)
        self.shortcut_label = QLabel()
        self.shortcut_label.setWordWrap(True)
        layout.addWidget(self.shortcut_label)
        self.status = QLabel("Stopped • demo selected")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.meters = []
        for side in ("L", "R"):
            row = QHBoxLayout()
            row.addWidget(QLabel(side))
            meter = QProgressBar()
            meter.setRange(0, 100)
            meter.setValue(0)
            row.addWidget(meter)
            self.meters.append(meter)
            layout.addLayout(row)
        self.metrics = QLabel("Processing: —   Pipeline: —   Dropped: 0")
        layout.addWidget(self.metrics)
        self.messages = QTextEdit()
        self.messages.setReadOnly(True)
        self.messages.document().setMaximumBlockCount(160)
        layout.addWidget(self.messages, 1)
        self.hotkey = EmergencyHotkey(self.emergency_stop)
        QApplication.instance().installNativeEventFilter(self.hotkey)
        self.shortcut_label.setText(self.hotkey.message)
        self.tray = QSystemTrayIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaVolume), self)
        menu = QMenu(self)
        for title, callback in [("Show controls", self.show_controls), ("Emergency stop / hide overlay", self.emergency_stop),
                                ("Quit", self.close)]:
            action = QAction(title, self)
            action.triggered.connect(callback)
            menu.addAction(action)
        self.tray.setContextMenu(menu)
        self.tray.setToolTip("ARC Audio Companion • Ctrl+Alt+F10 to stop")
        self.tray.show()
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self.messages.append("System tray unavailable. Keep controls accessible for the Stop button.")
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(33)
        self.devices_ready.connect(self.on_devices)
        self.mode.currentIndexChanged.connect(self.mode_changed)
        self.mode_changed()
        self.options_changed()
        self.toggle_overlay(True)
        self.messages.append("Demo repeats: left burst at 1 s, center activity at 3.5 s, right burst at 6.5 s.")
        if warning:
            self.messages.append(warning)
        self.messages.append(self.hotkey.message)
        from .local_model import default_model_path
        if not default_model_path().exists():
            self.messages.append("Local ARC model is not trained yet. Use 'Label training videos' to confirm audio tracks, "
                                 "mark target sounds and reviewed background, then run train-local. "
                                 "See TRAINING.md. These cues cannot establish enemy identity.")

    def show_controls(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def open_labeler(self):
        root = Path(__file__).resolve().parents[2]
        manifest = root / "training" / "dataset.json"
        if not manifest.exists():
            self.messages.append("Run: python -m arc_audio prepare-training training, then open labels again.")
            return
        success, _pid = QProcess.startDetached(sys.executable, ["-m", "arc_audio", "label-training", str(manifest)], str(root))
        if not success:
            self.messages.append("Could not launch label review. Run: python -m arc_audio label-training")

    def refresh_monitors(self):
        previous = self.monitors.currentData()
        self.monitors.blockSignals(True)
        self.monitors.clear()
        for screen in QGuiApplication.screens():
            rect = screen.geometry()
            self.monitors.addItem(f"{screen.name()} • {rect.width()}×{rect.height()} at {rect.x()},{rect.y()} "
                                  f"• {screen.devicePixelRatio():g}×", screen.name())
            if self.screen_connections.get(screen.name()) is not screen:
                screen.geometryChanged.connect(lambda _rect: self.monitor_changed())
                self.screen_connections[screen.name()] = screen
        current_names = {screen.name() for screen in QGuiApplication.screens()}
        self.screen_connections = {name: screen for name, screen in self.screen_connections.items()
                                   if name in current_names}
        preferred = previous or self.settings.monitor
        index = self.monitors.findData(preferred)
        self.monitors.setCurrentIndex(max(0, index))
        self.monitors.blockSignals(False)
        self.monitor_changed()

    def monitor_changed(self, *_args):
        name = self.monitors.currentData()
        screens = QGuiApplication.screens()
        if not screens:
            self.overlay.hide()
            return
        screen = next((s for s in screens if s.name() == name), screens[0])
        self.overlay.target(screen)
        if hasattr(self, "spins"):
            self.options_changed()

    def mode_changed(self):
        live = self.mode.currentData() == "live"
        if self.mode.currentData() != "demo" and self.target_only.isChecked() and self.backend.currentData() == "baseline":
            self.backend.setCurrentIndex(self.backend.findData("local"))
        self.devices.setEnabled(live)
        self.refresh.setEnabled(live)
        self.live_opt_in.setEnabled(live)
        self.file_button.setEnabled(self.mode.currentData() == "file")
        if live and self.devices.count() == 0:
            self.refresh_devices()

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose stereo PCM WAV", "", "Stereo PCM WAV (*.wav)",
                                            options=QFileDialog.Option.DontUseNativeDialog)
        if path:
            self.file_path = path
            self.file_label.setText(path)

    def refresh_devices(self):
        if self.device_scan and self.device_scan.is_alive():
            return
        self.refresh.setEnabled(False)
        self.status.setText("Enumerating compatible stereo WASAPI loopback devices…")

        def run():
            try:
                devices, error = loopback_devices(), ""
            except Exception as exc:
                devices, error = [], str(exc)
            self.devices_ready.emit(devices, error)

        self.device_scan = threading.Thread(target=run, daemon=True)
        self.device_scan.start()

    def on_devices(self, devices, error):
        self.devices.clear()
        for device in devices:
            self.devices.addItem(f"{device.name} • {device.rate} Hz", device)
        for index in range(self.devices.count()):
            if self.devices.itemData(index).name == self.settings.device_name:
                self.devices.setCurrentIndex(index)
                break
        self.refresh.setEnabled(self.session is None and self.mode.currentData() == "live")
        message = error or (f"{len(devices)} compatible stereo outputs found." if devices else
                            "No compatible stereo outputs. Set the Windows output to stereo, reconnect it, then Refresh.")
        self.status.setText(message)
        self.messages.append(message)

    def options_changed(self, *_args):
        values = {name: spin.value() for name, spin in self.spins.items()}
        device = self.devices.currentData()
        self.settings = replace(self.settings, **values, monitor=self.monitors.currentData() or "",
                                device_name=device.name if device else self.settings.device_name,
                                backend=self.backend.currentData(), target_only=self.target_only.isChecked())
        self.overlay.settings = self.settings
        self.centralWidget().setStyleSheet(f"font-size: {max(10, round(12 * self.settings.ui_scale))}px;")
        try:
            save_settings(self.settings)
        except OSError as exc:
            if hasattr(self, "messages"):
                self.messages.append(f"Cannot save settings: {exc}. Check permissions for {data_dir()}.")

    def toggle_overlay(self, visible):
        if visible:
            try:
                self.overlay.show()
            except OSError as exc:
                self.messages.append(f"Windows overlay style setup failed: {exc}. Stop and restart the app.")
                self.overlay.hide()
        else:
            self.overlay.hide()

    def lock_inputs(self, locked):
        for widget in (self.mode, self.backend, self.target_only, self.label_button, self.file_button, self.devices, self.refresh,
                       self.live_opt_in, self.spins["sensitivity"], self.spins["min_confidence"]):
            widget.setEnabled(not locked)
        self.start_button.setEnabled(not locked)
        if not locked:
            self.mode_changed()

    def start_analysis(self):
        if self.session:
            return
        mode = self.mode.currentData()
        if mode != "demo" and self.target_only.isChecked() and self.backend.currentData() == "baseline":
            self.messages.append("Baseline cannot identify these sound classes. Choose the trained Local ARC model, "
                                 "or AST for generic footsteps/gunfire only (no reload class). "
                                 "Uncheck the target-only option only for activity diagnostics.")
            return
        if mode == "file" and not self.file_path:
            self.messages.append("Load a stereo PCM WAV first.")
            return
        if mode == "live" and not self.live_opt_in.isChecked():
            self.messages.append("Live input requires the explicit acknowledgement checkbox above.")
            return
        if mode == "live" and self.devices.currentData() is None:
            self.messages.append("Refresh devices and choose a compatible stereo output first.")
            return
        self.options_changed()
        log_path = data_dir() / "logs" / f"{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:6]}.jsonl"
        self.session = Session(mode, self.settings, log_path, self.file_path, self.devices.currentData())
        self.pending.clear()
        self.overlay.clear()
        self.visible.setChecked(True)
        self.lock_inputs(True)
        self.messages.append(f"Session: {mode} / {self.settings.backend}. Event log: {log_path}")
        if self.settings.backend == "ast":
            self.messages.append("AST needs 10.24 seconds of continuous audio. Clip classes have unknown direction.")
            self.messages.append("AST has no verified reload class. It cannot identify enemy versus self/teammate.")
        elif self.settings.backend == "local":
            self.messages.append("Experimental local model: 1-second context, 250 ms updates. "
                                 "Only classes included during training, direction unknown. Scores are not enemy probabilities.")
        self.session.start()

    def emergency_stop(self):
        self.visible.setChecked(False)
        self.overlay.clear()
        self.pending.clear()
        if self.session:
            self.session.stop()
        self.status.setText("Emergency stop requested • overlay hidden • waiting for workers to close")

    def poll(self):
        if self.overlay.native_error and self.overlay.native_error != self.last_overlay_error:
            self.last_overlay_error = self.overlay.native_error
            self.messages.append(self.last_overlay_error)
            self.visible.setChecked(False)
        session = self.session
        if session:
            if not session.stop_event.is_set():
                self.status.setText(session.source.status if session.source.ident else session.status)
                for meter, value in zip(self.meters, session.levels):
                    # Log-scaled RMS, -60 to 0 dBFS.
                    import math
                    db = 20 * math.log10(max(value, 1e-6))
                    meter.setValue(round(max(0, min(100, (db + 60) / 60 * 100))))
                    meter.setFormat(f"{db:.1f} dBFS")
                while not session.events.empty():
                    event = session.events.get(0)
                    if event:
                        self.pending.append(event)
                self.pending = self.pending[-24:]
                now = time.monotonic()
                waiting = []
                for event in self.pending:
                    if event.present_at > now:
                        waiting.append(event)
                    elif now - event.present_at <= 2.0:
                        self.overlay.add(event)
                        self.messages.append(f"{event.audio_time:7.2f}s  {event.label}  {event.confidence:.0%}  "
                                             f"{event.direction}  ({event.processing_ms:.1f} ms)")
                self.pending = waiting
            self.metrics.setText(f"Processing: {session.processing_ms:.1f} ms   "
                                 f"Pipeline: {session.pipeline_ms:.1f} ms   Dropped / xruns: {session.dropped}")
            if session.done.is_set():
                if session.error:
                    self.messages.append(f"ERROR: {session.error}")
                self.status.setText(session.error or "Stopped / end of recording")
                self.session = None
                self.pending.clear()
                self.lock_inputs(False)
                for meter in self.meters:
                    meter.setValue(0)
                if self.quit_pending:
                    self.close()

    def closeEvent(self, event: QCloseEvent):
        if self.session and not self.session.done.is_set():
            self.quit_pending = True
            self.emergency_stop()
            event.ignore()
            return
        self.options_changed()
        self.hotkey.close()
        QApplication.instance().removeNativeEventFilter(self.hotkey)
        self.overlay.close()
        self.tray.hide()
        event.accept()


def run_ui(smoke_seconds=0):
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("ARC Audio Companion")
    app.setQuitOnLastWindowClosed(True)
    window = ControlWindow()
    window.show()
    if smoke_seconds:
        window.backend.setCurrentIndex(0)
        QTimer.singleShot(100, window.start_analysis)
        QTimer.singleShot(round(smoke_seconds * 1000), window.close)
    return run_event_loop(app, window.close)
