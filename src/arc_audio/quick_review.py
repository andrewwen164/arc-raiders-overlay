"""One-second, multiple-choice gunfire review in the existing desktop labeler.

These are human window-presence answers, not precisely timed event annotations.
No interval editor or implicit reviewed-background conversion is involved.
"""

import copy
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .dataset import load_dataset, media_path, save_dataset
from .presence_review import (
    answer_presence,
    bind_presence_media,
    presence_answer,
    presence_summary,
    review_window,
    set_review_split,
    undo_presence,
)
from .suggestions import current_suggestions, load_suggestions, suggestion_path


class QuickReviewWindow(QMainWindow):
    """Review proposals without manually marking timestamps or coverage."""

    def __init__(self, manifest):
        super().__init__()
        self.manifest = Path(manifest).resolve()
        self.document = load_dataset(self.manifest)
        self.proposals = load_suggestions(self.manifest)
        self.dataset_revision = self.manifest.read_bytes()
        sidecar = suggestion_path(self.manifest)
        self.proposal_revision = sidecar.read_bytes() if sidecar.exists() else None
        self.queue = []
        self.skipped_sources = set()
        self.index = -1
        self.clip = self.row = None
        self.heard = False
        self.preview_seek_pending = None
        self.play_requested = False
        self.playback_failed = False
        self.closed = False
        self.play_started_at = None
        self.window_span = None
        self._build_queue()

        self.setWindowTitle("ARC Audio - quick gunfire review")
        self.resize(820, 700)
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setSpacing(15)
        self.counter = QLabel()
        self.source = QLabel()
        self.source.setWordWrap(True)
        self.question = QLabel("Any gunfire in this entire 1-second sample?")
        self.question.setStyleSheet("font-size: 21px; font-weight: 600;")
        self.question.setWordWrap(True)
        self.context = QLabel()
        self.context.setWordWrap(True)
        for widget in (self.counter, self.source, self.question, self.context):
            layout.addWidget(widget)
        role_row = QHBoxLayout()
        role_row.addWidget(QLabel("Use this recording group for:"))
        self.role = QComboBox()
        self.role.addItem("Training", "train")
        self.role.addItem("Independent validation", "validation")
        self.role.activated.connect(self.change_role)
        role_row.addWidget(self.role, 1)
        layout.addLayout(role_row)

        self.source_box = QWidget()
        self.source_box.setMinimumHeight(64)
        source_layout = QVBoxLayout(self.source_box)
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.addWidget(QLabel("First, listen: is this the game-audio track you want to learn from?"))
        source_buttons = QHBoxLayout()
        self.confirm_button = QPushButton("Yes, game audio")
        self.skip_button = QPushButton("No, skip this source")
        self.source_unsure_button = QPushButton("Unsure, skip for now")
        self.confirm_button.clicked.connect(self.source_confirm_yes)
        self.skip_button.clicked.connect(self.skip_source)
        self.source_unsure_button.clicked.connect(self.skip_source)
        for button in (self.confirm_button, self.skip_button, self.source_unsure_button):
            button.setMinimumHeight(32)
            source_buttons.addWidget(button)
        source_layout.addLayout(source_buttons)
        layout.addWidget(self.source_box)

        self.replay_button = QPushButton("Replay sample (Space)")
        self.replay_button.clicked.connect(self.replay)
        layout.addWidget(self.replay_button)
        answers = QHBoxLayout()
        self.yes_button = QPushButton("Yes, gunfire (Y)")
        self.no_button = QPushButton("No gunfire (N)")
        self.unsure_button = QPushButton("Unsure (U)")
        for button, answer in ((self.yes_button, "yes"), (self.no_button, "no"), (self.unsure_button, "unsure")):
            button.setMinimumHeight(48)
            button.clicked.connect(lambda _checked=False, value=answer: self.save_answer(value))
            answers.addWidget(button)
        layout.addLayout(answers)

        weapon_row = QHBoxLayout()
        weapon_row.addWidget(QLabel("Optional weapon (human choice):"))
        self.weapons = QComboBox()
        self.weapons.addItem("Unknown / do not label weapon", None)
        names = set(self.document.get("weapons", []))
        for clip in self.document["clips"]:
            names.update(row["weapon"] for row in clip.get("presence_reviews", []) if row.get("weapon"))
        for name in sorted(names):
            self.weapons.addItem(name.replace("_", " ").title(), name)
        weapon_row.addWidget(self.weapons, 1)
        layout.addLayout(weapon_row)
        self.saved = QLabel()
        self.saved.setWordWrap(True)
        layout.addWidget(self.saved)
        self.progress = QLabel()
        self.progress.setWordWrap(True)
        layout.addWidget(self.progress)

        navigation = QHBoxLayout()
        self.previous_button = QPushButton("Previous (Left)")
        self.next_button = QPushButton("Next (Right)")
        self.undo_button = QPushButton("Undo this answer")
        self.previous_button.clicked.connect(self.previous)
        self.next_button.clicked.connect(self.next)
        self.undo_button.clicked.connect(self.undo_answer)
        for button in (self.previous_button, self.undo_button, self.next_button):
            navigation.addWidget(button)
        layout.addLayout(navigation)
        self.auto_advance = QCheckBox("Automatically play the next unanswered sample")
        self.auto_advance.setChecked(True)
        layout.addWidget(self.auto_advance)
        note = QLabel("Yes / No save only this one-second gunfire answer. Unsure supplies no training label. "
                      "Source weapon hints are suggestions, not recognition. Previous lets you change an answer.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(central)
        self.setCentralWidget(scroll)

        self.output = QAudioOutput(self)
        self.output.setVolume(.5)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.output)
        self.player.mediaStatusChanged.connect(self.media_ready)
        self.player.errorOccurred.connect(self.playback_error)
        self.boundary_timer = QTimer(self)
        self.boundary_timer.setInterval(20)
        self.boundary_timer.timeout.connect(self.check_boundary)
        self.boundary_timer.start()
        self.shortcuts = []
        for key, callback in (("Y", lambda: self.save_answer("yes")), ("N", lambda: self.save_answer("no")),
                              ("U", lambda: self.save_answer("unsure")), ("Space", self.replay),
                              ("Left", self.previous), ("Right", self.next)):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(callback)
            self.shortcuts.append(shortcut)
        QApplication.instance().focusChanged.connect(self.update_shortcuts)
        self.role.activated.connect(lambda _index: self.replay_button.setFocus())
        self.weapons.activated.connect(lambda _index: self.replay_button.setFocus())
        first = next((i for i, (clip, row) in enumerate(self.queue)
                      if presence_answer(clip, row) is None), 0)
        self.select(first)

    def _build_queue(self):
        for clip in self.document["clips"]:
            if clip.get("split") == "test":
                continue
            for row in current_suggestions(self.proposals, self.manifest, clip):
                if row["label"] != "gunshot":
                    continue
                if row["status"] != "pending" and not row.get("quick_review"):
                    continue
                try:
                    review_window(clip, row)
                except ValueError:
                    continue
                self.queue.append((clip, row))
        self.queue.sort(key=lambda item: (-item[1].get("quality_score", item[1].get("confidence", 0)),
                                         item[0]["file"], item[1]["start"]))
        shots = [item for item in self.queue if item[1].get("proposal_kind") != "gunshot_absence"]
        background = [item for item in self.queue if item[1].get("proposal_kind") == "gunshot_absence"]
        # Mix explicitly proposed backgrounds into the questions; their status
        # remains unknown until a human listens and answers.
        self.queue = []
        for index in range(0, len(shots), 4):
            self.queue.extend(shots[index:index + 4])
            if background:
                self.queue.append(background.pop(0))
        self.queue.extend(background)

    def update_controls(self):
        valid = self.clip is not None and self.window_span is not None
        confirmed = valid and self.clip.get("track_confirmed", False)
        can_answer = confirmed and self.heard and not self.playback_failed
        for button in (self.yes_button, self.no_button, self.unsure_button):
            button.setEnabled(bool(can_answer))
        self.confirm_button.setEnabled(bool(valid and self.heard and not self.playback_failed))
        self.source_box.setVisible(bool(valid and not confirmed))
        self.replay_button.setEnabled(valid)
        self.weapons.setEnabled(bool(confirmed))
        self.role.setEnabled(valid)
        self.undo_button.setEnabled(bool(valid and presence_answer(self.clip, self.row) is not None))
        self.previous_button.setEnabled(self.index > 0)
        self.next_button.setEnabled(valid and self.index + 1 < len(self.queue))
        if valid:
            answered = sum(presence_answer(clip, row) is not None for clip, row in self.queue)
            self.counter.setText(f"Sample {self.index + 1:,} of {len(self.queue):,}  |  {answered:,} answered")
        counts = presence_summary(self.document)
        self.progress.setText(f"Saved one-second training answers: {counts['train']['yes']} Yes / {counts['train']['no']} No. "
                              f"Validation: {counts['validation']['yes']} Yes / {counts['validation']['no']} No. "
                              "First fit needs at least 20/20 training and 5/5 independent validation answers; "
                              "these minimums do not establish accuracy.")

    def update_shortcuts(self, _previous=None, current=None):
        """Weapon/role multiple-choice navigation must never submit an answer."""
        current = current or QApplication.focusWidget()
        editing = False
        while current is not None:
            if isinstance(current, (QComboBox, QAbstractItemView)):
                editing = True
                break
            current = current.parentWidget()
        for shortcut in self.shortcuts:
            shortcut.setEnabled(not editing)

    def select(self, index):
        self.player.stop()
        self.heard = False
        self.playback_failed = False
        self.preview_seek_pending = None
        self.play_requested = False
        self.play_started_at = None
        self.window_span = None
        if not self.queue or not 0 <= index < len(self.queue):
            self.clip = self.row = None
            self.counter.setText("No eligible gunfire questions")
            self.source.setText("Create candidates first. Final test sources and legacy decisions are excluded.")
            self.context.clear()
            self.saved.clear()
            self.update_controls()
            return
        self.index = index
        self.clip, self.row = self.queue[index]
        self.window_span = review_window(self.clip, self.row)
        title = self.clip.get("source_metadata", {}).get("title") or self.clip["file"]
        self.source.setText(f"{title}\n{self.clip['split']} source - audio stream {self.clip['audio_stream']}")
        self.source.setToolTip(self.clip["file"])
        self.role.setCurrentIndex(max(0, self.role.findData(self.clip["split"])))
        hint = self.row.get("proposed_weapon") or "unknown"
        self.context.setText(f"{self.window_span['start']:.3f} - {self.window_span['end']:.3f} seconds. "
                             f"Source weapon hint: {hint}. Listen to the whole sample before answering.")
        answer = presence_answer(self.clip, self.row)
        self.saved.setText(f"Saved answer: {answer or 'not answered'}. You can answer again to change it.")
        saved_review = next((item for item in self.clip.get("presence_reviews", [])
                             if item["id"] == self.window_span["id"]), {})
        self.weapons.setCurrentIndex(max(0, self.weapons.findData(saved_review.get("weapon"))))
        self.status.setText("Playing the exact sample. Nothing is saved until you choose an answer.")
        self.player.setSource(QUrl.fromLocalFile(str(media_path(self.manifest, self.clip))))
        self.update_controls()
        self.replay_button.setFocus()
        QTimer.singleShot(0, self.replay)

    def media_ready(self, status):
        if not self.clip:
            return
        if (status == QMediaPlayer.MediaStatus.EndOfMedia and self.play_requested and self.window_span
                and self.player.position() >= round(self.window_span["end"] * 1000) - 20):
            self.sample_finished()
        if status in (QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia):
            streams = self.clip["audio_streams"]
            index = next((i for i, stream in enumerate(streams) if stream["index"] == self.clip["audio_stream"]), 0)
            self.player.setActiveAudioTrack(index)
            if self.play_requested and self.preview_seek_pending is not None:
                self.player.setPosition(self.preview_seek_pending)
                self.preview_seek_pending = None
                self.player.play()

    def replay(self):
        if self.closed or not self.window_span:
            return
        self.playback_failed = False
        self.heard = False
        self.play_started_at = None
        self.play_requested = True
        self.preview_seek_pending = round(self.window_span["start"] * 1000)
        self.update_controls()
        if self.player.mediaStatus() in (QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia,
                                         QMediaPlayer.MediaStatus.EndOfMedia):
            self.media_ready(QMediaPlayer.MediaStatus.LoadedMedia)

    def check_boundary(self):
        if (not self.window_span or not self.play_requested
                or self.player.playbackState() != QMediaPlayer.PlaybackState.PlayingState):
            return
        start_ms = round(self.window_span["start"] * 1000)
        if self.play_started_at is None and start_ms - 30 <= self.player.position() <= start_ms + 50:
            self.play_started_at = time.monotonic()
        if self.player.position() >= round(self.window_span["end"] * 1000) - 10:
            self.sample_finished()

    def sample_finished(self):
        self.player.pause()
        self.play_requested = False
        if self.play_started_at is None or time.monotonic() - self.play_started_at < .95:
            self.heard = False
            self.status.setText("The complete sample was not played from its start. Press Replay before answering.")
            self.update_controls()
            return
        self.heard = True
        self.status.setText("Sample finished. Choose Yes, No or Unsure; Space replays it.")
        self.update_controls()

    def playback_error(self, _error, message):
        self.playback_failed = True
        self.play_requested = False
        self.status.setText(f"Playback failed: {message}. Check the local media/output device and replay; no label saved.")
        self.update_controls()

    def _persist(self, change, *, proposals=True):
        sidecar = suggestion_path(self.manifest)
        before_document = copy.deepcopy(self.document)
        before_row = copy.deepcopy(self.row) if self.row is not None else None
        try:
            if self.manifest.read_bytes() != self.dataset_revision:
                raise ValueError("Dataset changed in another window. Close and reopen this labeler; newer labels are preserved.")
            current_revision = sidecar.read_bytes() if sidecar.exists() else None
            if proposals and current_revision != self.proposal_revision:
                raise ValueError("Candidates changed in another window. Close and reopen before answering.")
            if self.row is not None and not any(row is self.row for row in current_suggestions(
                    self.proposals, self.manifest, self.clip)):
                raise ValueError("This source audio or selected track changed. Regenerate candidates before reviewing it.")
            bind_presence_media(self.clip, media_path(self.manifest, self.clip))
            change()
            save_dataset(self.manifest, self.document)
            self.dataset_revision = self.manifest.read_bytes()
        except (OSError, ValueError) as exc:
            # Restore objects in place so queue references remain valid.
            for target, old in zip(self.document["clips"], before_document["clips"], strict=True):
                target.clear()
                target.update(old)
            if self.row is not None:
                self.row.clear()
                self.row.update(before_row)
            self.status.setText(f"Not saved: {exc}")
            self.update_controls()
            return False
        if proposals:
            try:
                save_dataset(sidecar, self.proposals)
                self.proposal_revision = sidecar.read_bytes()
            except OSError as exc:
                self.update_controls()
                self.status.setText(f"Human answer saved in dataset; candidate display save failed: {exc}. Reopen to reconcile.")
                return False
        return True

    def source_confirm_yes(self):
        if not self.clip or not self.heard or self.playback_failed:
            return
        if self._persist(lambda: self.clip.update(track_confirmed=True), proposals=False):
            self.status.setText("Game-audio track confirmed. Now choose Yes, No or Unsure for gunfire.")
            self.update_controls()

    def change_role(self, index):
        if not self.clip:
            return
        target = self.role.itemData(index)
        if target == self.clip["split"]:
            return
        if self._persist(lambda: set_review_split(self.document, self.clip, target), proposals=False):
            title = self.clip.get("source_metadata", {}).get("title") or self.clip["file"]
            self.source.setText(f"{title}\n{self.clip['split']} source - audio stream {self.clip['audio_stream']}")
            self.status.setText("Recording group role saved. All clips in that group stay together; held-out test sources are unchanged.")
        self.role.setCurrentIndex(max(0, self.role.findData(self.clip["split"])))

    def skip_source(self):
        if not self.clip:
            return
        self.skipped_sources.add(self.clip["file"])
        self.advance_unanswered()

    def save_answer(self, answer):
        if not self.clip or not self.clip.get("track_confirmed") or not self.heard or self.playback_failed:
            self.status.setText("Listen to the complete sample and confirm game audio first.")
            return
        weapon = self.weapons.currentData() if answer == "yes" else None
        if self._persist(lambda: answer_presence(self.clip, self.row, answer, weapon=weapon)):
            self.player.pause()
            self.play_requested = False
            self.saved.setText(f"Saved answer: {answer}. Previous or Undo lets you change it.")
            self.status.setText("Answer saved locally.")
            self.update_controls()
            if self.auto_advance.isChecked():
                self.advance_unanswered()

    def undo_answer(self):
        if self.clip and self._persist(lambda: undo_presence(self.clip, self.row)):
            self.saved.setText("Answer removed. Listen and choose again.")
            self.status.setText("Undo saved. Existing interval annotations were preserved.")
            self.update_controls()

    def advance_unanswered(self):
        indices = list(range(self.index + 1, len(self.queue))) + list(range(0, self.index + 1))
        target = next((i for i in indices if self.queue[i][0]["file"] not in self.skipped_sources
                       and presence_answer(*self.queue[i]) is None), None)
        if target is None:
            self.player.pause()
            self.status.setText("No unanswered samples remain in this session. Previous lets you revise answers; skipped sources return on reopen.")
            return
        self.select(target)

    def previous(self):
        if self.index > 0:
            self.select(self.index - 1)

    def next(self):
        if self.index + 1 < len(self.queue):
            self.select(self.index + 1)

    def closeEvent(self, event):
        self.closed = True
        self.boundary_timer.stop()
        self.player.stop()
        event.accept()  # Every deliberate answer is saved immediately; no close-time overwrite.
