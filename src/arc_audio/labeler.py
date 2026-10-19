"""Watch local clips, choose an audio track, and record human-reviewed intervals."""

import copy
import sys
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .annotations import SOURCE_TAGS, remove_interval, write_interval
from .dataset import (
    CLASS_NAMES,
    TARGET_CLASSES,
    load_dataset,
    merge_reviewed,
    reviewed_spans,
    save_dataset,
    validate_span,
)
from .qt_interrupts import run_event_loop
from .suggestions import (
    accept_suggestion,
    change_suggestion_status,
    current_suggestions,
    load_suggestions,
    suggestion_edit,
    suggestion_path,
)


class MediaImportWorker(QObject):
    finished = Signal(object)

    def __init__(self, manifest, document, paths, group):
        super().__init__()
        self.arguments = (manifest, document, paths, group)

    @Slot()
    def run(self):
        try:
            from .media_import import import_media
            result = import_media(*self.arguments)
        except Exception as exc:
            result = {"added": [], "skipped": [], "errors": [str(exc)]}
        self.finished.emit(result)


class LabelWindow(QMainWindow):
    def __init__(self, manifest):
        super().__init__()
        self.manifest = Path(manifest).resolve()
        self.document = load_dataset(self.manifest)
        self.dataset_revision = self.manifest.read_bytes()
        self.proposals = load_suggestions(self.manifest)
        sidecar = suggestion_path(self.manifest)
        self.proposal_revision = sidecar.read_bytes() if sidecar.exists() else None
        self.active_suggestion_id = None
        self.preview_end = None
        self.editing_interval_index = None
        self.import_thread = None
        self.quit_pending = False
        self.document["classes"] = list(TARGET_CLASSES)
        if not self.document["clips"]:
            raise ValueError("No clips in dataset. Run prepare-training first.")
        self.clip = None
        self.setWindowTitle("ARC sound labels • local video review")
        self.resize(1100, 920)
        root = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(root)
        self.setCentralWidget(scroll)
        layout = QVBoxLayout(root)
        note = QLabel("Label audible events in each reviewed stretch. Choose which sound classes you fully checked below. "
                      "Unreviewed audio is excluded. Background should include machinery, ARC bots, chat and ambience. "
                      "Source tags are human notes, not enemy predictions.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.clips = QComboBox()
        for clip in self.document["clips"]:
            self.clips.addItem(f"[{clip['split']}] {clip['file']}")
        clip_row = QHBoxLayout()
        clip_row.addWidget(self.clips, 1)
        self.import_button = QPushButton("Import video / WAV files...")
        self.import_button.clicked.connect(self.choose_import_files)
        clip_row.addWidget(self.import_button)
        layout.addLayout(clip_row)
        track_row = QHBoxLayout()
        track_row.addWidget(QLabel("Audio track"))
        self.tracks = QComboBox()
        track_row.addWidget(self.tracks, 1)
        self.confirmed = QCheckBox("I listened and confirmed this is game audio")
        track_row.addWidget(self.confirmed)
        layout.addLayout(track_row)
        self.video = QVideoWidget()
        self.video.setMinimumHeight(240)
        layout.addWidget(self.video, 3)
        self.player = QMediaPlayer(self)
        self.output = QAudioOutput(self)
        self.output.setVolume(.5)
        self.player.setAudioOutput(self.output)
        self.player.setVideoOutput(self.video)
        self.seek = QSlider(Qt.Orientation.Horizontal)
        self.seek.setRange(0, 0)
        self.seek.sliderReleased.connect(lambda: self.player.setPosition(self.seek.value()))
        self.seek.sliderMoved.connect(self.preview_position)
        layout.addWidget(self.seek)
        self.clock = QLabel("0.00 / 0.00 seconds")
        transport = QHBoxLayout()
        for title, callback in [("−2 s", lambda: self.player.setPosition(max(0, self.player.position() - 2000))),
                                ("Play / pause", self.toggle_play),
                                ("+2 s", lambda: self.player.setPosition(self.player.position() + 2000))]:
            button = QPushButton(title)
            button.clicked.connect(callback)
            transport.addWidget(button)
        self.speed = QComboBox()
        for speed in (.5, .75, 1, 1.25):
            self.speed.addItem(f"{speed:g}× speed", speed)
        self.speed.setCurrentIndex(2)
        self.speed.currentIndexChanged.connect(lambda: self.player.setPlaybackRate(self.speed.currentData()))
        transport.addWidget(self.speed)
        transport.addWidget(self.clock)
        layout.addLayout(transport)
        edit = QHBoxLayout()
        self.start, self.end = QDoubleSpinBox(), QDoubleSpinBox()
        for spin in (self.start, self.end):
            spin.setDecimals(2)
            spin.setSingleStep(.05)
        edit.addWidget(QLabel("Start (s)"))
        edit.addWidget(self.start)
        start_button = QPushButton("Mark start here")
        start_button.clicked.connect(lambda: self.start.setValue(self.player.position() / 1000))
        edit.addWidget(start_button)
        edit.addWidget(QLabel("End (s)"))
        edit.addWidget(self.end)
        end_button = QPushButton("Mark end here")
        end_button.clicked.connect(lambda: self.end.setValue(self.player.position() / 1000))
        edit.addWidget(end_button)
        layout.addLayout(edit)
        actions = QHBoxLayout()
        self.labels = QComboBox()
        for label in TARGET_CLASSES:
            self.labels.addItem(CLASS_NAMES[label], label)
        self.sources = QComboBox()
        self.sources.addItems(list(SOURCE_TAGS))
        actions.addWidget(QLabel("Sound"))
        actions.addWidget(self.labels)
        actions.addWidget(QLabel("Source, if confirmed by video"))
        actions.addWidget(self.sources)
        add_button = QPushButton("Add manual sound interval")
        add_button.setToolTip("Add your own interval using Start, End and Sound above; no AI suggestion required.")
        add_button.clicked.connect(self.add_interval)
        actions.addWidget(add_button)
        review_button = QPushButton("Mark range fully reviewed")
        review_button.clicked.connect(self.add_reviewed)
        actions.addWidget(review_button)
        layout.addLayout(actions)
        layout.addWidget(QLabel("Classes fully checked in the range you are marking reviewed:"))
        review_grid = QGridLayout()
        self.review_checks = {}
        review_classes = self.document.get("review_classes", list(TARGET_CLASSES))
        for index, label in enumerate(TARGET_CLASSES):
            checkbox = QCheckBox(CLASS_NAMES[label])
            checkbox.setChecked(label in review_classes)
            checkbox.toggled.connect(self.update_review_coverage)
            self.review_checks[label] = checkbox
            review_grid.addWidget(checkbox, index // 4, index % 4)
        layout.addLayout(review_grid)
        whole_review = QHBoxLayout()
        self.whole_review_button = QPushButton("Mark whole clip fully reviewed")
        self.whole_review_button.clicked.connect(lambda: self.add_reviewed(whole_clip=True))
        whole_review.addWidget(self.whole_review_button)
        self.review_coverage = QLabel()
        self.review_coverage.setWordWrap(True)
        whole_review.addWidget(self.review_coverage, 1)
        layout.addLayout(whole_review)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Type", "Start", "End", "Sound", "Source"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.cellDoubleClicked.connect(lambda _row, _column: self.edit_selected_interval())
        self.table.horizontalHeader().setStretchLastSection(True)
        annotation_panel = QWidget()
        annotation_layout = QVBoxLayout(annotation_panel)
        annotation_layout.setContentsMargins(0, 0, 0, 0)
        annotation_layout.addWidget(self.table)
        interval_actions = QHBoxLayout()
        for title, callback in [("Edit selected interval", self.edit_selected_interval),
                                ("Save interval changes", self.save_interval_changes)]:
            button = QPushButton(title)
            button.clicked.connect(callback)
            interval_actions.addWidget(button)
        annotation_layout.addLayout(interval_actions)
        delete = QPushButton("Delete selected annotation / reviewed range")
        delete.clicked.connect(self.delete_rows)
        annotation_layout.addWidget(delete)
        proposal_panel = QWidget()
        proposal_layout = QVBoxLayout(proposal_panel)
        proposal_layout.setContentsMargins(0, 0, 0, 0)
        self.suggestion_note = QLabel()
        self.suggestion_note.setWordWrap(True)
        proposal_layout.addWidget(self.suggestion_note)
        self.suggestion_table = QTableWidget(0, 5)
        self.suggestion_table.setHorizontalHeaderLabels(["Current label", "Start", "End", "Original AI score", "Review status"])
        self.suggestion_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.suggestion_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.suggestion_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.suggestion_table.horizontalHeader().setStretchLastSection(True)
        self.suggestion_table.setColumnWidth(0, 240)
        self.suggestion_table.cellDoubleClicked.connect(lambda _row, _column: self.review_suggestion())
        self.suggestion_table.currentCellChanged.connect(self.suggestion_selection_changed)
        proposal_layout.addWidget(self.suggestion_table)
        self.accept_hint = QLabel()
        self.accept_hint.setWordWrap(True)
        self.accept_hint.setStyleSheet("color: #9a5700; font-weight: 600;")
        proposal_layout.addWidget(self.accept_hint)
        proposal_actions = QHBoxLayout()
        for title, callback in [("Review / edit selected suggestion", self.review_suggestion),
                                ("Accept / save selected suggestion", self.accept_selected_suggestion),
                                ("Reject selected AI suggestion", self.reject_selected_suggestion)]:
            button = QPushButton(title)
            button.clicked.connect(callback)
            if callback == self.accept_selected_suggestion:
                self.accept_button = button
            proposal_actions.addWidget(button)
        proposal_layout.addLayout(proposal_actions)
        reopen = QPushButton("Undo decision / return selected suggestion to pending")
        reopen.clicked.connect(self.reopen_selected_suggestion)
        proposal_layout.addWidget(reopen)
        self.review_tabs = QTabWidget()
        self.review_tabs.addTab(annotation_panel, "Saved labels / reviewed ranges")
        self.review_tabs.addTab(proposal_panel, "AI suggestions")
        self.review_tabs.setCurrentIndex(1 if self.proposals["clips"] else 0)
        layout.addWidget(self.review_tabs, 1)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.clips.currentIndexChanged.connect(self.select_clip)
        self.tracks.currentIndexChanged.connect(self.select_track)
        self.confirmed.toggled.connect(self.confirm_track)
        self.player.positionChanged.connect(self.position_changed)
        self.player.durationChanged.connect(self.duration_changed)
        self.player.mediaStatusChanged.connect(self.media_ready)
        self.player.errorOccurred.connect(lambda _error, message: self.status.setText(f"Playback error: {message}"))
        self.select_clip(0)

    def save(self):
        try:
            self.check_dataset_revision()
            save_dataset(self.manifest, self.document)
            self.dataset_revision = self.manifest.read_bytes()
            self.status.setText(f"Saved locally: {self.manifest}")
            return True
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Could not save labels", str(exc))
            return False

    def check_dataset_revision(self):
        if self.manifest.read_bytes() != self.dataset_revision:
            raise ValueError("This dataset changed in another window. Close this window and reopen the labeler to load the latest labels.")

    def apply_annotation_change(self, callback, *, with_proposals=False):
        """Rollback failed label saves and always rebuild sorted table indices."""
        self.check_dataset_revision()
        if with_proposals:
            self.check_proposal_revision()
        before_clip, before_proposals = copy.deepcopy(self.clip), copy.deepcopy(self.proposals)

        def restore():
            self.clip.clear()
            self.clip.update(before_clip)
            self.proposals = before_proposals

        try:
            try:
                callback()
            except Exception:
                restore()
                raise
            if not self.save():
                restore()
                return False
            if with_proposals:
                try:
                    self.save_proposals()
                except (OSError, ValueError) as exc:
                    raise ValueError(f"Sound labels were saved, but the suggestion decision could not be saved: {exc}. Review that row again to retry; linked labels will not be duplicated.") from exc
            return True
        finally:
            self.refresh_table()

    def choose_import_files(self):
        paths, _filter = QFileDialog.getOpenFileNames(
            self, "Import local recordings", str(self.manifest.parent),
            "Audio/video recordings (*.mp4 *.mkv *.mov *.wav)", options=QFileDialog.Option.DontUseNativeDialog)
        if not paths:
            return
        group, ok = QInputDialog.getText(
            self, "Recording session", "Session name for these recordings (same real session = same name).\n"
            "An existing session keeps its split; a new session starts in train:")
        if ok:
            self.start_media_import(paths, group)

    def start_media_import(self, paths, group):
        if self.import_thread is not None:
            self.status.setText("An import is already running; wait for it to finish.")
            return
        try:
            self.check_dataset_revision()
            if not group.strip():
                raise ValueError("Enter the real recording session name before importing")
            self.import_thread = QThread(self)
            self.import_worker = MediaImportWorker(self.manifest, copy.deepcopy(self.document), paths, group)
            self.import_worker.moveToThread(self.import_thread)
            self.import_thread.started.connect(self.import_worker.run)
            self.import_worker.finished.connect(self.complete_media_import)
            self.import_worker.finished.connect(self.import_thread.quit)
            self.import_worker.finished.connect(self.import_worker.deleteLater)
            self.import_thread.finished.connect(self.import_finished)
            self.import_button.setEnabled(False)
            self.status.setText("Importing local recordings in the background; original files are preserved...")
            self.import_thread.start()
        except (OSError, ValueError) as exc:
            self.status.setText(str(exc))

    @Slot(object)
    def complete_media_import(self, result):
        try:
            self.check_dataset_revision()
            first = len(self.document["clips"])
            self.document["clips"].extend(result["added"])
            if result["added"]:
                if not self.save():
                    del self.document["clips"][first:]
                    self.status.setText("Recordings were copied, but the dataset could not be saved. Copied files remain in imported/; resolve the save error and import those files again.")
                    return
                self.clips.blockSignals(True)
                for clip in result["added"]:
                    self.clips.addItem(f"[{clip['split']}] {clip['file']}")
                self.clips.blockSignals(False)
                self.clips.setCurrentIndex(first)
            messages = [f"Imported {len(result['added'])} recording(s), skipped {len(result['skipped'])}."]
            messages.extend(result["errors"])
            messages.extend(result["skipped"])
            self.status.setText("\n".join(messages) + "\nListen to and confirm the selected audio track before labeling.")
        except (OSError, ValueError) as exc:
            self.status.setText(f"Import could not update the dataset: {exc}. Copied files are retained locally.")

    @Slot()
    def import_finished(self):
        thread = self.import_thread
        self.import_thread = None
        self.import_worker = None
        self.import_button.setEnabled(True)
        if thread is not None:
            thread.deleteLater()
        if self.quit_pending:
            QTimer.singleShot(0, self.close)

    def select_clip(self, index):
        self.player.stop()
        self.preview_end = None
        self.clip = self.document["clips"][index]
        self.tracks.blockSignals(True)
        self.tracks.clear()
        for stream in self.clip["audio_streams"]:
            title = stream.get("title") or "untitled"
            self.tracks.addItem(f"Stream {stream['index']} • {title} • {stream['channels']} ch", stream["index"])
        self.tracks.setCurrentIndex(self.tracks.findData(self.clip["audio_stream"]))
        self.tracks.blockSignals(False)
        self.confirmed.blockSignals(True)
        self.confirmed.setChecked(bool(self.clip.get("track_confirmed")))
        self.confirmed.blockSignals(False)
        for spin in (self.start, self.end):
            spin.setRange(0, self.clip["duration"])
        self.start.setValue(0)
        self.end.setValue(min(5, self.clip["duration"]))
        self.seek.setRange(0, round(self.clip["duration"] * 1000))
        self.player.setSource(QUrl.fromLocalFile(str(self.manifest.parent / self.clip["file"])))
        self.refresh_table()
        self.status.setText(f"Group {self.clip['group']} / {self.clip['split']}. Listen to the tracks and confirm game audio first.")

    def media_ready(self, status):
        if status in (QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia):
            self.player.setActiveAudioTrack(self.tracks.currentIndex())

    def select_track(self, _index):
        selected = self.tracks.currentData()
        if selected is None or not self.clip or selected == self.clip["audio_stream"]:
            return
        if self.clip.get("intervals") or self.clip.get("reviewed"):
            QMessageBox.information(self, "Track already labelled", "Delete this clip's labels/reviewed ranges before changing tracks; their timestamps refer to the selected audio.")
            self.tracks.blockSignals(True)
            self.tracks.setCurrentIndex(self.tracks.findData(self.clip["audio_stream"]))
            self.tracks.blockSignals(False)
            return
        self.clip["audio_stream"] = selected
        self.confirmed.setChecked(False)
        self.clip["track_confirmed"] = False
        self.player.setActiveAudioTrack(self.tracks.currentIndex())
        self.refresh_suggestions()
        self.save()

    def confirm_track(self, value):
        self.clip["track_confirmed"] = value
        self.save()
        self.update_accept_state()

    def toggle_play(self):
        self.preview_end = None
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def preview_position(self, position):
        self.clock.setText(f"{position / 1000:.2f} / {self.clip['duration']:.2f} seconds")

    def position_changed(self, position):
        if self.preview_end is not None and position >= self.preview_end:
            self.player.pause()
            self.preview_end = None
        if not self.seek.isSliderDown():
            self.seek.setValue(position)
            self.preview_position(position)

    def duration_changed(self, duration):
        self.seek.setRange(0, duration)

    def selected_span(self):
        if not self.clip.get("track_confirmed"):
            raise ValueError("Listen to and confirm the game-audio track first")
        row = {"start": self.start.value(), "end": self.end.value()}
        validate_span(row, self.clip["duration"])
        return row

    def add_interval(self):
        try:
            self.check_dataset_revision()
            row = self.selected_span()
            if self.apply_annotation_change(lambda: write_interval(
                    self.clip, row["start"], row["end"], self.labels.currentData(), self.sources.currentText())):
                self.review_tabs.setCurrentIndex(0)
                self.status.setText("Manual sound interval saved. No AI suggestion required; confirm fully reviewed coverage separately.")
        except (OSError, ValueError) as exc:
            self.status.setText(str(exc))

    def edit_selected_interval(self):
        try:
            selected = self.table.selectionModel().selectedRows()
            if len(selected) != 1:
                raise ValueError("Select one saved sound interval to edit")
            kind, index = self.row_keys[selected[0].row()]
            if kind != "intervals":
                raise ValueError("Select a sound interval. To change reviewed coverage, delete it and review the new range.")
            self.editing_interval_index = index
            self.active_suggestion_id = None
            row = self.clip["intervals"][index]
            self.start.setValue(row["start"])
            self.end.setValue(row["end"])
            self.labels.setCurrentIndex(self.labels.findData(row["label"]))
            self.sources.setCurrentText(row.get("source", "unknown"))
            self.preview_interval(row)
            self.status.setText("Editing a saved interval. Adjust Start, End, Sound and Source above, then click Save interval changes.")
        except ValueError as exc:
            self.status.setText(str(exc))

    def save_interval_changes(self):
        try:
            self.check_dataset_revision()
            if self.editing_interval_index is None:
                raise ValueError("Use Edit selected interval first")
            previous = self.clip["intervals"][self.editing_interval_index]
            proposal_id = previous.get("proposal_id")
            proposal = next((row for row in self.suggestion_rows if row["id"] == proposal_id), None)
            if proposal:
                self.check_proposal_revision()
            def update():
                interval = write_interval(self.clip, self.start.value(), self.end.value(),
                                          self.labels.currentData(), self.sources.currentText(),
                                          index=self.editing_interval_index)
                if proposal:
                    proposal["status"] = "accepted"
                    proposal["last_edit"] = dict(interval)

            if self.apply_annotation_change(update, with_proposals=proposal is not None):
                self.status.setText("Interval changes saved. Affected class/time coverage needs review again; unrelated review coverage is preserved.")
        except (OSError, ValueError) as exc:
            self.status.setText(str(exc))

    def preview_interval(self, row):
        self.player.setPosition(round(max(0, row["start"] - .5) * 1000))
        self.preview_end = round(min(self.clip["duration"], row["end"] + .5) * 1000)
        self.player.play()

    def update_review_coverage(self):
        if not self.clip:
            return
        classes = [label for label, checkbox in self.review_checks.items() if checkbox.isChecked()]
        if not classes:
            self.review_coverage.setText("Select the sound classes you fully checked.")
            return
        spans = reviewed_spans(self.clip, tuple(classes))
        seconds = sum(row["end"] - row["start"] for row in spans)
        labels = [row for row in self.clip["intervals"] if row["label"] in classes]
        included = sum(any(span["start"] <= row["start"] and row["end"] <= span["end"] for span in spans)
                       for row in labels)
        self.review_coverage.setText(
            f"Checked classes: {seconds:.2f} / {self.clip['duration']:.2f} seconds reviewed; "
            f"{included}/{len(labels)} sound labels fully inside reviewed coverage.")

    def add_reviewed(self, whole_clip=False):
        try:
            self.check_dataset_revision()
            if whole_clip:
                if not self.clip.get("track_confirmed"):
                    raise ValueError("Listen to and confirm the game-audio track first")
                row = {"start": 0, "end": self.clip["duration"]}
            else:
                row = self.selected_span()
            selected = [label for label, checkbox in self.review_checks.items() if checkbox.isChecked()]
            if not selected:
                raise ValueError("Select the sound classes you fully checked before marking a reviewed range")
            answer = QMessageBox.question(self, "Confirm full review", "Have you labelled ALL audible events for these classes in this entire range?\n\n"
                                          + f"{self.clip['file']}\n{row['start']:.2f} to {row['end']:.2f} seconds"
                                          + (" (whole clip)" if whole_clip else "") + "\n\n"
                                          + ", ".join(CLASS_NAMES[label] for label in selected)
                                          + "\n\nUnlabelled parts become negative examples for these classes only.")
            if answer != QMessageBox.StandardButton.Yes:
                return
            row["classes"] = selected
            self.apply_annotation_change(lambda: self.clip.update(
                reviewed=merge_reviewed(self.clip["reviewed"] + [row])))
        except (OSError, ValueError) as exc:
            self.status.setText(str(exc))

    def refresh_table(self):
        self.editing_interval_index = None
        self.table.setRowCount(0)
        self.row_keys = []
        for kind in ("intervals", "reviewed"):
            for index, row in enumerate(self.clip[kind]):
                at = self.table.rowCount()
                self.table.insertRow(at)
                for column, value in enumerate([kind, f"{row['start']:.2f}", f"{row['end']:.2f}",
                                                row.get("label", ", ".join(row.get("classes", ["footsteps", "reload", "gunshot"]))),
                                                row.get("source", "")]):
                    self.table.setItem(at, column, QTableWidgetItem(value))
                self.row_keys.append((kind, index))
        self.refresh_suggestions()
        self.update_review_coverage()

    def refresh_suggestions(self):
        self.active_suggestion_id = None
        self.suggestion_rows = current_suggestions(self.proposals, self.manifest, self.clip)
        self.suggestion_table.blockSignals(True)
        self.suggestion_table.setRowCount(0)
        for row in self.suggestion_rows:
            edit = suggestion_edit(self.clip, row)
            at = self.suggestion_table.rowCount()
            self.suggestion_table.insertRow(at)
            name = CLASS_NAMES[edit["label"]]
            if edit["label"] != row["label"]:
                name += f" (AI: {CLASS_NAMES[row['label']]})"
            for column, value in enumerate([name, f"{edit['start']:.2f}",
                                            f"{edit['end']:.2f}", f"{row['confidence']:.3f}", row["status"]]):
                item = QTableWidgetItem(value)
                item.setToolTip(f"Original AI proposal: {CLASS_NAMES[row['label']]} / {row['start']:.2f}-{row['end']:.2f}s. Score belongs to that original class/window.")
                self.suggestion_table.setItem(at, column, item)
        self.suggestion_table.blockSignals(False)
        self.update_accept_state()
        pending = sum(row["status"] == "pending" for row in self.suggestion_rows)
        self.suggestion_note.setText(
            f"AI review: {pending} pending suggestions for the selected track. Scores are not accuracy probabilities. "
            "Review/edit any row, including accepted/rejected ones; accept/update, reject, or undo its decision. "
            "AI scores belong to the original proposed class/window. Check misses before marking ranges reviewed. "
            "Proposals for changed media/other tracks are hidden.")

    def update_accept_state(self):
        if not self.clip.get("track_confirmed"):
            hint = "To accept: listen to the selected audio track, then tick 'I listened and confirmed this is game audio' above."
        elif self.suggestion_table.currentRow() < 0 or self.active_suggestion_id is None:
            hint = "Select a suggestion row to load its Start, End and Sound fields, then click Accept."
        else:
            hint = "Ready to accept. Check the Start, End and Sound fields above; Review / edit plays the sound if needed."
        ready = bool(self.clip.get("track_confirmed") and self.active_suggestion_id is not None
                     and self.suggestion_table.currentRow() >= 0)
        self.accept_button.setEnabled(ready)
        self.accept_button.setToolTip(hint)
        self.accept_hint.setText(hint)

    def load_suggestion_fields(self, row):
        edit = suggestion_edit(self.clip, row)
        self.active_suggestion_id = row["id"]
        self.editing_interval_index = None
        self.start.setValue(edit["start"])
        self.end.setValue(edit["end"])
        self.labels.setCurrentIndex(self.labels.findData(edit["label"]))
        self.sources.setCurrentText(edit.get("source", "unknown"))
        return edit

    def suggestion_selection_changed(self, current_row, _column, previous_row, _previous_column):
        try:
            row = self.selected_suggestion()
            if current_row != previous_row or self.active_suggestion_id != row["id"]:
                self.load_suggestion_fields(row)
        except (OSError, ValueError) as exc:
            self.active_suggestion_id = None
            self.status.setText(str(exc))
        self.update_accept_state()

    def selected_suggestion(self):
        index = self.suggestion_table.currentRow()
        if index < 0:
            raise ValueError("Select an AI suggestion first")
        row = self.suggestion_rows[index]
        # Recheck the media stamp rather than accepting a stale proposal.
        if row not in current_suggestions(self.proposals, self.manifest, self.clip):
            raise ValueError("Media or track changed; regenerate suggestions")
        return row

    def review_suggestion(self):
        try:
            row = self.selected_suggestion()
            edit = self.load_suggestion_fields(row)
            self.update_accept_state()
            self.preview_interval(edit)
            self.status.setText("Playing suggestion with context. Adjust interval/class above; acceptance requires confirmed game audio.")
        except (OSError, ValueError) as exc:
            self.status.setText(str(exc))

    def check_proposal_revision(self):
        sidecar = suggestion_path(self.manifest)
        current = sidecar.read_bytes() if sidecar.exists() else None
        if current != self.proposal_revision:
            raise ValueError("Suggestions changed in another process. Reopen the labeler before continuing review; nothing overwritten.")

    def save_proposals(self):
        self.check_proposal_revision()
        sidecar = suggestion_path(self.manifest)
        save_dataset(sidecar, self.proposals)
        self.proposal_revision = sidecar.read_bytes()

    def accept_selected_suggestion(self):
        try:
            self.check_dataset_revision()
            self.check_proposal_revision()
            row = self.selected_suggestion()
            if self.active_suggestion_id != row["id"]:
                self.load_suggestion_fields(row)
            self.selected_span()  # Require a valid span and a human-confirmed audio track.
            if self.apply_annotation_change(lambda: accept_suggestion(
                    self.clip, row, self.start.value(), self.end.value(), self.labels.currentData(),
                    self.sources.currentText()), with_proposals=True):
                self.status.setText("Accepted/updated and saved as a human label. Check affected ranges for missed events before marking fully reviewed.")
        except (OSError, ValueError) as exc:
            self.status.setText(str(exc))
            self.update_accept_state()

    def reject_selected_suggestion(self):
        self.set_selected_decision("rejected")

    def reopen_selected_suggestion(self):
        self.set_selected_decision("pending")

    def set_selected_decision(self, status):
        try:
            self.check_dataset_revision()
            self.check_proposal_revision()
            row = self.selected_suggestion()
            if self.apply_annotation_change(lambda: change_suggestion_status(self.clip, row, status), with_proposals=True):
                self.status.setText(f"Decision changed to {status}. Any linked label was removed; affected class/time coverage needs review. You can edit or accept this row again.")
        except (OSError, ValueError) as exc:
            self.status.setText(str(exc))

    def delete_rows(self):
        try:
            self.check_dataset_revision()
            self.check_proposal_revision()
            keys = {self.row_keys[index.row()] for index in self.table.selectionModel().selectedRows()}
            intervals = [row for index, row in enumerate(self.clip["intervals"]) if ("intervals", index) in keys]
            def delete():
                self.clip["reviewed"] = [row for index, row in enumerate(self.clip["reviewed"]) if ("reviewed", index) not in keys]
                for interval in intervals:
                    proposal = next((row for row in self.suggestion_rows if row["id"] == interval.get("proposal_id")), None)
                    if proposal:
                        change_suggestion_status(self.clip, proposal, "pending")
                    else:
                        remove_interval(self.clip, interval)

            if self.apply_annotation_change(delete, with_proposals=True):
                self.status.setText("Selection deleted. Linked suggestions were reopened and affected review coverage cleared.")
        except (OSError, ValueError) as exc:
            self.status.setText(str(exc))

    def closeEvent(self, event):
        if self.import_thread is not None:
            self.quit_pending = True
            self.player.stop()
            self.status.setText("Closing after the current import finishes; original recordings and labels are preserved.")
            event.ignore()
            return
        self.player.stop()
        try:
            self.check_dataset_revision()
        except (OSError, ValueError):
            answer = QMessageBox.question(self, "Dataset changed externally",
                                          "Close this stale window without saving? The newer dataset on disk will be preserved.")
            if answer == QMessageBox.StandardButton.Yes:
                event.accept()
            else:
                event.ignore()
            return
        if self.save():
            event.accept()
        else:
            event.ignore()


def run_labeler(manifest, smoke_seconds=0):
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = LabelWindow(manifest)
    window.show()
    if smoke_seconds:
        QTimer.singleShot(round(smoke_seconds * 1000), window.close)
    return run_event_loop(app, window.close)
