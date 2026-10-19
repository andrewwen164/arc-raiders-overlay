"""Watch local clips, choose an audio track, and record human-reviewed intervals."""

import copy
import sys
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, Qt, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
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
        self.preview_seek_pending = None
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
        self.video.setMinimumHeight(180)
        self.video.setMaximumHeight(300)
        self.video.hide()
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
        weapon_row = QHBoxLayout()
        weapon_row.addWidget(QLabel("Weapon (gunfire only; human confirmation)"))
        self.weapons = QComboBox()
        self.weapons.activated.connect(lambda _index: self.suggestion_table.setFocus())
        self.weapons.setToolTip("Source metadata proposes a weapon; this is not an audio weapon classifier. Leave unknown when unsure.")
        weapon_row.addWidget(self.weapons, 1)
        self.correct_weapon_button = QPushButton("Correct Weapon")
        self.correct_weapon_button.clicked.connect(self.focus_weapon_correction)
        weapon_row.addWidget(self.correct_weapon_button)
        self.save_weapon_button = QPushButton("Save weapon correction / accept")
        self.save_weapon_button.clicked.connect(self.accept_selected_suggestion)
        weapon_row.addWidget(self.save_weapon_button)
        layout.addLayout(weapon_row)
        self.populate_weapons()
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
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Type", "Start", "End", "Sound", "Source", "Weapon"])
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
        self.create_review_filters(proposal_layout)
        self.source_details = QLabel("Select a candidate to see its source context.")
        self.source_details.setTextFormat(Qt.TextFormat.PlainText)
        self.source_details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.source_details.setWordWrap(True)
        proposal_layout.addWidget(self.source_details)
        self.suggestion_table = QTableWidget(0, 6)
        self.suggestion_table.setHorizontalHeaderLabels(["Current label", "Start", "End", "Detection / model score", "Review status", "Weapon"])
        self.suggestion_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.suggestion_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.suggestion_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.suggestion_table.horizontalHeader().setStretchLastSection(True)
        self.suggestion_table.setColumnWidth(0, 240)
        self.suggestion_table.setColumnWidth(3, 160)
        self.suggestion_table.setColumnWidth(4, 110)
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
        navigation = QHBoxLayout()
        for title, step in (("Previous (Left)", -1), ("Next (Right)", 1)):
            button = QPushButton(title)
            button.clicked.connect(lambda _checked=False, direction=step: self.navigate_candidate(direction))
            navigation.addWidget(button)
        self.auto_advance = QCheckBox("Auto-advance after Accept / Reject")
        navigation.addWidget(self.auto_advance)
        self.autoplay = QCheckBox("Autoplay next candidate")
        self.autoplay.setChecked(True)
        navigation.addWidget(self.autoplay)
        proposal_layout.addLayout(navigation)
        proposal_layout.addWidget(QLabel("Keys: A Accept · R Reject · W Correct Weapon · Left/Right Previous/Next · Space Play/Pause. Keys do not act while editing fields."))
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
        QApplication.instance().installEventFilter(self)

    def create_review_filters(self, layout):
        grid = QGridLayout()
        self.weapon_filter, self.source_filter = QComboBox(), QComboBox()
        self.status_filter, self.duplicate_filter = QComboBox(), QComboBox()
        self.sort_candidates = QComboBox()
        self.weapon_filter.addItem("All weapons", None)
        self.weapon_filter.addItem("Unknown / ambiguous", "__unknown__")
        for index in range(1, self.weapons.count()):
            name = self.weapons.itemData(index)
            self.weapon_filter.addItem(name, name)
        self.source_filter.addItem("All sources", None)
        self.source_filter.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.source_filter.setMinimumContentsLength(12)
        for clip in self.document["clips"]:
            metadata = clip.get("source_metadata", {})
            title = metadata.get("title") or metadata.get("source_title") or clip["file"]
            self.source_filter.addItem(str(title), clip["file"])
        for status in (None, "pending", "accepted", "rejected", "corrected"):
            self.status_filter.addItem(status or "All statuses", status)
        self.duplicate_filter.addItem("All duplicate groups", None)
        groups = {row["duplicate_group_id"] for entry in self.proposals["clips"]
                  for row in entry["suggestions"] if row.get("duplicate_group_id")}
        for group in sorted(groups):
            self.duplicate_filter.addItem(group, group)
        for title, value in (("Chronological", "time"), ("Highest quality first", "quality"),
                             ("Weapon", "weapon"), ("Source", "source")):
            self.sort_candidates.addItem(title, value)
        self.minimum_quality, self.minimum_detection = QDoubleSpinBox(), QDoubleSpinBox()
        self.minimum_quality.setRange(0, 100)
        self.minimum_detection.setRange(0, 1)
        self.minimum_detection.setSingleStep(.05)
        fields = (("Weapon", self.weapon_filter), ("Source", self.source_filter),
                  ("Status", self.status_filter), ("Duplicate group", self.duplicate_filter),
                  ("Minimum quality", self.minimum_quality), ("Minimum detection", self.minimum_detection),
                  ("Sort", self.sort_candidates))
        for index, (title, widget) in enumerate(fields):
            grid.addWidget(QLabel(title), index // 3, index % 3 * 2)
            grid.addWidget(widget, index // 3, index % 3 * 2 + 1)
            if isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self.review_filter_changed)
                widget.activated.connect(lambda _index: self.suggestion_table.setFocus())
            else:
                widget.valueChanged.connect(self.review_filter_changed)
        layout.addLayout(grid)
        self.review_progress = QLabel()
        self.review_progress.setWordWrap(True)
        layout.addWidget(self.review_progress)

    def all_candidates(self):
        return [(clip, row) for clip in self.document["clips"]
                for row in current_suggestions(self.proposals, self.manifest, clip)]

    def candidate_matches(self, clip, row):
        edit = suggestion_edit(clip, row)
        weapon = edit.get("weapon")
        selected_weapon = self.weapon_filter.currentData()
        if selected_weapon == "__unknown__" and weapon:
            return False
        if selected_weapon not in (None, "__unknown__") and selected_weapon != weapon:
            if weapon or selected_weapon not in row.get("possible_weapons", []):
                return False
        return (self.source_filter.currentData() in (None, clip["file"])
                and self.status_filter.currentData() in (None, row["status"])
                and self.duplicate_filter.currentData() in (None, row.get("duplicate_group_id"))
                and row.get("quality_score", 0) >= self.minimum_quality.value()
                and row.get("detection_score", row["confidence"]) >= self.minimum_detection.value())

    def candidate_sort_key(self, item):
        clip, row = item
        edit = suggestion_edit(clip, row)
        base = (clip["file"], edit["start"], row["id"])
        mode = self.sort_candidates.currentData()
        if mode == "quality":
            return (-row.get("quality_score", 0), -row.get("detection_score", row["confidence"]), *base)
        if mode == "weapon":
            return (edit.get("weapon") or "", *base)
        if mode == "source":
            return (row.get("source_title", clip["file"]), *base)
        return base

    def candidate_queue(self):
        return sorted((item for item in self.all_candidates() if self.candidate_matches(*item)),
                      key=self.candidate_sort_key)

    def review_filter_changed(self, *_args):
        current_id = self.active_suggestion_id
        queue = self.candidate_queue()
        self.refresh_suggestions()
        selected = next((item for item in queue if item[1]["id"] == current_id), queue[0] if queue else None)
        if selected:
            self.select_candidate(*selected)
        elif hasattr(self, "status"):
            self.status.setText("No candidates match these filters. Lower thresholds or choose All statuses / sources.")

    def select_candidate(self, clip, row, *, play=False):
        index = self.document["clips"].index(clip)
        if self.clips.currentIndex() != index:
            self.clips.setCurrentIndex(index)
        for at, candidate in enumerate(self.suggestion_rows):
            if candidate["id"] == row["id"]:
                self.suggestion_table.selectRow(at)
                self.suggestion_table.scrollToItem(self.suggestion_table.item(at, 0))
                self.review_tabs.setCurrentIndex(1)
                if play:
                    self.review_suggestion()
                return

    def navigate_candidate(self, direction):
        queue = self.candidate_queue()
        if not queue:
            self.status.setText("No candidates match the current filters.")
            return
        current = next((index for index, item in enumerate(queue) if item[1]["id"] == self.active_suggestion_id), None)
        target = (0 if direction > 0 else len(queue) - 1) if current is None else current + direction
        if not 0 <= target < len(queue):
            self.status.setText("Reached the end of the filtered review queue.")
            return
        self.select_candidate(*queue[target], play=self.autoplay.isChecked())

    def advance_after_decision(self, identity, previous_queue):
        if not self.auto_advance.isChecked():
            return
        pending = [item for item in self.candidate_queue() if item[1]["status"] == "pending"]
        if not pending:
            self.player.pause()
            self.status.setText("Decision saved. No pending candidates remain in the filtered queue.")
            return
        old_ids = [item[1]["id"] for item in previous_queue]
        at = old_ids.index(identity) if identity in old_ids else -1
        following = set(old_ids[at + 1:])
        selected = next((item for item in pending if item[1]["id"] in following), pending[0])
        self.select_candidate(*selected, play=self.autoplay.isChecked())

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.KeyPress and self.isActiveWindow() and not event.isAutoRepeat():
            focus = QApplication.focusWidget()
            editing = isinstance(focus, (QAbstractSpinBox, QLineEdit, QComboBox, QTextEdit, QPlainTextEdit, QSlider))
            editing = editing or (isinstance(focus, QCheckBox) and event.key() == Qt.Key.Key_Space)
            modal = QApplication.activeModalWidget()
            if not editing and modal is None and event.modifiers() == Qt.KeyboardModifier.NoModifier:
                actions = {Qt.Key.Key_A: self.accept_selected_suggestion, Qt.Key.Key_R: self.reject_selected_suggestion,
                           Qt.Key.Key_W: self.focus_weapon_correction, Qt.Key.Key_Left: lambda: self.navigate_candidate(-1),
                           Qt.Key.Key_Right: lambda: self.navigate_candidate(1), Qt.Key.Key_Space: self.toggle_play}
                callback = actions.get(event.key())
                if callback is not None and self.review_tabs.currentIndex() == 1:
                    callback()
                    return True
        return super().eventFilter(watched, event)

    def populate_weapons(self, selected=None):
        """Manifest vocabulary plus legacy human values; unknown stays available."""
        names = set(self.document.get("weapons", []))
        for clip in self.document["clips"]:
            names.update(row["weapon"] for row in clip["intervals"] if row.get("weapon"))
        for entry in self.proposals["clips"]:
            for row in entry["suggestions"]:
                names.update(row.get("possible_weapons", []))
                if row.get("proposed_weapon"):
                    names.add(row["proposed_weapon"])
        if selected:
            names.add(selected)
        self.weapons.clear()
        self.weapons.addItem("Unknown / unsure", None)
        for name in sorted(names):
            self.weapons.addItem(name, name)
        self.weapons.setCurrentIndex(max(0, self.weapons.findData(selected)))

    def focus_weapon_correction(self):
        if self.active_suggestion_id is None:
            self.status.setText("Select a suggestion first, then choose its weapon and save the correction.")
            return
        if self.selected_suggestion().get("proposal_kind") == "gunshot_absence":
            self.status.setText("A background proposal asks whether this whole range has no gunfire; it has no weapon to correct.")
            return
        self.weapons.setFocus()
        self.weapons.showPopup()
        self.status.setText("Choose the audible weapon, or Unknown / unsure, then Save weapon correction / accept. This creates a human gunfire label.")

    def show_source_details(self, row):
        metadata = self.clip.get("source_metadata", {})
        weapon = row.get("proposed_weapon") or "unknown / ambiguous"
        possible = ", ".join(row.get("possible_weapons", []))
        title = row.get("source_title") or metadata.get("title") or metadata.get("source_title") or self.clip["file"]
        timestamp = row.get("source_timestamp", row["start"])
        lines = [f"Source: {title}", f"Source timestamp: {timestamp:.2f}s | Proposed weapon: {weapon}"
                 + (f" | Possible weapons: {possible}" if possible else "")]
        url = row.get("source_url") or metadata.get("source_url") or metadata.get("url")
        if url:
            lines.append(f"URL: {url}")
        for key, name in (("manifest_quality", "Manifest quality"), ("manifest_notes", "Manifest notes")):
            if row.get(key) is not None:
                lines.append(f"{name}: {row[key]}")
        scores = []
        for key, name in (("detection_score", "Detection heuristic"), ("quality_score", "Quality /100"),
                          ("duplicate_group_id", "Duplicate group")):
            if row.get(key) is not None:
                value = f"{row[key]:.3f}" if key == "detection_score" else f"{row[key]:.1f}" if key == "quality_score" else row[key]
                scores.append(f"{name}: {value}")
        lines.append(" | ".join(scores + [f"Status: {row['status']}"]))
        lines.append("Source context proposes weapon identity; it is not verified weapon recognition.")
        if row.get("proposal_kind") == "gunshot_absence":
            lines.insert(0, "BACKGROUND QUESTION: Did you listen to the entire Start–End range and confirm NO gunfire? "
                         "Acceptance reviews only gunshot absence; other classes remain unknown. "
                         "Zero detection score is a sampling heuristic, not a trained negative prediction.")
        self.source_details.setText("\n".join(lines))

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
        before_clip = copy.deepcopy(self.clip)
        # Annotation callbacks below only edit this clip and its selected-track
        # proposal rows. Keep a full rollback for that scope without copying
        # immutable candidate features for every unrelated source on each click.
        before_proposals = dict(self.proposals, clips=list(self.proposals["clips"]))
        if with_proposals:
            for index, entry in enumerate(self.proposals["clips"]):
                if entry["file"] == self.clip["file"] and entry["audio_stream"] == self.clip["audio_stream"]:
                    before_proposals["clips"][index] = copy.deepcopy(entry)

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
        self.video.hide()
        self.preview_end = None
        self.preview_seek_pending = None
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
        self.populate_weapons()
        self.seek.setRange(0, round(self.clip["duration"] * 1000))
        self.player.setSource(QUrl.fromLocalFile(str(self.manifest.parent / self.clip["file"])))
        self.refresh_table()
        self.status.setText(f"Group {self.clip['group']} / {self.clip['split']}. Listen to the tracks and confirm game audio first.")

    def media_ready(self, status):
        if status in (QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia):
            # Acquired remote sources use bestaudio. An old source's queued
            # readiness notification must not reopen a blank video pane while
            # a newly selected audio-only source is still loading.
            metadata = self.clip.get("source_metadata", {})
            allowed_video = self.clip.get("has_video", not (self.clip.get("collection_source_id")
                                                          and not metadata.get("original_path")))
            self.video.setVisible(bool(allowed_video and self.player.hasVideo()))
            self.player.setActiveAudioTrack(self.tracks.currentIndex())
            if self.preview_seek_pending is not None:
                self.player.setPosition(self.preview_seek_pending)
                self.preview_seek_pending = None
                self.player.play()

    def select_track(self, _index):
        selected = self.tracks.currentData()
        if selected is None or not self.clip or selected == self.clip["audio_stream"]:
            return
        if (self.clip.get("intervals") or self.clip.get("reviewed") or self.clip.get("presence_reviews")
                or self.clip.get("presence_states")):
            QMessageBox.information(self, "Track already labelled", "This track has saved labels, reviewed ranges or quick answers. Undo/remove its annotations before changing tracks; they refer to the selected audio.")
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
        self.preview_seek_pending = None
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
                    self.clip, row["start"], row["end"], self.labels.currentData(), self.sources.currentText(),
                    weapon=self.weapons.currentData() or "")):
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
            self.populate_weapons(row.get("weapon"))
            self.preview_interval(row)
            self.update_accept_state()
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
            proposal = next((row for row in current_suggestions(self.proposals, self.manifest, self.clip)
                             if row["id"] == proposal_id), None)
            if proposal:
                self.check_proposal_revision()
            def update():
                interval = write_interval(self.clip, self.start.value(), self.end.value(),
                                          self.labels.currentData(), self.sources.currentText(),
                                          index=self.editing_interval_index, weapon=self.weapons.currentData() or "")
                if proposal:
                    proposed_weapon = proposal.get("proposed_weapon", proposal.get("weapon"))
                    proposal["status"] = "corrected" if interval.get("weapon") != proposed_weapon else "accepted"
                    proposal["last_edit"] = dict(interval)

            if self.apply_annotation_change(update, with_proposals=proposal is not None):
                self.status.setText("Interval changes saved. Affected class/time coverage needs review again; unrelated review coverage is preserved.")
        except (OSError, ValueError) as exc:
            self.status.setText(str(exc))

    def preview_interval(self, row):
        first = row.get("preview_start", row["start"] - .5)
        last = row.get("preview_end", row["end"] + .5)
        position = round(max(0, first) * 1000)
        self.preview_end = round(min(self.clip["duration"], last) * 1000)
        if self.player.mediaStatus() in (QMediaPlayer.MediaStatus.NoMedia, QMediaPlayer.MediaStatus.LoadingMedia):
            self.preview_seek_pending = position
            return
        self.preview_seek_pending = None
        self.player.setPosition(position)
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
                                                row.get("source", ""), row.get("weapon", "")]):
                    self.table.setItem(at, column, QTableWidgetItem(value))
                self.row_keys.append((kind, index))
        self.refresh_suggestions()
        self.update_review_coverage()

    def refresh_suggestions(self):
        self.active_suggestion_id = None
        # Stat each source once for a refresh, never once per candidate. Large
        # collections can contain thousands of proposals for the same source.
        all_rows = self.all_candidates()
        matching = [item for item in all_rows if self.candidate_matches(*item)]
        self.suggestion_rows = [row for clip, row in matching if clip is self.clip]
        self.suggestion_rows.sort(key=lambda row: self.candidate_sort_key((self.clip, row)))
        self.suggestion_table.blockSignals(True)
        self.suggestion_table.setRowCount(0)
        for row in self.suggestion_rows:
            edit = suggestion_edit(self.clip, row)
            at = self.suggestion_table.rowCount()
            self.suggestion_table.insertRow(at)
            name = "Background / no gunfire (verify)" if row.get("proposal_kind") == "gunshot_absence" else CLASS_NAMES[edit["label"]]
            if edit["label"] != row["label"]:
                name += f" (AI: {CLASS_NAMES[row['label']]})"
            for column, value in enumerate([name, f"{edit['start']:.2f}",
                                            f"{edit['end']:.2f}", f"{row['confidence']:.3f}", row["status"], edit.get("weapon") or "unknown"]):
                item = QTableWidgetItem(value)
                origin = "Signal-processing" if "detection_score" in row else "AI"
                item.setToolTip(f"Original {origin} proposal: {CLASS_NAMES[row['label']]} / {row['start']:.2f}-{row['end']:.2f}s. Score belongs to that original class/window.")
                self.suggestion_table.setItem(at, column, item)
        self.suggestion_table.blockSignals(False)
        self.source_details.setText("Select a candidate to see its source context.")
        self.update_accept_state()
        pending = sum(row["status"] == "pending" for row in self.suggestion_rows)
        totals = {status: 0 for status in ("pending", "accepted", "rejected", "corrected")}
        for _clip, candidate in all_rows:
            totals[candidate["status"]] += 1
        reviewed = totals["accepted"] + totals["rejected"] + totals["corrected"]
        self.review_progress.setText(
            f"All sources: {reviewed} reviewed · {totals['accepted']} accepted · {totals['rejected']} rejected · "
            f"{totals['corrected']} corrected · {totals['pending']} pending. "
            f"{len(matching)} match filters; {len(self.suggestion_rows)} in this clip. Next traverses sources.")
        self.suggestion_note.setText(
            f"Candidate review: {pending} pending suggestions for the selected track. Scores are ranking signals, not accuracy probabilities. "
            "Review/edit any row, including accepted/rejected ones; accept/update, reject, or undo its decision. "
            "Source weapon names require human confirmation. Check misses before marking ranges reviewed. "
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
        active = next((row for row in self.suggestion_rows if row["id"] == self.active_suggestion_id), {})
        negative = active.get("proposal_kind") == "gunshot_absence"
        self.accept_button.setEnabled(ready)
        self.save_weapon_button.setEnabled(ready and not negative)
        self.correct_weapon_button.setEnabled(ready and not negative)
        self.weapons.setEnabled(not negative)
        self.labels.setEnabled(not negative)
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
        self.populate_weapons(None if row.get("proposal_kind") == "gunshot_absence" else edit.get("weapon"))
        self.show_source_details(row)
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
            preview = dict(edit)
            for key in ("preview_start", "preview_end"):
                if key in row:
                    preview[key] = row[key]
            self.preview_interval(preview)
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
            negative = row.get("proposal_kind") == "gunshot_absence"
            if negative:
                answer = QMessageBox.question(
                    self, "Confirm reviewed gunfire absence",
                    f"Have you listened to the ENTIRE {self.start.value():.2f}–{self.end.value():.2f} second range "
                    "and confirmed NO gunfire?\n\n"
                    "Accepting creates reviewed gunshot background only. Other sound classes remain unknown. "
                    "The detection score does not prove absence.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
                if answer != QMessageBox.StandardButton.Yes:
                    self.status.setText("Background not accepted. Listen to the whole range before confirming no gunfire.")
                    return
            previous_queue = self.candidate_queue()
            if self.apply_annotation_change(lambda: accept_suggestion(
                    self.clip, row, self.start.value(), self.end.value(), self.labels.currentData(),
                    self.sources.currentText(), weapon=self.weapons.currentData() or ""), with_proposals=True):
                self.status.setText("Verified no gunfire: only this range's gunshot background is reviewed; other classes remain unknown."
                                    if negative else "Accepted/updated and saved as a human label. Check affected ranges for missed events before marking fully reviewed.")
                self.advance_after_decision(row["id"], previous_queue)
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
            previous_queue = self.candidate_queue()
            if self.apply_annotation_change(lambda: change_suggestion_status(self.clip, row, status), with_proposals=True):
                self.status.setText(f"Decision changed to {status}. Any linked label was removed; affected class/time coverage needs review. You can edit or accept this row again.")
                if status == "rejected":
                    self.advance_after_decision(row["id"], previous_queue)
        except (OSError, ValueError) as exc:
            self.status.setText(str(exc))

    def delete_rows(self):
        try:
            self.check_dataset_revision()
            self.check_proposal_revision()
            keys = {self.row_keys[index.row()] for index in self.table.selectionModel().selectedRows()}
            intervals = [row for index, row in enumerate(self.clip["intervals"]) if ("intervals", index) in keys]
            reviews = [row for index, row in enumerate(self.clip["reviewed"]) if ("reviewed", index) in keys]
            def delete():
                unfiltered = current_suggestions(self.proposals, self.manifest, self.clip)
                for review in reviews:
                    proposal = next((row for row in unfiltered if row["id"] == review.get("proposal_id")
                                     and row.get("proposal_kind") == "gunshot_absence"), None)
                    if proposal:
                        change_suggestion_status(self.clip, proposal, "pending")
                    elif review in self.clip["reviewed"]:
                        self.clip["reviewed"].remove(review)
                for interval in intervals:
                    proposal = next((row for row in unfiltered
                                     if row["id"] == interval.get("proposal_id")), None)
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


def run_labeler(manifest, smoke_seconds=0, *, advanced=False):
    app = QApplication.instance() or QApplication(sys.argv[:1])
    if not advanced:
        from .quick_review import QuickReviewWindow
        window = QuickReviewWindow(manifest)
    else:
        window = LabelWindow(manifest)
    window.show()
    if smoke_seconds:
        QTimer.singleShot(round(smoke_seconds * 1000), window.close)
    return run_event_loop(app, window.close)
