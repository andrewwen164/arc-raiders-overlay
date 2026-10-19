# Train player-activity sound classes on your recordings

The local backend can be trained for any selected subset of these classes. These are candidate cues to evaluate, not claims of reliable recognition. The first pilot now has human gunfire labels; no trained ARC model has been created yet. Label counts alone do not establish reviewed background or independent validation data. Use the readiness check below for the current saved state.

| Label | Audible event to annotate |
|---|---|
| `footsteps` | Human walking/running steps |
| `reload` | Weapon reload sequence |
| `gunshot` | Gunfire or a consistently grouped burst |
| `landing` | Landing/fall impact or clearly attributable fall-related vocalization |
| `snap_hook` | Snap Hook use, verified against video |
| `zipline` | Zip-line use, verified against video |
| `climb_vault` | Audible climbing/vaulting sequence |
| `healing` | Audible healing/consumable use |

Label audible events only. Seeing a player falling does not prove that an identifiable falling sound is present. Ordinary impacts, ARC movement, doors and inventory handling are important confusers. Ambiguous audio should remain unreviewed for that class until resolved.

## What is in this folder

For planned clean gunshot recordings, see [GUNSHOT_RECORDING.md](GUNSHOT_RECORDING.md): documented distance effects, an explicit collection matrix, recording metadata, and a test for whether a single-distance model generalizes. Proposed metre values are sampling positions, not verified game audio thresholds.

The guide also includes a [small paired suppressor experiment](GUNSHOT_RECORDING.md#suppressors-without-an-exhaustive-attachment-matrix): start with two weapons and two distances, log the exact silencer type/tier, and label audible shots `gunshot`. Broaden attachment coverage using independent evaluation rather than collecting every combination at once.

The initial inventory found **19 MP4s / 95.04 minutes**, each with **four stereo AAC audio tracks at 48 kHz**. Track titles do not establish which one is game-only audio. The generated `training/dataset.json` lists these streams without assigning semantic labels or confirming a track.

The initial split has 17 train clips, one validation clip and one test clip. Clips from the same recording date are kept together. This is a conservative default grouping, not proof that sessions/maps are independent: edit the `group` fields if clips share a session across dates, and keep every group in a single split. Training validates this constraint. Do not move clips between splits after inspecting test results.

Raw media remains in place. Inventory reads headers; training decodes only the selected audio track into memory. No extracted gameplay WAVs, network requests, uploads or paid services are used.

## Start reviewing

The first controlled collection is prepared separately: see [1st test/README.md](1st%20test/README.md). It contains seven listener-side clips with Silencer II conditions; Ferro 50 m silenced is missing. Open it with:

```powershell
.\.venv\Scripts\python.exe -m arc_audio label-training '.\1st test\dataset.json'
```

This pilot has review-only pretrained gunfire suggestions and defaults to checking only Gunfire. The original mixed-gameplay dataset below remains separate and its existing annotations are preserved.

To check the current saved pilot without fitting a model or changing annotations:

```powershell
.\.venv\Scripts\python.exe scripts\audit_training.py '.\1st test\dataset.json'
```

The report at `artifacts/pilot-readiness/report.json` counts complete reviewed one-second windows on the trainer's 250 ms grid, positive/background readiness, labels inside/outside coverage and independent train/validation groups. It decodes selected train/validation tracks into RAM only, does not decode test audio, and does not run model inference. A successful audit command can still report `trainable: false`; read its blockers. Overlapping windows are not independent examples.

Close the labeler with X or **Ctrl+C in its launching terminal** (clear terminal text selection first). Ctrl+C saves through the normal close path and returns the terminal prompt. During import it waits for the worker to finish and the new clips to save before closing; it does not kill the copy mid-write. Windows Ctrl+Break is a fallback. An already-running older instance needs closing/relaunching once to load this fix.

The media dependency is installed in this checkout. On a new environment install it with:

```powershell
Set-Location -LiteralPath 'C:\Users\andre\OneDrive\Desktop\Codebase\arc raiders overlay'
uv pip install --python .venv\Scripts\python.exe 'av>=16,<20'
.\.venv\Scripts\python.exe -m arc_audio prepare-training training
.\.venv\Scripts\python.exe -m arc_audio label-training training\dataset.json
```

You can also open **Label training videos…** from the main control window while analysis is stopped. The labeler plays the original local video with sound. Play/pause, seek, ±2 seconds, and slower playback help locate events. Position values are seconds on the video's timeline, including audio-track offsets.

For each clip you review:

1. Listen to the audio-track choices. Select the track containing the game audio you want to learn from, then check **I listened and confirmed this is game audio**. Do not assume the default track is game-only. The app cannot infer the meaning of untitled tracks. You cannot change tracks after adding labels without deleting that clip's existing annotations/reviewed ranges first.
2. Use **Mark start here** and **Mark end here**, or enter exact seconds. Choose a sound class, then **Add manual sound interval**. No AI suggestion is required. The saved row appears in **Saved labels / reviewed ranges**. Overlapping event labels are allowed. Use the full audible span; group a continuous burst of footsteps/shooting into one interval consistently.
3. Optionally tag `self`, `teammate`, or `other_raider` only when the video confirms the source; otherwise leave `unknown`. These are reviewer notes. They are not classifier output, and the model does not learn or claim enemy identity from them. `other_raider` does not necessarily mean enemy.
4. After labelling all events for the classes you checked, select the whole stretch and select those classes under **Classes fully checked**, then click **Mark range fully reviewed**. The confirmation lists the exact classes. Unlabelled portions inside become negatives for **only those classes**. All eight boxes start checked; uncheck any you did not fully review. Legacy ranges without a `classes` field cover only footsteps/reload/gunshot; re-review additional classes before training them. Reviews with different class scopes remain separate even when they overlap. Audio outside reviewed ranges is excluded entirely.

   If you have checked the entire clip, use **Mark whole clip fully reviewed** instead. It explicitly confirms 0 to the clip duration for the checked classes, independently of the last sound's editor bounds. It preserves existing labels and does not change the editor fields. The adjacent readout shows reviewed seconds and how many sound labels are fully inside coverage. Labeling every shot does not automatically mark the pauses between shots reviewed; checked pauses supply important negative examples.
5. Include reviewed background-only ranges containing none of the target classes: ARC machine movement, ambience, music, environmental impacts, wind, machinery, voice/chat, doors, item handling and similar confusers. Audible human footsteps remain a footsteps label even if they are your own or a teammate's. An enemy-only objective would require a separate, independently evaluated task.
6. Review examples in both train and validation recording groups. Leave the test recording group for independent annotation/evaluation. Save is automatic after each edit. Double-click a saved interval (or use **Edit selected interval**), change its fields above, and click **Save interval changes**. Select rows and use Delete to remove mistakes. Edits/deletions clear affected class/time review coverage so removed labels do not silently become negative examples; check those stretches again. The original recordings are not modified.

Start with a few short, thoroughly reviewed stretches rather than trying to label all 95 minutes at once. Capture different surfaces, movement speeds, distances/mixes, weapons, reload styles and background conditions. Multiple independent examples and sessions matter much more than many overlapping windows of one sound.

## Add your own intervals or recordings

To mark a sound the model missed, set Start/End, select a supported Sound class and Source, then click **Add manual sound interval**. This adds an annotation on the existing timeline, without cutting/re-encoding a new audio/video file. Confirm the game-audio track first. Manual intervals can be edited and deleted through **Saved labels / reviewed ranges**, including clips with no AI suggestions.

For new recording files, click **Import video / WAV files...** beside the clip selector and choose MP4, MKV, MOV or WAV files. Enter a session name: clips from the same real recording session need the same name. An existing session retains its split; a new session starts in train. This import does not automatically establish independent validation/test recordings or generate model suggestions.

Files outside the dataset folder are copied into its `imported/` subfolder with collision-safe names; original files and existing labels are preserved. Files already inside the dataset are referenced in place. Copying and header probing run in a worker so the UI remains responsive. New recordings appear in the dropdown after saving and require track confirmation and annotation. Unchanged duplicate imports are skipped; a previously imported source that changed must be renamed to import a distinct version. Unsupported/audio-free files produce actionable errors. The labeler waits for an active import to finish before closing. Atomic copy publication requires hard-link support in the destination filesystem (tested here on Windows/OneDrive NTFS); an unsupported filesystem reports an import error.

## Optional model-assisted review

`suggest-labels <manifest>` scores local selected tracks with cached AST weights, outside the Qt UI process. It writes `<manifest-stem>.suggestions.json`, never human labels or reviewed ranges. By default it skips unconfirmed tracks and always excludes the test split. Explicit `--allow-unconfirmed-track` creates provisional candidates only; acceptance still requires human track confirmation. No network download is performed. Use `--model-dir <existing AST cache>` to reuse an existing cache; omit it for the app's default model location.

AST proposals support `--classes gunshot footsteps` (gunshot alone by default). They do not provide reload, Snap Hook, landing or other ARC-specific classes. The initial implementation uses experimental two-second windows/hops, feature-padded to the fixed 1024-frame model input; `--window-seconds` supports 0.5-10.24 seconds and `--hop-seconds` 0.25-window seconds. Short windows are not established as accurate ARC event localization. Live AST behavior remains full-context 10.24 seconds. The default 0.2 cutoff is an uncalibrated candidate filter, not an accuracy claim.

In the labeler, open **AI suggestions** and single-click any row (pending, accepted or rejected). This loads its start/end/class/source fields above. **Review / edit selected suggestion** or double-click plays it with context; this is optional, not an extra acceptance requirement. Edit the fields as needed, then use **Accept / save selected suggestion**. Re-accepting updates the linked human interval rather than adding a duplicate. Previously edited bounds/class/source are restored when you revisit a row. Original AI scores remain associated with their original class/window, even if the human class changes.

Accept is enabled only when a row is selected and **I listened and confirmed this is game audio** is checked for that clip. Listen to the selected stream and tick that checkbox yourself; the app does not infer confirmation. A message beside Accept explains the missing requirement while it is disabled. Reject can discard a proposal before track confirmation because it creates no human-positive label. A successful acceptance immediately shows `accepted` in the table and saves a linked interval.

**Reject selected AI suggestion** can reverse an acceptance and removes only its linked label. **Undo decision / return selected suggestion to pending** works on either accepted or rejected rows; it removes a linked label and keeps your last edit ready for reconsideration. Deleting a linked saved label also returns its proposal to pending. Unrelated manual labels remain intact. Changes to a label's class or bounds, deletion, rejection after acceptance and undo invalidate affected class/time review coverage, preserving unrelated classes and times. Source-only edits preserve existing coverage.

Only explicit acceptance saves a human interval; rejected or pending proposals supply no training examples. No action automatically adds reviewed coverage. Listen for missed events and independently mark fully checked ranges afterward. Avoid duplicate labels for one event/burst and keep final test annotation independent of model suggestions. Changes autosave; stale windows cannot overwrite newer dataset/decision files. A failed dataset save rolls back the edit, and a later decision-sidecar failure reports that sound labels saved but the decision needs retrying. Table/editor selection is reset after either failure so retries cannot target a different sorted row.

Proposals include scores, processing time, media fingerprints and model/config hashes. Stale/other-track proposals are hidden. Reruns preserve decisions and add new IDs; they do not erase lower-scoring candidates from previous runs. Close the labeler while regenerating; the generator aborts rather than overwriting concurrently changed labels or proposals. The trainer/evaluator read only the dataset's human intervals and class-scoped reviewed ranges, never this sidecar.

## Train and use the model

```powershell
.\.venv\Scripts\python.exe -m arc_audio training-status training\dataset.json
.\.venv\Scripts\python.exe -m arc_audio train-local training\dataset.json
.\.venv\Scripts\python.exe -m arc_audio ui
```

By default training selects the original three classes. Select a larger task after reviewing it in both train and validation groups:

```powershell
.\.venv\Scripts\python.exe -m arc_audio train-local training\dataset.json --classes footsteps reload gunshot landing snap_hook
```

Use `--classes all` for all eight, or name any supported subset. Every selected class needs verified positive and negative examples in both splits. A training window must be fully reviewed for the entire selected task. Minimum counts are 20 positive and negative one-second windows per class in training, five of each in validation, and 20/5 background windows containing none of the selected classes. These are technical minimums, not evidence of sufficient data or accuracy. Training refuses collections missing the required coverage, examples or independent validation data and does not create dummy weights.

After successful training, select **C · Local ARC model**. **Only trained player-activity sound classes** is enabled by default for recorded/live input. The local backend emits only classes actually fitted and stored in its model; adding label choices does not teach an existing three-class model the new sounds. Baseline activity/transient cues cannot leak through the filter. Demo retains the original synthetic activity display. Recorded/live input selects the local backend when target-only is enabled, which gives a missing-model error until trained. Uncheck the filter and select Baseline for diagnostics. Generic AST still supports only footsteps/gunfire.

Weights are saved atomically to `%LOCALAPPDATA%\ArcAudioCompanion\models\arc-targets.npz`, with an adjacent training report. Use `train-local --output <path>` for experiments; custom model paths can be passed to evaluation, while the UI loads the default cache path. Model data uses numeric arrays and JSON metadata, not pickle or executable code. Training does not require PyTorch or pretrained downloads.

## What the model actually does

It pools log-mel energy statistics across four temporal sections of a **one-second mono, 16 kHz window**, then fits one class-balanced, regularized logistic head per selected class using SciPy. It learns associations between features and human labels. Multiple classes can be active together; background has no positive class. Scores are uncalibrated and a threshold can suppress all output. Existing three-class model files remain compatible.

Inference attempts an update every 250 ms after one second of continuous context, with an 800 ms per-class alert cooldown. Audio gaps reset context. Numeric fit and feature extraction are intentionally approachable for learning, but this shallow classifier may fail to separate similar gameplay sounds. It has no prior acoustic knowledge comparable to a pretrained network. Reloads in particular can resemble item/equipment handling. Measure performance before relying on it; no ARC-specific performance is claimed yet.

The local backend shows direction **unknown**, because a class present in a mixed one-second window cannot reliably be assigned the whole mix's stereo balance. It cannot establish exact direction, distance, number of people, self/teammate/enemy identity, or whether an audible player can see you. Generic footsteps or gunfire may be your own. Source tags are preserved for future confusion analysis; they do not solve that problem.

## Evaluate on held-out recordings

```powershell
.\.venv\Scripts\python.exe -m arc_audio evaluate-training training\dataset.json --split validation --output artifacts\validation-events.json
.\.venv\Scripts\python.exe -m arc_audio evaluate-training training\dataset.json --split test --output artifacts\test-events.json
```

Only confirmed, reviewed regions are scored. Matching is performed separately for each file using the existing exact-class, one-alert-per-interval maximum matching and ±0.15 s tolerance. A prediction is timestamped at the end of its trailing window; delayed or repeated alerts are penalized under those rules. Each disconnected reviewed region gets fresh model context. Entire target intervals should be within reviewed ranges; partially reviewed labels are excluded from truth. Short boundary fragments may not have enough context to produce a detection. The report includes event precision/recall, per-class counts, per-file onset delays, raw event metadata and processing latency. No cross-file timestamp matching is allowed.

The report saved during fitting contains **window-level validation metrics**, not event accuracy. The test split is not decoded or used during fitting, normalization or threshold selection; test evaluation rejects a group listed in the model's training/validation provenance. Tune thresholds on validation only; keep test results independent. Null metric values mean no denominator, not perfect accuracy.

## Other useful cues and the enemy question

Landing, Snap Hook, zip-line, climbing/vaulting and healing sounds are now optional annotation/training classes. They still need verified examples, confuser negatives and held-out evaluation. Generic doors, looting and inventory sounds remain outside the current taxonomy. These cues can also be your own or a teammate's.

Embark describes item-specific handling effects and state-dependent ARC-machine sound patterns in its [audio design overview](https://arcraiders.com/news/soundscapes-blog). This motivates collecting confuser examples as well as positives; it does not establish that this classifier can separate them reliably.

The honest output is **possible player-related audio activity**. Nearby-enemy identification requires source information this app does not have. Positional mixing, echoes, machinery, your own movement, teammates and non-player gunfire complicate inference. Developer policy/anti-cheat boundaries from the main README still apply; this change adds no game integration or gameplay automation.

## Verification

This update was tested with synthetic training fixtures, model fit/save/load, runtime warm-up and gap resets, class names, negative-label masking, per-group split validation, and held-out event evaluation. Tests do not establish gameplay accuracy. The real media was inventoried; the first two seconds of each of the first clip's four tracks decoded successfully into RAM. The labeler loaded the real MP4 in a Windows smoke check. No human event labels were invented and no trained gameplay model was produced.

Media APIs were checked against [PyAV audio/resampling documentation](https://pyav.basswood.io/docs/stable/api/audio.html) and [Qt's local media player API](https://doc.qt.io/qtforpython-6/PySide6/QtMultimedia/QMediaPlayer.html). PyAV 18.1.0 was installed and used with the existing Python 3.11 environment.
