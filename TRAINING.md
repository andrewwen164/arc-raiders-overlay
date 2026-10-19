# Train player-activity sound classes on your recordings

The local backend can be trained for any selected subset of these classes. These are candidate cues to evaluate, not claims of reliable recognition. The first pilot now has human gunfire labels; no trained ARC model has been created yet. Label counts alone do not establish reviewed background or independent validation data. Use the readiness check below for the current saved state.

## Quick gunfire review: Yes / No / Unsure

This is now the default labeling view:

```powershell
.\.venv\Scripts\python.exe -m arc_audio label-training .\training\dataset.json
```

The app plays a brief sample from a proposed sound, then asks **Any gunfire in this sample?** Answer **Yes (Y)**, **No (N)** or **Unsure (U)**. Space replays it; Previous/Left and Undo let you change an answer. Automatic playback/advance is on by default. No timestamp entry, clip cutting, manual interval creation or **Mark range fully reviewed** step is needed for this workflow.

For each new source, listen and answer the multiple-choice game-audio confirmation first. It is never automatically confirmed. Weapon selection is optional multiple choice and starts at **Unknown**, even when a source suggests a name. Only a positive answer with your explicit weapon choice records weapon metadata. No/Unsure do not identify a weapon. Background proposals are interleaved with shot candidates and ask the same neutral question; they are not assumed negative.

Each Yes/No is saved as one exact, sample-aligned **one-second window presence label** in `presence_reviews`: Yes means gunfire occurs somewhere in that second; No means no gunfire in that second. It creates no precise event onset/offset and labels no surrounding audio or other sound class. Unsure and Undo have durable nontraining states in `presence_states`, so reopening after a candidate-sidecar save failure does not resurrect an old answer. The dataset is authoritative. Existing manual intervals/reviewed ranges are preserved, and conflicting answers are refused. Media fingerprints and selected-track checks protect answers from changed sources.

Playback uses Qt's source timeline, codec seeking and timers, so its boundaries are approximate; these are presence questions, not sample-accurate event annotations. Controls unlock after playback and source confirmation. Replay or choose Unsure if a boundary, mixed sound, music, commentary or distant/suppressed shot makes the answer unclear. Keyboard shortcuts pause while a dropdown has focus.

Use the source-role multiple choice to mark a genuinely separate recording group **Independent validation**; other reviewed sources can remain **Training**. This explicitly changes the whole group. Nothing is shuffled automatically, and final test sources remain excluded from quick review. Different uploads can reuse the same footage, so separate URLs alone do not establish independent recordings.

Check progress, then train the first gunfire/background experiment after sufficient answers exist:

```powershell
.\.venv\Scripts\python.exe -m arc_audio training-status .\training\dataset.json
.\.venv\Scripts\python.exe -m arc_audio train-local .\training\dataset.json `
  --classes gunshot --output .\artifacts\quick-review-experiment\gunfire.npz
```

The current technical minimum is **20 Yes + 20 No training windows**, plus **5 Yes + 5 No validation windows** from independent groups. This is a minimum for running an experiment, not enough evidence to claim accuracy. You do not need to review every proposal before trying an initial fit. Quick answers feed training directly, exactly once per answered second; the trainer does not generate sliding labels around a Yes or infer background around it. Multi-class training cannot use a gunfire-only No to label other classes absent.

The report distinguishes presence-window data from legacy intervals. Its validation metrics describe the sampled questions, not event-level recall, false alarms over all gameplay or accuracy on every weapon/distance. The event evaluator still requires actual event intervals and exhaustive reviewed spans; it will not invent those from Yes/No answers. This experiment output does not replace the default runtime model or activate itself in the companion. No actual model is trained until you run the command with sufficient human answers.

The older interval editor remains optional for advanced work:

```powershell
.\.venv\Scripts\python.exe -m arc_audio label-training .\training\dataset.json --advanced
```

The interval/review-range instructions below describe that optional editor, not a prerequisite for quick gunfire review.

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

## Advanced interval review (optional)

The first controlled collection is prepared separately: see [1st test/README.md](1st%20test/README.md). It contains seven listener-side clips with Silencer II conditions; Ferro 50 m silenced is missing. Open it with:

```powershell
.\.venv\Scripts\python.exe -m arc_audio label-training '.\1st test\dataset.json' --advanced
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
.\.venv\Scripts\python.exe -m arc_audio label-training training\dataset.json --advanced
```

The main control window's **Label training videos** button opens quick review while analysis is stopped. Use the explicit `--advanced` command above for this optional interval editor. It plays the original local video with sound. Play/pause, seek, ±2 seconds, and slower playback help locate events. Position values are seconds on the video's timeline, including audio-track offsets.

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

## Building the weapon dataset from public source videos

You can collect weapon-shot proposals from the curated `arc_raiders_audio_seed_manifest.csv` without recording gameplay or cutting videos yourself. This extends the existing dataset and PySide labeler; it does **not** train a weapon classifier. Close the labeler before collection, then use PowerShell from this repository:

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[collection]'

.\.venv\Scripts\python.exe -m arc_audio build-weapon-dataset `
  .\arc_raiders_audio_seed_manifest.csv `
  --dataset .\training\dataset.json

.\.venv\Scripts\python.exe -m arc_audio weapon-dataset-status `
  .\training\dataset.json

.\.venv\Scripts\python.exe -m arc_audio label-training `
  .\training\dataset.json
```

For an existing uv environment missing pip, run `uv --cache-dir .cache\uv pip install --python .venv\Scripts\python.exe pip` first. The collection extra includes PyAV, yt-dlp with its default JavaScript support packages, and imageio-ffmpeg. An installed supported Deno or Node runtime improves YouTube compatibility; the collector enables available local runtimes and does not install one or fetch remote JavaScript components. See the [yt-dlp installation documentation](https://github.com/yt-dlp/yt-dlp#installation) and [JavaScript runtime documentation](https://github.com/yt-dlp/yt-dlp/wiki/EJS).

Inspect the manifest without writing files or accessing the network:

```powershell
.\.venv\Scripts\python.exe -m arc_audio build-weapon-dataset `
  .\arc_raiders_audio_seed_manifest.csv --dataset .\training\dataset.json --dry-run
```

For a small first run, append `--source-url 'https://www.youtube.com/watch?v=VB9FH35YGUY'`, or use `--max-sources 1` for the first unique URL. `--source-url` selects an exact URL in the CSV. All imported weapon names remain available for corrections even when collecting a subset.

### What acquisition does

CSV columns are `weapon,source_title,source_url,start,end,quality,notes`; extra columns and every original cell are retained in source metadata. Times accept blank, seconds, `MM:SS` and `HH:MM:SS`, including fractional seconds. Local media paths, relative to the CSV directory, are also supported. YouTube URLs are canonicalized by video ID, so multiple weapons/ranges from one video use one media file and one source group.

The explicit command downloads the best available **audio-only** stream with yt-dlp. It preserves the original container and full source timeline in `training\sources\`; it does not make thousands of permanent WAVs. An alternative `--storage` must stay inside the dataset directory. Processing respects manifest start/end ranges and caps decoding at the last needed end. The entire compressed audio track may still download once, including portions outside the processing range; sources are canonical originals, not separately trimmed exports. No audio is uploaded and the cache is excluded from Git. Keep downloaded material local and use sources you have permission to process; collection does not grant redistribution rights.

Reruns reuse fingerprint-verified downloads and deterministic candidate IDs. They preserve accepted/rejected/corrected decisions, human intervals, track confirmations and reviewed coverage. Source failures are reported and checkpointed individually; later URLs continue. The command exits 1 if any source failed, so read the summary rather than assuming the whole batch failed. Retry the same command after resolving a download error. A `.collection.lock` prevents concurrent collectors; remove a stale lock only after confirming no collector is running. The labeler detects external dataset edits and refuses to overwrite newer labels.

### Candidate detection and weapon proposals

This stage uses NumPy/SciPy **signal-processing heuristics**, not AST or a trained weapon recognizer. It combines adaptive RMS/noise-floor rise, spectral flux, onset prominence, crest and high-frequency content. It deliberately over-generates: speech, music, impacts, reloads, machinery and interface sounds can qualify. Quiet/suppressed/overlapping gunfire can be missed. `detection_score` is a heuristic ranking in 0–1 and `quality_score` is a review priority in 0–100; neither is a gunshot probability or accuracy measurement.

Onsets can remain separate at approximately 70 ms spacing, including automatic bursts. Suggested event spans are roughly 150 ms or shorter at the next onset; these are editable boundaries, not measured complete shot tails. Playback defaults to 0.35 seconds before and 1.00 second after the onset, clipped to the manifest range. Configure with `--pre`, `--post` and `--min-spacing`. Absolute source timestamps are preserved. Every candidate records provenance and lightweight RMS, peak, zero-crossing, spectral centroid/bandwidth/rolloff and log-mel summaries.

`gunshot` remains the sound class. A single weapon associated with the source/range becomes `proposed_weapon`; it is context, not acoustic recognition. Mixed gameplay can contain opponents' weapons as well. When overlapping source rows list different weapons, candidates are **unknown/ambiguous** with possible names, instead of duplicating every onset for every weapon. In the supplied seed, 26 untimestamped rows reference one all-weapons showcase; those require a human weapon choice or better curated ranges. No weapon names or timestamp ranges are invented.

### Human review and annotation format

The normal `label-training` command now opens [Yes / No / Unsure review](#quick-gunfire-review-yes--no--unsure). It saves directly trainable window-presence answers without manual time marks or coverage steps. The interval acceptance workflow below remains available only through `--advanced`.

In **AI suggestions**, select a source/candidate, listen, confirm the audio track, then Accept, Reject or choose a weapon and **Save weapon correction / accept**. Acceptance writes a human interval; rejection leaves the candidate out of training. Decisions remain reversible. Unknown weapon is allowed for generic gunfire. Refine Start/End if the proposed boundary misses the audible event. A suggested annotation after human acceptance is:

```json
{"start": 142.31, "end": 142.46, "label": "gunshot", "weapon": "anvil", "source": "unknown", "proposal_id": "stable-candidate-id"}
```

`weapon` is optional; old version-1 datasets still load. Weapon identifiers are lowercase with underscores, for example `il_toro`. Other sound classes cannot carry weapon identity. The dataset records manifest rows in `weapon_sources`, vocabulary in `weapons`, and source provenance on each imported clip. Candidate scores/context/features/status remain in `dataset.suggestions.json`; valid statuses now include `corrected`. Correcting weapon creates a real accepted gunshot annotation with that human-selected weapon. The existing trainer continues learning **event classes** and ignores weapon metadata; no 26-class training was added.

Review controls work across sources: **A** accepts, **R** rejects, **W** opens the weapon correction dropdown, **Left/Right** select Previous/Next, and **Space** plays/pauses. Shortcuts operate in the suggestions tab with this app focused; typing fields, dropdowns, sliders and modal dialogs keep normal keyboard behavior. After choosing a correction, click **Save weapon correction / accept**. Enable **Auto-advance after Accept / Reject** and optional **Autoplay** to move to the next matching pending candidate. Track confirmation is required separately for each source; it is never silently checked.

Filter by weapon, source, review status, duplicate group, minimum quality and minimum detection score. Sort by quality, time, weapon or source; use **Next** to start the filtered queue. Unknown/ambiguous proposals remain available when a possible weapon is selected, but selecting that filter does not confirm identity. Counters show global decisions and matching candidates. Use All statuses to revisit a decision. Filters never change annotations. Audio-only downloads hide the empty video area. The current JSON sidecar saves the whole proposal bundle per decision; a measured 10,518-candidate / 29 MB collection took about 1.2 seconds per acceptance with auto-advance on this machine, mostly persistence. This is not an inference or game latency measurement.

Accepting candidates does **not** mark whole windows or clips fully reviewed. Missed events still require checking before marking a range reviewed; rejected suggestions do not establish background truth. Existing class-scoped review rules continue to apply.

All ranges from a source video stay in the same group and compatible split. New source groups start in train; the collector never reshuffles existing groups and never generates proposals for reserved test sources. Reserve independent source/session groups for validation and test before future model development. Reuploads, compilations and shared footage across different channels may still leak the same underlying recording: grouping by URL cannot prove recording independence. Avoid treating many shots from one showcase as independent source diversity.

### Quality, similarity and diversity

Quality additionally considers local background level, SNR, nearby competing transients, spectral distinctiveness and clipping. Dedupe compares centered log-mel shape using cosine similarity and a bounded 64-neighbor search, default threshold 0.995. It stores `duplicate_group_id` and `duplicate_similarity`, retaining **every** candidate and decision. A similar spectrum can represent different real shots; similarity groups are advisory, can miss pairs, and do not establish copied footage or independent recordings. Distinct nearby shots within one recording are protected from collapsing into a duplicate group. `burst_group_id` separately links nearby onsets and is scoped to its source and absolute first onset. Optional duplicate-scoring failures are reported without discarding collected candidates.

`weapon-dataset-status` reports registered/available sources, downloaded/failed videos, pending/accepted/rejected/corrected proposals, accepted human shot counts, duplicate groups and known channels per weapon. Accepted corrections are credited to the human-selected weapon. Unknown-weapon proposals are counted separately. Warnings flag fewer than three reviewed videos, fewer than two known channels, or one video providing over 70% of accepted shots. These are conservative collection checks, not sufficient-data guarantees. Many candidate windows from one uploader do not establish generalization. Check duplicate footage and reserve independent sessions before training.

### Optional discovery: metadata stays unverified

Only after the curated workflow works, search for more possible sources:

```powershell
.\.venv\Scripts\python.exe -m arc_audio discover-weapon-sources `
  --weapon 'Anvil' --max-results 10 `
  --output .\artifacts\anvil-unverified-sources.json
```

This searches YouTube through yt-dlp and saves metadata marked `verification: unverified`. It downloads no media, creates no labels and never trains. Inspect the source, relevance, other audible weapons and usable shooting ranges before manually adding verified rows to the curated CSV. Discovery JSON is not an ingestion CSV; CSV rows explicitly marked `verification=unverified` are refused. Optional `--dataset .\training\dataset.json` registers these unverified references for status reporting only, separately from curated acquisition inputs. Close the labeler before that metadata update. Discovery registration is idempotent and never starts collection.

### Optional background / hard-negative review

Once weapon-shot collection is working, propose ranges away from existing transient peaks and human gunshot intervals:

```powershell
.\.venv\Scripts\python.exe -m arc_audio suggest-weapon-background `
  .\training\dataset.json --seconds 2 --margin 0.75 --max-per-source 8
```

This writes **pending** `gunshot_absence` proposals only, within registered source ranges on non-test clips with matching fingerprints. No quiet stretch becomes truth automatically. These are background candidates, not proven hard confusers; listen for missed/suppressed gunfire. Their detection score is zero by construction, not a trained negative prediction. Filter thresholds can hide them. Linked review provenance is retained through review merges and interval edits so a failed sidecar save can still be reversed safely.

In the existing labeler, the row says **Background / no gunfire (verify)**. Accept requires a separate, default-No confirmation that you listened to the entire edited range and heard **no gunfire**. It adds reviewed coverage for `gunshot` only and creates **no positive sound interval**. Other classes remain unknown. Acceptance refuses overlap with any saved gunshot; trim or reject the proposal instead. Reject/Undo removes its linked absence review. If later manual review merges erased provenance, Undo conservatively clears affected gunshot coverage for re-review, retaining other classes. Deleting a linked reviewed row reopens its proposal. A background decision never establishes a weapon identity.

### Collection troubleshooting

| Problem | Action |
|---|---|
| Missing yt-dlp/PyAV | Install `.[collection]` with this venv's Python, then restart the command. |
| Missing FFmpeg | Ordinary audio-only acquisition needs no conversion. If yt-dlp reports FFmpeg is needed, install an FFmpeg build and pass `--ffmpeg-location 'C:\Tools\ffmpeg\bin'`. `imageio-ffmpeg` also supplies a local executable, whose path is printed by `.\.venv\Scripts\python.exe -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"`. |
| Private/deleted/unavailable video | Replace or remove that curated source; other rows still process. |
| Age/region restriction or HTTP 403 | Update the collection packages; check supported local Node/Deno runtime and source availability. The collector does not bypass restrictions or automatically load browser cookies. Use an accessible permitted source or local fallback if it remains unavailable. |
| No audio stream | Inspect the original source; use a source with audible audio. No silent file is substituted. |
| No candidates | Zero proposals is not proof of no shots. Check track/range and listen; add missed intervals in the labeler or improve the source. |
| Too many false positives | Use shorter, verified shooting ranges and clean sources; prioritize higher scores during review. Music and commentary can dominate transient detection. |
| Cannot read cached media | Check local permissions and OneDrive availability. The collector reports unreadable caches instead of repeatedly downloading them. |
| Save/lock/concurrent-edit error | Close the labeler and other collectors, retain the source cache, then rerun. Never delete human annotation files as a reset. |

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
