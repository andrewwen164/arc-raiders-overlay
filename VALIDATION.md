# Validation performed

Executed on **Windows, 64-bit CPython 3.11.17**, October 18, 2026. No game was launched, accessed, or tested. These checks do not establish ARC Raiders compatibility, developer permission, or ban safety.

## Automated checks

- `python -m pytest -q`: **35 passed**. Synthetic fixtures only. Covers stereo balance/antiphase/silence, baseline event classes/confidence/cooldown, bounded buffering, thread wake-up, stale discard, PCM decoding, antialiased resampling, WAV validation, settings round-trip/corruption, annotation validation, maximum event matching, local logging, and clean worker shutdown. Missing optional weights fail explicitly without downloading; baseline does not construct AST.
- `python -m ruff check src tests scripts`: passed.
- `python -m compileall -q src scripts`: passed.
- `python -m arc_audio ui --smoke-seconds 8`: visible Qt control/overlay windows opened, deterministic demo ran, and the app exited successfully.
- Synthetic nine-second stereo WAV generation, offline baseline analysis, event JSONL writing and annotation matching completed. Four alerts were produced for three hand-labelled intervals, including a false-positive transient on the sustained sound's onset. This is deliberately not a gameplay accuracy result.

## Windows integration smoke

`python scripts/windows_smoke.py` completed successfully with no errors. Machine-readable details are in `artifacts/windows-smoke.json`; own-window renders are in `artifacts/controls.png` and `artifacts/overlay.png`. The controls and the rendered transient indicator were visually inspected.

Verified:

- WASAPI enumeration returned two compatible stereo output loopbacks, both 48 kHz: Realtek speakers and LG/NVIDIA display audio.
- Nine-second synthetic WAV playback on the Windows default output reached EOF with four logged detections and no playback error.
- Concurrent, explicitly selected whole-device loopback captured nonzero synthetic playback levels, completed analysis, and shut down with no capture error. Captured samples were not written to disk.
- Native overlay extended styles included layered, transparent, no-activate, and tool-window flags. Showing the overlay preserved the current foreground window.
- Both physical QScreen monitor geometries were applied successfully: 2560×1440 at (0,0), scale 1.0; 1707×960 logical at (-2560,0), scale 1.5. A separate negative-coordinate geometry assertion passed. This checks placement/DPI plumbing; it does not prove perfect visual alignment in every mixed-DPI arrangement.
- A real synthetic detection reached the visible overlay and rendered its text/icon.
- Ctrl+Alt+F10 registration succeeded. A standard `WM_HOTKEY` thread message exercised Qt's native event-filter dispatch, stopped the worker and hid the overlay. **This did not synthesize or test an actual keyboard chord while a different program had focus.**

## Optional pretrained classifier

Installed CPU PyTorch and the optional extra. Explicitly downloaded four upstream files, pinned to revision `f826b80d28226b62986cc218e5cec390b1096902`. Safetensors file size: **346,404,948 bytes**. The model/config/feature extractor loaded locally without remote code.

Ran offline AST evaluation on a 12-second synthetic stereo fixture with `HF_HUB_OFFLINE=1`. The detector waited until 10.25 seconds of source audio (50 ms chunk resolution) before its first inference. Actual model labels were checked against `Walk, footsteps` and `Gunshot, gunfire`; feature configuration was checked for mono 16 kHz and 1024 frames. Scores used sigmoid.

For this plumbing check only, minimum confidence was explicitly set to **0.0**. It emitted the two genuine but very low model scores (approximately 0.000026 and 0.000044), both with unknown direction and the trailing clip interval. These are **not positive footsteps/gunshot examples**, and do not validate semantic accuracy. They would be suppressed by the default 0.6 threshold. No random predictions or heuristics were used as model output.

The slowest measured detector call was approximately **3.07 seconds**; two full model passes took most of the computation time. CPU/model loading is additional and not included in that number. Thus, this model is unsuitable here for rapid live indicators: long initial context and inference stalls can reset streaming context when old chunks are discarded. Offline evaluation remains functional. The upstream NumPy feature extractor emitted a warning about an empty mel filter; inference completed. No TorchAudio implementation or numerical equivalence comparison was tested.

## Compass-style overlay update

The overlay now draws broad pulsing left/right arcs near screen center, compact event icons, and a separate centered unknown-direction badge. Duplicate cues renew in place. After this change, all 35 tests, Ruff, compilation and the Windows integration smoke passed again. A static fixture preview (`python scripts/preview_overlay.py`) was rendered and visually inspected at `artifacts/compass-preview.png`; it contains deliberately constructed UI events, not detector predictions. The live synthetic demo also delivered and rendered a real baseline cue during the Windows smoke check. Game behavior and accurate angular localization remain untested/unsupported as described below.

Model verification used a project-local test cache at `artifacts/classifier-data/ArcAudioCompanion/models/ast-audioset`; the normal app cache is separate. No model is silently copied into the user's normal cache. To reuse this test cache in a single PowerShell session:

```powershell
$env:LOCALAPPDATA = Join-Path (Get-Location) 'artifacts\classifier-data'
.\.venv\Scripts\python.exe -m arc_audio evaluate .\artifacts\classifier-demo.wav --backend ast
```

Close that PowerShell session to restore the inherited environment. Normal setup uses the explicit download command in the README.

## Installed versions

| Dependency | Tested version |
|---|---|
| PySide6 / Qt bindings | 6.11.2 |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |
| PyAudioWPatch | 0.2.12.9 |
| pytest | 9.1.1 |
| Ruff | 0.15.22 |
| PyTorch | 2.14.1+cpu |
| Transformers | 4.57.6 |
| huggingface-hub | 0.36.2 |
| safetensors | 0.8.0 |

## Still needs user verification

- Actual Ctrl+Alt+F10 presses while a different app has focus, hotkey-conflict handling on your normal desktop, and the tray-menu fallback.
- Mouse clicks and typing through visible indicators, including over the chosen borderless fullscreen game. Native styles/nonactivation were tested; actual game input was not.
- Visual position/size on your monitor arrangement and after hotplug/DPI changes; these scenarios have code support but were not all visually exercised.
- Your headphones/output format, audio-device disconnection/reconnection, Bluetooth delay, playback/indicator perceptual sync, and a device that is idle for 15 seconds. Format/error paths and synthetic audio were tested; not every physical driver scenario was simulated.
- Representative gameplay recordings and honest labelled event metrics. No ARC-specific dataset, trained classifier, accuracy claim, or semantic validation exists.
- Exclusive fullscreen, protected/composited surfaces, anti-cheat behavior and game policy permission. Exclusive fullscreen support is not guaranteed; no bypass is provided.
- Packaging/signing an executable. No packaged executable has been built or claimed to be verified.

## Local training and target-only update

Current full-suite result: **43 passed**; Ruff and compilation passed.

The training workflow adds human interval review, local supervised fitting, a constrained three-class backend, reload icon, and filtering of non-target alerts before recorded/live logging or display. Supported trained classes are footsteps, reload and gunshot. These are sound classes, not enemy/self/teammate classifications.

Inventory: 19 local MP4s, 95.04 minutes, four stereo AAC/48 kHz streams each. No tracks are human-confirmed and there are zero event labels or reviewed spans. The `train-local` command was run against that inventory and correctly refused; it created no gameplay model. First-two-second in-memory decode checks succeeded for each of the first clip's four streams. No raw audio was exported or uploaded.

Training tests use synthetic tones only: actual optimizer fit, numeric weights save/load without pickle, runtime label restriction, context warm-up/gap reset, group split checks, negative-label masking, and held-out event evaluation. They do not validate sound recognition on gameplay. The original Windows smoke passed again after this update, using baseline diagnostics explicitly so synthetic playback still produces activity cues.

The Windows labeler smoke (`python scripts/training_smoke.py`) played an original MP4 locally, advanced to 2333 ms with 141 rendered frames, switched to audio track index 1, and added/deleted a reload annotation on a disposable manifest. It reported no player error and did not annotate the real dataset. Details: `artifacts/training-smoke.json`. PyAV 18.1.0 was used. A model can only be fitted to gameplay after human track confirmation and sound/background labeling as documented in `TRAINING.md`.

## Expanded sound-class update

The annotation vocabulary now includes landing, Snap Hook, zip-line, climbing/vaulting and healing, alongside the original three. Training accepts a selected subset or `--classes all`; models emit only classes stored in their fitted metadata. Existing three-class numeric model files remain loadable.

Reviewed intervals record their checked classes. Legacy intervals without that field cover only the original three; tests verify they do not become negatives for new classes. Merging preserves differing review scopes. Additional tests fit/load/run/evaluate a one-class optional model using synthetic tones only and reject training an unreviewed new class. These verify the software contracts, not actual Snap Hook or landing recognition.

Full suite after expansion: **47 passed**, Ruff/compilation passed. Labeler playback/track switching/annotation CRUD and the Windows overlay/playback/loopback smoke passed again. The real dataset still contains no human-confirmed tracks, reviewed spans or labels, so no ARC-trained model or gameplay performance result is claimed.

## First controlled pilot and model-assisted review

The following section records the initial pilot run; review decisions may have changed since that run. See the later editable-review update below for the current software verification.

Prepared the seven MP4s in `1st test` separately from the original mixed-gameplay manifest. Total duration: **75.88 seconds**. User-confirmed setup notes: friend shooting, listener-side recording at the named ranges, Silencer II, intended track 1. Track selection remains provisional until confirmed in the labeler. Ferro 50 m silenced is explicitly missing. All seven clips share `first-pilot-session / train`; they were not scattered across splits to manufacture independent validation data. Original media and existing mixed-gameplay annotations were preserved.

Ran actual, offline, cached-only AST inference on selected stream 1: **40 windows / 10 pending gunfire candidate windows** at cutoff 0.2. Two-second audio windows/hops were used, with feature padding to the model's fixed 1024-frame input. Weight/config/feature SHA-256 values and per-clip timings are stored in `1st test/dataset.suggestions.json`. Per-clip mean processing time was approximately 0.93-0.98 s/window; max 1.15 s. This measures offline inference computation, not event latency or precision. The model produced zero qualifying candidates for Ferro 10 m silenced; it must still be manually inspected. No ground-truth labels, accuracy scores, per-shot counts, suppressor attenuation claims or ARC-trained model were produced.

The `suggest-labels` CLI and the labeler's AI suggestions tab are implemented. Proposals are a separate sidecar and cannot train the model automatically. Explicit acceptance requires a human-confirmed track and valid edited interval/class/source. Acceptance adds one annotation without marking any coverage reviewed; rejection adds neither labels nor background. IDs prevent duplicate acceptance after an interrupted sidecar save. Media stamps/track IDs hide stale proposals; generator and reviewer checks prevent overwriting concurrent edits. Test clips are always excluded from proposal generation.

Full suite: **59 passed**, including short/final-window coverage, score handling, proposal validation, pending/rejected exclusion from truth, acceptance/idempotence, class-scoped review preservation, stale media/other-track rejection, explicit provisional-track processing, test-split exclusion, and concurrent-edit protection. Ruff and compilation passed. Synthetic model test doubles are used only in tests and clearly labelled; the pilot proposals came from the actual cached AST weights.

`scripts/suggestions_smoke.py` passed on Windows using an original pilot MP4 and a disposable manifest with explicit UI fixtures. It decoded 105 video frames, advanced beyond 2 seconds, blocked acceptance before confirmation, saved edited acceptance/rejection, kept reviewed coverage empty, checked the pilot's Gunfire-only review default and blocked concurrent proposal writes. The real pilot manifest was unchanged. Playback was muted for this automated check; correct audible track selection/perceptual sync still requires user review. The UI was rendered and inspected at `artifacts/suggestions-review.png`; results are in `artifacts/suggestions-smoke.json`. No game session, live overlay or gameplay input behavior was tested in this update. Live AST still requires full 10.24-second context; the short-window experiment is offline annotation assistance only.

## Editable review and recording imports

Current full suite: **82 passed**, including new reversible-decision, scoped review invalidation, file-import and Qt editor regression tests. Ruff and compilation passed. No model download, training run or real annotation mutation was needed for this update.

Suggestions can be edited after acceptance or rejection, re-accepted in place, rejected after acceptance, or returned to pending. Last edited boundaries/class/source persist in the sidecar. Only linked human labels are removed during reversal; unrelated manual annotations remain intact. Saved manual intervals can be added, edited and deleted. Affected class/time review coverage is invalidated when truth changes, preserving unrelated scopes. Source-only edits preserve coverage. Original AI scores remain identified with the original class/window.

An independent current-file review found and reproduced a failed-save regression: sorting an edited interval before a failed save left its old row index active, so a retry could overwrite another label. Transactions now restore the clip/proposals on initial save failure and always reset sorted editor/table indices, including after later sidecar failure. Regression tests cover failed sorted edits, failed deletion and a successful dataset write followed by failed decision-sidecar write. The latter reports the partial save explicitly and remains recoverable through proposal IDs. Stale dataset/sidecar writes are refused.

The media importer uses a Qt worker for local copying and PyAV header probes. External MP4/MKV/MOV/WAV recordings are copied without overwrite into `imported/`; in-dataset media is referenced. Existing group splits and annotations are preserved, unchanged duplicates skipped and new rows require human track confirmation. Synthetic tests cover collision handling, invalid/audio-free files, invalid groups, changed external sources, document nonmutation, session split inheritance and UI dropdown updates after worker completion. Atomic publication was also checked with disposable files inside this Windows OneDrive workspace. Unsupported hard-link filesystems report an import error.

The Windows suggestions smoke passed with **104 decoded frames**, using original pilot video playback muted and synthetic review candidates in a disposable manifest. It exercised acceptance edits, rejection after acceptance, reacceptance, undo to pending, manual add/edit/delete, Gunfire-only defaults and concurrent sidecar protection. The real pilot manifest remained unchanged. Updated UI rendering was inspected at `artifacts/suggestions-review.png`. The additional import UI test used synthetic WAV data and temporary datasets; no real recordings were imported for a test. Existing human decisions were not reset. No live overlay/game behavior or recognition accuracy was tested in this update.

## Console Ctrl+C shutdown

The later acceptance usability update is verified separately below; this section records the interrupt-specific run.

Full suite: **86 passed**, Ruff and compilation passed. `qt_interrupts.py` now services SIGINT/SIGBREAK through a 100 ms Qt timer and invokes the existing close path instead of raising KeyboardInterrupt inside Qt callbacks. On Windows it explicitly clears launcher-inherited ignore-Ctrl+C state using SetConsoleCtrlHandler(NULL, FALSE). Python signal handlers are restored after the event loop exits. Normal shutdown still waits for audio workers/current inference or labeler imports, preserves labels, and respects save/conflict errors. File selectors use Qt dialogs so timer-driven shutdown can dismiss them.

`scripts/interrupt_smoke.py` passed six Windows checks. Five child app cases received actual CTRL_C_EVENT events in dedicated hidden consoles: idle controls, active demo analysis, an open file dialog, idle labeler, and labeler with an active synthetic import. These cases intentionally set ignore-Ctrl+C before startup to verify that production code clears it. All exited with code 130, no KeyboardInterrupt traceback or live audio workers; controls hid their overlay/tray, and labeler cases preserved fixture labels. The active import completed and saved both fixture clips before exit. Measured shutdown in this run was approximately 0.31-1.20 seconds; this is not a guarantee for large imports or long model operations.

The sixth check launched the actual `launch.ps1` from a hidden interactive Windows PowerShell with process-only ExecutionPolicy Bypass, sent Ctrl+C, then successfully executed another command in that same shell. Its explicit final exit returned zero. Initial launcher probes exposed script-policy/inherited-console-state behavior; these were resolved and the full six-case smoke passed afterward. Failed disposable child processes were cleaned up without targeting user app instances. Results: `artifacts/interrupt-smoke.json`. No real labels or media were modified. Ctrl+Break was covered through Python signal tests, not a separate physical-keypress test.

Implementation references: Python documents [deferred execution of signal handlers](https://docs.python.org/3/library/signal.html#execution-of-python-signal-handlers); Microsoft documents [console control events](https://learn.microsoft.com/en-us/windows/console/generateconsolectrlevent) and [the inherited ignore-Ctrl+C attribute](https://learn.microsoft.com/en-us/windows/console/setconsolectrlhandler). The smoke sender is a separate helper attached only to its owned isolated test console; it never broadcasts to the user's launching terminal.

## Accept button usability

This section records the acceptance-specific run; the later review-readiness audit below reflects subsequent human annotation.

The prior Accept handler required an explicit Review action and human track confirmation, whereas Reject needed neither. Missing requirements were only reported in the bottom status text. Inspection at this update found all seven pilot tracks still unconfirmed; no confirmations or labels were invented to work around this.

A single row selection now loads its fields without requiring playback/Review first. Accept is disabled with an adjacent explanation until the selected clip's game-audio checkbox is checked and a valid row is selected. Confirmation enables it immediately; direct button clicks save the linked human interval and accepted table status for pending, rejected and accepted rows. Switching rows loads the new row's fields; clicking another column in the same row preserves draft edits. Existing track confirmation, media-staleness, review-mask and persistence checks remain active.

Full suite: **92 passed**; Ruff/compilation passed. Regression tests exercise actual QPushButton clicks without a Review call, all three decision states, the missing-confirmation hint/enabling transition, row switching and same-row draft preservation. Windows suggestions smoke passed using disposable UI fixtures and an original MP4 (muted), including single-select acceptance, edited reacceptance, rejection reversal, manual annotation CRUD and real-manifest preservation. The new inline hint/button layout was rendered for inspection at `artifacts/suggestions-review.png`. No real pilot labels or decisions were changed by these checks.

## Human review readiness audit

After the user reported review completion, the actual pilot contained seven confirmed Stream 1 selections, 20 gunfire intervals and 19.81 seconds of reviewed coverage over 75.88 seconds of media. Every clip had only one reviewed range, usually the last shot. Annotation containment was seven fully reviewed labels, one partially reviewed and twelve outside reviewed coverage. A read-only specialist audit and `scripts/audit_training.py` independently counted **47 positive / 5 negative complete one-second windows** on the actual trainer's 250 ms grid, decoding selected audio into RAM only. No inference, exports, labels, review scopes, groups or splits were changed by the audit.

All seven remain `first-pilot-session / train`, with no independent validation/test data. `train-local '1st test/dataset.json' --classes gunshot --output artifacts/pilot-readiness/arc-targets.npz` correctly refused with `No reviewed validation clips`; no experiment or default runtime weights were created. The report at `artifacts/pilot-readiness/report.json` records manifest SHA-256, per-clip coverage/counts and blockers. Original mixed-gameplay data was inspected separately: one confirmed track, three gunfire labels, one Snap Hook label and no reviewed coverage at this snapshot; it was not merged or repurposed as validation.

Added **Mark whole clip fully reviewed**, with explicit time bounds/class confirmation and an adjacent coverage/contained-label readout. It does not use the last shot's editor bounds or add labels. Actual user confirmation is still required. Tests cover whole-clip bounds, class scope, unchanged annotations/editor values, cancellation, and partially reviewed coverage feedback. Full suite: **95 passed**; Ruff/compilation passed. Existing labels are preserved; final training still requires sufficient confirmed coverage/background and an independent validation session.

## Public weapon-source acquisition — October 18, 2026

Fresh full suite: **257 passed**. Ruff (`src tests scripts`) and compileall (`src scripts`) passed. Synthetic tests cover CSV timestamps/idempotency, multi-weapon ambiguity, actual signal-processing transient proposals including 70 ms rapid fire, manifest-range limits, source failures/cache reruns, optional weapon schema/backward compatibility, acceptance/rejection/correction, scoped backgrounds, source/test grouping, advisory dedupe, diversity reporting and unverified discovery. No recognition accuracy was measured and no final weapon classifier was trained.

The real seed dry run read **41 rows, 26 weapon names and 14 unique YouTube videos**. The initial batch acquired 13 audio-only sources and isolated one HTTP 403 failure. Enabling the installed Node 24 runtime for yt-dlp resolved the failed source on retry. A subsequent full cached run used **14 cached sources, zero downloads, zero failures and zero new shot candidates**. The existing dataset now has **10,518 pending transient/weapon-context proposals**, with **135 possible-duplicate groups (343 candidates)**. Separate background generation added **47 pending absence proposals**. A rerun after those background proposals preserved all rows and created zero new candidates. Downloaded compressed audio remains in ignored `training/sources/`; no permanent per-shot WAV corpus was generated. Counts describe proposals, not verified shots.

All 19 pre-existing clip objects were compared against a pre-collection snapshot and remained exactly equal, including intervals, reviews, tracks, groups and splits. All new proposals remain pending and new source tracks remain unconfirmed. `1st test/dataset.json` was not merged or repurposed. The 26 overlapping untimestamped showcase rows produce one ambiguous candidate set instead of 26 falsely named copies.

Native Windows Qt playback and Accept/Reject/Correct Weapon were exercised on disposable copies using acquired public audio. Original dataset/sidecar hashes remained unchanged. Legacy MP4 suggestion smoke also passed (latest specialist run: 95 video frames, 2,033 ms playback). Reviewed screenshots include `artifacts/weapon-review-phase3.png`, `artifacts/weapon-review-full-size.png`, `artifacts/weapon-review-background.png` and `artifacts/weapon-review-smoke/`. Full-size measurement before backgrounds: 10,518 candidates / 29.3 MB sidecar; approximately 644 ms initialization, 192 ms filtering and 1,213 ms acceptance plus auto-advance, including 970 ms atomic JSON persistence. These are desktop review timings, not inference or live-game latency claims.

Independent review reproduced and prompted fixes for overlong weapon names, globally colliding burst IDs, filtered-out linked proposal edits/deletes and interrupted background-decision saves. Linked negative-review provenance now survives manual coverage merges and interval splits; rejection removes its own reviewed coverage even when the original decision sidecar write failed. Regression tests also preserve unrelated class coverage and manual positive labels. Background acceptance requires a default-No whole-range confirmation, refuses saved gunshot overlap and creates no positive event interval.

Real optional discovery returned two source references marked unverified without downloading their media or changing datasets. Discovery registration is metadata-only, and ingestion rejects explicitly unverified CSV rows. Background generation uses existing peaks/fingerprints and does not classify silence as negative truth. No audio uploads, paid APIs, game integration, trained weapon-ID model or gameplay/anti-cheat test was introduced.

The Windows interrupt smoke passed again: five isolated real Ctrl+C app cases exited 130 and the PowerShell launcher accepted another command in the same terminal afterward. No user terminal received a broadcast event. This check used disposable media/settings/labels; the game was not launched.

## Quick multiple-choice gunfire review — October 19, 2026

Fresh full suite: **304 passed**; Ruff (`src tests scripts`) and compilation (`src scripts`) passed. `label-training` now opens a short-sample **Yes / No / Unsure** view; `--advanced` retains the interval editor. Track confirmation, optional weapon identity and source training/validation role use multiple choice. Final test sources remain excluded. No main dataset/sidecar answers, track confirmations or split assignments were invented during implementation.

Exact one-second `presence_reviews` store human gunfire presence/absence separately from event intervals and exhaustive reviewed ranges. Unsure/Undo have durable nontraining `presence_states`, preventing a failed informational sidecar write from restoring a prior answer. Schema tests cover duplicate windows, sample alignment, class/track mismatches, optional weapon identity, manual-truth conflicts and grouping. Media size/mtime binds local quick answers to their source; collected-source SHA-256 is additionally verified before training. Changed media is refused rather than relabeled silently.

The trainer uses each answered second once, deduplicates matching legacy-grid windows, refuses contradictory truth and never extrapolates to neighboring audio or other classes. Positive/background minimums and independent train/validation groups remain required; test media is neither decoded nor hashed during fitting. Synthetic fixtures fitted/reloaded a temporary one-class model using 40 train and 10 validation windows. Reports distinguish presence windows from legacy intervals. These checks do not measure ARC event recall, gameplay accuracy, weapon recognition or onset precision; no real model/default runtime weights were created.

Native Windows synthetic playback checked the listen gate, explicit track confirmation, Yes→No→Unsure→Undo and optional weapon selection. Qt regressions cover media-ready seeking, keyboard focus, stale media, concurrent edits, save rollback and sidecar-failure recovery. A fresh read-only real-collection launch opened `QuickReviewWindow` with **10,565 questions**, completed one muted source sample and closed successfully. Original dataset and sidecar bytes remained unchanged. The screenshot was visually inspected: `artifacts/quick-review-live-smoke/preview.png`; report and reproducible check: `artifacts/quick-review-live-smoke/report.json` and `scripts/quick_review_smoke.py`. Questions target exact source windows, while Qt playback uses codec/timer granularity rather than sample-accurate cutting.

The completed interrupt smoke for this implementation passed **seven Windows checks**, including the quick view: six isolated Ctrl+C child cases returned 130, and the PowerShell launcher accepted a subsequent command. Advanced import cleanup remains covered. This was completed before the interrupted documentation step; the current resume reran the full suite and read-only playback smoke. No user terminal was signaled, raw recording created or game launched. Independent review findings about Unsure/Undo recovery, media binding and state-only track changes were fixed and regression-tested.
