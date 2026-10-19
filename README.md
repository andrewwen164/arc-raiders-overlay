# ARC Audio Companion

For the reusable Codex project agents and example prompts, see [CODEX_WORKFLOW.md](CODEX_WORKFLOW.md).

A Windows desktop learning project that turns local stereo audio into brief, rough visual cues. Python 3.11+, PySide6, NumPy, SciPy and PyAudioWPatch. **Demo is the default every launch. No game API is used.**

The baseline detects **sound activity** and **transients**, not footsteps or enemies. An optional, explicitly downloaded AudioSet AST model can classify generic footsteps/gunshots in long audio clips. Neither backend is trained or validated on ARC Raiders.

**Training-folder update:** a third, locally trainable backend supports selected player-activity classes: footsteps, reloads, gunfire, landing impacts, Snap Hook, zip-line, climbing/vaulting and healing sounds. The video-labeling tool records intervals and which classes were fully reviewed. See [TRAINING.md](TRAINING.md) for review, class selection, training and evaluation commands. Collections now have partial human annotations; use `training-status` and the readiness audit for current counts and missing coverage. No ARC-trained weights or gameplay accuracy claim exist yet. Target-only filtering is enabled for recorded/live input; choose the local backend after training, or disable filtering for baseline diagnostics. Demo remains a synthetic baseline test. Audio alone cannot establish enemy identity.

**First recording pilot:** [1st test/README.md](1st%20test/README.md) covers seven listener-side gunfire clips, Silencer II comparisons, the missing Ferro 50 m silenced condition, and a prepared AI-suggestion review workflow. Proposals require human acceptance and do not train the model automatically. See [model-assisted review](TRAINING.md#optional-model-assisted-review) for the `suggest-labels` command and its short-window limitations.

The labeler supports reversible AI decisions, editing saved intervals, manual intervals without suggestions, and importing additional local video/WAV recordings. See [adding and editing training data](TRAINING.md#add-your-own-intervals-or-recordings). Importing external recordings copies them locally into the dataset; it does not modify the originals or start live capture.

## Scope and permission

This uses standard WASAPI whole-output-device loopback, Win32 desktop window styles and `RegisterHotKey`. It does not read game memory, inject code, install rendering/input hooks, change game files, inspect packets, automate gameplay, or bypass anti-cheat.

**Permission for live-match use has not been established. This is not approved by Embark and is not guaranteed safe from bans.** Developer policy approval means the developer permits a particular use. Anti-cheat detection is a separate technical question: using documented desktop APIs does not prove permission or prevent detection. Obtain explicit guidance from the developer before using it in live matches. Start with synthetic or your own recorded audio.

All inference is local. There are no paid APIs, accounts, audio uploads, or runtime network calls. Package installation and the explicit model-download command require Internet access. Settings, model files and detection logs are local. The app **never writes captured raw audio** and has no recording feature; its only WAV-writing command creates a synthetic fixture on explicit request. Audio exists temporarily in bounded RAM buffers (50 ms chunks; AST retains about 10.24 seconds).

## Setup — exact PowerShell commands

Use **64-bit CPython 3.11 or 3.12 on Windows 10/11** for the recommended setup. The declared range is 3.11–3.14; the verified runtime for this implementation is 3.11.17. Native ARM64 is not verified; use x64 Python if no PyAudioWPatch wheel is available. Install Python with its Windows launcher if `py` is missing, then open a new PowerShell window.

```powershell
Set-Location -LiteralPath 'C:\Users\andre\OneDrive\Desktop\Codebase\arc raiders overlay'
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
.\.venv\Scripts\python.exe -m arc_audio ui
```

No environment activation or administrator privileges are needed. If this checkout already has the tested `.venv`, use its Python directly; do not recreate it while the app is running.

For the included launcher:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\launch.ps1
```

The execution policy flag applies to that PowerShell process; it does not change machine policy. In a managed environment, follow the organization's script policy or run the Python command directly.

To close either the control window or labeler, focus the launching terminal and press **Ctrl+C**. If text is selected in the terminal, press Escape to clear the selection first. Ctrl+Break is also supported on Windows. This requests normal shutdown: analysis stops, overlays close and labels save; an active import or model operation may need time to finish. Save errors/stale datasets still require resolution rather than discarding data. Direct Python commands return exit code 130 for an interrupt, without a KeyboardInterrupt traceback. You can reuse the same terminal afterward. Ctrl+Alt+F10 remains the separate global emergency stop that hides the overlay and stops analysis.

Alternative setup with an already installed `uv` (the method used here):

```powershell
uv venv --seed --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -e '.[dev]'
.\.venv\Scripts\python.exe -m arc_audio ui
```

`--seed` includes pip for the optional installation commands below. If you already created an unseeded uv environment, use `uv pip install --python .venv\Scripts\python.exe pip` first, or use `uv pip install --python .venv\Scripts\python.exe ...` instead of `python -m pip install ...`.

## Try it without a game

1. Launch the app, leave **Demo** and **A · Baseline** selected, and click **Start analysis**.
2. Synthetic audio loops every nine seconds: left burst at 1 s, centered burst/activity at 3.5 s, right burst at 6.5 s. Demo is intentionally silent; detections come from the real baseline processing generated samples, not random/model predictions.
3. Choose the overlay monitor. The overlay uses a compass-style layout around screen center: broad pulsing side arcs, compact event icons and short labels. Repeated cues of the same type renew in place to reduce clutter. Unknown direction appears as a separate centered cue inside the ring, with no directional arc. The overlay is transparent and click-through; use the control window to change it. Indicators disappear with a timer-driven fade.
4. Switch focus to a normal desktop application and press **Ctrl+Alt+F10**. Analysis should stop and the overlay should disappear. This is a Windows global hotkey, not a control-window-only shortcut.
5. The tray menu offers **Emergency stop / hide overlay**, **Show controls**, and **Quit**. The tray icon may be in Windows' hidden-icons area. Hotkey registration success/failure is displayed in the controls. If another program owns the shortcut, the label includes the Windows error and the tray/Stop fallback. There is no hidden keyboard hook.

An automatic eight-second demo smoke run (opens visible windows and then closes):

```powershell
.\.venv\Scripts\python.exe -m arc_audio ui --smoke-seconds 8
```

To render a static preview of the compass layout without capturing or playing audio, run `python scripts/preview_overlay.py` using the venv Python. It writes `artifacts/compass-preview.png` from explicitly constructed UI fixtures, not model predictions.

## Recorded WAV playback

Create an entirely synthetic stereo test recording:

```powershell
.\.venv\Scripts\python.exe -m arc_audio make-demo .\artifacts\demo.wav --seconds 9
```

For the synthetic tone fixture, choose **Recorded stereo WAV**, load the file, **uncheck Only trained player-activity sound classes**, select the baseline, and Start. This is an activity/playback diagnostic, not a semantic sound test. For your recordings after supervised training, keep target-only filtering enabled and select the local model instead. Playback goes to the **current Windows default output**. Analysis receives the same source frames as playback. Indicators are scheduled against PortAudio's estimated DAC timing, rather than running ahead at offline analysis speed. Bluetooth latency, driver timestamp quality and physical output buffering can affect sync; this is not sample-accurate acoustic calibration. Stop also aborts playback; a small amount of audio already handed to a driver may finish.

Supported recordings: uncompressed, two-channel PCM WAV, integer 8/16/24/32-bit, 8–192 kHz. Float WAV, compressed WAV and mono/surround files are rejected with an export hint. Prefer **stereo PCM16 at 48 kHz**. Source analysis preserves the WAV rate; playback uses the default output rate with antialiased polyphase resampling when necessary. Resampling at block boundaries may introduce small edge artifacts. Offline evaluation does not need an audio output device.

Sensitivity and minimum score are locked during analysis; Stop to change them. Opacity, lifetime, scale, monitor and visibility can change while running. Sensitivity affects only the baseline. The UI scale affects controls and indicator sizing; Windows DPI scaling is applied separately by Qt.

## Live loopback — explicitly selected

Choose **Live WASAPI loopback**, acknowledge the live-match notice, refresh devices, select a stereo loopback output and Start. Live input is never saved as the startup mode and cannot start without that acknowledgement.

Keep target-only filtering enabled and select your trained local model for your selected sound classes. Until it is trained, use demo, or deliberately uncheck the filter and select Baseline to diagnose audio capture. Baseline activity does not identify semantic sound classes or enemies. Generic AST supports only footsteps/gunfire.

```powershell
.\.venv\Scripts\python.exe -m arc_audio devices
```

Only stereo loopback devices accepting float32 at their native default sample rate are offered. Set the Windows output channel layout to stereo if necessary; the app intentionally does not guess a surround channel map. Device indices are transient: Refresh after changing devices. A selected device's name is checked again before opening.

This is **whole-device capture, not game/process-specific capture**. Discord, system notifications, music, a video, and any application routed to that output affect detection. There is no process picker or process isolation. Switching the Windows default device does not move an active capture; Stop and reselect. Stream errors stop the session. No packets produces an idle/disconnect warning after 3 seconds and an error after 15 seconds; some drivers legitimately withhold packets during silence. Reconnect/play audio, Refresh, then Start.

## Detection backends and limits

**A. Baseline (always available):** 50 ms stereo chunks; adaptive RMS noise floor, onset rise and positive spectral flux; a shared 650 ms cooldown; user sensitivity and score thresholds. Its confidence is a heuristic strength score, **not a calibrated probability or accuracy claim**. A sustained sound can generate periodic activity cues. Onsets of music or machinery can be transients. The first 300 ms is warm-up.

Stereo direction compares left/right channel energy. A normalized difference beyond ±0.35 produces rough left/right balance; weak or balanced evidence is **unknown**. It cannot establish azimuth, front/back, elevation, distance, number of people, or teammate/enemy identity. Mixed sources and virtual surround can make even left/right balance misleading. No exact enemy positions are drawn.

The visual layout is inspired by directional sound indicators in games. Side arcs are fixed, broad **left/right balance symbols**, not measured angular sectors. There are no front/back cues, compass bearings or distance markers. The center of the display remains clear for your game's crosshair. Changing the visual style does not upgrade the baseline into a footsteps/gunshot classifier; those icons appear only for actual optional-model events.

**B. Optional pretrained AST:** [MIT's AudioSet checkpoint](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593), implemented through the maintained [Transformers AST interface](https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer). The checkpoint is an older, fixed research model; the library integration is maintained. The model card declares **BSD-3-Clause**, permitting this use subject to its terms. Keep upstream attribution/license notices when redistributing weights; this repo does not redistribute them. Transformers is Apache-2.0; other dependencies have their own licenses.

The verified [label map](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593/blob/main/config.json) has class 53 `Walk, footsteps` and class 427 `Gunshot, gunfire`; these alone map to `footsteps` and `gunshot`. No heuristic output is relabelled as either. The [feature configuration](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593/blob/main/preprocessor_config.json) specifies mono 16 kHz, 128 mel bins and 1024 frames. This implementation waits for **10.24 seconds of continuous audio**, uses the upstream normalization/feature extractor, and attempts an update each second. The extractor pads/truncates to 1024 frames. Multi-label logits use sigmoid, not a two-class softmax. Scores are not calibrated for ARC Raiders.

Download is explicit and optional. Safetensors weights are approximately **346 MB (330 MiB)** plus small configs/model card; PyTorch and its dependencies add a much larger installation. Only safetensors is loaded, with `local_files_only=True` and no remote code. The downloader fixes the source revision to `f826b80`. Nothing downloads when opening or selecting the backend.

```powershell
.\.venv\Scripts\python.exe -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e '.[classifier]'
.\.venv\Scripts\python.exe -m arc_audio download-model --accept-download
.\.venv\Scripts\python.exe -m arc_audio evaluate .\artifacts\demo.wav --backend ast
```

Use a recording **longer than 10.24 seconds** to obtain AST predictions; the nine-second demo WAV deliberately produces none. Generate a longer fixture with `--seconds 15` to test model plumbing, not semantic accuracy. Once downloaded, select **B · Optional AST** in the UI. Missing packages/weights produce an actionable error; there is no fallback masquerading as a trained model.

Expected delay is **10.24 seconds of initial context plus feature extraction/inference time**, with subsequent updates at most once per second. CPU performance is hardware-dependent; inspect the measured processing/pipeline times. The AST output describes a whole trailing clip, not a precisely located onset, and its direction is always unknown because class-specific channel attribution is unavailable. Audio gaps reset model context. On slower CPUs, inference can overrun the 200 ms audio queue and repeated resets can make interactive AST impractical; offline evaluation is the preferred model workflow. Alerts more than two seconds late at UI delivery are logged but not displayed. Optional inference cannot be interrupted mid-forward-pass; emergency stop hides the overlay immediately and requests capture stop, then the worker finishes/cleans up.

Generic models can confuse the player's sounds, teammates, machinery, echoes and environmental noise. No ARC-specific accuracy, enemy identification, or game approval is claimed.

## Evaluation and annotations

```powershell
.\.venv\Scripts\python.exe -m arc_audio evaluate .\artifacts\demo.wav --annotations .\examples\annotations.json --output .\artifacts\detections.jsonl
.\.venv\Scripts\python.exe -m arc_audio evaluate 'C:\Recordings\my-session.wav' --annotations 'C:\Recordings\my-session.labels.json' --output .\artifacts\my-detections.jsonl
```

Annotations use source-file seconds, not wall time:

```json
{
  "version": 1,
  "intervals": [
    {"start": 12.30, "end": 12.85, "label": "footsteps", "direction": "any"},
    {"start": 21.00, "end": 21.20, "label": "gunshot", "direction": "unknown"}
  ]
}
```

Allowed labels: `sound activity`, `transient`, `footsteps`, `gunshot`. Direction is `any` (also the default), `unknown`, `left`, or `right`. A baseline alert does **not** match a footsteps annotation. Label acoustic activity/transient intervals for baseline evaluation and semantic classes for AST. Use `--backend ast` for semantic evaluation.

Matching uses **exact class**, optional direction, and a detection's `audio_time` point inside `[start - tolerance, end + tolerance]`. Default tolerance is 0.15 s (`--tolerance`). Maximum-cardinality one-to-one bipartite matching avoids greedy failures on overlapping intervals. Each interval matches at most one alert; repeated alerts are false positives, unmatched intervals false negatives. Precision/recall are null when their denominator is zero. The command reports overall and per-class values, measured processing mean/p50/p95/max, and matched delay from annotation onset (clamped at zero for early alerts). For AST this strict point rule uses the clip **end**, not arbitrary overlap with its ten-second context; late clip detections may be counted as misses. This intentionally exposes its latency limitation. Logs also include `window_start`/`window_end` for later clip-level analyses.

The synthetic annotation example intentionally includes a sustained center sound; its abrupt onset can cause an extra transient false positive. Synthetic results check the machinery and are **not gameplay accuracy estimates**. Without annotations, the command reports timing and detection counts, and explicitly says accuracy was not computed. Offline timing excludes capture, playback, queueing and display, and is separate from onset delay.

For later ARC-specific work:

1. Use a permitted local recording workflow to collect your own stereo gameplay audio. This app does not record. Keep original timestamps/rate/channel order, export PCM16 WAV, and avoid normalization that changes channel balance. Record only with permission; remove/private-label chat as appropriate.
2. In an audio editor, listen and mark intervals for each class, plus silence and hard negatives (music, own movement, teammate movement, machinery). Preserve uncertain labels separately; do not guess enemy identity from audio. Save the JSON beside each WAV.
3. Split recordings by session/map before tuning; keep a held-out set. Tune thresholds on development recordings, then report event metrics, duplicates, misses and latency on held-out recordings. Keep whole-device contamination and overlapping sounds in the notes.
4. Inspect false positives/negatives and class balance before considering fine-tuning. A future ARC-specific model requires a permitted, labelled dataset and independent evaluation; collecting examples alone does not establish accuracy.

## Architecture, privacy and diagnostics

`src/arc_audio/` contains `audio.py` (sources/resampling), `detection.py` (baseline), `classifier.py` (optional model), `events.py`, `buffering.py`, `pipeline.py` (workers), `overlay.py`, `windows.py`, `ui.py`, `settings.py`, and `evaluation.py`. `__main__.py` is the CLI.

Capture/playback and inference run outside Qt's UI thread. A four-chunk queue evicts oldest data, and chunks more than 250 ms old are discarded. Detection delivery is also bounded; UI polling avoids an unbounded Qt signal backlog. The overlay retains at most 12 indicators. Audio overflow/underflow flags, discarded data and event delivery overflow contribute to the Dropped/xruns counter. This is a chunk/notification count, not a count of missing samples. Displayed processing time measures the detector call; pipeline time estimates delay since the captured/audible chunk end and excludes display/compositor delay and AST context length. Input levels show RMS in dBFS. No worker calls Qt widgets.

By default, `%LOCALAPPDATA%\ArcAudioCompanion\` contains validated `settings.json`, `logs\*.jsonl`, and optional `models\ast-audioset\`. Writes to settings use temporary-file replacement. Corrupt settings produce a visible warning and safe defaults. Event logs contain UTC timestamp, source seconds, class, score, rough direction, backend, measured processing/pipeline latency, and optional classification window. They contain **no raw audio**. Each session gets a unique log. Delete old logs/model caches manually when no longer needed; there is no automatic retention policy.

## Troubleshooting and Windows verification

- **Ctrl+C does not close an older running instance:** Close that window with X once, then relaunch to load the interrupt fix. Focus the terminal and clear any text selection before pressing Ctrl+C. Qt file selectors are used so interrupts stay responsive while a picker is open. If an import/model operation is finishing or a label-save error is visible, resolve/wait for it rather than force-killing the process.
- **No devices / invalid format:** Refresh, choose the output used by the game, set stereo in Windows sound settings, disable exclusive ownership by another app, and reconnect. No microphone permission is needed for output loopback. Some Bluetooth modes/drivers do not expose compatible stereo loopback.
- **No alerts:** Watch input levels/status, lower minimum confidence, increase baseline sensitivity, and test demo first. AST requires a full uninterrupted context and optional weights. A high score threshold can intentionally suppress everything.
- **Lag / drops:** Stop background audio/tasks, use baseline, and prefer a wired 48 kHz output. Stale audio is discarded, so missing cues are preferable to a growing delay. File playback continues even if inference misses chunks.
- **WAV load error:** Export two-channel uncompressed integer PCM16 WAV. A file extension alone does not guarantee that format. Choose a working Windows default output for playback.
- **Invisible overlay:** Check Show overlay, monitor, opacity, and Windows desktop compositing. The overlay targets QScreen's full logical geometry (including negative coordinates), sets its screen, and lets Qt handle per-monitor DPI. Move it across mixed-DPI monitors and confirm placement. Unplugging a display falls back to an available screen.
- **Clicks/focus:** Verify with another desktop app before any game: click and type through an indicator, confirm the foreground window stays the same when cues appear, and try the global emergency shortcut while that app has focus. Overlay windows use transparent-input/no-focus Qt flags plus Win32 layered/transparent/no-activate/tool-window styles and nonactivating topmost positioning.
- **Fullscreen:** Borderless fullscreen/windowed applications are the target. **Exclusive fullscreen support is not guaranteed.** Protected surfaces, driver/compositor behavior and another program's topmost window can obscure it. There is no rendering hook or bypass workaround.
- **Hotkey failure:** Read the registration status/error, close software using Ctrl+Alt+F10 and restart, or use tray → Emergency stop. If the tray is unavailable keep the Stop button accessible. Closing controls requests shutdown; it waits without blocking the UI. Drivers that fail to close within three seconds produce an error; quit/reconnect rather than restarting that device repeatedly.
- **Model setup failure:** Keep baseline usable. Reinstall the optional extra in the same venv, check free space/network for the explicit download, and remove only the model cache if incomplete. No cloud fallback is used.

Checks to run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m compileall -q src
.\.venv\Scripts\python.exe scripts\windows_smoke.py
```

Synthetic tests cover stereo direction, silence, debounce, bounded buffering, stale discard, PCM/resampling, settings validation, matching and worker shutdown. `windows_smoke.py` uses only synthetic playback/loopback, checks native window styles, verifies nonactivation against its own control window, and posts a standard hotkey thread message to verify dispatch. It does not synthesize key presses or test a game. It reports details, including skips, in `artifacts/windows-smoke.json`. Model inference and actual keyboard input require separate checks; see `VALIDATION.md` for checks actually executed in this checkout. No executable packaging has been attempted; it remains a follow-up.

Dependency choices were checked against [PyAudioWPatch documentation](https://github.com/s0d3s/PyAudioWPatch), [its Windows wheels](https://pypi.org/project/PyAudioWPatch/), [Qt input/focus flags](https://doc.qt.io/qtforpython-6/PySide6/QtCore/Qt.html), [Qt transparency requirements](https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QWidget.html), and [Microsoft RegisterHotKey](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-registerhotkey). The actual installed dependency versions and Windows checks are recorded in `VALIDATION.md`.
