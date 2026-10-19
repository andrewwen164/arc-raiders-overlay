# ARC Audio Companion project instructions

## Project context

This is a Windows desktop app using Python 3.11+, PySide6, NumPy, SciPy and
PyAudioWPatch. Read README.md for behavior and boundaries, TRAINING.md for dataset
and model workflows, and VALIDATION.md for historical verification. Inspect the
current files before making claims about dataset readiness or model accuracy.
For collection work, also read GUNSHOT_RECORDING.md and 1st test/README.md.

## Lead agent and delegation

The main chat owns requirements, architecture, shared interfaces, integration,
and the final result. Use the project agents in .codex/agents/ for substantial,
bounded tasks when specialization or independent work helps:

- data_curator: recording plans, coverage, track selection, annotation and splits.
- training_evaluator: features, fitting, model artifacts, metrics and experiments.
- desktop_ui: Qt controls, labeler, overlay and desktop/audio integration.
- reviewer: independent correctness review with evidence; no file edits.

Delegate independent audits or work in separate files when useful. Keep small
changes and dependent steps in the main agent. Do not start every specialist for
every task. Give each assignment an objective, relevant inputs, allowed files,
validation expectations and an expected result. Specialists should return to the
lead rather than create additional agents themselves.

Assign one writer per file or mutable artifact. Serialize shared changes to
events.py, dataset/model formats, settings, the CLI and documentation. The lead
must check producers and consumers together when changing these contracts, then
integrate and validate the combined result. Use a reviewer for substantial changes
to training/evaluation semantics, schemas or worker lifecycle when useful.

## Architecture and behavior

- src/arc_audio/audio.py and buffering.py own audio input and bounded buffering;
  pipeline.py owns worker coordination. Keep capture and inference off Qt's UI
  thread. Workers must not call widgets. Preserve bounded queues, stale-data
  handling, gap resets, prompt stop/hide and orderly shutdown.
- events.py defines shared chunks/detections; detection.py, classifier.py and
  local_model.py implement distinct backends. Keep baseline activity, generic AST
  predictions and locally trained classes distinguishable.
- ui.py, overlay.py, labeler.py and windows.py own the desktop experience.
  Preserve demo startup, click-through/nonactivating overlay behavior and the
  emergency stop. Do not infer semantic direction from the whole mixed signal.
- Use standard desktop/audio APIs. Preserve the project's no-game-integration
  boundary: no memory access, injection, hooks, packet inspection or gameplay
  automation. Keep inference local and model downloads explicit.

## Data and experiment integrity

- Preserve original recordings, human annotations and review decisions. Keep
  training/dataset.json and 1st test/dataset.json separate unless a task explicitly
  requests a merge. Inspect actual manifests; documentation may describe an older
  state. Do not regenerate inventories over existing annotations casually.
- Never invent track confirmations, labels or reviewed coverage. Model suggestions
  remain proposals in their sidecar until human review; they are not ground truth.
  Unreviewed audio is not background. Negative labels require reviewed coverage
  for the relevant classes. Preserve uncertainty and class-specific review scopes.
- Group splits by independent recording session; adjacent windows, clips or
  variants from one session are not independent data. Fit preprocessing on train,
  tune on validation, and reserve test for independent evaluation. Do not reshuffle
  groups or use test suggestions to make an experiment pass.
- Use separate experiment outputs instead of overwriting the default runtime
  model. Record input/model/config provenance. Distinguish window metrics from
  event metrics, synthetic checks from gameplay accuracy, and processing time
  from event/display latency. Scores do not establish calibrated probabilities,
  enemy identity, weapon identity, exact direction or distance.

## Validation

Use the existing .venv directly from PowerShell. Do not recreate it or install
dependencies unless needed for the task. Select checks appropriate to the change:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m compileall -q src scripts
```

For relevant desktop changes, inspect scripts/preview_overlay.py,
scripts/windows_smoke.py, scripts/training_smoke.py and scripts/suggestions_smoke.py
and run the applicable checks. Use disposable fixtures for mutating smoke tests;
never alter real labels for a test. Visually inspect meaningful UI changes and
report checks that could not run. For documentation/agent configuration changes,
validate syntax, paths and consistency without launching audio or retraining.

Report the result, changed files, checks actually executed and remaining limits.
Update the relevant user docs when behavior changes. Do not present historical
counts in VALIDATION.md as fresh test results.
