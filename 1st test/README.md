# First listener-side gunfire pilot

Seven original MP4s, **75.88 seconds** in total. User-reported setup: your friend was shooting, you recorded from the listener's position, suppressed takes use **Silencer II**, and track 1 is the intended game-audio track. Distances and weapon conditions come from filenames plus those notes, not independent measurements. Keep track confirmation pending until you listen in the labeler.

**Ferro at 50 m with Silencer II is missing.** Continue with the other seven conditions. No audio, labels, or synthetic substitute has been created for the missing setup, and a paired Ferro suppressor comparison at 50 m is not possible from this collection.

## Open the prepared review

From the project folder:

```powershell
.\.venv\Scripts\python.exe -m arc_audio label-training '.\1st test\dataset.json'
```

1. Listen on **Stream 1**, then check **I listened and confirmed this is game audio** for each clip. If another stream is correct, change it first; suggestions generated for a different stream will be hidden and must be regenerated.
2. Open **AI suggestions**. Single-click any row, including accepted/rejected ones, to load its fields. **Review / edit selected suggestion** or double-click optionally plays the window with half a second of context. Playback pauses after that context.
3. Adjust **Start**, **End**, sound class, and source tag above. Use **Accept / save selected suggestion**, **Reject selected AI suggestion**, or **Undo decision / return selected suggestion to pending**. If Accept is disabled, read the message beside it: confirm the selected game-audio track using the checkbox above, then select a row. No separate Review action is required. Re-acceptance updates the same linked label; rejecting/undoing an acceptance removes that label while retaining the last edit for reconsideration. Scores belong to the original AI class/window and are not probabilities of correctness. Pending/accepted/rejected states remain visible.
4. Listen through the **whole short clip** and manually add missed sounds. Neighboring windows can refer to the same event/burst: label each audible event or consistently grouped burst once rather than accepting duplicates. Adjust coarse boundaries to the actual audible span. The first clip (Ferro 10 m silenced) has zero model suggestions and needs manual labeling.
5. With only **Gunfire** checked under **Classes fully checked**, select a stretch you have completely inspected and click **Mark range fully reviewed**. If you checked the entire video, use **Mark whole clip fully reviewed** to include its pauses as well as the shot labels. This does not reuse the last shot's Start/End values. The readout beside it shows reviewed seconds and how many labels are fully inside reviewed coverage. Include checked background-only stretches. Accepting proposals does not mark a range reviewed or establish that unlabelled audio is background.

Human labels and reviewed spans save to `dataset.json`; model proposals and decisions save separately to `dataset.suggestions.json`. The original videos are untouched. The existing `training/dataset.json` was not merged, replaced, or annotated.

For missed sounds, use **Add manual sound interval** with the Start/End/Sound fields above. It works without an AI suggestion and opens **Saved labels / reviewed ranges** so you can see the saved row. Double-click a saved interval to edit it, then click **Save interval changes**. To add new recordings, use **Import video / WAV files...** at the top and enter the real session name; see [import details](../TRAINING.md#add-your-own-intervals-or-recordings). Corrections/removals invalidate affected class/time review coverage, so check those stretches again before marking them fully reviewed.

## Initial model pass

Real local inference used the existing cached MIT AudioSet AST model; no weights or audio were uploaded, and no new model download was needed. Two-second mono 16 kHz audio windows were scored with a two-second hop, plus an end-aligned final window. Short inputs are padded by the upstream feature extractor to its 1024-frame input. This differs from the live AST backend, which still requires 10.24 seconds of continuous audio. Short-window performance on ARC Raiders is unmeasured. These are window-level class suggestions, not detected individual shots or exact onset/offset estimates.

The initial proposal cutoff was **0.2**. Forty windows yielded **10 gunfire candidates**; current pending/accepted/rejected counts change as you review them:

| Clip | Candidate windows |
|---|---:|
| Ferro 10 m silenced | 0 |
| Ferro 10 m unsilenced | 2 |
| Ferro 50 m unsilenced | 1 |
| Stitcher 10 m silenced | 1 |
| Stitcher 10 m unsilenced | 4 |
| Stitcher 50 m silenced | 1 |
| Stitcher 50 m unsilenced | 1 |

Those counts are not counts of actual shots or correct detections. No precision/recall was measured for the initial model pass. Human annotation now exists, but there is no independent, fully reviewed evaluation session yet. Zero suggestions does not mean no shots were audible. Per-window inference took approximately 0.93-0.98 seconds on average per clip, with a maximum about 1.15 seconds; these are offline processing times, not live detection latency. See `dataset.suggestions.json` for scores, timings, file fingerprints, and model/config SHA-256 provenance.

## Regenerate proposals if needed

Close the labeler before generation. This command uses the already available project cache and never downloads a model:

```powershell
.\.venv\Scripts\python.exe -m arc_audio suggest-labels '.\1st test\dataset.json' --model-dir '.\artifacts\classifier-data\ArcAudioCompanion\models\ast-audioset' --classes gunshot --window-seconds 2 --hop-seconds 2 --threshold 0.2
```

By default only human-confirmed tracks in train/validation are processed. The first pass explicitly used `--allow-unconfirmed-track` to make provisional suggestions for the selected track before your review. This flag does not allow acceptance without confirmation. Test clips are always skipped. Missing weights cause an actionable error; there is no automatic download or heuristic replacement. For a new machine, follow the optional classifier installation/download instructions in the main README.

Reruns add suggestions while preserving existing decisions for identical model/input/window IDs. They do not clear old pending suggestions when changing the cutoff. Media changes invalidate old proposals; changing tracks hides proposals from the other stream. Keep both sidecars with the videos when moving the folder; touching/re-encoding media may require regeneration because fingerprints change.

## Next collection

An October 18 review-readiness audit found **20 gunfire intervals, seven confirmed tracks, and 19.81 / 75.88 seconds of reviewed coverage**. Seven labels were fully inside coverage, one partially inside and twelve outside. Most clips had only the last sound's range marked reviewed. Actual trainer-grid counts were 47 positive and five background windows; the current training minimum is 20 of each, plus separate validation data. These are readiness counts, not accuracy or independent examples. Run `scripts/audit_training.py '1st test/dataset.json'` to refresh this snapshot after your next edits.

Existing sound labels are preserved. If you listened through a whole clip and labeled every audible gunshot, explicitly use **Mark whole clip fully reviewed** with only Gunfire checked. No coverage was silently expanded by the audit. A training attempt with `--classes gunshot` and a separate artifact output correctly refused due to missing reviewed validation clips and created no model. After completing coverage, review gunfire and pauses from another real recording session for validation; an existing gameplay recording from a different session can supply that data. Keep a further session for final testing.

All seven clips are grouped as **first-pilot-session / train**. Different weapons, distances and attachments from this one session do not create independent validation/test sessions. Do not split these clips just to make training pass. Collect separate sessions, annotate targets and confusers, and reserve independent gameplay for validation and final evaluation. This pilot only covers gunfire; it supplies no verified footsteps/reload/Snap Hook labels. No ARC-specific classifier has been fitted from it yet.

`pilot-report.json` records the available/missing conditions and numeric track diagnostics. Whole-clip levels are affected by timing, ambience, shot count, decoding and downmixing; they do not measure silencer attenuation. Decoded AAC/downmixed peaks may exceed 0 dBFS and alone do not prove the original capture clipped. Neither track's semantic role is established by correlation or level statistics.
