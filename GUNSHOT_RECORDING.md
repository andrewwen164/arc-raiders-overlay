# ARC Raiders gunshot distance research and collection guide

Research checked October 18, 2026. This is a collection/testing proposal, not a measurement of ARC Raiders' current audio curves. No game session or controlled audio comparison was performed for this research.

## Conclusion

Collect multiple listener distances. There is no verified single reference distance that establishes recognition of every weapon at every audible range. Turning down a close sample is not a substitute for collecting genuine distant examples. Train for the conditions the app will actually encounter, then measure generalization on separate recordings.

The existing app detects the `gunshot` sound class. It does not acoustically identify a weapon by name or estimate its distance. The labeler now supports human weapon metadata and context-based proposals from curated public sources; see [public-source collection](TRAINING.md#building-the-weapon-dataset-from-public-source-videos). A trained weapon recognizer remains a separate future task.

## What the sources establish

- **ARC-specific source recordings:** Embark describes gun recordings captured at the weapon, an intermediate location, and hundreds of metres away. This establishes that its source material includes multiple distances; it does not reveal runtime blending distances, attenuation curves, or which layers each weapon uses. [Embark audio design overview](https://arcraiders.com/news/soundscapes-blog).
- **ARC-specific spatial processing:** Epic's interview with Embark describes the game's use of spatialization and acoustic simulation to communicate distance and environment. It does not publish per-weapon acoustic parameters. [Embark developer interview](https://www.unrealengine.com/developer-interviews/embark-studios-build-the-award-winning-arc-raiders-with-unreal-engine).
- **Game-audio mechanisms to test:** Unreal documents distance-related level changes, frequency filtering, reverb sends, and obstruction-dependent filtering/level reduction. These explain why volume alone need not reproduce distance. They are engine capabilities, not proof of ARC's specific implementation or settings. [Unreal attenuation documentation](https://dev.epicgames.com/documentation/en-us/unreal-engine/sound-attenuation-in-unreal-engine).
- **Weapon variants/versioning:** Embark's August 18, 2026 update lists new firing audio associated with Tier 3 magazine upgrades. Record the game version and complete loadout, rather than assuming the weapon name determines every sound. [Official update 1.42.0](https://arcraiders.com/news/live-update-1-42-0).
- **Related ML research:** A published gunshot dataset collected in another FPS includes firearm identity, source/receiver distances and directions. It is relevant experimental design, not evidence of ARC-specific recognition accuracy. [BGG / Enemy Spotted paper](https://arxiv.org/abs/2210.05917).

I did not find a published controlled per-weapon ARC dataset or measured range-transition table sufficient to answer exactly where each gun changes character. Sound-range videos, subjective player reports and damage-range tables are not interchangeable with that evidence. No exact audibility limit, filter cutoff, layer-switch distance or recognition percentage is asserted here.

## Questions to examine in your actual recordings

Compare the same gun across distances for changes in attack sharpness, spectral balance, low-level mechanical detail, tail/echo, and masking. Listen again after matching playback loudness temporarily; a persistent difference then suggests more than simple gain reduction. Preserve the unmodified originals for training.

Repeat with and without an obstruction, and in an open area versus a reflective/enclosed location. A muffled sound can have multiple explanations, so do not label distance from how muffled or quiet it seems. Use the known recording setup as distance ground truth.

Rhythm and burst pattern are useful candidate identity features, but they are not unique identifiers. A missed/obscured shot or similar firing cadence can make them ambiguous. Distinguish detecting some gunfire from identifying the exact gun, and support an unknown-weapon outcome if weapon identification is added later.

## A manageable first experiment

Start with three weapons whose sounds you believe are contrasting. Use one fixed, unsuppressed configuration for each. Have a cooperating shooter and a stationary, separate listener; record **on the listener's machine**. Keep the listener orientation and audio settings fixed during the first pass.

Use these proposed sampling positions where practical:

| Listener-to-shooter distance | Purpose |
|---|---|
| About 10 m | Close external/listener example |
| About 25 m | Additional nearby example |
| About 50 m | Intermediate example |
| About 100 m | Distant example |
| About 200 m | Longer-range example, only if audible |

These are collection bins, not known game cutoffs, effective weapon ranges, or guaranteed audibility limits. Record approximate distances honestly if the game cannot establish precise measurement. An inaudible shot is not a positive audible-gunshot training example; retain it in the experiment log as an audibility observation.

Record ten separated shots per weapon/position, ideally over at least two takes. Leave several seconds of background between shots so each report and tail can be reviewed. This pilot is **3 weapons × 5 distances × 10 shots = 150 shots**, before optional bursts. It tests whether broader collection is worthwhile; it is not enough by itself to claim robust recognition.

Once that first pass works:

1. Add automatic/burst sequences, where supported, and log firing mode.
2. Repeat selected distances with the listener facing sideways/away while positions stay fixed.
3. Add indoor, outdoor and partially blocked cases. Change one factor at a time before collecting combined variations.
4. Treat suppressed configurations and sound-affecting upgrades as separate conditions. Record exact configuration rather than assuming equivalent gain changes.
5. Keep bullet fly-bys, nearby impacts and ricochets separately noted. In an in-game test, aim away from the listener initially so these do not become accidental identifying features of a weapon.
6. Collect realistic mixes afterward: movement, weather, machinery/ARC sounds, overlapping players and background shots. Clean positives help inspect the task, while ordinary gameplay is needed to measure behavior under actual use conditions.

Keep own-weapon playback examples separate from listener-side examples. Do not rely on a recording made by the shooter alone to represent the remote shots the companion will hear.

## Suppressors without an exhaustive attachment matrix

Include suppressed gunfire as a condition to collect and evaluate. Do not assume a suppressed example is equivalent to turning down an unsuppressed recording. The exact changes to per-shot layers, filtering and audible range have not been verified here; compare actual listener-side recordings instead of inventing those transformations.

Start with this small paired experiment, using compatible weapons and a silencer you have available:

| Factor | First-pass choice |
|---|---|
| Weapons | Two with contrasting firing patterns |
| Listener distances | Two of the existing positions, such as about 10 m and 50 m |
| Muzzle condition | Unsuppressed versus one exact silencer type/tier |
| Repetitions | Ten separated shots per setup; add short bursts where supported |

This is **2 weapons x 2 distances x 2 conditions x 10 shots = 80 shots**, excluding bursts. Reuse matching unsuppressed takes from the distance pilot; then only 40 suppressed shots are additional. These counts are a manageable pilot, not a sufficient-data claim. Retain inaudible attempts in the collection log, but do not label them as audible gunshot positives.

Record on the separate listener's machine. Hold weapon upgrade, magazine, Kinetic Converter status, location, positions, orientation and audio settings constant within each pair. Keep the converter off for the first comparison so two attachment effects do not change together. Record the full silencer name/tier in `attachments`; label audible examples `gunshot` in the existing labeler. A separate silenced-gunshot class is not required for the current generic gunfire task.

Next, spot-check other available silencer tiers/types on the same setup, including shotgun-specific variants if collecting compatible shotguns. Do not assume tiers have identical sound profiles or differ only in volume. Expand to more weapons, ranges and combined attachments when independent evaluation reveals misses or when a configuration has no coverage. Before claiming coverage of every gun, collect independent examples of those guns; a two-weapon pilot cannot establish universal generalization.

Reserve separate takes/sessions for evaluation and compare suppressed versus unsuppressed misses at a threshold chosen on validation data. Include quiet unsuppressed shots and background confusers too, so the model cannot succeed merely by associating loudness with the label. The current evaluator does not automatically break down results by attachment; keep those conditions in separate evaluation collections or add metadata-aware reporting later. Keep all paired takes from the same recording session in the same split.

Prioritize clean suppressed/unsuppressed coverage and genuine listener distances before recording every silencer x converter x magazine combination. Collect representative combined configurations afterward and use held-out results to decide where more data is needed. No such comparisons or gameplay accuracy tests have been performed as part of this guide update.

## Recording consistency and notes

Preserve stereo, original rate and channel order. Prefer the isolated game-audio track, keep recording gain/settings fixed, avoid clipping, and retain the original files before editing. Log in-game audio preset/output mode, any sound-altering Windows/headset processing, volume settings, and game version. Avoid applying loudness normalization or artificial effects to the only copy.

Keep a simple sidecar recording log. Suggested columns:

```text
file,take_id,session_id,game_version,weapon,weapon_upgrade,attachments,firing_mode,listener_distance_m,distance_method,listener_orientation,shooter_orientation,environment,occlusion,audio_track,audio_preset,source,notes
```

Mark unavailable information `unknown`. One record per constant setup is fine; if a five-minute file changes weapon, distance or room, include the relevant start/end timestamps in the notes or make separate takes. Describe `self`/`teammate`/`other_raider` accurately: recording a teammate is legitimate sound-class data but does not establish an enemy-only classifier.

This recording log is collection metadata. The labeler now has an optional human weapon field for `gunshot`, but no distance controls. The training code does not automatically use this recording-log CSV; it differs from the public-source seed CSV supported by `build-weapon-dataset`. Continue marking audible intervals as `gunshot`. Per-distance reports and weapon-name training still need a separate implementation.

## How to test whether one recording distance is enough

1. Reserve entire independent recording sessions/takes for evaluation, including examples of every distance. Do not randomly scatter overlapping windows from one firing sequence across train and test sets.
2. As a deliberately restricted baseline, train using one distance only. Measure event precision/recall separately at each other recorded distance.
3. Train a second version with multiple distances, keeping the independent test recordings and thresholds fixed. Compare misses and false alerts by distance and environment.
4. If weapon identification is added, additionally report confusion by weapon and distance. Correctly detecting gunfire is not evidence of correctly identifying the gun.
5. If two adjacent distances sound unexpectedly different, collect intermediate positions there. Do not infer exact engine layer thresholds from one take.

The app's current evaluator reports event metrics on reviewed intervals, not distance-stratified or weapon-ID metrics. Those extra breakdowns require the metadata integration mentioned above. The present shallow log-mel model may have difficulty generalizing across these conditions; more recordings improve the evidence but do not guarantee that this model architecture is sufficient.

Artificial gain/filter/reverb augmentation can be investigated after genuine multi-distance recordings exist. Keep every augmented version of a take in the same split as its source. Use real distant/blocked recordings to judge whether augmentation helped, rather than treating a synthetic filter as a verified ARC distance simulation.
