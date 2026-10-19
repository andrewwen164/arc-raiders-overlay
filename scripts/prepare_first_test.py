"""Inventory the named seven-clip pilot without inventing event annotations."""

import json
import re
from pathlib import Path

import numpy as np

from arc_audio.dataset import load_dataset, prepare_dataset, read_media_audio, save_dataset

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "1st test"


def main():
    manifest = FOLDER / "dataset.json"
    new = not manifest.exists()
    manifest, _ = prepare_dataset(FOLDER)
    original = manifest.read_bytes()
    doc = load_dataset(manifest)
    doc.setdefault("review_classes", ["gunshot"])
    report = {"notice": "Filename conditions and user-reported setup, not verified event labels. "
                        "All pilot clips share one session/train split. No raw audio exports or uploads.",
              "setup": {"perspective": "listener", "shooter": "friend / teammate",
                        "silencer": "Silencer II", "provisional_audio_stream": 1,
                        "source": "user report"},
              "clips": [], "missing_conditions": []}
    observed = set()
    for clip in doc["clips"]:
        match = re.fullmatch(r"(ferro|stitcher) (10|50)m? (silenced|unsilenced)\.mp4", clip["file"], re.I)
        if not match:
            continue
        weapon, distance, state = match.groups()
        suppressed = state.lower() == "silenced"
        condition = {"weapon": weapon.lower(), "reported_distance_m": int(distance),
                     "suppressed": suppressed,
                     "silencer": "Silencer II" if suppressed else "none",
                     "perspective": "listener", "source": "teammate",
                     "condition_source": "filename plus user report; distance not independently measured"}
        observed.add((weapon.lower(), int(distance), suppressed))
        if new:
            clip.update(group="first-pilot-session", split="train", audio_stream=1)
        clip.setdefault("collection", condition)
        tracks, signals = [], []
        for stream in clip["audio_streams"]:
            signal = read_media_audio(manifest.parent / clip["file"], stream["index"], rate=8000)
            rms = float(np.sqrt(np.mean(signal.astype(np.float64) ** 2)))
            peak = float(np.max(np.abs(signal)))
            tracks.append({"audio_stream": stream["index"], "mono_rms_dbfs": 20 * np.log10(max(rms, 1e-12)),
                           "mono_peak_dbfs": 20 * np.log10(max(peak, 1e-12)), "decoded_seconds": len(signal) / 8000})
            signals.append(signal)
        correlation = None
        if len(signals) == 2:
            count = min(map(len, signals))
            a, b = (signal[:count] for signal in signals)
            if np.std(a) > 1e-10 and np.std(b) > 1e-10:
                correlation = float(np.corrcoef(a, b)[0, 1])
        report["clips"].append({"file": clip["file"], "seconds": clip["duration"],
                                "condition": condition, "tracks": tracks,
                                "mono_track_correlation": correlation})
    for weapon in ("ferro", "stitcher"):
        for distance in (10, 50):
            for suppressed in (False, True):
                if (weapon, distance, suppressed) not in observed:
                    report["missing_conditions"].append({"weapon": weapon, "distance_m": distance,
                                                         "suppressed": suppressed})
    if manifest.read_bytes() != original:
        raise RuntimeError("Manifest changed during inventory; close the labeler before rerunning")
    save_dataset(manifest, doc)
    report["total_seconds"] = sum(clip["seconds"] for clip in report["clips"])
    save_dataset(FOLDER / "pilot-report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
