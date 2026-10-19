"""Human answers to exact audio windows, separate from event timestamps.

Yes means at least one audible gunshot somewhere in that second. No means none
in the entire second. Neither answer supplies an onset or labels nearby audio.
"""

import hashlib
import json
import math
from pathlib import Path

from .dataset import reviewed_spans, validate_span, validate_weapon

RATE = 16000
WINDOW_FRAMES = RATE


def review_window(clip: dict, row: dict) -> dict:
    """Choose a reproducible, sample-aligned one-second question."""
    if row.get("label") != "gunshot":
        raise ValueError("Quick review currently asks about gunfire only")
    duration = clip["duration"]
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 1:
        raise ValueError("Quick review needs at least one second of source audio")
    onset = row.get("source_timestamp", (row["start"] + row["end"]) / 2)
    if isinstance(onset, bool) or not isinstance(onset, (int, float)) or not math.isfinite(onset):
        raise ValueError("Invalid candidate source timestamp")
    before = .5 if row.get("proposal_kind") == "gunshot_absence" else .35
    first = max(0, min(round((onset - before) * RATE), math.floor(duration * RATE) - WINDOW_FRAMES))
    token = [clip["file"], clip["audio_stream"], first, first + WINDOW_FRAMES, "gunshot_presence_v1"]
    identity = "presence-" + hashlib.sha256(json.dumps(token).encode()).hexdigest()[:24]
    return {"id": identity, "start": first / RATE, "end": (first + WINDOW_FRAMES) / RATE}


def validate_presence_reviews(clip: dict) -> None:
    rows = clip.get("presence_reviews", [])
    if not isinstance(rows, list):
        raise ValueError("presence_reviews must be an array of human window answers")
    seen_ids, seen_spans = set(), set()
    for row in rows:
        validate_span(row, clip["duration"])
        identity = row.get("id")
        first, last = round(row["start"] * RATE), round(row["end"] * RATE)
        if (not isinstance(identity, str) or not identity or identity in seen_ids
                or (first, last) in seen_spans or last - first != WINDOW_FRAMES
                or abs(first - row["start"] * RATE) > 1e-5 or abs(last - row["end"] * RATE) > 1e-5
                or row["end"] > clip["duration"] + 1e-9 or row.get("label") != "gunshot"
                or not isinstance(row.get("present"), bool)
                or isinstance(row.get("audio_stream"), bool) or row.get("audio_stream") != clip["audio_stream"]):
            raise ValueError("Presence reviews need unique, exact sample-aligned one-second gunshot answers on the selected track")
        proposals = row.get("proposal_ids")
        if (not isinstance(proposals, list) or not proposals
                or any(not isinstance(value, str) or not value for value in proposals)
                or len(set(proposals)) != len(proposals)):
            raise ValueError("Presence reviews require their source question IDs")
        validate_weapon(row.get("weapon"))
        if not row["present"] and row.get("weapon") is not None:
            raise ValueError("An absence answer cannot identify a weapon")
        seen_ids.add(identity)
        seen_spans.add((first, last))
    states = clip.get("presence_states", [])
    if not isinstance(states, list):
        raise ValueError("presence_states must be an array of unanswered/unsure windows")
    for state in states:
        validate_span(state, clip["duration"])
        identity = state.get("id")
        first, last = round(state["start"] * RATE), round(state["end"] * RATE)
        if (not isinstance(identity, str) or not identity or identity in seen_ids
                or (first, last) in seen_spans or last - first != WINDOW_FRAMES
                or abs(first - state["start"] * RATE) > 1e-5 or abs(last - state["end"] * RATE) > 1e-5
                or state["end"] > clip["duration"] + 1e-9
                or state.get("answer") not in ("unsure", "unanswered")
                or isinstance(state.get("audio_stream"), bool) or state.get("audio_stream") != clip["audio_stream"]):
            raise ValueError("Presence states need unique exact one-second unanswered/unsure windows on the selected track")
        seen_ids.add(identity)
        seen_spans.add((first, last))
    if "presence_media" in clip:
        stamp = clip["presence_media"]
        if (not isinstance(stamp, dict) or any(isinstance(stamp.get(key), bool)
                or not isinstance(stamp.get(key), int) or stamp[key] < 0 for key in ("size", "mtime_ns"))):
            raise ValueError("Quick-review media fingerprint needs size and mtime_ns integers")


def presence_answer(clip: dict, row: dict) -> str | None:
    window = review_window(clip, row)
    found = next((review for review in clip.get("presence_reviews", []) if review["id"] == window["id"]), None)
    if found is not None:
        return "yes" if found["present"] else "no"
    state = next((state for state in clip.get("presence_states", []) if state["id"] == window["id"]), None)
    if state is not None:
        return "unsure" if state["answer"] == "unsure" else None
    # The dataset is authoritative if the informational sidecar write failed.
    status = row.get("quick_review", {})
    return "unsure" if status.get("answer") == "unsure" and status.get("window_id") == window["id"] else None


def answer_presence(clip: dict, row: dict, answer: str, *, weapon: str | None = None) -> None:
    if answer not in ("yes", "no", "unsure"):
        raise ValueError("Choose Yes, No or Unsure")
    if not clip.get("track_confirmed"):
        raise ValueError("Listen and answer Yes to the game-audio track question first")
    window = review_window(clip, row)
    if weapon == "":
        weapon = None
    validate_weapon(weapon)
    if answer != "yes" and weapon is not None:
        raise ValueError("Only a Yes gunfire answer can have an optional weapon choice")
    overlaps = any(interval["label"] == "gunshot"
                   and max(window["start"], interval["start"]) < min(window["end"], interval["end"])
                   for interval in clip.get("intervals", []))
    fully_reviewed = any(span["start"] <= window["start"] and window["end"] <= span["end"]
                         for span in reviewed_spans(clip, ("gunshot",)))
    if (answer == "no" and overlaps) or (answer == "yes" and fully_reviewed and not overlaps):
        raise ValueError("This answer conflicts with an existing manual gunshot label/review. Choose Unsure to skip it; previous labels are preserved")
    previous = next((review for review in clip.get("presence_reviews", []) if review["id"] == window["id"]), None)
    remaining = [review for review in clip.get("presence_reviews", []) if review["id"] != window["id"]]
    if answer != "unsure":
        record = dict(window, label="gunshot", present=answer == "yes", audio_stream=clip["audio_stream"],
                      proposal_ids=sorted(set((previous or {}).get("proposal_ids", [])) | {row["id"]}),
                      review_kind="window_presence")
        if weapon is not None:
            record["weapon"] = weapon
        remaining.append(record)
    clip["presence_reviews"] = sorted(remaining, key=lambda review: review["start"])
    clip["presence_states"] = [state for state in clip.get("presence_states", []) if state["id"] != window["id"]]
    if answer == "unsure":
        clip["presence_states"].append(dict(window, answer="unsure", audio_stream=clip["audio_stream"]))
    row["quick_review"] = {"answer": answer, "window_id": window["id"],
                           "start": window["start"], "end": window["end"]}
    if weapon is not None:
        row["quick_review"]["weapon"] = weapon


def undo_presence(clip: dict, row: dict) -> None:
    window = review_window(clip, row)
    identity = window["id"]
    clip["presence_reviews"] = [review for review in clip.get("presence_reviews", []) if review["id"] != identity]
    clip["presence_states"] = [state for state in clip.get("presence_states", []) if state["id"] != identity]
    clip["presence_states"].append(dict(window, answer="unanswered", audio_stream=clip["audio_stream"]))
    row.pop("quick_review", None)


def bind_presence_media(clip: dict, path: Path | str) -> None:
    """Bind answers to the local media identity without hashing large video on UI."""
    info = Path(path).stat()
    stamp = {"size": info.st_size, "mtime_ns": info.st_mtime_ns}
    if clip.get("presence_media") is not None and clip["presence_media"] != stamp:
        raise ValueError("Media changed since these quick answers. Preserve the old source/answers and import the changed recording separately")
    clip["presence_media"] = stamp


def presence_training_rows(clip: dict, classes: tuple[str, ...]) -> list[dict]:
    """Only the exact answered task; other classes remain unknown."""
    validate_presence_reviews(clip)
    if tuple(classes) != ("gunshot",) or not clip.get("track_confirmed"):
        return []
    return list(clip.get("presence_reviews", []))


def set_review_split(document: dict, clip: dict, split: str) -> None:
    """Explicit multiple choice updates an entire non-test source/session group."""
    if split not in ("train", "validation"):
        raise ValueError("Choose Training or Validation; reserved test groups stay held out")
    group = [candidate for candidate in document["clips"] if candidate["group"] == clip["group"]]
    if not group or any(candidate["split"] == "test" for candidate in group):
        raise ValueError("A reserved test group cannot be moved by quick review")
    for candidate in group:
        candidate["split"] = split


def presence_summary(document: dict) -> dict:
    return {split: {"yes": sum(review["present"] for clip in document["clips"]
                               if clip["split"] == split and clip.get("track_confirmed")
                               for review in clip.get("presence_reviews", [])),
                    "no": sum(not review["present"] for clip in document["clips"]
                              if clip["split"] == split and clip.get("track_confirmed")
                              for review in clip.get("presence_reviews", []))}
            for split in ("train", "validation", "test")}
