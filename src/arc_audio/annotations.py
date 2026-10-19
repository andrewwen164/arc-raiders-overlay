"""Human interval corrections and conservative class-scoped review invalidation."""

from .dataset import CORE_CLASSES, TARGET_CLASSES, merge_reviewed, validate_span

SOURCE_TAGS = ("unknown", "self", "teammate", "other_raider")


def invalidate_review(clip, interval):
    """A correction must not silently turn an unchecked absence into a negative."""
    result = []
    for review in clip.get("reviewed", []):
        classes = list(review.get("classes", CORE_CLASSES))
        first = max(review["start"], interval["start"])
        last = min(review["end"], interval["end"])
        if first >= last or interval["label"] not in classes:
            result.append(review)
            continue
        if review["start"] < first:
            result.append({"start": review["start"], "end": first, "classes": classes})
        remaining = [label for label in classes if label != interval["label"]]
        if remaining:
            result.append({"start": first, "end": last, "classes": remaining})
        if last < review["end"]:
            result.append({"start": last, "end": review["end"], "classes": classes})
    clip["reviewed"] = merge_reviewed(result)


def write_interval(clip, start, end, label, source="unknown", *, index=None, proposal_id=None):
    if not clip.get("track_confirmed"):
        raise ValueError("Listen to and confirm the game-audio track first")
    interval = {"start": start, "end": end, "label": label, "source": source}
    validate_span(interval, clip["duration"])
    if label not in TARGET_CLASSES or source not in SOURCE_TAGS:
        raise ValueError("Choose a supported sound class and source tag")
    previous = clip["intervals"][index] if index is not None else None
    if previous:
        interval = dict(previous, **interval)  # Keep proposal linkage/other provenance.
    if proposal_id:
        interval["proposal_id"] = proposal_id
    same_span_and_class = previous and all(previous[key] == interval[key] for key in ("start", "end", "label"))
    if not same_span_and_class:
        if previous:
            invalidate_review(clip, previous)
        invalidate_review(clip, interval)
    if index is None:
        clip["intervals"].append(interval)
    else:
        clip["intervals"][index] = interval
    clip["intervals"].sort(key=lambda row: row["start"])
    return interval


def remove_interval(clip, interval):
    invalidate_review(clip, interval)
    clip["intervals"].remove(interval)
