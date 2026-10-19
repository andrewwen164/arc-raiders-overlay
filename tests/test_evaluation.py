import json

import pytest

from arc_audio.audio import write_demo
from arc_audio.evaluation import evaluate, match_events, read_annotations


def detection(at, label="transient", direction="left"):
    return {"audio_time": at, "label": label, "direction": direction}


def test_duplicate_alert_is_false_positive_and_wrong_class_does_not_match():
    truth = [{"start": 1, "end": 1.2, "label": "transient", "direction": "left"}]
    result = match_events([detection(1.1), detection(1.2), detection(1.1, "footsteps")], truth)
    assert result["true_positives"] == 1
    assert result["false_positives"] == 2
    assert result["recall"] == 1
    assert result["precision"] == pytest.approx(1 / 3)


def test_matching_finds_maximum_in_overlapping_intervals():
    truth = [{"start": 1, "end": 2, "label": "transient"},
             {"start": 1, "end": 1.3, "label": "transient"}]
    # First point can match both; second can match only the first interval.
    assert match_events([detection(1.1), detection(1.9)], truth, 0)["true_positives"] == 2


def test_direction_and_tolerance():
    truth = [{"start": 1, "end": 2, "label": "transient", "direction": "right"}]
    assert match_events([detection(.9, direction="right")], truth, .15)["true_positives"] == 1
    assert match_events([detection(1.5)], truth)["false_negatives"] == 1
    assert match_events([], [])["precision"] is None
    with pytest.raises(ValueError):
        match_events([], [], -1)


def test_annotations_reject_reversed_or_nonfinite_intervals(tmp_path):
    path = tmp_path / "a.json"
    for start, end in [(2, 1), (0, float("inf")), (-1, 2)]:
        path.write_text(json.dumps({"version": 1, "intervals": [{"start": start, "end": end, "label": "transient"}]}))
        with pytest.raises(ValueError):
            read_annotations(path)
    for document in ([], {"version": 1, "intervals": ["invalid"]}):
        path.write_text(json.dumps(document))
        with pytest.raises(ValueError):
            read_annotations(path)


def test_end_to_end_offline_demo(tmp_path):
    wav = tmp_path / "fixture.wav"
    output = tmp_path / "detections.jsonl"
    write_demo(wav, 9)
    report = evaluate(wav, output=output)
    assert 3 <= report["detections"] < 10
    assert report["processing_ms_per_chunk"]["count"] == 180
    events = [json.loads(row) for row in output.read_text().splitlines()]
    assert {e["direction"] for e in events} == {"left", "right", "unknown"}
    assert all(e["processing_ms"] >= 0 and e["timestamp"] for e in events)
