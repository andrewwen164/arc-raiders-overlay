import numpy as np
import pytest

from arc_audio import classifier


def test_absent_weights_fail_without_download(tmp_path, monkeypatch):
    monkeypatch.setattr(classifier, "model_dir", lambda: tmp_path / "no-model")
    with pytest.raises(RuntimeError, match="download-model --accept-download"):
        classifier.ASTDetector()


def test_backend_names_and_baseline_does_not_import_optional_model(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Baseline must not create the optional classifier")
    monkeypatch.setattr(classifier, "ASTDetector", forbidden)
    assert classifier.make_detector("baseline").min_confidence == .6
    with pytest.raises(ValueError, match="Unknown backend"):
        classifier.make_detector("guess")


def test_offline_ast_window_rejects_invalid_input_before_optional_inference():
    detector = classifier.ASTDetector.__new__(classifier.ASTDetector)
    for signal in (np.zeros(0), np.zeros((16, 2)), np.zeros(164000), np.array([float("nan")])):
        with pytest.raises(ValueError):
            detector.score_window(signal)
