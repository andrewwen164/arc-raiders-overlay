"""Optional AudioSet AST backend, cached-only inference. No implicit downloads."""

import json
import os
from collections import deque
from pathlib import Path

import numpy as np

from .audio import resample
from .detection import Debounce
from .events import Detection, utc_now
from .settings import data_dir

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
# Immutable upstream revision containing safetensors; verified model files/labels.
MODEL_REVISION = "f826b80d28226b62986cc218e5cec390b1096902"
WINDOW_SECONDS = 10.24  # 1024 log-mel frames at 10 ms; extractor pads/truncates
TARGET_LABELS = {"Walk, footsteps": "footsteps", "Gunshot, gunfire": "gunshot"}


def model_dir():
    return data_dir() / "models" / "ast-audioset"


def download_model():
    """Only this explicitly invoked command can contact the model host."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from huggingface_hub import snapshot_download

    # An immutable upstream commit fixes all four files, including the label map.
    location = snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=str(model_dir()),
                                 allow_patterns=["config.json", "preprocessor_config.json",
                                                 "model.safetensors", "README.md"])
    config = json.loads((model_dir() / "config.json").read_text())
    if not set(TARGET_LABELS).issubset(set(config["id2label"].values())):
        raise RuntimeError("Downloaded model does not expose the expected sound-event classes")
    return location


class ASTDetector:
    def __init__(self, sensitivity=1.0, min_confidence=0.6, path=None):
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        location = Path(path) if path is not None else model_dir()
        if not (location / "model.safetensors").exists():
            raise RuntimeError("AST weights are absent. Install .[classifier], then run: "
                               "python -m arc_audio download-model --accept-download")
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        self.torch = torch
        torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2) - 1)))
        self.extractor = ASTFeatureExtractor.from_pretrained(str(location), local_files_only=True)
        self.model = ASTForAudioClassification.from_pretrained(str(location), local_files_only=True,
                                                              use_safetensors=True).eval()
        actual = self.model.config.id2label
        self.labels = {int(i): TARGET_LABELS[label] for i, label in actual.items() if label in TARGET_LABELS}
        if len(self.labels) != 2 or self.extractor.sampling_rate != 16000 or self.extractor.max_length != 1024:
            raise RuntimeError("Unexpected AST labels or feature configuration; delete cache and redownload")
        self.min_confidence = min_confidence
        self.parts = deque()
        self.frames = 0
        self.previous_end = None
        self.last_inference = -1e9
        self.debounce = Debounce(cooldown=2.0)

    def score_window(self, signal):
        """Offline mono 16 kHz scores. Short inputs are feature-padded, not stretched.

        Short-window annotation suggestions are experimental; streaming still waits
        for the full 10.24 seconds. Scores refer to a window, never an exact onset.
        """
        signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim != 1 or not 0 < len(signal) <= round(WINDOW_SECONDS * 16000):
            raise ValueError("AST expects a nonempty mono window of at most 10.24 seconds at 16 kHz")
        if not np.isfinite(signal).all():
            raise ValueError("AST audio must contain finite samples")
        features = self.extractor(signal, sampling_rate=16000, return_tensors="pt")
        with self.torch.inference_mode():
            scores = self.model(**features).logits.sigmoid()[0].cpu().numpy()
        return {label: float(scores[index]) for index, label in self.labels.items()}

    def process(self, chunk):
        duration = len(chunk.samples) / chunk.sample_rate
        if self.previous_end is not None and abs(chunk.position - duration - self.previous_end) > 0.02:
            # Don't concatenate noncontiguous audio after queue overflow.
            self.parts.clear()
            self.frames = 0
        self.previous_end = chunk.position
        mono = chunk.samples.mean(axis=1)
        x = resample(mono, chunk.sample_rate, 16000)
        self.parts.append(x)
        self.frames += len(x)
        required = round(WINDOW_SECONDS * 16000)
        while self.parts and self.frames - len(self.parts[0]) >= required:
            self.frames -= len(self.parts.popleft())
        if self.frames < required or chunk.position - self.last_inference < 1.0:
            return []
        self.last_inference = chunk.position
        signal = np.concatenate(self.parts)[-required:]
        scores = self.score_window(signal)
        result = []
        for label, confidence in scores.items():
            if confidence >= self.min_confidence and self.debounce.accept(label, chunk.position):
                # Clip classification cannot attribute a channel balance to a particular
                # class among overlapping sounds. Always show unknown direction.
                result.append(Detection(utc_now(), chunk.position, label, confidence, "unknown", "ast",
                                        window_start=chunk.position - WINDOW_SECONDS,
                                        window_end=chunk.position, present_at=chunk.available_at))
        return result


def make_detector(backend, sensitivity=1.0, min_confidence=0.6):
    if backend == "local":
        from .local_model import LocalDetector
        return LocalDetector(sensitivity, min_confidence)
    if backend == "ast":
        return ASTDetector(sensitivity, min_confidence)
    if backend == "baseline":
        from .detection import BaselineDetector
        return BaselineDetector(sensitivity, min_confidence)
    raise ValueError(f"Unknown backend: {backend}")
