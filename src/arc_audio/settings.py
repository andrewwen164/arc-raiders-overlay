import json
import math
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path


def data_dir() -> Path:
    root = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".local" / "share")))
    return root / "ArcAudioCompanion"


@dataclass(frozen=True)
class Settings:
    sensitivity: float = 1.0
    min_confidence: float = 0.6
    opacity: float = 0.85
    lifetime: float = 1.4
    ui_scale: float = 1.0
    monitor: str = ""
    device_name: str = ""
    backend: str = "baseline"
    target_only: bool = True
    # Input mode deliberately is not persisted: every launch starts in demo.

    def __post_init__(self):
        limits = {"sensitivity": (0.25, 3.0), "min_confidence": (0, 1),
                  "opacity": (0.1, 1), "lifetime": (0.3, 5), "ui_scale": (0.6, 2.0)}
        for name, (low, high) in limits.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be between {low} and {high}")
        if self.backend not in ("baseline", "ast", "local"):
            raise ValueError("Unknown backend")
        if not isinstance(self.target_only, bool):
            raise ValueError("target_only must be a boolean")
        if not isinstance(self.monitor, str) or not isinstance(self.device_name, str):
            raise ValueError("Monitor and device names must be strings")


def load_settings(path: Path | None = None) -> tuple[Settings, str]:
    path = path or data_dir() / "settings.json"
    if not path.exists():
        return Settings(), ""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Expected a JSON object")
        known = {f.name for f in fields(Settings)}
        return Settings(**{k: v for k, v in raw.items() if k in known}), ""
    except (OSError, ValueError, TypeError) as exc:
        return Settings(), f"Settings could not be loaded; defaults restored: {exc}"


def save_settings(settings: Settings, path: Path | None = None):
    path = path or data_dir() / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
    temporary.replace(path)
