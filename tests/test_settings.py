import json
from dataclasses import asdict

import pytest

from arc_audio.settings import Settings, load_settings, save_settings


@pytest.mark.parametrize("key,value", [("sensitivity", 0), ("min_confidence", 1.1), ("opacity", float("nan")),
                                        ("lifetime", float("inf")), ("ui_scale", "large"),
                                        ("backend", "random"), ("monitor", 3), ("opacity", True)])
def test_invalid_setting(key, value):
    with pytest.raises(ValueError):
        Settings(**{key: value})


def test_settings_roundtrip_and_safe_mode(tmp_path):
    path = tmp_path / "settings.json"
    expected = Settings(opacity=.5, monitor="negative-coordinate display")
    save_settings(expected, path)
    actual, warning = load_settings(path)
    assert actual == expected and not warning
    assert "mode" not in asdict(actual)


def test_corruption_reports_fallback(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"opacity": 900}')
    settings, warning = load_settings(path)
    assert settings == Settings()
    assert "defaults restored" in warning
    path.write_text("[]")
    assert load_settings(path)[1]
    path.write_text(json.dumps({"future_field": "ignored", "opacity": .8}))
    assert load_settings(path)[0].opacity == .8
