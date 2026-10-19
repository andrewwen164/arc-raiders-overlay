"""Discovery returns unverified metadata only; no network or downloads here."""

import json
import sys
import types

import pytest

from arc_audio import weapon_discovery
from arc_audio.weapon_discovery import discover_weapon_sources


def fake_search(monkeypatch, responses):
    calls = []

    class Downloader:
        def __init__(self, options):
            assert options["extract_flat"] == "in_playlist"
            assert options["skip_download"] is True
            assert not options.get("remote_components")
            assert not options.get("cookiefile")
            assert options["js_runtimes"] == {"node": {"path": "C:/tools/node.exe"}}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def extract_info(self, query, download):
            assert download is False
            calls.append(query)
            response = responses[len(calls) - 1] if len(calls) <= len(responses) else {"entries": []}
            if isinstance(response, Exception):
                raise response
            return response

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=Downloader))
    monkeypatch.setattr(weapon_discovery.shutil, "which", {"node": "C:/tools/node.exe"}.get)
    return calls


def video(identity, **fields):
    return {"id": identity, "title": "Synthetic metadata", **fields}


def test_search_is_bounded_metadata_only_and_json_safe(monkeypatch):
    calls = fake_search(monkeypatch, [{"entries": [video("aaaaaaaaaaa", channel="Source channel",
                                                       channel_id="channel-id", uploader="Uploader", duration=62),
                                                 video("bbbbbbbbbbb"), video("ccccccccccc")]}])
    result = discover_weapon_sources("Il Toro", 2)
    assert result["version"] == 1 and result["verification"] == "unverified"
    assert result["weapon"] == "il_toro"
    assert len(result["sources"]) == 2 and len(calls) == 1
    assert calls[0] == "ytsearch2:ARC Raiders il toro weapon showcase"
    row = result["sources"][0]
    assert row["weapon"] == "il_toro" and row["verification"] == "unverified"
    assert row["channel"] == "Source channel" and row["uploader"] == "Uploader"
    assert row["source_url"] == "https://www.youtube.com/watch?v=aaaaaaaaaaa"
    assert row["duration"] == 62
    assert result["errors"] == []
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_duplicate_ids_fill_from_later_query(monkeypatch):
    calls = fake_search(monkeypatch, [
        {"entries": [video("aaaaaaaaaaa"), video("aaaaaaaaaaa")]},
        {"entries": [video("aaaaaaaaaaa"), video("bbbbbbbbbbb")]}])
    result = discover_weapon_sources("Ferro", 2)
    assert [row["id"] for row in result["sources"]] == ["aaaaaaaaaaa", "bbbbbbbbbbb"]
    assert len(calls) == 2 and calls[1] == "ytsearch1:ARC Raiders ferro gameplay"


def test_query_failures_are_isolated_and_retained(monkeypatch):
    calls = fake_search(monkeypatch, [RuntimeError("synthetic unavailable search"),
                                     {"entries": [video("aaaaaaaaaaa")]}])
    result = discover_weapon_sources("Anvil", 1)
    assert len(calls) == 2 and len(result["sources"]) == 1
    assert result["errors"] == [{"query": "ARC Raiders anvil weapon showcase",
                                  "error": "synthetic unavailable search"}]


def test_malformed_entries_do_not_become_sources(monkeypatch):
    calls = fake_search(monkeypatch, [{"entries": [None, "string", {}, {"id": "bad"},
                                                 {"url": "https://example.org/not-a-youtube-video"},
                                                 {"url": "https://youtu.be/aaaaaaaaaaa", "duration": float("nan")},
                                                 video("bbbbbbbbbbb", duration=True, channel={"bad": 1}),
                                                 video("ccccccccccc", duration=10 ** 500)]}])
    result = discover_weapon_sources("Ferro", 10)
    assert len(calls) == 5
    assert [row["id"] for row in result["sources"]] == ["aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc"]
    assert all(row["duration"] is None for row in result["sources"])
    assert result["sources"][1]["channel"] is None
    json.dumps(result, allow_nan=False)


def test_invalid_playlist_and_all_failed_queries_are_bounded(monkeypatch):
    calls = fake_search(monkeypatch, [None, {"entries": "bad"}, {"entries": None},
                                     RuntimeError("fourth"), RuntimeError("fifth")])
    result = discover_weapon_sources("Stitcher", 5)
    assert len(calls) == 5 and len(result["errors"]) == 5
    assert result["sources"] == []


@pytest.mark.parametrize(("weapon", "maximum"), [("", 1), ("!!!", 1), (None, 1), ("a" * 81, 1),
                                                   ("Ferro\n", 1), ("Ferro", 0), ("Ferro", 101),
                                                   ("Ferro", True), ("Ferro", 1.5), ("Ferro", "2")])
def test_invalid_parameters_fail_before_network(monkeypatch, weapon, maximum):
    calls = fake_search(monkeypatch, [])
    with pytest.raises(ValueError):
        discover_weapon_sources(weapon, maximum)
    assert calls == []


def test_missing_collection_dependency_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "yt_dlp", None)
    result = discover_weapon_sources("Anvil", 2)
    assert result["sources"] == [] and "Missing yt-dlp" in result["errors"][0]["error"]
    assert result["queries"] == []
