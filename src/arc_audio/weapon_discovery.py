"""Explicit YouTube search returning unverified source candidates only.

This module never downloads media, writes datasets, creates sound labels or
establishes that a video's title correctly identifies its audible weapon.
"""

from __future__ import annotations

import itertools
import math
import re
import shutil

from .weapon_sources import normalize_weapon, source_key

_QUERY_SUFFIXES = ("weapon showcase", "gameplay", "gun", "recoil", "guide")


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _source(entry: object, weapon: str, query: str) -> dict | None:
    if not isinstance(entry, dict):
        return None
    identity = entry.get("id")
    if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9_-]{11}", identity):
        identity = None
        for field in ("webpage_url", "url"):
            url = entry.get(field)
            if not isinstance(url, str) or not url.startswith(("https://", "http://")):
                continue
            try:
                candidate = source_key(url)
            except ValueError:
                continue
            if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
                identity = candidate
                break
    if identity is None:
        return None
    duration = entry.get("duration")
    try:
        valid_duration = (not isinstance(duration, bool) and isinstance(duration, (float, int))
                          and math.isfinite(duration) and duration >= 0)
    except OverflowError:
        valid_duration = False
    if not valid_duration:
        duration = None
    return {"id": identity, "source_url": "https://www.youtube.com/watch?v=" + identity,
            "source_title": _text(entry.get("title")) or "", "weapon": weapon,
            "channel": _text(entry.get("channel")), "channel_id": _text(entry.get("channel_id")),
            "uploader": _text(entry.get("uploader")), "duration": duration,
            "verification": "unverified", "query": query}


def discover_weapon_sources(weapon: str, max_results: int = 10) -> dict:
    """Search at most five templates, capped at 100 unique unverified videos.

    Uses lightweight playlist metadata only. Duplicate IDs do not consume the
    result allowance; subsequent queries attempt to fill it. Failures are kept
    per query, and other templates may still succeed. Missing yt-dlp is reported
    as an actionable error in the same JSON-safe result.
    """
    if not isinstance(weapon, str) or any(ord(character) < 32 for character in weapon):
        raise ValueError("Weapon must be a name without control characters")
    normalized = normalize_weapon(weapon)
    if isinstance(max_results, bool) or not isinstance(max_results, int) or not 1 <= max_results <= 100:
        raise ValueError("max_results must be an integer between 1 and 100")
    result = {"version": 1, "verification": "unverified", "weapon": normalized,
              "sources": [], "errors": [], "queries": []}
    try:
        import yt_dlp
    except ImportError:
        result["errors"].append({"query": None, "error":
                                 "Missing yt-dlp; install the collection extra: pip install -e '.[collection]'"})
        return result
    options = {"extract_flat": "in_playlist", "skip_download": True, "quiet": True,
               "socket_timeout": 20, "retries": 1, "extractor_retries": 1,
               "fragment_retries": 0, "ignoreerrors": False}
    runtimes = {name: {"path": executable} for name in ("deno", "node")
                if (executable := shutil.which(name))}
    if runtimes:
        options["js_runtimes"] = runtimes
    seen = set()
    for suffix in _QUERY_SUFFIXES:
        remaining = max_results - len(result["sources"])
        if remaining <= 0:
            break
        query = f"ARC Raiders {normalized.replace('_', ' ')} {suffix}"
        result["queries"].append(query)
        try:
            with yt_dlp.YoutubeDL(options) as downloader:
                document = downloader.extract_info(f"ytsearch{remaining}:{query}", download=False)
                if not isinstance(document, dict):
                    raise ValueError("Search returned no playlist metadata")
                entries = document.get("entries")
                if entries is None or isinstance(entries, (dict, str, bytes)):
                    raise ValueError("Search returned invalid video entries")
                # ytsearchN is already capped, but also bound iteration for malformed
                # extractor results and avoid trusting arbitrary generators to end.
                for entry in itertools.islice(entries, remaining * 3):
                    row = _source(entry, normalized, query)
                    if row is None or row["id"] in seen:
                        continue
                    seen.add(row["id"])
                    result["sources"].append(row)
                    if len(result["sources"]) >= max_results:
                        break
        except Exception as exc:
            result["errors"].append({"query": query, "error": str(exc)})
    return result
