"""Where the competitor binaries and the private capture live.

These were absolute paths under one developer's home directory, hardcoded across
five files. That breaks for every other user and publishes a username, so they
are resolved here: environment variable first, then a documented default, then a
clear error naming the variable.

The room1 scene is a private capture and is deliberately NOT vendored, so its
resolver returns None when unset and callers skip rather than fail.
"""
from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_THIRD_PARTY = Path.home() / "third_party"


def third_party() -> Path:
    return Path(os.environ.get("METAL_GAUSS_THIRD_PARTY", _DEFAULT_THIRD_PARTY))


def _resolve(env: str, *rel: str) -> str:
    return os.environ.get(env) or str(third_party().joinpath(*rel))


def brush_bin() -> str:
    return _resolve("METAL_GAUSS_BRUSH", "brush",
                    "brush-app-aarch64-apple-darwin", "brush_app")


def spirula_bin() -> str:
    return _resolve("METAL_GAUSS_SPIRULA", "spirula-studio", "build", "spirula")


def msplat_bin() -> str:
    return os.environ.get("METAL_GAUSS_MSPLAT", "/tmp/cmp_msplat/bin/msplat-train")


def room1(kind: str):
    """Private real-capture scene. None when unset, so callers can skip."""
    root = os.environ.get("METAL_GAUSS_ROOM1")
    if not root:
        return None
    return {"colmap": f"{root}/02_poses/sparse/1",
            "images": f"{root}/01_frames/images",
            "ply": f"{root}/03_splats/exports/splat_30000.ply"}[kind]
