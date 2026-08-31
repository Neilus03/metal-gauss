"""Stamp a benchmark payload with the code and machine that produced it.

Trainer results carry a `resolved` block written by train.py --report. Kernel
benchmarks have no trainer config to resolve, but they still need to say which
commit and which machine produced them -- bench/results/stages_latest.json was
a bare JSON list with no metadata of any kind, which means its numbers could
not be attributed to a commit even in principle.
"""
from __future__ import annotations

import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _git(*a) -> str | None:
    try:
        return subprocess.run(("git",) + a, cwd=ROOT, capture_output=True,
                              text=True, timeout=5).stdout.strip()
    except Exception:
        return None


def env() -> dict:
    try:
        import torch
        tv = torch.__version__
    except Exception:
        tv = None
    return {"git": _git("rev-parse", "--short", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
            "torch": tv,
            "platform": platform.platform(),
            "machine": platform.machine()}


def stamp(payload, **config) -> dict:
    """Wrap a result in {schema, env, config, data}.

    Accepts a bare list, which is what stages_latest.json was, so callers do
    not have to restructure their result to gain provenance.
    """
    return {"schema": 1, "env": env(), "config": config, "data": payload}
