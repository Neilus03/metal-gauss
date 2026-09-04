"""Tests for selecting a precompiled Metal library safely.

These tests are deliberately CPU-only. They cover the source/digest contract
without requiring a Metal device or the Xcode Metal compiler, so stale-library
handling is checked in ordinary CI as well as on an Apple Silicon machine.
"""
from __future__ import annotations

import hashlib

import metal_gauss.metal_backend as backend


def _write_sources(root, contents):
    csrc = root / "csrc"
    csrc.mkdir()
    for name, text in zip(backend._METAL_SOURCES, contents):
        (csrc / name).write_text(text)


def test_matching_source_digest_selects_library(tmp_path, monkeypatch):
    contents = [f"kernel_{i}\n" for i in range(len(backend._METAL_SOURCES))]
    _write_sources(tmp_path, contents)
    lib = tmp_path / "csrc" / "metal_gauss.metallib"
    lib.write_bytes(b"not a library for this CPU-only test")

    monkeypatch.setattr(backend, "_HERE", tmp_path)
    source = backend._metal_source()
    digest = hashlib.sha256(source.encode()).hexdigest()
    lib.with_suffix(lib.suffix + ".sha256").write_text(digest + "\n")

    assert backend._precompiled_library(source) == lib


def test_changed_source_rejects_stale_library(tmp_path, monkeypatch):
    contents = [f"kernel_{i}\n" for i in range(len(backend._METAL_SOURCES))]
    _write_sources(tmp_path, contents)
    lib = tmp_path / "csrc" / "metal_gauss.metallib"
    lib.write_bytes(b"not a library for this CPU-only test")

    monkeypatch.setattr(backend, "_HERE", tmp_path)
    source = backend._metal_source()
    lib.with_suffix(lib.suffix + ".sha256").write_text(
        hashlib.sha256((source + "changed").encode()).hexdigest() + "\n"
    )

    assert backend._precompiled_library(source) is None


def test_force_source_bypasses_matching_library(tmp_path, monkeypatch):
    contents = [f"kernel_{i}\n" for i in range(len(backend._METAL_SOURCES))]
    _write_sources(tmp_path, contents)
    lib = tmp_path / "csrc" / "metal_gauss.metallib"
    lib.write_bytes(b"not a library for this CPU-only test")

    monkeypatch.setattr(backend, "_HERE", tmp_path)
    source = backend._metal_source()
    lib.with_suffix(lib.suffix + ".sha256").write_text(
        hashlib.sha256(source.encode()).hexdigest() + "\n"
    )
    monkeypatch.setenv("METAL_GAUSS_FORCE_SOURCE", "1")

    assert backend._precompiled_library(source) is None
