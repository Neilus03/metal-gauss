"""CPU-only tests for the trainer's step-dependent schedule resolution."""

from __future__ import annotations

import pytest

from metal_gauss.schedule import resolve_training_schedule


def _resolve(**overrides):
    values = {
        "steps": 30_000,
        "steps_scaler": 0.1,
        "budget": None,
        "start_active": 150_000,
        "relocate_every": 100,
        "eval_every": 1_000,
        "sh_warmup": 1_000,
        "resolution_schedule": None,
        "filter_3d_every": 0,
        "export_every": 0,
    }
    values.update(overrides)
    return resolve_training_schedule(**values)


def test_steps_scaler_resolves_defaults_from_scaled_steps():
    got = _resolve()

    assert got == {
        "steps": 3_000,
        "budget": 100_000,
        "start_active": 50_000,
        "relocate_every": 10,
        "eval_every": 100,
        "sh_warmup": 100,
        "resolution_schedule": 1_000,
        "filter_3d_every": 0,
        "export_every": 0,
    }


def test_steps_scaler_scales_explicit_step_intervals_but_not_budget():
    got = _resolve(budget=500_000, resolution_schedule=2_000,
                   filter_3d_every=250, export_every=100)

    assert got["steps"] == 3_000
    assert got["budget"] == 500_000
    assert got["resolution_schedule"] == 200
    assert got["filter_3d_every"] == 25
    assert got["export_every"] == 10


def test_zero_intervals_remain_disabled():
    got = _resolve(sh_warmup=0, filter_3d_every=0, export_every=0)

    assert got["sh_warmup"] == 0
    assert got["filter_3d_every"] == 0
    assert got["export_every"] == 0


def test_unit_scaler_keeps_original_defaults_and_explicit_values():
    got = _resolve(steps_scaler=1.0, resolution_schedule=777,
                   budget=123_000, start_active=100_000, relocate_every=37,
                   eval_every=211,
                   sh_warmup=59, filter_3d_every=31, export_every=43)

    assert got == {
        "steps": 30_000,
        "budget": 123_000,
        "start_active": 100_000,
        "relocate_every": 37,
        "eval_every": 211,
        "sh_warmup": 59,
        "resolution_schedule": 777,
        "filter_3d_every": 31,
        "export_every": 43,
    }


def test_steps_scaler_must_be_positive_and_finite():
    for value in (0.0, -0.5, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="greater than zero"):
            _resolve(steps_scaler=value)
