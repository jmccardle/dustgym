"""Tests for the per-planet Body registry (bodies.py). Numpy-only (no gymnasium) -> runs in the core suite.

Pins: gravity is exact per body; params_for_body reproduces the existing lunar() factory; the
Lyasko-corrected Bekker moduli are monotonic in gravity; and wheel load scales with body gravity.
"""
from __future__ import annotations

import dataclasses

import pytest

from terrain_authority import bodies as B
from terrain_authority import terramechanics as tm


def test_moon_matches_lunar_factory():
    # params_for_body("moon") must equal the pre-existing TerramechanicsParams.lunar()
    assert dataclasses.asdict(B.params_for_body("moon")) == dataclasses.asdict(tm.TerramechanicsParams.lunar())


def test_earth_is_baseline_identity():
    # at Earth g the Lyasko correction is the identity -> the un-reduced baseline
    assert dataclasses.asdict(B.params_for_body("earth")) == dataclasses.asdict(
        tm.TerramechanicsParams.from_constants())


def test_known_gravities():
    assert B.BODIES["moon"].g == 1.62
    assert B.BODIES["earth"].g == 9.81
    assert B.BODIES["moon"].g < B.BODIES["mars"].g < B.BODIES["earth"].g


def test_bekker_moduli_monotonic_in_gravity():
    # Lyasko: lower gravity -> lower k_phi and cohesion
    order = ["ceres", "moon", "mars", "earth"]
    kphi = [B.params_for_body(n).k_phi for n in order]
    coh = [B.params_for_body(n).cohesion for n in order]
    assert kphi == sorted(kphi)
    assert coh == sorted(coh)


def test_wheel_load_scales_with_gravity():
    # weight = m*g per wheel -> Mars load > Moon load by the gravity ratio
    moon = tm.static_wheel_load_n(g=B.BODIES["moon"].g)
    mars = tm.static_wheel_load_n(g=B.BODIES["mars"].g)
    assert mars == pytest.approx(moon * (B.BODIES["mars"].g / B.BODIES["moon"].g), rel=1e-9)


def test_get_body_case_insensitive_and_errors():
    assert B.get_body("MARS").name == "mars"
    assert B.get_body(B.BODIES["moon"]).name == "moon"   # pass-through
    with pytest.raises(KeyError):
        B.get_body("pluto")
