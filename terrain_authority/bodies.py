"""bodies.py — per-planet constants for the dustgym environments.

A ``Body`` is (surface gravity, regolith terramechanics) for a planetary body. Gravity is the
rigorously known, body-specific quantity (textbook values, [FIXED]); it drives wheel load
(weight = m*g) and, through the Lyasko-2010 reduced-gravity law, the Bekker frictional modulus and
cohesion. So a body's regolith parameters are the repo's Earth-era Bekker baseline
(:meth:`TerramechanicsParams.from_constants`) Lyasko-corrected to that body's gravity:

    params_for_body("moon")  == TerramechanicsParams.lunar()    # baseline reduced to 1/6 g
    params_for_body("earth") == TerramechanicsParams.from_constants()  # Lyasko identity at 1 g
    params_for_body("mars")  == baseline reduced to Mars g (3.72)

HONESTY (no fabricated constants): gravity is exact for every body. The Bekker MODULI are a
lunar/Earth-analog regolith scaled by the documented gravity law -- they are NOT a body-specific
in-situ soil fit. Body-specific soil composition/cohesion is flagged ``[UNKNOWN]`` in ``provenance``
until a sourced dataset is dropped in (Moon and Earth baselines are sourced in constants.py; Mars
and the icy/dwarf bodies carry gravity-only fidelity).
"""
from __future__ import annotations

from dataclasses import dataclass

from . import constants as K
from .terramechanics import TerramechanicsParams, lyasko_reduce

G_EARTH = 9.81


@dataclass(frozen=True)
class Body:
    name: str          # canonical key (lowercase)
    label: str         # display name
    g: float           # surface gravity [m/s^2], textbook/[FIXED]
    provenance: str    # gravity source + regolith fidelity caveat


BODIES = {
    "moon": Body("moon", "Moon", 1.62,
                 "g: Apollo/LRO [FIXED]. Regolith: repo Bekker baseline (constants.py) Lyasko-reduced "
                 "to 1/6 g == TerramechanicsParams.lunar() (sourced)."),
    "mars": Body("mars", "Mars", 3.721,
                 "g: 3.721 m/s^2 [FIXED]. Regolith: Earth-era Bekker baseline Lyasko-corrected to Mars g; "
                 "Mars-specific in-situ Bekker moduli [UNKNOWN] (gravity-only fidelity)."),
    "earth": Body("earth", "Earth", 9.81,
                  "g: 9.81 m/s^2 [FIXED]. Regolith: repo Earth/Apollo-era Bekker baseline (Lyasko identity); "
                  "validation/sanity body."),
    "ceres": Body("ceres", "Ceres", 0.27,
                  "g: Dawn-derived ~0.27 m/s^2 [FIXED]. Regolith [UNKNOWN] (gravity-only fidelity)."),
    "europa": Body("europa", "Europa", 1.314,
                   "g: 1.314 m/s^2 [FIXED]. Icy regolith [UNKNOWN] (gravity-only fidelity)."),
    "enceladus": Body("enceladus", "Enceladus", 0.113,
                      "g: 0.113 m/s^2 [FIXED]. Icy regolith [UNKNOWN] (gravity-only fidelity)."),
}

DEFAULT_BODY = "moon"


def get_body(name) -> Body:
    """Resolve a body by name (case-insensitive) or pass a Body through unchanged."""
    if isinstance(name, Body):
        return name
    key = str(name).strip().lower()
    if key not in BODIES:
        raise KeyError(f"unknown body {name!r}; known: {sorted(BODIES)}")
    return BODIES[key]


def params_for_body(name) -> TerramechanicsParams:
    """Terramechanics params for a body: the Earth-era Bekker baseline Lyasko-corrected to its gravity.

    At g = g_earth this is the identity (the baseline); at lower g it reduces k_phi and cohesion per
    Lyasko-2010 (see terramechanics.lyasko_reduce). params_for_body("moon") reproduces .lunar()."""
    body = get_body(name)
    base = TerramechanicsParams.from_constants()
    return lyasko_reduce(base, g=body.g, g_earth=G_EARTH)
