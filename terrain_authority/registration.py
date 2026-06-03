"""registration.py — register the lunar-vehicle envs as a Gymnasium suite (namespace ``Lunar``).

Makes the envs discoverable through ``gymnasium.make("Lunar/<Env>-v0")``. Importing the package
registers them (the documented Gymnasium third-party pattern)::

    import lunar_sim_gym            # or: import terrain_authority  -- either registers Lunar/*
    import gymnasium as gym
    env = gym.make("Lunar/RoverDrive-v0")

(The ``[project.entry-points."gymnasium.envs"]`` hook in pyproject.toml is a forward-compatible
plugin entry for Gymnasium versions that auto-load entry points; current Gymnasium needs the import
above. ``gymnasium.register_envs(lunar_sim_gym)`` is the explicit, lint-friendly equivalent.)

Each ID is constructible with NO user arguments (the construction/scheduling envs get a default
challenge / layout here), and any constructor arg can still be overridden via ``gym.make(id, **kw)``.
register_envs() is a no-op when gymnasium is absent, so the bare-numpy core stays importable.
"""
from __future__ import annotations

LUNAR_ENV_IDS = [
    "Lunar/RoverDrive-v0",     # closed-loop unicycle drive over terramechanics (slip, sinkage)
    "Lunar/Construct-v0",      # goal-conditioned cut/fill to a target heightmap (terrain-matching reward)
    "Lunar/SkillMacro-v0",     # skill-macro construction: pick a cell + cut/dump toward target
    "Lunar/Scheduler-v0",      # multi-objective construction scheduling (borrow pits -> build sites)
]

_REGISTERED = False


def _default_challenge():
    """A small, self-contained flatten-a-pad challenge used as the default for the construction envs."""
    from . import challenge as ch
    return ch.Challenge(
        id="lunar_default", name="Flatten a construction pad", difficulty_tier=2,
        map=ch.MapSpec(seed=0, base="bumps", grid=48, roughness_m=0.004),
        objective=ch.Objective(type="flatten_pad", region=(16, 16, 32, 32), tolerance_m=0.01),
        constraints=ch.Constraints(max_time_steps=400),
    )


def _scheduler_kwargs():
    from . import ipex_specs as ix
    return dict(
        grid=64, cell_m=0.5,
        borrows=[(4, 4, 12, 12), (52, 52, 60, 60)],
        builds=[(10, 40, 14, 44), (40, 10, 44, 14), (44, 44, 48, 48)],
        fill_delta_m=0.10, mound_height_m=0.30, drum_capacity_kg=120.0, max_legs=40,
        travel_cost_per_cell=ix.drive_energy_per_m() * 0.5,   # grounded: 135 J/m * cell_m
        dig_cost_per_kg=ix.dig_energy_per_kg(),               # grounded: 4151 J/kg
        randomize=True,
    )


def register_envs():
    """Register the Lunar/* environments with Gymnasium. Idempotent; no-op without gymnasium."""
    global _REGISTERED
    if _REGISTERED:
        return
    try:
        from gymnasium.envs.registration import register, registry
    except Exception:
        return
    dc = _default_challenge()
    specs = [
        # (id, "module:Class", default kwargs, max_episode_steps)
        ("Lunar/RoverDrive-v0", "terrain_authority.rover_env:RoverSimEnv", {}, 2000),
        ("Lunar/Construct-v0", "terrain_authority.terrain_target_env:TerrainTargetEnv",
         {"challenge": dc}, None),
        ("Lunar/SkillMacro-v0", "terrain_authority.skill_env:SkillMacroEnv",
         {"challenge": dc, "discrete_cells": 8}, None),
        ("Lunar/Scheduler-v0", "terrain_authority.scheduler_env:SchedulerEnv",
         _scheduler_kwargs(), None),
    ]
    for env_id, entry_point, kwargs, max_steps in specs:
        if env_id in registry:
            continue
        register(id=env_id, entry_point=entry_point, kwargs=kwargs, max_episode_steps=max_steps)
    _REGISTERED = True
