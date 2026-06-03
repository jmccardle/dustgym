"""lunar_sim_gym — a Gymnasium suite for lunar surface vehicles + autonomous construction.

Importing this package REGISTERS the ``Lunar/*`` environments with Gymnasium (the documented
third-party pattern), so::

    import lunar_sim_gym                       # registers Lunar/* on import
    import gymnasium as gym
    env = gym.make("Lunar/RoverDrive-v0")

The environments live on the mass-conserving terramechanics authority in ``terrain_authority``
(IPEx / Lunar Autonomy Challenge lineage). This package is the thin Gymnasium-facing layer.
"""
from __future__ import annotations

from terrain_authority.registration import LUNAR_ENV_IDS, register_envs

register_envs()

__all__ = ["register_envs", "LUNAR_ENV_IDS"]
