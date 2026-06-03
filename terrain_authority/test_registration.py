"""Tests for the Lunar/* Gymnasium suite registration.

Verifies every registered ID is gym.make-able with no args, round-trips reset/step under the
gym.make wrappers (OrderEnforcing / PassiveEnvChecker / TimeLimit), and passes the official
env_checker. Skipped automatically where gymnasium is not installed (the bare-numpy core).
"""
from __future__ import annotations

import pytest

gym = pytest.importorskip("gymnasium")

import lunar_sim_gym  # noqa: E402,F401  -- importing the suite registers Lunar/* on import
from terrain_authority.registration import LUNAR_ENV_IDS  # noqa: E402


@pytest.mark.parametrize("env_id", LUNAR_ENV_IDS)
def test_registered(env_id):
    from gymnasium.envs.registration import registry
    assert env_id in registry


@pytest.mark.parametrize("env_id", LUNAR_ENV_IDS)
def test_gym_make_roundtrip(env_id):
    env = gym.make(env_id)                      # no args -> defaults make it constructible
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs), env_id
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert isinstance(reward, float)
    env.close()


@pytest.mark.parametrize("env_id", LUNAR_ENV_IDS)
def test_check_env(env_id):
    from gymnasium.utils.env_checker import check_env
    check_env(gym.make(env_id).unwrapped, skip_render_check=True)


def test_override_kwargs():
    # any constructor arg can still be overridden through gym.make
    env = gym.make("Lunar/Scheduler-v0", max_legs=25)
    assert env.unwrapped.max_legs == 25
    env.close()
