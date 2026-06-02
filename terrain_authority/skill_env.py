"""SkillMacroEnv — skill-macro construction RL env (M2 pivot, 2026-06-02).

Implements the `state -> skill -> parameters` abstraction the taxonomy specifies, instead
of raw `state -> action` (which raw drive+drum PPO barely learned). One action = one
purposeful construction macro: SELECT a cell in the work region + a mode, and the macro
services that cell's disc TOWARD the target (cut excess into the drum, or dump deficit
from the drum), via the conserved authority (mass conservation guaranteed). This collapses
the long horizon to a few dozen cell-selection decisions, which is tractable for RL (and a
greedy selector already solves it — see test_skill_env).

Action (Box(3), [-1,1]): [row_frac, col_frac, mode] -> region cell + (mode>0 cut / <0 dump).
Observation: the region target-error field downsampled to obs_k x obs_k + drum fill.
Reward: potential-based RMSE reduction toward target, minus a small per-macro cost, +bonus.
Gymnasium-optional (bare-numpy core; gym.Env + spaces.Box when gymnasium present).
"""
from __future__ import annotations

import dataclasses

import numpy as np

from . import challenge as chmod
from .column_state import StateLabel

try:
    import gymnasium as _gym
    from gymnasium import spaces as _spaces
    _HAS_GYM = True
    _BASE = _gym.Env
except Exception:                              # pragma: no cover
    _gym = None; _spaces = None; _HAS_GYM = False; _BASE = object

HAS_GYM = _HAS_GYM


class SkillMacroEnv(_BASE):
    metadata = {"render_modes": []}

    def __init__(self, challenge, *, obs_k: int = 8, disc_cells: float = 2.0,
                 cut_per_macro_m: float = 0.04, match_scale: float = 50.0,
                 macro_cost: float = 0.01):
        super().__init__()
        self.challenge = challenge
        self.obj = challenge.objective
        self.region = self.obj.region
        self.tol = self.obj.tolerance_m
        self.grid = challenge.map.grid
        self.cell_m = challenge.map.cell_m
        self.max_macros = challenge.constraints.max_time_steps
        self.obs_k = int(obs_k)
        self.disc = float(disc_cells)
        self.cut_per_macro_m = float(cut_per_macro_m)
        self.match_scale = float(match_scale)
        self.macro_cost = float(macro_cost)
        self.obs_dim = self.obs_k * self.obs_k + 1
        self.action_dim = 3

        self.inst = None; self.cs = None; self._steps = 0; self._rmse = 0.0; self._m0 = 0.0
        if _HAS_GYM:
            self.action_space = _spaces.Box(-1.0, 1.0, shape=(self.action_dim,), dtype=np.float32)
            hi = np.full(self.obs_dim, 1.0e3, dtype=np.float32)
            self.observation_space = _spaces.Box(-hi, hi, dtype=np.float32)

    # -- helpers -------------------------------------------------------------

    def _err(self):
        return self.cs.derive_height() - self.inst.target_height

    def _rmse_region(self):
        return chmod.terrain_rmse(self.cs.derive_height(), self.inst.target_height, self.region)

    def _disc_mask(self, rc):
        r0, c0 = rc
        rr = np.arange(self.grid)[:, None] - r0
        cc = np.arange(self.grid)[None, :] - c0
        m = (rr * rr + cc * cc) <= self.disc * self.disc
        # confine to the work region
        reg = np.zeros_like(m)
        a, b, c, d = self.region
        reg[a:c, b:d] = True
        return m & reg

    def _obs(self):
        err = self._err()
        a, b, c, d = self.region
        rows = np.linspace(a, c - 1, self.obs_k).round().astype(int)
        cols = np.linspace(b, d - 1, self.obs_k).round().astype(int)
        patch = err[np.ix_(rows, cols)].ravel()
        drum = self.cs.drum_inventory / max(1.0, self.cs.grid_mass())
        return np.concatenate([patch, [drum]]).astype(np.float32)

    # -- gym API -------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if _HAS_GYM:
            super().reset(seed=seed)
        c = self.challenge
        if seed is not None:
            c = dataclasses.replace(c, map=dataclasses.replace(c.map, seed=seed))
        self.inst = chmod.realize(c)
        self.cs = self.inst.cs
        self._m0 = self.cs.total_mass()
        self._steps = 0
        self._rmse = self._rmse_region()
        return self._obs(), {"rmse": self._rmse}

    def step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        rf = (float(np.clip(a[0], -1, 1)) + 1) / 2
        cf = (float(np.clip(a[1], -1, 1)) + 1) / 2
        mode = float(np.clip(a[2], -1, 1))
        r0, c0, r1, c1 = self.region
        rc = (r0 + rf * (r1 - 1 - r0), c0 + cf * (c1 - 1 - c0))
        mask = self._disc_mask(rc)

        if mask.any():
            h = self.cs.derive_height()
            tgt = self.inst.target_height
            if mode > 1.0 / 3.0:                         # Excavate toward target (cut excess -> drum)
                excess = np.maximum(h - tgt, 0.0)
                mpc = np.minimum(excess, self.cut_per_macro_m) * self.cs.density
                self.cs.cut_to_inventory(mask, mpc)
                self.cs.state_label[mask] = StateLabel.EXCAVATED
            elif mode < -1.0 / 3.0 and self.cs.drum_inventory > 0.0:   # Dump toward target (drum -> deficit)
                deficit = np.minimum(np.maximum(tgt - h, 0.0), self.cut_per_macro_m)
                want_kg = float((deficit * self.cs.density)[mask].sum()) * self.cs.cell_area
                self.cs.dump_from_inventory(mask, min(want_kg, self.cs.drum_inventory))

        self._steps += 1
        new = self._rmse_region()
        reward = (self._rmse - new) * self.match_scale - self.macro_cost
        self._rmse = new
        success = new <= self.tol
        if success:
            reward += 1.0
        terminated = bool(success)
        truncated = self._steps >= self.max_macros
        return self._obs(), float(reward), terminated, truncated, {"success": success, "rmse": new,
                                                                    "drum": self.cs.drum_inventory}


def greedy_action(env: SkillMacroEnv):
    """Greedy cell-selector: cut the highest above-target cell; if none above tol, dump into
    the lowest below-target cell. Demonstrates the macro action space solves flatten."""
    err = env._err()
    r0, c0, r1, c1 = env.region
    sub = err[r0:r1, c0:c1]
    if sub.max() > env.tol:
        idx = np.unravel_index(np.argmax(sub), sub.shape); mode = 1.0
    else:
        idx = np.unravel_index(np.argmin(sub), sub.shape); mode = -1.0
    rf = idx[0] / max(1, r1 - 1 - r0)
    cf = idx[1] / max(1, c1 - 1 - c0)
    return [rf * 2 - 1, cf * 2 - 1, mode]              # map [0,1]->[-1,1] for the action space
