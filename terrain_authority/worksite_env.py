"""worksite_env.py — RL controller over John McCardle's WorkSite seam (reconciliation, 2026-06-03).

WorkSite (terrain_authority/worksite.py, PR #5) is the streaming execution engine: a coarse base +
rover-following fine window with a GLOBAL drum ledger, exposing `.flatten()/.dump()/.drive()/.relax()`
"shaped so an RL policy can drive the SAME seam later -- the controller is the only stub." This is that
controller: a Gymnasium env whose actions are WorkSite construction verbs, executed on the real window.

Task (cut-haul-fill, the genuine planning regime): flatten a bumpy PAD to a level (dig -> ledger) and
build a BERM elsewhere by dumping that material (ledger -> grid). The drum ledger couples them: you
cannot dump more than you have dug, so the binding decision is WHEN to switch cut->dump (batching), the
same finding as the standalone SchedulerEnv -- but now mass flows through John's conserved WorkSite ledger
(`inventory_kg`), not an ad-hoc env. Action = Discrete(2): 0 flatten the next pad slice, 1 dump the next
berm slice. Mass is conserved by WorkSite (grid + inventory_kg invariant).

Gymnasium-optional (bare-numpy core; gym.Env + spaces when present).
"""
from __future__ import annotations

import numpy as np

from . import constants as K

try:
    import gymnasium as _gym
    from gymnasium import spaces as _spaces
    _HAS_GYM = True
    _BASE = _gym.Env
except Exception:                              # pragma: no cover
    _gym = None; _spaces = None; _HAS_GYM = False; _BASE = object

HAS_GYM = _HAS_GYM


def _bumpy_base(n_base=8, base_cell_m=0.5, roughness_m=0.03, seed=0):
    """A small synthetic coarse base with surface bumps (so the pad has excess to cut)."""
    from .column_state import ColumnState
    cs = ColumnState(width=n_base, height=n_base, cell_m=base_cell_m)
    rng = np.random.default_rng(seed)
    cs.mass_areal += (rng.random((n_base, n_base)) * roughness_m) * cs.density   # bumps as mass
    return cs


class WorkSiteConstructEnv(_BASE):
    metadata = {"render_modes": []}

    def __init__(self, *, n_base=8, base_cell_m=0.5, fine_cell_m=0.1, roughness_m=0.15,
                 berm_delta_m=0.025, n_slices=6, max_steps=24, tol_frac=0.15,
                 match_scale=10.0, step_cost=0.05, seed=0):
        super().__init__()
        self.n_base = int(n_base); self.base_cell_m = float(base_cell_m)
        self.fine_cell_m = float(fine_cell_m); self.roughness_m = float(roughness_m)
        self.berm_delta_m = float(berm_delta_m); self.n_slices = int(n_slices)
        self.max_steps = int(max_steps); self.tol_frac = float(tol_frac)
        self.match_scale = float(match_scale); self.step_cost = float(step_cost)
        self._seed0 = int(seed)
        self.ws = None; self.fine = None
        self.pad_rows = None; self.berm_rows = None; self.pad_target = 0.0
        self.berm_target = None; self._pad_done = None; self._berm_done = None
        self._steps = 0
        self.obs_dim = 4
        if _HAS_GYM:
            self.action_space = _spaces.Discrete(2)            # 0 = flatten pad slice, 1 = dump berm slice
            hi = np.full(self.obs_dim, 1.0e3, dtype=np.float32)
            self.observation_space = _spaces.Box(-hi, hi, dtype=np.float32)

    # -- geometry -----------------------------------------------------------
    def _slice_mask(self, rows, k):
        m = np.zeros((self.fine.height, self.fine.width), bool)
        r = rows[k]
        m[r[0]:r[1], self._cwin[0]:self._cwin[1]] = True
        return m

    def _pad_excess_kg(self):
        h = self.fine.derive_height()
        tot = 0.0
        for (r0, r1) in self.pad_rows:
            tot += float(np.maximum(h[r0:r1, self._cwin[0]:self._cwin[1]] - self.pad_target, 0.0).sum())
        return tot

    def _berm_deficit_kg(self):
        h = self.fine.derive_height()
        tot = 0.0
        for (r0, r1) in self.berm_rows:
            tot += float(np.maximum(self.berm_target - h[r0:r1, self._cwin[0]:self._cwin[1]], 0.0).sum())
        return tot

    # -- gym API ------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if _HAS_GYM:
            super().reset(seed=seed)
        s = self._seed0 if seed is None else int(seed)
        from .worksite import WorkSite
        base = _bumpy_base(self.n_base, self.base_cell_m, self.roughness_m, seed=s)
        self.ws = WorkSite(base, world_x0=0.0, world_y0=0.0, fine_cell_m=self.fine_cell_m)
        self.ws.open_window((self.n_base / 2.0, self.n_base / 2.0), radius_m=self.base_cell_m * self.n_base)
        self.fine = self.ws.fine
        H, W = self.fine.height, self.fine.width
        # central work column; pad = top band, berm = bottom band, each split into n_slices rows
        self._cwin = (W // 4, 3 * W // 4)
        pad_band = (H // 8, H // 2); berm_band = (H // 2, 7 * H // 8)
        self.pad_rows = self._split(pad_band, self.n_slices)
        self.berm_rows = self._split(berm_band, self.n_slices)
        h = self.fine.derive_height()
        pad_h = h[pad_band[0]:pad_band[1], self._cwin[0]:self._cwin[1]]
        self.pad_target = float(pad_h.min())                       # flatten the pad down to its lowest
        berm_h = h[berm_band[0]:berm_band[1], self._cwin[0]:self._cwin[1]]
        self.berm_target = float(berm_h.mean() + self.berm_delta_m)  # raise the berm by delta
        self._pad_done = [False] * self.n_slices
        self._berm_done = [False] * self.n_slices
        self._steps = 0
        self._pad0 = max(1e-9, self._pad_excess_kg())
        self._berm0 = max(1e-9, self._berm_deficit_kg())
        self._prev = self._pad_excess_kg() / self._pad0 + self._berm_deficit_kg() / self._berm0
        self._m0 = self.ws.total_mass()                            # grid + ledger invariant
        return self._obs(), {"inventory_kg": self.ws.inventory_kg}

    def _split(self, band, n):
        edges = np.linspace(band[0], band[1], n + 1).round().astype(int)
        return [(int(edges[i]), int(edges[i + 1])) for i in range(n)]

    def _obs(self):
        pad_left = self._pad_excess_kg() / self._pad0
        berm_left = self._berm_deficit_kg() / self._berm0
        cap = max(1e-9, self._berm0 * K.RHO_SPOIL * self.fine_cell_m ** 2 / max(1, self.n_slices))
        inv = self.ws.inventory_kg / max(1.0, self._m0)
        return np.array([pad_left, berm_left, float(inv), 1.0 - self._steps / self.max_steps],
                        dtype=np.float32)

    def step(self, action):
        a = int(action) if np.isscalar(action) else int(np.asarray(action).ravel()[0])
        if a == 0:                                                 # flatten next undone pad slice
            k = next((i for i, d in enumerate(self._pad_done) if not d), None)
            if k is not None:
                self.ws.flatten(self._slice_mask(self.pad_rows, k), self.pad_target)
                self._pad_done[k] = True
        else:                                                      # dump next undone berm slice
            k = next((i for i, d in enumerate(self._berm_done) if not d), None)
            if k is not None:
                mask = self._slice_mask(self.berm_rows, k)
                h = self.fine.derive_height()
                need_m = np.maximum(self.berm_target - h, 0.0)
                want_kg = float(need_m[mask].sum()) * K.RHO_SPOIL * self.fine_cell_m ** 2
                placed = self.ws.dump(mask, kg=min(want_kg, self.ws.inventory_kg))
                if placed > 0 and self._berm_deficit_slice(k) <= self.tol_frac * self._berm_slice0(k):
                    self._berm_done[k] = True
        self._steps += 1
        cur = self._pad_excess_kg() / self._pad0 + self._berm_deficit_kg() / self._berm0
        reward = (self._prev - cur) * self.match_scale - self.step_cost
        self._prev = cur
        success = (self._pad_excess_kg() <= self.tol_frac * self._pad0
                   and self._berm_deficit_kg() <= self.tol_frac * self._berm0)
        if success:
            reward += 5.0
        terminated = bool(success)
        truncated = self._steps >= self.max_steps
        info = {"success": success, "inventory_kg": self.ws.inventory_kg, "steps": self._steps,
                "pad_excess": self._pad_excess_kg(), "berm_deficit": self._berm_deficit_kg()}
        return self._obs(), float(reward), terminated, truncated, info

    def _berm_deficit_slice(self, k):
        h = self.fine.derive_height(); r0, r1 = self.berm_rows[k]
        return float(np.maximum(self.berm_target - h[r0:r1, self._cwin[0]:self._cwin[1]], 0.0).sum())

    def _berm_slice0(self, k):
        return max(1e-9, self._berm0 / self.n_slices)


def greedy_worksite(env: WorkSiteConstructEnv):
    """Batch policy on the WorkSite seam: flatten pad slices to build the ledger, then dump into the berm.
    Flatten while the ledger can't yet cover the next berm slice; dump once it can."""
    if any(not d for d in env._pad_done):
        # need enough material for a berm slice? estimate one slice's kg
        h = env.fine.derive_height()
        k = next((i for i, d in enumerate(env._berm_done) if not d), None)
        if k is not None:
            mask = env._slice_mask(env.berm_rows, k)
            need = float(np.maximum(env.berm_target - h, 0.0)[mask].sum()) * K.RHO_SPOIL * env.fine_cell_m ** 2
            if env.ws.inventory_kg < need:
                return 0                                            # flatten more first
        else:
            return 0
    if any(not d for d in env._berm_done) and env.ws.inventory_kg > 0:
        return 1
    return 0 if any(not d for d in env._pad_done) else 1
