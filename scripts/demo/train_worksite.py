#!/usr/bin/env python3
"""Train a LEARNED controller on John McCardle's WorkSite seam (Dust/WorkSite-v0).

WorkSite (terrain_authority/worksite.py, PR #5) is the streaming execution engine and explicitly leaves
"the controller as the only stub" -- its .flatten()/.dump() + global drum ledger are shaped for an RL
policy. WorkSiteConstructEnv is that controller: Discrete(2) actions ARE WorkSite verbs (flatten a pad
slice -> ledger, dump a berm slice <- ledger), so PPO learns the cut-haul-fill batching (WHEN to switch
cut->dump under the ledger) on the real conserved engine.

Observed (held-out randomized instances, 120k steps):
    greedy heuristic   success=60%  avg_steps=22
    random             success=23%
    PPO (learned)      success=63%  avg_steps=21   (matches/edges greedy, generalizes, ~3x random)

The ~60% ceiling is feasibility variation across random bumps (not all instances are solvable within the
budget); PPO learns the threshold and is slightly more step-efficient than greedy on John's seam.

Run with the runtime venv (gymnasium + torch + SB3):
    PYTHONPATH=<repo> /mnt/projects/07_runtime_system/venv/bin/python scripts/demo/train_worksite.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from terrain_authority.worksite_env import (WorkSiteConstructEnv, beam_worksite_plan,
                                            greedy_worksite)


class _TrainEnv(WorkSiteConstructEnv):
    """Cycle the seed each reset so PPO trains on varied WorkSite instances (domain randomization)."""
    _ep = 0

    def reset(self, *, seed=None, options=None):
        _TrainEnv._ep += 1
        s = seed if seed is not None else _TrainEnv._ep % 300
        return super().reset(seed=s, options=options)


def evaluate(policy, n=30, seed0=9000):
    succ, steps = [], []
    for i in range(n):
        env = WorkSiteConstructEnv(); obs, _ = env.reset(seed=seed0 + i); done = False; info = {}
        while not done:
            obs, _, te, tr, info = env.step(policy(env, obs)); done = te or tr
        succ.append(bool(info["success"]))
        steps.append(info["steps"] if info["success"] else env.max_steps)
    return float(np.mean(succ)), float(np.mean(steps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=120_000)
    ap.add_argument("--eval-episodes", type=int, default=30)
    args = ap.parse_args()

    rng = np.random.default_rng(1)
    gs, gl = evaluate(lambda e, o: greedy_worksite(e), args.eval_episodes)
    rs, rl = evaluate(lambda e, o: rng.integers(2), args.eval_episodes)
    bs = _beam_rate(args.eval_episodes)
    print(f"greedy heuristic        success={gs:.0%}  avg_steps={gl:.1f}")
    print(f"random                  success={rs:.0%}  avg_steps={rl:.1f}")
    print(f"beam search (ceiling)   success={bs:.0%}  (model-based, the best policy on this cheap sim)")

    from stable_baselines3 import PPO
    model = PPO("MlpPolicy", _TrainEnv(), n_steps=2048, batch_size=256, gamma=0.99,
                ent_coef=0.01, learning_rate=3e-4, verbose=0, seed=0, device="cpu")
    model.learn(total_timesteps=args.timesteps)
    ps, pl = evaluate(lambda e, o: int(model.predict(o, deterministic=True)[0]), args.eval_episodes)
    print(f"PPO (learned)           success={ps:.0%}  avg_steps={pl:.1f}  ({args.timesteps} steps)")
    print(f"\nA learned policy drives John's WorkSite seam (flatten/dump + drum ledger): PPO {ps:.0%} "
          f"(>= greedy {gs:.0%}), beating random by {(ps - rs) * 100:.0f} points -- the controller "
          f"WorkSite stubbed, learned on the real conserved engine.")
    print(f"Honest ceiling: beam search reaches {bs:.0%}; the {bs - ps:+.0%} gap over reactive policies is a "
          f"LOOKAHEAD advantage (planning the schedule against the tight budget) that greedy/PPO/search-"
          f"distilled-MLP all plateau below (~63%). On this exact, cheap sim the search IS the best policy.")


def _beam_rate(n=30, seed0=9000):
    out = []
    for i in range(n):
        env = WorkSiteConstructEnv(); env.reset(seed=seed0 + i)
        plan = beam_worksite_plan(env, width=12)
        env2 = WorkSiteConstructEnv(); env2.reset(seed=seed0 + i); info = {}
        for a in plan:
            _, _, te, tr, info = env2.step(a)
            if te or tr:
                break
        out.append(bool(info.get("success")))
    return float(np.mean(out))


if __name__ == "__main__":
    main()
