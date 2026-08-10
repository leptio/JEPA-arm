"""Exploratory data collection in SIMULATION (directive §3.4, §4.1).

§3.4 requires exploration to run first in simulation; that is the only place it runs
here. Every trajectory logs the observation stream, executed actions, and the policy
identity (§4.1). Collection is safety-gated (actions clamped by SafetyMonitor) and
seeded/deterministic (§4.4). A SafetyHalt aborts the episode and is recorded.

Stored as a compressed .npz of transitions (o_t, a_t, o_{t+1}, arm_id) plus a JSON
manifest with provenance.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np

from ..envs.arm_env import ArmEnv
from ..safety import SafetyHalt
from .policies import make_policy


def collect(arm: str, policy_name: str, n_episodes: int, horizon: int,
            safety_config: str, seed: int, out_dir: str,
            control_hz: float = 10.0, joint_encoding: str = "raw",
            obstacle: dict = None, max_goal_delta: float = None) -> dict:
    rng = np.random.default_rng(seed)
    env = ArmEnv(arm, safety_config, control_hz=control_hz, seed=seed,
                 joint_encoding=joint_encoding, obstacle=obstacle,
                 max_goal_delta=max_goal_delta)
    policy = make_policy(policy_name)

    obs_t, act_t, obs_tp1 = [], [], []
    n_halts = 0
    for ep in range(n_episodes):
        o = env.reset(seed=seed * 100003 + ep)
        policy.reset(env, rng)
        try:
            for _ in range(horizon):
                a = policy.act(env, o, rng)
                o2, info = env.step(a)
                obs_t.append(o); act_t.append(env_pad_action(env, a)); obs_tp1.append(o2)
                o = o2
        except SafetyHalt:
            n_halts += 1
            continue

    obs_t = np.asarray(obs_t, dtype=np.float32)
    act_t = np.asarray(act_t, dtype=np.float32)
    obs_tp1 = np.asarray(obs_tp1, dtype=np.float32)
    arm_id = np.full((len(obs_t),), env.emb.index, dtype=np.int64)

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    npz = out / f"{arm}__{policy.name}__seed{seed}.npz"
    np.savez_compressed(npz, obs_t=obs_t, act_t=act_t, obs_tp1=obs_tp1, arm_id=arm_id)
    manifest = {
        "arm": arm, "policy": policy.name, "seed": int(seed),
        "n_episodes": n_episodes, "horizon": horizon,
        "n_transitions": int(len(obs_t)), "n_safety_halts": n_halts,
        "env_spec": env.spec(), "safety_summary": env.safety.summary(),
        "npz": npz.name,
    }
    (out / f"{npz.stem}.manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def env_pad_action(env: ArmEnv, a: np.ndarray) -> np.ndarray:
    """Pad a per-arm action to the canonical CANON_ACT_DIM (=MAX_DOF)."""
    from ..envs.embodiment import CANON_ACT_DIM
    out = np.zeros(CANON_ACT_DIM, dtype=np.float32)
    out[: len(a)] = a
    return out
