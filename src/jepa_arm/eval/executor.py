"""Shared episode executor so EVERY method is measured identically (fair H1/H2/H3).

A method supplies a `decide_fn(z_latent, step) -> action_np`. The executor runs the
closed loop on the (simulated) env, encoding observations into the model latent, and
records the common metric set: success, environment interactions (§6.1 sample-efficiency
unit), control energy, jerk (smoothness §6.1), the end-effector path (for the optimality
gap vs the classical planner §6.1), constraint/safety events, and any SafetyHalt (§8.1).
"""
from __future__ import annotations
import numpy as np
import torch

from ..safety import SafetyHalt


@torch.no_grad()
def execute_episode(env, model, decide_fn, max_steps: int) -> dict:
    dev = model.device
    z = model.encode(torch.from_numpy(env.obs()).float().to(dev).unsqueeze(0))[0]
    interactions = 0
    energy = 0.0
    jerk = 0.0
    prev_a = np.zeros(env.nu)
    ee0, _ = env.ee_pose()
    path = [ee0.copy()]
    success = False
    halted = False
    info = {"dist": float(np.linalg.norm(ee0 - env.ee_goal))}
    for step in range(max_steps):
        a = np.asarray(decide_fn(z, step), dtype=np.float64)[: env.nu]
        try:
            o2, info = env.step(a)
        except SafetyHalt:
            halted = True
            break
        interactions += 1
        energy += float(np.sum(a ** 2))
        jerk += float(np.sum((a - prev_a) ** 2))
        prev_a = a
        path.append(info["ee_pos"].copy())
        z = model.encode(torch.from_numpy(o2).float().to(dev).unsqueeze(0))[0]
        if info["success"]:
            success = True
            break
    path = np.asarray(path)
    path_len = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))) if len(path) > 1 else 0.0
    return {
        "success": bool(success),
        "interactions": int(interactions),
        "final_dist": float(info["dist"]),
        "energy": float(energy),
        "jerk": float(jerk),
        "ee_path_len": path_len,
        "ee_path": path.tolist(),
        "safety_halt": bool(halted),
        "safety_summary": env.safety.summary(),
    }
