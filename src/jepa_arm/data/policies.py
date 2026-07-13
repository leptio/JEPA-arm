"""Behavior policies for data collection (directive §2.3.1, §4.1).

The policy identity is logged for every trajectory. Two DELIBERATELY DIFFERENT policies
are provided so the backward pathway can be tested for policy-confounding (§2.3.1): if
B's predicted predecessors shift between policy A and policy B beyond TAU_POLICY, B is
flagged confounded.

  policy_A "smoothed_random" : correlated random-walk exploration (broad state coverage)
  policy_B "goal_biased"     : servo toward sampled goals (narrow, task-directed coverage)

Both are safety-gated by the env's SafetyMonitor (actions are clamped before actuation).
"""
from __future__ import annotations
import numpy as np


class Policy:
    name = "base"

    def reset(self, env, rng):
        pass

    def act(self, env, obs, rng) -> np.ndarray:
        raise NotImplementedError


class SmoothedRandom(Policy):
    """Policy A: temporally correlated random actions (Ornstein-Uhlenbeck-like)."""
    name = "policy_A_smoothed_random"

    def __init__(self, theta: float = 0.15, sigma: float = 0.5):
        self.theta, self.sigma = theta, sigma
        self._a = None

    def reset(self, env, rng):
        self._a = np.zeros(env.nu)

    def act(self, env, obs, rng):
        self._a = (1 - self.theta) * self._a + self.sigma * rng.normal(size=env.nu)
        return np.clip(self._a, -1.0, 1.0)


class GoalBiased(Policy):
    """Policy B: proportional servo toward a periodically-resampled joint goal. Produces a
    markedly different state-visitation distribution than policy A (task-directed)."""
    name = "policy_B_goal_biased"

    def __init__(self, resample_every: int = 40, gain: float = 3.0, noise: float = 0.15):
        self.resample_every, self.gain, self.noise = resample_every, gain, noise
        self._goal = None
        self._k = 0

    def reset(self, env, rng):
        self._goal = env.random_config(rng)
        self._k = 0

    def act(self, env, obs, rng):
        if self._k % self.resample_every == 0:
            self._goal = env.random_config(rng)
        self._k += 1
        err = self._goal - env.q
        vmax = env.limits.joint_vel_max[: env.nu] * env.limits.speed_cap_frac
        a = self.gain * err / (vmax * env.dt + 1e-9)
        a = a + self.noise * rng.normal(size=env.nu)
        return np.clip(a, -1.0, 1.0)


def make_policy(name: str) -> Policy:
    return {"policy_A": SmoothedRandom, "policy_B": GoalBiased,
            SmoothedRandom.name: SmoothedRandom, GoalBiased.name: GoalBiased}[name]()
