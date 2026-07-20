"""Classical sampling-based motion planner: RRT-Connect (directive §5.2c).

Non-learned reference. Plans a collision-free joint-space path from the current config to
the goal joint config using RRT-Connect with MuJoCo collision checking, then shortcuts it.
It does not use the world model at all; its path length is the reference for the path
optimality gap (§6.1) and it is a feasibility floor. Execution follows the path under the
same safety-gated position/velocity controller as every other method, so success and
smoothness are measured on identical footing.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np


@dataclass
class RRTConfig:
    max_iters: int = 4000
    step_size: float = 0.25          # rad, tree extension step
    goal_bias: float = 0.1
    shortcut_iters: int = 200
    max_env_steps: int = 400


class _Tree:
    def __init__(self, root):
        self.nodes = [root]
        self.parent = [-1]

    def nearest(self, q):
        d = [np.linalg.norm(n - q) for n in self.nodes]
        return int(np.argmin(d))

    def add(self, q, parent):
        self.nodes.append(q)
        self.parent.append(parent)
        return len(self.nodes) - 1

    def path_to(self, idx):
        out = []
        while idx != -1:
            out.append(self.nodes[idx])
            idx = self.parent[idx]
        return out[::-1]


def _steer(a, b, step):
    d = b - a
    n = np.linalg.norm(d)
    return b.copy() if n <= step else a + d / n * step


class RRTConnectPlanner:
    def __init__(self, cfg: RRTConfig | None = None):
        self.cfg = cfg or RRTConfig()

    def plan(self, env, rng) -> tuple[list | None, dict]:
        cfg = self.cfg
        start = env.q.copy()
        goal = env.q_goal.copy()
        lo = np.where(np.isfinite(env.limits.joint_pos_low[: env.nq]),
                      env.limits.joint_pos_low[: env.nq], -np.pi)
        hi = np.where(np.isfinite(env.limits.joint_pos_high[: env.nq]),
                      env.limits.joint_pos_high[: env.nq], np.pi)
        if not (env.collision_free(start) and env.collision_free(goal)):
            return None, {"reason": "endpoint_in_collision"}

        ta, tb = _Tree(start), _Tree(goal)
        t0 = time.time()
        for it in range(cfg.max_iters):
            q_rand = goal if rng.random() < cfg.goal_bias else rng.uniform(lo, hi)
            # extend ta toward q_rand
            ia = ta.nearest(q_rand)
            q_new = _steer(ta.nodes[ia], q_rand, cfg.step_size)
            if not self._edge_free(env, ta.nodes[ia], q_new):
                ta, tb = tb, ta
                continue
            ina = ta.add(q_new, ia)
            # connect tb toward q_new
            ib = tb.nearest(q_new)
            q_c = tb.nodes[ib]
            connected = False
            while True:
                q_step = _steer(q_c, q_new, cfg.step_size)
                if not self._edge_free(env, q_c, q_step):
                    break
                ib = tb.add(q_step, ib)
                q_c = q_step
                if np.linalg.norm(q_step - q_new) < 1e-6:
                    connected = True
                    break
            if connected:
                pa = ta.path_to(ina)
                pb = tb.path_to(ib)
                path = pa + pb[::-1]
                # ta/tb may have swapped; ensure path starts at `start`
                if np.linalg.norm(path[0] - start) > 1e-6:
                    path = path[::-1]
                path = self._shortcut(env, path, rng)
                return path, {"iters": it + 1, "plan_wall_s": time.time() - t0,
                              "path_nodes": len(path)}
            ta, tb = tb, ta
        return None, {"reason": "max_iters", "plan_wall_s": time.time() - t0}

    def _edge_free(self, env, a, b, res: float = 0.05):
        n = max(2, int(np.linalg.norm(b - a) / res))
        for s in np.linspace(0, 1, n):
            if not env.collision_free(a + s * (b - a)):
                return False
        return True

    def _shortcut(self, env, path, rng):
        path = [p.copy() for p in path]
        for _ in range(self.cfg.shortcut_iters):
            if len(path) <= 2:
                break
            i = rng.integers(0, len(path) - 1)
            j = rng.integers(i + 1, len(path))
            if j - i <= 1:
                continue
            if self._edge_free(env, path[i], path[j]):
                path = path[: i + 1] + path[j:]
        return path


class RRTController:
    """Follows an RRT joint path with the env's safety-gated controller."""
    def __init__(self, env, path):
        self.path = path
        self.idx = 1
        self.env = env

    def act(self, env, step):
        if self.idx >= len(self.path):
            self.idx = len(self.path) - 1
        target = self.path[self.idx]
        err = target - env.q
        vmax = env.limits.joint_vel_max[: env.nu] * env.limits.speed_cap_frac
        a = np.clip(err / (vmax * env.dt + 1e-9), -1, 1)
        if np.linalg.norm(err) < 0.05 and self.idx < len(self.path) - 1:
            self.idx += 1
        return a


def solve_rrt(env, seed: int, cfg: RRTConfig | None = None) -> dict:
    from ..eval.executor import execute_episode
    import torch
    rng = np.random.default_rng(seed)
    planner = RRTConnectPlanner(cfg)
    t0 = time.time()
    path, info = planner.plan(env, rng)
    plan_wall = time.time() - t0
    if path is None:
        return {"method": "rrt", "success": False, "interactions": 0,
                "planning_wall_s": plan_wall, "planning_flops": 0,
                "rrt_failed": True, "reason": info.get("reason"),
                "final_dist": None, "energy": 0.0, "jerk": 0.0,
                "ee_path_len": None, "ee_path": [], "safety_halt": False,
                "safety_summary": env.safety.summary()}
    ctrl = RRTController(env, path)
    cfg = cfg or RRTConfig()

    def decide(z_cur, step):
        return ctrl.act(env, step)

    # RRT doesn't use the model latent; execute_episode still encodes for uniformity but
    # decide ignores z. Pass a lightweight stand-in via env's model-free loop:
    m = _execute_modelfree(env, decide, cfg.max_env_steps)
    m.update({"method": "rrt", "planning_wall_s": plan_wall, "planning_flops": 0,
              "rrt_path_nodes": len(path), "rrt_failed": False})
    return m


def _execute_modelfree(env, decide_fn, max_steps: int) -> dict:
    """Executor for non-learned methods (no world-model encoding needed)."""
    from ..safety import SafetyHalt
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
        a = np.asarray(decide_fn(None, step), dtype=np.float64)[: env.nu]
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
        if info["success"]:
            success = True
            break
    path = np.asarray(path)
    path_len = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))) if len(path) > 1 else 0.0
    return {"success": bool(success), "interactions": int(interactions),
            "final_dist": float(info["dist"]), "energy": float(energy), "jerk": float(jerk),
            "ee_path_len": path_len, "ee_path": path.tolist(),
            "safety_halt": bool(halted), "safety_summary": env.safety.summary()}
