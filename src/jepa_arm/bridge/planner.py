"""Planning-as-inference and bridge-driven control (directive §2.5, §2.6, §2.7).

PathIntegralController: desirability-weighted (path-integral / MPPI) control. It samples
action sequences, rolls them out THROUGH the forward model's action channel (the
interventional quantity p(outcome | do(a)), §2.5), reweights by exp(-cost/lambda), and
takes the reweighted expectation over actions. It never conditions on historically
successful trajectories (the confounded observational quantity p(a | outcome), §2.5, §8.2).

forward_reachable: the [HARD] backward-frontier validation (§2.6). A candidate waypoint
proposed by the backward pathway is only allowed to influence a command if the FORWARD
model can reach it from the current latent within tolerance. The backward pathway
proposes; the forward pathway disposes.

Latent distance to the goal is used only as a search heuristic and is NOT assumed
admissible (§2.7); reported solutions are validated-feasible, with an optimality gap
measured against the classical planner (§5.2c).
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
import torch


def mlp_flops(sizes) -> int:
    """FLOPs (multiply-adds*2) of one forward pass of an MLP with given layer sizes."""
    f = 0
    for i in range(len(sizes) - 1):
        f += 2 * sizes[i] * sizes[i + 1]
    return f


def model_call_flops(cfg) -> dict:
    d, a, w, h = cfg.latent_dim, cfg.act_dim, cfg.w_dim, cfg.hidden
    return {
        "encode": mlp_flops([cfg.obs_dim, h, h, d]),
        "decode": mlp_flops([d + a + w, h, h, d]),
    }


@dataclass
class MPPIConfig:
    horizon: int = 12
    n_samples: int = 256
    lam: float = 0.5                 # temperature
    sigma: float = 0.4               # action noise std
    ctrl_cost: float = 0.01
    running_cost: float = 0.1        # weight on per-step distance to target
    iters: int = 2                   # refinement iterations per decision


class PathIntegralController:
    def __init__(self, model, cfg: MPPIConfig, active_dof: int):
        self.model = model
        self.cfg = cfg
        self.dev = model.device
        self.act_dim = model.cfg.act_dim
        self.active = torch.zeros(self.act_dim, device=self.dev)
        self.active[:active_dof] = 1.0
        self._decode_flops = model_call_flops(model.cfg)["decode"]
        self.flops = 0                # accumulated planning FLOPs (H2)

    @torch.no_grad()
    def _rollout_cost(self, z0, mean_seq, z_target):
        cfg = self.cfg
        N, H = cfg.n_samples, cfg.horizon
        noise = torch.randn(N, H, self.act_dim, device=self.dev) * cfg.sigma * self.active
        actions = torch.clamp(mean_seq.unsqueeze(0) + noise, -1.0, 1.0)
        z = z0.repeat(N, 1)
        cost = torch.zeros(N, device=self.dev)
        for t in range(H):
            z = self.model.forward_pred.decode(z, actions[:, t], torch.zeros(N, self.model.cfg.w_dim, device=self.dev))
            cost = cost + cfg.running_cost * (z - z_target).pow(2).sum(-1)
            cost = cost + cfg.ctrl_cost * actions[:, t].pow(2).sum(-1)
        cost = cost + (z - z_target).pow(2).sum(-1)      # terminal
        self.flops += N * H * self._decode_flops
        return actions, cost

    @torch.no_grad()
    def plan_sequence(self, z0, z_target, mean_seq=None):
        """Return the desirability-weighted action sequence (path-integral update)."""
        H = self.cfg.horizon
        if mean_seq is None:
            mean_seq = torch.zeros(H, self.act_dim, device=self.dev)
        for _ in range(self.cfg.iters):
            actions, cost = self._rollout_cost(z0, mean_seq, z_target)
            weights = torch.softmax(-cost / self.cfg.lam, dim=0)     # desirability weights
            mean_seq = (weights.view(-1, 1, 1) * actions).sum(0)     # E_w[a] (§2.5)
        return mean_seq

    @torch.no_grad()
    def act(self, z0, z_target, mean_seq=None):
        seq = self.plan_sequence(z0, z_target, mean_seq)
        return seq[0], seq        # first action (receding horizon), full seq for warm start


@torch.no_grad()
def forward_reachable(model, controller: PathIntegralController, z_from, z_wp,
                      steps: int = 12, tol: float = 0.5):
    """§2.6 forward-frontier validation, entirely in latent space under F.
    Returns (reachable, min_dist)."""
    z = z_from.reshape(1, -1)
    best = float("inf")
    mean_seq = None
    for _ in range(steps):
        a, mean_seq = controller.act(z, z_wp.reshape(1, -1), mean_seq)
        z = model.forward_pred.predict_mean(z, a.unsqueeze(0))
        controller.flops += controller._decode_flops
        d = float((z - z_wp.reshape(1, -1)).norm().item())
        best = min(best, d)
        mean_seq = torch.cat([mean_seq[1:], mean_seq[-1:]], 0)   # shift warm start
    return best < tol, best


@dataclass
class BridgePlannerConfig:
    mppi: MPPIConfig = None
    forward_validate: bool = True          # ABL-noFV sets False (§5.3)
    validate_steps: int = 10
    validate_tol_frac: float = 0.6         # tol as frac of start->goal latent dist / T
    advance_tol_frac: float = 0.5          # advance waypoint when within this frac
    max_env_steps: int = 300


class BridgePlanner:
    """Full bridge-driven policy executed on the real (simulated) env."""
    def __init__(self, model, bridge, cfg: BridgePlannerConfig, active_dof: int,
                 backward_trust_weight: float = 1.0):
        self.model = model
        self.bridge = bridge
        self.cfg = cfg or BridgePlannerConfig()
        if self.cfg.mppi is None:
            self.cfg.mppi = MPPIConfig()
        self.controller = PathIntegralController(model, self.cfg.mppi, active_dof)
        self.active_dof = active_dof
        self.bwd_trust = backward_trust_weight
        self.bridge.set_active_dof(active_dof)

    @torch.no_grad()
    def solve(self, env) -> dict:
        from ..eval.executor import execute_episode
        dev = self.model.device
        t_plan0 = time.time()
        z = self.model.encode(torch.from_numpy(env.obs()).float().to(dev).unsqueeze(0))[0]
        z_goal = self.model.encode(torch.from_numpy(env.goal_obs()).float().to(dev).unsqueeze(0))[0]
        d_sg = float((z - z_goal).norm().item())

        waypoints, clouds, diag = self.bridge.plan(z, z_goal)
        # ---- §2.6 backward-frontier forward-validation --------------------------
        tol = self.cfg.validate_tol_frac * d_sg / max(1, len(waypoints) - 1)
        validated = [z]
        n_proposed = len(waypoints) - 1
        n_reject = 0
        zprev = z
        for wp in waypoints[1:]:
            if self.cfg.forward_validate:
                ok, _ = forward_reachable(self.model, self.controller, zprev, wp,
                                          steps=self.cfg.validate_steps, tol=max(tol, 0.3))
                if not ok:
                    n_reject += 1
                    continue
            validated.append(wp)
            zprev = wp
        validated.append(z_goal)
        plan_wall = time.time() - t_plan0

        # ---- execute on env, tracking validated waypoints (shared executor) ------
        adv_tol = self.cfg.advance_tol_frac * d_sg / max(1, len(validated) - 1)
        state = {"wp_idx": 1, "mean_seq": None}

        def decide(z_cur, step):
            target = validated[min(state["wp_idx"], len(validated) - 1)]
            a_t, mean_seq = self.controller.act(z_cur.unsqueeze(0), target.unsqueeze(0),
                                                state["mean_seq"])
            state["mean_seq"] = torch.cat([mean_seq[1:], mean_seq[-1:]], 0)
            if float((z_cur - target).norm().item()) < adv_tol and state["wp_idx"] < len(validated) - 1:
                state["wp_idx"] += 1
            return a_t[: self.active_dof].cpu().numpy()

        m = execute_episode(env, self.model, decide, self.cfg.max_env_steps)
        m.update({
            "method": "bridge",
            # total planning wall (bridge construction + validation + per-decision MPPI),
            # so H2 compares like-for-like against the baseline's whole-episode timing.
            "planning_wall_s": time.time() - t_plan0,
            "bridge_construction_wall_s": plan_wall,
            "planning_flops": int(self.controller.flops + self.bridge.flops),
            "n_waypoints_proposed": n_proposed,
            "n_waypoints_rejected_by_forward_validation": n_reject,
            "waypoint_rejection_frac": n_reject / max(1, n_proposed),
            "bridge_ess_mean": diag.get("ess_mean", float("nan")),
            "backward_trust_weight": self.bwd_trust,
        })
        return m
