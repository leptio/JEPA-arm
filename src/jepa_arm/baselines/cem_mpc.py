"""Forward-only CEM/MPC baseline in the same latent space (directive §5.2a).

This is the primary comparator for H1 (sample efficiency) and H2 (planning compute). It
plans by the Cross-Entropy Method: sample action sequences, roll them out THROUGH the
forward model (interventional p(outcome|do(a)), §2.5), keep the elite fraction by cost,
refit a Gaussian, iterate; execute the first action (receding horizon). No backward
predictor, no bridge. ABL-noB (§5.3) recovers exactly this behavior.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
import torch

from ..models.world_model import decode_flops


@dataclass
class CEMConfig:
    horizon: int = 20
    n_samples: int = 256
    n_elite: int = 32
    iters: int = 4
    init_std: float = 0.6
    ctrl_cost: float = 0.01
    running_cost: float = 0.1
    max_env_steps: int = 250


class CEMController:
    def __init__(self, model, cfg: CEMConfig, active_dof: int):
        self.model = model
        self.cfg = cfg
        self.dev = model.device
        self.act_dim = model.cfg.act_dim
        self.active = torch.zeros(self.act_dim, device=self.dev)
        self.active[:active_dof] = 1.0
        self._decode_flops = decode_flops(model.cfg)
        self.flops = 0

    @torch.no_grad()
    def _cost(self, z0, actions, z_target):
        N, H = actions.shape[0], actions.shape[1]
        z = z0.repeat(N, 1)
        cost = torch.zeros(N, device=self.dev)
        for t in range(H):
            z = self.model.forward_pred.decode(
                z, actions[:, t], torch.zeros(N, self.model.cfg.w_dim, device=self.dev))
            cost += self.cfg.running_cost * (z - z_target).pow(2).sum(-1)
            cost += self.cfg.ctrl_cost * actions[:, t].pow(2).sum(-1)
        cost += (z - z_target).pow(2).sum(-1)
        self.flops += N * H * self._decode_flops
        return cost

    @torch.no_grad()
    def act(self, z0, z_target, warm=None):
        cfg = self.cfg
        H = cfg.horizon
        mean = warm if warm is not None else torch.zeros(H, self.act_dim, device=self.dev)
        std = torch.full((H, self.act_dim), cfg.init_std, device=self.dev)
        for _ in range(cfg.iters):
            noise = torch.randn(cfg.n_samples, H, self.act_dim, device=self.dev)
            actions = torch.clamp((mean.unsqueeze(0) + std.unsqueeze(0) * noise) * self.active, -1, 1)
            cost = self._cost(z0, actions, z_target)
            elite = actions[torch.topk(-cost, cfg.n_elite).indices]
            mean = elite.mean(0)
            std = elite.std(0) + 1e-3
        return mean[0], mean


class ForwardOnlyMPC:
    def __init__(self, model, cfg: CEMConfig, active_dof: int, method_name: str = "cem_mpc"):
        self.model = model
        self.cfg = cfg
        self.controller = CEMController(model, cfg, active_dof)
        self.active_dof = active_dof
        self.method_name = method_name

    @torch.no_grad()
    def solve(self, env) -> dict:
        from ..eval.executor import execute_episode
        dev = self.model.device
        t0 = time.time()
        z_goal = self.model.encode(
            torch.from_numpy(env.goal_obs()).float().to(dev).unsqueeze(0))[0]
        state = {"warm": None}

        def decide(z_cur, step):
            a, mean = self.controller.act(z_cur.unsqueeze(0), z_goal.unsqueeze(0), state["warm"])
            state["warm"] = torch.cat([mean[1:], mean[-1:]], 0)
            return a[: self.active_dof].cpu().numpy()

        m = execute_episode(env, self.model, decide, self.cfg.max_env_steps)
        m.update({"method": self.method_name,
                  "planning_wall_s": time.time() - t0,
                  "planning_flops": int(self.controller.flops)})
        return m
