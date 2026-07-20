"""Forward model + learned value function baseline (directive §5.2b).

This baseline isolates backward *value* propagation from backward *state* reconstruction:
it learns a goal-conditioned value V(z, z_goal) approximating the negative cost-to-go, by
model-based fitted value iteration IN LATENT SPACE using the frozen forward predictor F
(Bellman backups over sampled actions). It then plans with short-horizon forward shooting
whose terminal cost is -V (so value information "propagates backward" from the goal),
WITHOUT ever reconstructing predecessor states as the double-ended bridge does.

Comparing this against the bridge tells us whether any bridge advantage comes from
backward *state* reconstruction specifically, or merely from propagating value.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
import torch
import torch.nn as nn

from ..models.world_model import mlp, decode_flops
from .cem_mpc import CEMController, CEMConfig


@dataclass
class ValueConfig:
    hidden: int = 256
    gamma: float = 0.97
    fvi_iters: int = 40
    batch: int = 1024
    n_action_samples: int = 16
    success_tol_latent: float = 0.5     # latent radius counted as "at goal"
    lr: float = 1e-3


class GoalValue(nn.Module):
    def __init__(self, latent_dim, hidden):
        super().__init__()
        self.net = mlp([2 * latent_dim, hidden, hidden, 1])

    def forward(self, z, g):
        return self.net(torch.cat([z, g], -1)).squeeze(-1)


def train_value(model, dataset, cfg: ValueConfig, seed: int = 0) -> GoalValue:
    """Model-based fitted value iteration in latent space (backward value propagation)."""
    dev = model.device
    torch.manual_seed(seed)
    V = GoalValue(model.cfg.latent_dim, cfg.hidden).to(dev)
    Vt = GoalValue(model.cfg.latent_dim, cfg.hidden).to(dev)
    Vt.load_state_dict(V.state_dict())
    opt = torch.optim.Adam(V.parameters(), lr=cfg.lr)
    with torch.no_grad():
        allz = model.encode(dataset.obs_t.to(dev))
    N = allz.shape[0]
    for it in range(cfg.fvi_iters):
        idx = torch.randint(0, N, (cfg.batch,), device=dev)
        gidx = torch.randint(0, N, (cfg.batch,), device=dev)
        z = allz[idx]; g = allz[gidx]
        with torch.no_grad():
            # Bellman target: min over sampled actions of step_cost + gamma V_target(F(z,a),g)
            best = None
            for _ in range(cfg.n_action_samples):
                a = (torch.rand(cfg.batch, model.cfg.act_dim, device=dev) * 2 - 1)
                zn = model.forward_pred.predict_mean(z, a)
                at_goal = (zn - g).norm(dim=-1) < cfg.success_tol_latent
                step_cost = torch.ones(cfg.batch, device=dev)      # 1 step
                boot = torch.where(at_goal, torch.zeros_like(step_cost),
                                   cfg.gamma * Vt(zn, g))
                q = step_cost + boot
                best = q if best is None else torch.minimum(best, q)
            target = torch.where((z - g).norm(dim=-1) < cfg.success_tol_latent,
                                 torch.zeros_like(best), best)
        pred = V(z, g)
        loss = ((pred - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 5 == 0:
            Vt.load_state_dict(V.state_dict())
    V.eval()
    return V


class ValueGuidedMPC:
    def __init__(self, model, value: GoalValue, cfg: CEMConfig, active_dof: int,
                 value_weight: float = 1.0):
        self.model = model
        self.value = value
        self.cfg = cfg
        self.controller = CEMController(model, cfg, active_dof)
        self.active_dof = active_dof
        self.vw = value_weight
        self._decode_flops = decode_flops(model.cfg)
        # monkey-patch the terminal cost to include -V (value guidance)
        self._orig_cost = self.controller._cost

        @torch.no_grad()
        def cost_with_value(z0, actions, z_target):
            N, H = actions.shape[0], actions.shape[1]
            z = z0.repeat(N, 1)
            c = torch.zeros(N, device=self.model.device)
            for t in range(H):
                z = self.model.forward_pred.decode(
                    z, actions[:, t], torch.zeros(N, self.model.cfg.w_dim, device=self.model.device))
                c += self.cfg.running_cost * (z - z_target).pow(2).sum(-1)
                c += self.cfg.ctrl_cost * actions[:, t].pow(2).sum(-1)
            c += (z - z_target).pow(2).sum(-1)
            c += self.vw * self.value(z, z_target.repeat(N, 1))   # terminal value-to-go
            self.controller.flops += N * H * self._decode_flops
            return c

        self.controller._cost = cost_with_value

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
        m.update({"method": "value_fn",
                  "planning_wall_s": time.time() - t0,
                  "planning_flops": int(self.controller.flops)})
        return m
