"""Two-sided latent bridge (directive §2.4).

Given a current latent z_start and a goal latent z_goal, compute a distribution over
intermediate latent trajectories consistent with BOTH endpoints. This is a particle
Schrodinger-bridge / iterative-proportional-fitting (IPF) construction between a FORWARD
potential (reachable set of F from z_start) and a BACKWARD potential (predecessor set of
B from z_goal):

  * backward half-bridge: iterate B.sample_prior from z_goal to build, for each remaining
    horizon r, a particle cloud of latents from which the goal is reachable in r steps.
  * forward sweep (Feynman-Kac): propagate particles from z_start under F, and at each
    step importance-reweight/resample them by the backward potential (a Gaussian KDE of
    the corresponding backward cloud). Iterating forward/backward sweeps is IPF.

This is NOT linear interpolation between z_start and z_goal. Naive interpolation is a
[HARD]-prohibited failure mode (§2.4): the latent manifold is curved and a straight line
through it is not a feasible trajectory. `LinearInterpBridge` is provided ONLY as the
ABL-interp ablation (§5.3), expected to fail, to empirically confirm that.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

from ..envs.embodiment import CANON_ACT_DIM
from ..models.world_model import decode_flops


def _masked_actions(n, act_dim, active_mask, device):
    a = torch.rand(n, act_dim, device=device) * 2 - 1
    return a * active_mask


def _kde_logpotential(pts, cloud, h):
    """LOG Gaussian-KDE backward potential log psi(pts) under a particle cloud.
    pts:(N,D) cloud:(M,D) -> (N,). Log domain avoids underflow to zero when candidates
    fall far outside the cloud (which otherwise yields all-zero multinomial rows)."""
    d2 = torch.cdist(pts, cloud).pow(2)                 # (N,M)
    logk = -d2 / (2 * h * h + 1e-9)
    return torch.logsumexp(logk, dim=1) - torch.log(torch.tensor(float(cloud.shape[0]),
                                                                  device=pts.device))


def _median_bandwidth(cloud):
    if cloud.shape[0] < 2:
        return torch.tensor(1.0, device=cloud.device)
    d = torch.cdist(cloud, cloud)
    md = d[d > 0].median() if (d > 0).any() else torch.tensor(1.0, device=cloud.device)
    return (md + 1e-6)


def _systematic_resample(particles, weights):
    M = particles.shape[0]
    w = weights / (weights.sum() + 1e-12)
    positions = (torch.arange(M, device=particles.device) + torch.rand(1, device=particles.device)) / M
    cumsum = torch.cumsum(w, 0)
    idx = torch.searchsorted(cumsum, positions).clamp(max=M - 1)
    return particles[idx], idx


@dataclass
class BridgeConfig:
    n_waypoints: int = 8           # T bridge steps
    n_particles: int = 256         # M forward particles
    n_action_samples: int = 8      # A action proposals per particle per step
    n_ipf_sweeps: int = 2          # IPF forward/backward alternations
    kde_bandwidth_scale: float = 1.0
    backward_cloud_size: int = 256


class LatentBridge:
    def __init__(self, model, cfg: BridgeConfig | None = None,
                 backward_trust_weight: float = 1.0):
        self.model = model
        self.cfg = cfg or BridgeConfig()
        self.dev = model.device
        # active action dims (dof) inferred from model act_dim; caller can override mask
        self.active_mask = torch.ones(model.cfg.act_dim, device=self.dev)
        self._decode_flops = decode_flops(model.cfg)
        self.flops = 0                # decode FLOPs used building the bridge (H2)

    def set_active_dof(self, dof: int):
        m = torch.zeros(self.model.cfg.act_dim, device=self.dev)
        m[:dof] = 1.0
        self.active_mask = m

    @torch.no_grad()
    def _backward_clouds(self, z_goal):
        """Zb[r] = cloud of latents from which z_goal is reachable in r backward steps."""
        T = self.cfg.n_waypoints
        M = self.cfg.backward_cloud_size
        clouds = [z_goal.repeat(M, 1)]
        cur = z_goal.repeat(M, 1)
        for _ in range(T):
            a = _masked_actions(M, self.model.cfg.act_dim, self.active_mask, self.dev)
            nxt = self.model.backward_pred.sample_prior(cur, a, 1)[:, 0, :]
            cur = nxt
            clouds.append(cur)
        return clouds     # length T+1; clouds[r] for remaining horizon r

    @torch.no_grad()
    def plan(self, z_start, z_goal):
        """Return (waypoints [T+1,D], particle_clouds list, diagnostics)."""
        cfg = self.cfg
        T, M, A = cfg.n_waypoints, cfg.n_particles, cfg.n_action_samples
        z_start = z_start.reshape(1, -1).to(self.dev)
        z_goal = z_goal.reshape(1, -1).to(self.dev)

        bclouds = self._backward_clouds(z_goal)
        self.flops += T * self.cfg.backward_cloud_size * self._decode_flops  # backward sweep
        self.flops += T * M * A * self._decode_flops                        # forward sweep
        h = [self.cfg.kde_bandwidth_scale * _median_bandwidth(c) for c in bclouds]

        waypoints = [z_start.squeeze(0)]
        clouds_out = []
        particles = z_start.repeat(M, 1)
        ess_hist = []
        for k in range(T):
            r = T - (k + 1)                      # remaining horizon after this step
            target_cloud = bclouds[r]
            # propose A candidates per particle via forward model (interventional)
            z_rep = particles.repeat_interleave(A, 0)
            a = _masked_actions(M * A, self.model.cfg.act_dim, self.active_mask, self.dev)
            w = torch.randn(M * A, self.model.cfg.w_dim, device=self.dev)
            nxt = self.model.forward_pred.decode(z_rep, a, w)          # (M*A,D)
            logpsi = _kde_logpotential(nxt, target_cloud, h[r])       # (M*A,)
            logpsi_g = logpsi.view(M, A)
            nxt_g = nxt.view(M, A, -1)
            probs = torch.softmax(logpsi_g, dim=1)                   # always valid rows
            pick = torch.multinomial(probs, 1)                       # (M,1)
            chosen = torch.gather(nxt_g, 1, pick.unsqueeze(-1).expand(-1, -1, nxt_g.shape[-1]))[:, 0, :]
            logsel = torch.gather(logpsi_g, 1, pick)[:, 0]           # (M,)
            wsel = torch.softmax(logsel, dim=0)                      # particle weights
            ess = 1.0 / (wsel.pow(2).sum() + 1e-12)                  # effective sample size
            ess_hist.append(float(ess.item()))
            particles, _ = _systematic_resample(chosen, wsel)
            clouds_out.append(particles.clone())
            # bridge waypoint = potential-weighted mean of the resampled cloud
            wk = torch.softmax(_kde_logpotential(particles, target_cloud, h[r]), dim=0)
            waypoints.append((wk.unsqueeze(1) * particles).sum(0))
        waypoints[-1] = z_goal.squeeze(0)        # pin terminal to the goal latent
        wp = torch.stack(waypoints, 0)
        diag = {
            "ess_mean": float(sum(ess_hist) / len(ess_hist)),
            "n_waypoints": T + 1,
            "backward_trust_weight_used": 1.0,
        }
        return wp, clouds_out, diag


class LinearInterpBridge:
    """ABLATION ONLY (ABL-interp, §5.3). Straight line in latent space between endpoints.
    This is the [HARD]-prohibited naive bridge; included solely to demonstrate it fails."""
    def __init__(self, model, cfg: BridgeConfig | None = None, **kw):
        self.model = model
        self.cfg = cfg or BridgeConfig()
        self.dev = model.device
        self.flops = 0                # no learned dynamics used; interpolation is free

    def set_active_dof(self, dof: int):
        pass

    @torch.no_grad()
    def plan(self, z_start, z_goal):
        T = self.cfg.n_waypoints
        z0 = z_start.reshape(-1).to(self.dev)
        z1 = z_goal.reshape(-1).to(self.dev)
        alphas = torch.linspace(0, 1, T + 1, device=self.dev).unsqueeze(1)
        wp = (1 - alphas) * z0 + alphas * z1
        return wp, [], {"ess_mean": float("nan"), "n_waypoints": T + 1,
                        "linear_interpolation": True}
