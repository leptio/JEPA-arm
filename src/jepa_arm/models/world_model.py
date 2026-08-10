"""Double-ended action-conditioned JEPA world model (directive §2.1, §2.2, §2.3).

Components
----------
E   : encoder  o_t -> z_t                       (§2.1; NO pixel reconstruction)
E~  : target encoder (EMA of E, stop-grad)      provides JEPA prediction targets
F   : forward predictor  (z_t, a_t, w)  -> z_{t+1}   (§2.2; action channel mandatory)
B   : backward predictor (z_{t+1}, a_t, w') -> z_t   (§2.3; MULTI-MODAL proposal, not a fn)

F and B are conditional VAEs. The latent w (resp. w') "selects among admissible modes"
(directive's w_t / w'_t). At train time an inference network q(w | ...) is used (ELBO);
at test time w is drawn from the prior N(0, I), so B yields a *distribution* over
predecessors whose effective mode count is measurable (§2.3.2).

Collapse of the latent (the trivial JEPA failure where E maps everything to a constant so
F predicts perfectly) is prevented by a VICReg variance+covariance regularizer on z, not
by reconstructing observations (§2.1).
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F_

from ..envs.embodiment import CANON_OBS_DIM, CANON_ACT_DIM


def _mlp_flops(sizes) -> int:
    """FLOPs (2*MACs) of one forward pass of an MLP with the given layer sizes."""
    return sum(2 * sizes[i] * sizes[i + 1] for i in range(len(sizes) - 1))


def decode_flops(cfg) -> int:
    return _mlp_flops([cfg.latent_dim + cfg.act_dim + cfg.w_dim, cfg.hidden, cfg.hidden,
                       cfg.latent_dim])


def encode_flops(cfg) -> int:
    return _mlp_flops([cfg.obs_dim, cfg.hidden, cfg.hidden, cfg.latent_dim])


def mlp(sizes, act=nn.SiLU, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    if out_act is not None:
        layers.append(out_act())
    return nn.Sequential(*layers)


@dataclass
class WMConfig:
    obs_dim: int = CANON_OBS_DIM
    act_dim: int = CANON_ACT_DIM
    latent_dim: int = 64
    hidden: int = 256
    w_dim: int = 8            # mode latent dimensionality (directive w_t / w'_t)
    use_mode_latent: bool = True   # ABL-noW sets this False
    beta_kl: float = 1e-3          # forward-predictor KL weight
    # v2: the backward predictor gets a MUCH stronger (annealed) KL so its posterior tracks
    # the prior; the decoder then ignores w' on near-invertible transitions instead of
    # injecting spurious predecessor spread (v1's diffuse-predecessor pathology, FINDINGS.md).
    beta_kl_backward: float = 1e-3
    vicreg_var: float = 1.0
    vicreg_cov: float = 0.04
    ema_tau: float = 0.01


class Encoder(nn.Module):
    def __init__(self, cfg: WMConfig):
        super().__init__()
        self.net = mlp([cfg.obs_dim, cfg.hidden, cfg.hidden, cfg.latent_dim])

    def forward(self, o):
        return self.net(o)


class CVAEPredictor(nn.Module):
    """Conditional VAE mapping (z_cond, a) -> z_target, with mode latent w.

    Forward predictor:  z_cond=z_t,     z_target=z_{t+1}
    Backward predictor: z_cond=z_{t+1}, z_target=z_t
    """
    def __init__(self, cfg: WMConfig):
        super().__init__()
        self.cfg = cfg
        d, a, w, h = cfg.latent_dim, cfg.act_dim, cfg.w_dim, cfg.hidden
        # inference network q(w | z_cond, a, z_target)
        self.enc = mlp([d + a + d, h, h, 2 * w])
        # decoder z_target_hat = g(z_cond, a, w)   (residual in latent space)
        self.dec = mlp([d + a + w, h, h, d])

    def posterior(self, z_cond, a, z_target):
        mu_logvar = self.enc(torch.cat([z_cond, a, z_target], dim=-1))
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, -8.0, 8.0)
        return mu, logvar

    def decode(self, z_cond, a, w):
        delta = self.dec(torch.cat([z_cond, a, w], dim=-1))
        return z_cond + delta               # residual prediction stabilizes learning

    def sample_w(self, mu, logvar):
        if not self.cfg.use_mode_latent:
            return torch.zeros_like(mu)
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def elbo(self, z_cond, a, z_target):
        mu, logvar = self.posterior(z_cond, a, z_target)
        w = self.sample_w(mu, logvar)
        z_hat = self.decode(z_cond, a, w)
        recon = F_.mse_loss(z_hat, z_target, reduction="none").sum(-1)
        if self.cfg.use_mode_latent:
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)
        else:
            kl = torch.zeros_like(recon)
        return z_hat, recon.mean(), kl.mean()

    @torch.no_grad()
    def sample_prior(self, z_cond, a, n_samples: int = 1):
        """Sample predecessors/successors from the prior over w (test-time multimodality)."""
        B = z_cond.shape[0]
        zc = z_cond.repeat_interleave(n_samples, 0)
        aa = a.repeat_interleave(n_samples, 0)
        if self.cfg.use_mode_latent:
            w = torch.randn(B * n_samples, self.cfg.w_dim, device=z_cond.device)
        else:
            w = torch.zeros(B * n_samples, self.cfg.w_dim, device=z_cond.device)
        z = self.decode(zc, aa, w)
        return z.view(B, n_samples, -1)

    def predict_mean(self, z_cond, a):
        """Deterministic nominal prediction (w = prior mean = 0). Used for planning
        rollouts where we want the expected next latent given an action."""
        w = torch.zeros(z_cond.shape[0], self.cfg.w_dim, device=z_cond.device)
        return self.decode(z_cond, a, w)


def vicreg(z, cfg: WMConfig):
    """Variance + covariance regularization to prevent latent collapse (§2.1)."""
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    var_loss = torch.mean(F_.relu(cfg.vicreg_var - std))
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / (z.shape[0] - 1)
    off = cov - torch.diag(torch.diag(cov))
    cov_loss = (off.pow(2).sum()) / z.shape[1]
    return var_loss, cfg.vicreg_cov * cov_loss


class JEPAWorldModel(nn.Module):
    def __init__(self, cfg: WMConfig, device: str = "cuda"):
        super().__init__()
        self.cfg = cfg
        self.device = device if torch.cuda.is_available() else "cpu"
        self.encoder = Encoder(cfg)
        self.target_encoder = Encoder(cfg)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.forward_pred = CVAEPredictor(cfg)
        self.backward_pred = CVAEPredictor(cfg)
        self.to(self.device)

    def encode(self, o):
        return self.encoder(o)

    @torch.no_grad()
    def encode_target(self, o):
        return self.target_encoder(o)

    @torch.no_grad()
    def update_target(self):
        tau = self.cfg.ema_tau
        for pt, p in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            pt.mul_(1 - tau).add_(tau * p.data)

    def save(self, path):
        torch.save({"cfg": self.cfg.__dict__, "state": self.state_dict()}, path)

    @staticmethod
    def load(path, device="cuda"):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = WMConfig(**ckpt["cfg"])
        m = JEPAWorldModel(cfg, device=device)
        m.load_state_dict(ckpt["state"])
        m.eval()
        return m
