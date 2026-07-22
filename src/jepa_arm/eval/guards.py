"""Backward-pathway honesty guards (directive §2.2, §2.3.1, §2.3.2, §6.2).

The backward pathway is a HYPOTHESIS, never an oracle. These tests are what would
falsify trust in it:

  action_sensitivity_test  (§2.2)  : F must measurably, monotonically respond to a_t,
                                      else F is rejected as ignoring the action channel.
  policy_confounding_shift (§2.3.1): how much B's predicted predecessors move when the
                                      data-collection policy changes; if > TAU_POLICY,
                                      B is flagged CONFOUNDED and down-weighted.
  invertibility_scores     (§2.3.2): per-transition effective predecessor-mode count;
                                      high-destruction transitions are flagged so B is
                                      not trusted for point-estimate predecessors.
"""
from __future__ import annotations
import numpy as np
import torch

from ..envs.embodiment import CANON_ACT_DIM


def _sample_batch(dataset, n, dev):
    idx = torch.randint(0, len(dataset), (min(n, len(dataset)),))
    o = dataset.obs_t[idx].to(dev)
    a = dataset.act_t[idx].to(dev)
    o2 = dataset.obs_tp1[idx].to(dev)
    return o, a, o2


@torch.no_grad()
def action_sensitivity_test(model, dataset, n: int = 1024, eps_small: float = 0.15,
                            eps_large: float = 0.30) -> dict:
    """Intervene on a_t and measure the change in the forward prediction (§2.2).

    Returns response magnitude in units of latent std, and the fraction of samples whose
    response grows with perturbation size (physically-consistent monotonicity)."""
    dev = model.device
    o, a, _ = _sample_batch(dataset, n, dev)
    z = model.encode(o)
    zstd = z.std(0).mean().clamp_min(1e-6)

    a_zero = torch.zeros_like(a)
    z_stay = model.forward_pred.predict_mean(z, a_zero)     # outcome of "do nothing"
    z_move = model.forward_pred.predict_mean(z, a)          # outcome of the actual action
    drift = (z_stay - z).norm(dim=-1)                       # action-INDEPENDENT drift
    move = (z_move - z).norm(dim=-1)
    action_effect = (z_move - z_stay).norm(dim=-1)          # do(a) vs do(0)
    # action_determinism: fraction of one-step motion actually driven by the action.
    # A forward predictor that IGNORES a_t (§2.2, must be rejected) yields ~0 here.
    action_determinism = (action_effect / (action_effect + drift + 1e-6)).median().item()

    # monotonicity of the intervention: bigger action -> bigger deviation from do-nothing.
    active = (dataset.act_t.abs().sum(0) > 0).float().to(dev)
    g = torch.randn(a.shape[0], CANON_ACT_DIM, device=dev) * active
    g = g / (g.norm(dim=-1, keepdim=True) + 1e-9)

    def resp(eps):
        z1 = model.forward_pred.predict_mean(z, torch.clamp(eps * g, -1.0, 1.0))
        return (z1 - z_stay).norm(dim=-1)

    monotone = (resp(eps_large) > resp(eps_small)).float().mean().item()
    return {
        "action_determinism": float(action_determinism),
        "mean_action_effect": float(action_effect.mean().item()),
        "mean_drift": float(drift.mean().item()),
        "mean_step_motion": float(move.mean().item()),
        "monotone_frac": float(monotone),
        "eps_small": eps_small, "eps_large": eps_large,
        "latent_std": float(zstd.item()),
    }


@torch.no_grad()
def invertibility_scores(model, dataset, n: int = 512, K: int = 32) -> dict:
    """Per-transition effective predecessor-mode count via the participation ratio of the
    predecessor-sample spread spectrum (§2.3.2). Higher => more information destroyed =>
    B must not be trusted as a point estimate for that transition."""
    dev = model.device
    o, a, o2 = _sample_batch(dataset, n, dev)
    z2 = model.encode(o2)
    eff = []
    for i in range(z2.shape[0]):
        samples = model.backward_pred.sample_prior(z2[i:i + 1], a[i:i + 1], K)[0]  # (K,D)
        s = samples - samples.mean(0, keepdim=True)
        cov = (s.T @ s) / (K - 1)
        lam = torch.linalg.eigvalsh(cov).clamp_min(0)
        pr = (lam.sum() ** 2) / (lam.pow(2).sum() + 1e-12)   # participation ratio
        eff.append(float(pr.item()))
    eff = np.array(eff)
    return {
        "effective_modes_mean": float(eff.mean()),
        "effective_modes_median": float(np.median(eff)),
        "effective_modes_p90": float(np.percentile(eff, 90)),
        "high_destruction_frac_ge3": float((eff >= 3.0).mean()),
        "hist": np.histogram(eff, bins=10, range=(1, 11))[0].tolist(),
        "K": K, "n": int(len(eff)),
    }


@torch.no_grad()
def policy_confounding_shift(model, backward_A, backward_B, query_dataset,
                             n: int = 1024) -> dict:
    """Shift of B's predicted predecessors between two data-collection policies (§2.3.1).

    backward_A, backward_B are backward predictors trained on policy-A and policy-B data
    respectively, sharing model's frozen encoder. On a common set of (z_{t+1}, a) queries
    we compare mean predicted predecessors, normalized by the predecessor spread under A.
    """
    dev = model.device
    o, a, o2 = _sample_batch(query_dataset, n, dev)
    z2 = model.encode(o2)
    pa = backward_A.predict_mean(z2, a)
    pb = backward_B.predict_mean(z2, a)
    # normalize by per-dim predecessor spread under policy A
    spread = backward_A.sample_prior(z2[:256], a[:256], 16).reshape(-1, pa.shape[-1])
    scale = spread.std(0).mean().clamp_min(1e-6)
    shift = ((pa - pb).norm(dim=-1) / scale).mean().item()
    return {"policy_shift_normalized": float(shift)}


def evaluate_backward_guards(model, backward_A, backward_B, dataset_neutral,
                             tau_policy: float, K_invert: int = 32) -> dict:
    """Bundle the backward-pathway reports and apply the pre-registered confounding
    flag + down-weight (§2.3.1). Returns a report incl. the trust weight to apply."""
    inv = invertibility_scores(model, dataset_neutral, K=K_invert)
    conf = policy_confounding_shift(model, backward_A, backward_B, dataset_neutral)
    confounded = bool(conf["policy_shift_normalized"] > tau_policy)
    report = {
        "invertibility": inv,
        "policy_confounding": conf,
        "tau_policy": tau_policy,
        "confounded_flag": confounded,
        "backward_trust_weight": 0.25 if confounded else 1.0,
        "review_required": confounded,   # §8.1 halt/review condition
    }
    return report
