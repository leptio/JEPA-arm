"""Curriculum training of the double-ended JEPA world model (directive §4.2).

Phase 1: train encoder E + forward predictor F (JEPA latent-prediction ELBO + VICReg).
         Target encoder updated by EMA. NO observation reconstruction (§2.1).
Phase 2: ONLY after F passes the action-sensitivity intervention test (§2.2), freeze E
         and train the backward predictor B on the frozen latent space.

The bridge (§2.4) MUST NOT be trained/used until BOTH predictors meet acceptance
criteria; that ordering is enforced by run_experiment.py, which checks the gate flags
this module writes.
"""
from __future__ import annotations
from pathlib import Path
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models.world_model import JEPAWorldModel, WMConfig, vicreg
from ..eval.guards import action_sensitivity_test


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_world_model(dataset, cfg: WMConfig, out_dir: str, seed: int,
                      epochs_fwd: int = 12, epochs_bwd: int = 12,
                      batch_size: int = 512, lr: float = 1e-3,
                      action_sensitivity_min: float = 0.5) -> dict:
    dev = _device()
    torch.manual_seed(seed)
    model = JEPAWorldModel(cfg, device=dev)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # ---- Phase 1: encoder + forward -----------------------------------------
    opt = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.forward_pred.parameters()), lr=lr)
    hist_fwd = []
    t0 = time.time()
    for ep in range(epochs_fwd):
        agg = {"recon": 0.0, "kl": 0.0, "var": 0.0, "cov": 0.0, "n": 0}
        for o, a, o2, _ in dl:
            o, a, o2 = o.to(dev), a.to(dev), o2.to(dev)
            z = model.encode(o)
            # VICReg-JEPA: single encoder, stop-grad target in the SAME latent space, so
            # training and planning-time encoding are consistent (no online/EMA mismatch).
            # Collapse is prevented by the VICReg variance+covariance term, not by an EMA.
            z2_tgt = model.encode(o2).detach()
            _, recon, kl = model.forward_pred.elbo(z, a, z2_tgt)
            var_loss, cov_loss = vicreg(torch.cat([z, model.encode(o2)], 0), cfg)
            loss = recon + cfg.beta_kl * kl + var_loss + cov_loss
            opt.zero_grad(); loss.backward(); opt.step()
            bs = o.shape[0]
            agg["recon"] += recon.item() * bs; agg["kl"] += kl.item() * bs
            agg["var"] += var_loss.item() * bs; agg["cov"] += cov_loss.item() * bs
            agg["n"] += bs
        hist_fwd.append({k: agg[k] / agg["n"] for k in ("recon", "kl", "var", "cov")})

    # ---- Gate: action-sensitivity intervention test (§2.2) -------------------
    sens = action_sensitivity_test(model, dataset, n=1024)
    # Gate on action-determinism (§2.2): F must not ignore a_t. Rejects a predictor whose
    # one-step motion is dominated by action-independent drift.
    gate_passed = bool(sens["action_determinism"] >= action_sensitivity_min
                       and sens["monotone_frac"] >= 0.6)

    result = {
        "seed": seed, "phase1_epochs": epochs_fwd, "history_forward": hist_fwd,
        "action_sensitivity": sens, "action_sensitivity_gate": gate_passed,
        "action_sensitivity_min": action_sensitivity_min,
        "phase1_wall_s": round(time.time() - t0, 2),
    }

    # ---- Phase 2: backward (only if gate passed) -----------------------------
    if gate_passed:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        optb = torch.optim.Adam(model.backward_pred.parameters(), lr=lr)
        hist_bwd = []
        t1 = time.time()
        for ep in range(epochs_bwd):
            agg = {"recon": 0.0, "kl": 0.0, "n": 0}
            for o, a, o2, _ in dl:
                o, a, o2 = o.to(dev), a.to(dev), o2.to(dev)
                with torch.no_grad():
                    z = model.encode(o); z2 = model.encode(o2)
                _, recon, kl = model.backward_pred.elbo(z2, a, z)  # predict z_t from z_{t+1}
                loss = recon + cfg.beta_kl * kl
                optb.zero_grad(); loss.backward(); optb.step()
                bs = o.shape[0]
                agg["recon"] += recon.item() * bs; agg["kl"] += kl.item() * bs
                agg["n"] += bs
            hist_bwd.append({k: agg[k] / agg["n"] for k in ("recon", "kl")})
        result["phase2_epochs"] = epochs_bwd
        result["history_backward"] = hist_bwd
        result["phase2_wall_s"] = round(time.time() - t1, 2)
    else:
        result["phase2_skipped_reason"] = "forward action-sensitivity gate not passed"

    # Keep target_encoder == encoder so encode_target() is consistent for any consumer.
    model.target_encoder.load_state_dict(model.encoder.state_dict())

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "world_model.pt"
    model.save(ckpt)
    result["checkpoint"] = str(ckpt)
    result["backward_trained"] = gate_passed
    (out / "train_result.json").write_text(json.dumps(result, indent=2, default=float))
    return result
