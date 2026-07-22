"""Mechanical hypothesis evaluation (directive §5.1, §5.4, §0.3).

Reads the FROZEN thresholds from configs/experiment/prereg.yaml and emits
CONFIRMED / DISCONFIRMED / INCONCLUSIVE for H1-H4 with no human judgement in the loop,
so thresholds cannot be silently retuned after seeing results (§5.4). Each verdict
carries the baseline compared against, the seed count + variance, and the falsification
condition that would have disproven it (§0.3).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import yaml

from .metrics import mean_std, ci95_upper


def load_prereg(path: str = "configs/experiment/prereg.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def _verdict(confirmed: bool, inconclusive: bool = False) -> str:
    if inconclusive:
        return "INCONCLUSIVE"
    return "CONFIRMED" if confirmed else "DISCONFIRMED"


def eval_H1(interactions_ratio_per_arm: dict, pr: dict) -> dict:
    """H1 sample efficiency: N_bridge/N_forward-only <= threshold, CI upper < 1, on >=k arms."""
    cfg = pr["hypotheses"]["H1_sample_efficiency"]
    thr = cfg["interaction_ratio_max"]
    per_arm_pass = {}
    ratios = []
    for arm, seed_ratios in interactions_ratio_per_arm.items():
        ms = mean_std(seed_ratios)
        up = ci95_upper(seed_ratios)
        ratios += [r for r in seed_ratios if r == r]
        per_arm_pass[arm] = {
            "mean_ratio": ms["mean"], "std": ms["std"], "n_seeds": ms["n"],
            "ci95_upper": up,
            "pass": bool(ms["mean"] <= thr and (up < cfg["require_ci_upper_below"]
                                                if up == up else False)),
        }
    n_pass = sum(1 for v in per_arm_pass.values() if v["pass"])
    n_measurable = sum(1 for v in per_arm_pass.values() if v["n_seeds"] > 0)
    confirmed = n_pass >= cfg["min_arms_pass"]
    # If too few arms even produced a measurable interactions-to-threshold ratio (i.e.,
    # neither method reached SR_TARGET), H1 is not testable -> INCONCLUSIVE, not disproven.
    inconclusive = n_measurable < cfg["min_arms_pass"]
    return {
        "hypothesis": "H1_sample_efficiency",
        "baseline": "forward-only CEM/MPC (same latent space)",
        "threshold": thr, "min_arms_pass": cfg["min_arms_pass"],
        "per_arm": per_arm_pass, "n_arms_pass": n_pass, "n_arms_measurable": n_measurable,
        "verdict": _verdict(confirmed, inconclusive=inconclusive),
        "falsification": "ratio >= 1.0 or bridge never reaches SR_TARGET while baseline does, on >=2/3 arms",
    }


def eval_H2(flops_ratios: list[float], wall_ratios: list[float], pr: dict) -> dict:
    cfg = pr["hypotheses"]["H2_planning_compute"]
    f = mean_std(flops_ratios); w = mean_std(wall_ratios)
    fu = ci95_upper(flops_ratios); wu = ci95_upper(wall_ratios)
    confirmed = bool(f["mean"] <= cfg["flops_ratio_max"] and w["mean"] <= cfg["wallclock_ratio_max"]
                     and (fu < cfg["require_ci_upper_below"] if fu == fu else False)
                     and (wu < cfg["require_ci_upper_below"] if wu == wu else False))
    return {
        "hypothesis": "H2_planning_compute",
        "baseline": "forward-only shooting (CEM/MPC)",
        "flops_ratio": f, "flops_ci95_upper": fu,
        "wallclock_ratio": w, "wallclock_ci95_upper": wu,
        "threshold": {"flops": cfg["flops_ratio_max"], "wallclock": cfg["wallclock_ratio_max"]},
        "verdict": _verdict(confirmed),
        "falsification": "either resource ratio >= 1.0 at matched quality",
    }


def eval_H3(fwd_near: float, fwd_far: float, bridge_near: float, bridge_far: float,
            pr: dict) -> dict:
    cfg = pr["hypotheses"]["H3_long_horizon"]
    fwd_degr = (fwd_near - fwd_far) * 100.0
    bridge_degr = (bridge_near - bridge_far) * 100.0
    premise_met = fwd_degr >= cfg["forward_degradation_pp_min"]
    confirmed = bool(premise_met and bridge_degr <= cfg["bridge_max_degradation_pp"])
    return {
        "hypothesis": "H3_long_horizon",
        "baseline": "forward-only CEM/MPC vs task-distance (compounding-error) bins",
        "forward_degradation_pp": float(fwd_degr),
        "bridge_degradation_pp": float(bridge_degr),
        "premise_met": bool(premise_met),
        "threshold": cfg,
        "verdict": _verdict(confirmed, inconclusive=not premise_met),
        "falsification": "bridge degrades >10pp at long horizon, OR forward does not degrade >=20pp (premise unmet -> inconclusive)",
    }


def eval_H4(heldout_success_per_arm: dict, floor_success_per_arm: dict, pr: dict) -> dict:
    cfg = pr["hypotheses"]["H4_cross_embodiment"]
    margin_thr = cfg["margin_over_floor_pp"] / 100.0
    per_arm = {}
    for arm in heldout_success_per_arm:
        h = mean_std(heldout_success_per_arm[arm])
        fl = mean_std(floor_success_per_arm.get(arm, [0.0]))
        margin = h["mean"] - fl["mean"]
        per_arm[arm] = {"heldout_success": h, "floor_success": fl,
                        "margin": float(margin), "pass": bool(margin >= margin_thr)}
    confirmed = all(v["pass"] for v in per_arm.values()) and len(per_arm) > 0
    return {
        "hypothesis": "H4_cross_embodiment",
        "baseline": "straight-line joint-interpolation kinematic floor",
        "threshold_margin_pp": cfg["margin_over_floor_pp"],
        "per_arm": per_arm,
        "verdict": _verdict(confirmed),
        "falsification": "held-out success <= floor + 15pp on the held-out arm",
    }
