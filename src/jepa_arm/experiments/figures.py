"""Publication figures from logged study results (directive §7.1e: raw logs reproduce
every figure). Static light-mode PNGs. Colors use the validated colorblind-safe
categorical palette; hue follows the METHOD (entity), assigned in fixed order, never
cycled. One y-axis per panel, recessive grid, legend present, selective direct labels.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..eval.metrics import mean_std
from ..envs.embodiment import ALL_ARMS

# validated categorical palette (light), fixed order; hue follows the entity (method)
PAL = {"bridge": "#2a78d6", "cem_mpc": "#eb6834", "value_fn": "#1baf7a",
       "rrt": "#898781", "abl_noFV": "#eda100", "abl_interp": "#e34948", "abl_noW": "#4a3aa7"}
LABEL = {"bridge": "Bridge", "cem_mpc": "Fwd-only CEM", "value_fn": "Fwd+value",
         "rrt": "RRT (ref)", "abl_noFV": "no-fwd-valid", "abl_interp": "interp", "abl_noW": "no-w"}
INK, SEC, MUT, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURF)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=MUT, labelsize=9)
    ax.title.set_color(INK); ax.yaxis.label.set_color(SEC); ax.xaxis.label.set_color(SEC)


def _load(root: Path):
    per_arm = {}
    for arm in ALL_ARMS:
        d = root / "per_arm" / arm
        seeds = [json.loads(f.read_text()) for f in sorted(d.glob("seed*.json"))]
        if seeds:
            per_arm[arm] = seeds
    return per_arm


def fig_success_by_arm(per_arm, out: Path):
    methods = ["bridge", "cem_mpc", "value_fn", "rrt"]
    arms = list(per_arm)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    _style(ax)
    w = 0.2
    x = np.arange(len(arms))
    for i, m in enumerate(methods):
        means = [mean_std([c["summaries"][m]["success_rate"] for c in per_arm[a]])["mean"] for a in arms]
        errs = [mean_std([c["summaries"][m]["success_rate"] for c in per_arm[a]])["std"] for a in arms]
        bars = ax.bar(x + (i - 1.5) * w, np.array(means) * 100, w * 0.92, yerr=np.array(errs) * 100,
                      color=PAL[m], label=LABEL[m], zorder=3, capsize=2,
                      error_kw=dict(ecolor=MUT, lw=1))
        for b, mn in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{100*mn:.0f}",
                    ha="center", va="bottom", fontsize=7.5, color=SEC)
    ax.set_xticks(x); ax.set_xticklabels([a.upper() for a in arms])
    ax.set_ylabel("task success rate (%)"); ax.set_ylim(0, 105)
    ax.set_title("Success rate by arm and method (mean±std over seeds)")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight", facecolor=SURF); plt.close(fig)


def fig_h1_budget(per_arm, out: Path):
    arms = list(per_arm)
    fig, axes = plt.subplots(1, len(arms), figsize=(4 * len(arms), 3.6), dpi=150, sharey=True)
    if len(arms) == 1:
        axes = [axes]
    for ax, arm in zip(axes, arms):
        _style(ax)
        for m in ["bridge", "cem_mpc"]:
            curves = [c["budget_curve"][m] for c in per_arm[arm]]
            budgets = sorted({b for cv in curves for b, _ in cv})
            means, errs = [], []
            for bud in budgets:
                vals = [s for cv in curves for b, s in cv if b == bud]
                ms = mean_std(vals); means.append(ms["mean"] * 100); errs.append(ms["std"] * 100)
            ax.errorbar(budgets, means, yerr=errs, color=PAL[m], marker="o", ms=5, lw=2,
                        capsize=2, label=LABEL[m], zorder=3)
        ax.axhline(70, color=BASE, ls="--", lw=1, zorder=1)
        ax.text(budgets[0], 71, "SR target 70%", fontsize=7, color=MUT)
        ax.set_title(arm.upper()); ax.set_xlabel("env interactions (transitions)")
    axes[0].set_ylabel("success rate (%)"); axes[0].set_ylim(0, 105)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("H1: sample-efficiency curves — bridge vs forward-only", color=INK, y=1.02)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight", facecolor=SURF); plt.close(fig)


def fig_compute_tradeoff(per_arm, out: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=150)
    _style(ax)
    for m in ["bridge", "cem_mpc", "value_fn"]:
        xs, ys = [], []
        for arm in per_arm:
            f = mean_std([c["summaries"][m]["mean_planning_flops"] for c in per_arm[arm]])["mean"]
            sr = mean_std([c["summaries"][m]["success_rate"] for c in per_arm[arm]])["mean"]
            xs.append(f); ys.append(sr * 100)
        ax.scatter(xs, ys, s=70, color=PAL[m], label=LABEL[m], zorder=3, edgecolor=SURF, lw=1.5)
    ax.set_xlabel("planning FLOPs per decision episode"); ax.set_ylabel("success rate (%)")
    ax.set_title("H2: planning compute vs success (per arm)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight", facecolor=SURF); plt.close(fig)


def fig_backward(per_arm, out: Path):
    arms = list(per_arm)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    _style(ax)
    x = np.arange(len(arms)); w = 0.38
    shifts = [mean_std([c["backward_guards"]["policy_confounding"]["policy_shift_normalized"]
                        for c in per_arm[a] if "backward_guards" in c])["mean"] for a in arms]
    rej = [mean_std([c["summaries"]["bridge"]["waypoint_rejection_frac"] for c in per_arm[a]])["mean"] for a in arms]
    ax.bar(x - w/2, shifts, w, color=PAL["abl_interp"], label="policy-confounding shift", zorder=3)
    ax.bar(x + w/2, rej, w, color=PAL["bridge"], label="waypoint rejection frac (fwd-valid)", zorder=3)
    ax.axhline(0.30, color=BASE, ls="--", lw=1)
    ax.text(-0.4, 0.31, "τ_policy = 0.30", fontsize=7, color=MUT)
    ax.set_xticks(x); ax.set_xticklabels([a.upper() for a in arms])
    ax.set_ylabel("normalized value"); ax.set_title("Backward-pathway guards (§2.3.1, §2.6)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight", facecolor=SURF); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--root", default="results/study")
    args = ap.parse_args()
    root = Path(args.root) / args.tag
    per_arm = _load(root)
    if not per_arm:
        print("no per-arm cells yet"); return
    figs = root / "figures"; figs.mkdir(exist_ok=True)
    fig_success_by_arm(per_arm, figs / "success_by_arm.png")
    fig_h1_budget(per_arm, figs / "h1_budget_curves.png")
    fig_compute_tradeoff(per_arm, figs / "h2_compute_tradeoff.png")
    fig_backward(per_arm, figs / "backward_guards.png")
    print(f"wrote figures to {figs}")


if __name__ == "__main__":
    main()
