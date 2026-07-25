"""Generate REPORT.md from logged study results (directive §7.1d, §7.2, §6).

Every number traces to a per-cell results JSON + provenance.json. Aggregates over seeds
with mean ± sample std. Robust to partial runs (reports whatever cells exist). Emits the
hypothesis verdicts with the §0.3 triple: baseline, seed count + variance, falsification.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import numpy as np

from ..eval.metrics import mean_std
from ..envs.embodiment import ALL_ARMS
from ..eval import hypotheses as H
from ..experiments.run_experiment import aggregate, Scale


def _load_cells(root: Path):
    per_arm = {}
    for arm in ALL_ARMS:
        d = root / "per_arm" / arm
        seeds = {}
        for f in sorted(d.glob("seed*.json")):
            c = json.loads(f.read_text())
            seeds[c["seed"]] = c
        if seeds:
            per_arm[arm] = {"arm": arm, "seeds": seeds}
    cross = None
    cs = root / "cross_embodiment" / "cross_summary.json"
    if cs.exists():
        cross = json.loads(cs.read_text())
    else:  # reconstruct from cells if summary not written yet
        held = {}
        for f in sorted((root / "cross_embodiment").glob("*_seed*.json")) if (root / "cross_embodiment").exists() else []:
            c = json.loads(f.read_text())
            held.setdefault(c["held_out"], {"train_arms": c["train_arms"],
                                            "heldout_success": [], "floor_success": []})
            held[c["held_out"]]["heldout_success"].append(c["heldout_bridge"]["success_rate"])
            held[c["held_out"]]["floor_success"].append(c["floor"]["success_rate"])
        if held:
            cross = {"held_out": held}
    return per_arm, cross


def _agg_method(per_arm, arm, method, key):
    cells = per_arm[arm]["seeds"].values()
    return mean_std([c["summaries"][method][key] for c in cells if method in c["summaries"]])


METHODS = ["bridge", "cem_mpc", "value_fn", "rrt", "abl_noFV", "abl_interp", "abl_noW"]
METHOD_LABEL = {
    "bridge": "Bridge (double-ended)", "cem_mpc": "Forward-only CEM/MPC (=ABL-noB)",
    "value_fn": "Forward + value fn", "rrt": "RRT-Connect (classical ref)",
    "abl_noFV": "ABL: no forward-validation", "abl_interp": "ABL: linear interp bridge",
    "abl_noW": "ABL: no mode latent w",
}


def _fmt(ms, pct=False, nd=2):
    if ms["n"] == 0 or ms["mean"] != ms["mean"]:
        return "—"
    m, s = ms["mean"], ms["std"]
    if pct:
        return f"{100*m:.0f}±{100*s:.0f}%"
    if abs(m) >= 1e4:
        return f"{m:.2e}±{s:.1e}"
    return f"{m:.{nd}f}±{s:.{nd}f}"


def build(root: Path) -> str:
    per_arm, cross = _load_cells(root)
    scale = json.loads((root / "scale.json").read_text()) if (root / "scale.json").exists() else {}
    L = []
    L.append("# Results — Double-Ended JEPA Bridge, Multi-Arm Sim Testbed\n")
    L.append("> **SIMULATION ONLY.** Every arm is a MuJoCo twin; no real-hardware number "
             "appears here. See `HONESTY.md`. Numbers are mean ± sample std over seeds.\n")
    n_seed = len(next(iter(per_arm.values()))["seeds"]) if per_arm else 0
    L.append(f"**Scale executed:** seeds={scale.get('seeds')} · budgets={scale.get('budgets')} "
             f"· n_eval={scale.get('n_eval')} · MPPI {scale.get('mppi_samples')}smp×"
             f"{scale.get('mppi_horizon')}H×{scale.get('mppi_iters')}it · arms with results: "
             f"{list(per_arm.keys())} ({n_seed} seeds each).\n")

    # ---- Hypotheses -----------------------------------------------------------
    verdicts = None
    if per_arm and cross:
        try:
            sc = Scale(**{k: scale[k] for k in scale if k in Scale.__dataclass_fields__})
            verdicts = aggregate(per_arm, cross, sc, root)
        except Exception as e:
            L.append(f"_(hypothesis aggregation skipped: {e})_\n")
    if (root / "hypotheses.json").exists() and verdicts is None:
        verdicts = json.loads((root / "hypotheses.json").read_text())

    L.append("## Hypothesis verdicts (frozen thresholds, evaluated mechanically)\n")
    if verdicts:
        for k in ["H1", "H2", "H3", "H4"]:
            v = verdicts[k]
            L.append(f"### {k} — **{v['verdict']}**")
            L.append(f"- **Baseline:** {v['baseline']}")
            L.append(f"- **Falsification test:** {v['falsification']}")
            if k == "H1":
                for arm, d in v["per_arm"].items():
                    L.append(f"  - {arm}: mean N_bridge/N_fwd = "
                             f"{d['mean_ratio']:.2f} (±{d['std']:.2f}, n={d['n_seeds']}), "
                             f"CI95↑={d['ci95_upper']:.2f} → {'pass' if d['pass'] else 'fail'}")
            if k == "H2":
                L.append(f"  - FLOPs ratio bridge/fwd = {_fmt(v['flops_ratio'])} (CI95↑ {v['flops_ci95_upper']:.2f})")
                L.append(f"  - Wall ratio bridge/fwd  = {_fmt(v['wallclock_ratio'])} (CI95↑ {v['wallclock_ci95_upper']:.2f})")
            if k == "H3":
                L.append(f"  - forward degradation near→far = {v['forward_degradation_pp']:.1f} pp "
                         f"(premise met: {v['premise_met']}); bridge degradation = {v['bridge_degradation_pp']:.1f} pp")
            if k == "H4":
                for arm, d in v["per_arm"].items():
                    L.append(f"  - held-out {arm}: bridge {100*d['heldout_success']['mean']:.0f}% "
                             f"vs floor {100*d['floor_success']['mean']:.0f}% → margin {100*d['margin']:.0f}pp "
                             f"({'pass' if d['pass'] else 'fail'})")
            L.append("")
    else:
        L.append("_(insufficient cells for verdicts yet)_\n")

    # ---- Per-arm method comparison -------------------------------------------
    L.append("## Per-arm method comparison (success rate, mean±std over seeds)\n")
    L.append("| method | " + " | ".join(per_arm.keys()) + " |")
    L.append("|" + "---|" * (len(per_arm) + 1))
    for m in METHODS:
        row = [METHOD_LABEL[m]]
        for arm in per_arm:
            row.append(_fmt(_agg_method(per_arm, arm, m, "success_rate"), pct=True))
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    # ---- Planning compute (H2 detail) ----------------------------------------
    L.append("## Planning compute & quality (top budget)\n")
    L.append("| arm | method | success | interactions(succ) | FLOPs/ep | wall/ep (s) | opt.gap vs RRT | jerk |")
    L.append("|---|---|---|---|---|---|---|---|")
    for arm in per_arm:
        for m in ["bridge", "cem_mpc", "value_fn", "rrt"]:
            og = mean_std([c["optimality_gaps"][m]["mean_optimality_gap"]
                           for c in per_arm[arm]["seeds"].values()
                           if m in c.get("optimality_gaps", {})]) if m != "rrt" else {"mean": 0.0, "std": 0.0, "n": 1}
            L.append(f"| {arm} | {m} | {_fmt(_agg_method(per_arm,arm,m,'success_rate'),pct=True)} "
                     f"| {_fmt(_agg_method(per_arm,arm,m,'mean_interactions_success'),nd=0)} "
                     f"| {_fmt(_agg_method(per_arm,arm,m,'mean_planning_flops'))} "
                     f"| {_fmt(_agg_method(per_arm,arm,m,'mean_planning_wall_s'),nd=2)} "
                     f"| {_fmt(og,nd=2)} | {_fmt(_agg_method(per_arm,arm,m,'mean_jerk'),nd=0)} |")
    L.append("")

    # ---- Ablations ------------------------------------------------------------
    L.append("## Ablations (§5.3) — success rate vs full bridge\n")
    L.append("| arm | bridge(full) | noB(=CEM) | no-fwd-valid | linear-interp | no-w |")
    L.append("|---|---|---|---|---|---|")
    for arm in per_arm:
        L.append(f"| {arm} "
                 f"| {_fmt(_agg_method(per_arm,arm,'bridge','success_rate'),pct=True)} "
                 f"| {_fmt(_agg_method(per_arm,arm,'cem_mpc','success_rate'),pct=True)} "
                 f"| {_fmt(_agg_method(per_arm,arm,'abl_noFV','success_rate'),pct=True)} "
                 f"| {_fmt(_agg_method(per_arm,arm,'abl_interp','success_rate'),pct=True)} "
                 f"| {_fmt(_agg_method(per_arm,arm,'abl_noW','success_rate'),pct=True)} |")
    L.append("\n_ABL-interp is the [HARD]-prohibited linear-interpolation bridge (§2.4), "
             "included only to confirm it underperforms the proper two-sided bridge._\n")

    # ---- Backward pathway report (§6.2) --------------------------------------
    L.append("## Backward-pathway report (§6.2)\n")
    L.append("| arm | policy-shift (τ=0.30) | confounded? | eff. predecessor spread | high-destruction frac | waypoints rejected by fwd-valid |")
    L.append("|---|---|---|---|---|---|")
    for arm in per_arm:
        cells = list(per_arm[arm]["seeds"].values())
        shift = mean_std([c["backward_guards"]["policy_confounding"]["policy_shift_normalized"]
                          for c in cells if "backward_guards" in c and "policy_confounding" in c["backward_guards"]])
        conf = any(c["backward_guards"].get("confounded_flag") for c in cells if "backward_guards" in c)
        eff = mean_std([c["backward_guards"]["invertibility"]["effective_modes_mean"]
                        for c in cells if "backward_guards" in c and "invertibility" in c["backward_guards"]])
        hd = mean_std([c["backward_guards"]["invertibility"]["high_destruction_frac_ge3"]
                       for c in cells if "backward_guards" in c and "invertibility" in c["backward_guards"]])
        rej = mean_std([c["summaries"]["bridge"]["waypoint_rejection_frac"] for c in cells])
        L.append(f"| {arm} | {_fmt(shift,nd=3)} | {'YES ⚠' if conf else 'no'} "
                 f"| {_fmt(eff,nd=2)} | {_fmt(hd,pct=True)} | {_fmt(rej,pct=True)} |")
    L.append("\n_'Eff. predecessor spread' is a participation-ratio proxy for how diffuse "
             "B's predecessor distribution is (higher = more information destroyed / less "
             "point-invertible), per §2.3.2. It is a spread measure, not a literal mode count._\n")

    # ---- sim2sim robustness proxy --------------------------------------------
    L.append("## Sim-to-sim robustness proxy (NOT a sim-to-real number; HONESTY.md §2)\n")
    L.append("| arm | bridge SR (nominal) | bridge SR (perturbed dynamics) |")
    L.append("|---|---|---|")
    for arm in per_arm:
        nom = _agg_method(per_arm, arm, "bridge", "success_rate")
        s2s = mean_std([c["sim2sim_robustness_bridge"]["success_rate"]
                        for c in per_arm[arm]["seeds"].values() if "sim2sim_robustness_bridge" in c])
        L.append(f"| {arm} | {_fmt(nom,pct=True)} | {_fmt(s2s,pct=True)} |")
    L.append("")

    # ---- safety --------------------------------------------------------------
    L.append("## Safety events (§3.2, §8.1)\n")
    tot_halt = sum(c["summaries"][m]["constraint_violations"]
                   for arm in per_arm for c in per_arm[arm]["seeds"].values() for m in c["summaries"])
    tot_clamp = sum(c["summaries"][m]["safety_clamps"]
                    for arm in per_arm for c in per_arm[arm]["seeds"].values() for m in c["summaries"])
    L.append(f"- Safety-halts (watchdog trips) across all logged eval episodes: **{tot_halt}**")
    L.append(f"- Safety clamps (commanded action limited before actuation): **{tot_clamp}**\n")

    L.append("---\n_Generated by `make_report.py` from logged results; every figure/number "
             "traces to a per-cell JSON + provenance.json (§7.2)._")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--root", default="results/study")
    args = ap.parse_args()
    root = Path(args.root) / args.tag
    md = build(root)
    out = root / "REPORT.md"
    out.write_text(md, encoding="utf-8")
    print(f"wrote {out} ({len(md)} chars)")


if __name__ == "__main__":
    main()
