"""Experiment orchestrator (directive §4.3 regimes, §5 hypotheses, §5.3 ablations, §6 metrics).

Produces, with full provenance per run (§7):
  * per-arm regime (§4.3a): data-budget sweep (H1), full method comparison at the top
    budget (H2, optimality gap, smoothness, safety), task-distance bins (H3), and the
    four mandated ablations (§5.3).
  * cross-embodiment regime (§4.3b): shared-latent model trained on a subset of arms,
    evaluated on a held-out arm vs the kinematic floor (H4).
  * backward-pathway guards (§2.3.1, §2.3.2, §6.2): policy-confounding shift and
    invertibility, with the pre-registered TAU_POLICY flag + down-weight.

Every eval uses a FIXED task set per (arm, seed) so all methods are compared paired.
Runs are resumable: a cell whose result JSON already exists is skipped.
Scale is configurable; the exact scale used is recorded in each provenance.json
(HONESTY.md §3).
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import time
import numpy as np
import torch

from ..seeding import seed_everything
from ..provenance import RunContext, write_json
from ..data.collect import collect
from ..data.dataset import TransitionDataset, find_shards
from ..models.world_model import WMConfig, JEPAWorldModel
from ..train.train_jepa import train_world_model
from ..bridge.sampler import LatentBridge, LinearInterpBridge, BridgeConfig
from ..bridge.planner import BridgePlanner, BridgePlannerConfig, MPPIConfig
from ..baselines.cem_mpc import ForwardOnlyMPC, CEMConfig
from ..baselines.value_fn import train_value, ValueGuidedMPC, ValueConfig
from ..baselines.rrt_connect import solve_rrt, RRTConfig
from ..envs.arm_env import ArmEnv
from ..envs.embodiment import ALL_ARMS, get, CANON_ACT_DIM
from ..eval import metrics as M
from ..eval import guards as G
from ..eval import hypotheses as H

SAFETY = "configs/safety/default.yaml"
PREREG = "configs/experiment/prereg.yaml"
EVAL_SEED_BASE = 1_000_000


@dataclass
class Scale:
    seeds: list = field(default_factory=lambda: [0, 1, 2, 3, 4])
    budgets: list = field(default_factory=lambda: [1500, 4500, 9000])  # transitions
    n_eval: int = 12
    epochs_fwd: int = 25
    epochs_bwd: int = 18
    # planner budgets (kept equal across bridge/CEM where possible for fair H2)
    mppi_samples: int = 256
    mppi_horizon: int = 15
    mppi_iters: int = 2
    bridge_particles: int = 128
    bridge_action_samples: int = 6
    bridge_waypoints: int = 8
    bridge_cloud: int = 128
    max_env_steps: int = 250
    latent_dim: int = 64
    hidden: int = 256
    tag: str = "full"


def _mppi(sc: Scale):
    return MPPIConfig(horizon=sc.mppi_horizon, n_samples=sc.mppi_samples, iters=sc.mppi_iters,
                      sigma=0.6, lam=0.2)


def _cem(sc: Scale):
    # CEM optimizer iterations kept close to the bridge's MPPI iters for a fair H2 compute
    # comparison (CEM needs >=1 refinement iter to be a competent baseline; we use +1). The
    # exact per-decision budgets are recorded in provenance so H2 is not misread.
    return CEMConfig(horizon=sc.mppi_horizon, n_samples=sc.mppi_samples,
                     n_elite=max(8, sc.mppi_samples // 8), iters=sc.mppi_iters + 1,
                     max_env_steps=sc.max_env_steps)


def _bcfg(sc: Scale):
    return BridgeConfig(n_waypoints=sc.bridge_waypoints, n_particles=sc.bridge_particles,
                        n_action_samples=sc.bridge_action_samples, backward_cloud_size=sc.bridge_cloud)


# ----------------------------------------------------------------------------- data
def ensure_data(arm: str, seed: int, max_transitions: int, out_dir: str) -> list:
    """Collect (once) policy-A and policy-B data covering max_transitions; return shards."""
    shards = find_shards(out_dir, arm)
    if shards:
        return shards
    horizon = 80
    ep_A = int(np.ceil(0.66 * max_transitions / horizon))
    ep_B = int(np.ceil(0.34 * max_transitions / horizon))
    collect(arm, "policy_A", ep_A, horizon, SAFETY, seed, out_dir)
    collect(arm, "policy_B", ep_B, horizon, SAFETY, seed, out_dir)
    return find_shards(out_dir, arm)


def subset_dataset(shards: list, n: int, seed: int) -> TransitionDataset:
    ds = TransitionDataset(shards)
    if n < len(ds):
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(ds), generator=g)[:n]
        ds.obs_t = ds.obs_t[idx]; ds.act_t = ds.act_t[idx]
        ds.obs_tp1 = ds.obs_tp1[idx]; ds.arm_id = ds.arm_id[idx]; ds.src = ds.src[idx]
    return ds


# ----------------------------------------------------------------------------- eval tasks
def eval_tasks(arm: str, seed: int, n: int) -> list:
    env = ArmEnv(arm, SAFETY, seed=seed)
    tasks = []
    for i in range(n):
        env.reset(seed=EVAL_SEED_BASE + seed * 1000 + i)
        start = env.q.copy(); goal = env.q_goal.copy()
        ee0, _ = env.fk(start); eeg, _ = env.fk(goal)
        tasks.append({"start": start, "goal": goal,
                      "start_goal_ee_dist": float(np.linalg.norm(ee0 - eeg))})
    return tasks


def run_method_on_tasks(make_solver, arm: str, tasks: list, dyn=None) -> list:
    eps = []
    for t in tasks:
        env = ArmEnv(arm, SAFETY, seed=0, dyn=dyn)
        env.reset(start=t["start"], goal_q=t["goal"])
        solver = make_solver(env)
        r = solver(env) if callable(solver) else solver.solve(env)
        r["start_goal_ee_dist"] = t["start_goal_ee_dist"]
        eps.append(r)
    return eps


# ----------------------------------------------------------------------------- method factories
def method_bridge(model, sc, dof, forward_validate=True, interp=False, trust=1.0):
    bcfg = _bcfg(sc)
    bridge = (LinearInterpBridge(model, bcfg) if interp else LatentBridge(model, bcfg,
                                                                          backward_trust_weight=trust))
    pcfg = BridgePlannerConfig(mppi=_mppi(sc), forward_validate=forward_validate,
                               max_env_steps=sc.max_env_steps)
    return lambda env: BridgePlanner(model, bridge, pcfg, active_dof=dof, backward_trust_weight=trust)


def method_cem(model, sc, dof):
    return lambda env: ForwardOnlyMPC(model, _cem(sc), active_dof=dof, method_name="cem_mpc")


def method_value(model, value, sc, dof):
    return lambda env: ValueGuidedMPC(model, value, _cem(sc), active_dof=dof)


def method_rrt(sc):
    return lambda env: (lambda e: solve_rrt(e, seed=0, cfg=RRTConfig(max_env_steps=sc.max_env_steps + 150)))


def kinematic_floor(dof, sc):
    """H4 floor: straight-line joint interpolation start->goal, safety-clamped."""
    from ..baselines.rrt_connect import _execute_modelfree

    def make(env):
        def solver(e):
            start = e.q.copy()
            def decide(z, step):
                alpha = min(1.0, (step + 1) / 120.0)
                target = start + alpha * (e.q_goal - start)
                err = target - e.q
                vmax = e.limits.joint_vel_max[: e.nu] * e.limits.speed_cap_frac
                return np.clip(err / (vmax * e.dt + 1e-9), -1, 1)
            m = _execute_modelfree(e, decide, sc.max_env_steps + 150)
            m["method"] = "kinematic_floor"; m["planning_flops"] = 0; m["planning_wall_s"] = 0.0
            return m
        return solver
    return make


# ----------------------------------------------------------------------------- per-arm regime
def per_arm(arm: str, sc: Scale, root: Path) -> dict:
    dof = get(arm).dof
    out = root / "per_arm" / arm
    out.mkdir(parents=True, exist_ok=True)
    result = {"arm": arm, "dof": dof, "regime": "per_arm", "seeds": {}}

    for seed in sc.seeds:
        cell = out / f"seed{seed}.json"
        if cell.exists():
            result["seeds"][seed] = json.loads(cell.read_text())
            continue
        seed_everything(seed)
        t0 = time.time()
        data_dir = str(out / f"data_seed{seed}")
        shards = ensure_data(arm, seed, max(sc.budgets), data_dir)
        tasks = eval_tasks(arm, seed, sc.n_eval)

        cfg = WMConfig(latent_dim=sc.latent_dim, hidden=sc.hidden)
        # --- H1 data-budget sweep: bridge vs cem at each budget ------------------
        budget_curve = {"bridge": [], "cem_mpc": []}
        top_model = None
        for b in sc.budgets:
            ds = subset_dataset(shards, b, seed)
            mdir = str(out / f"model_seed{seed}_b{b}")
            tr = train_world_model(ds, cfg, out_dir=mdir, seed=seed,
                                   epochs_fwd=sc.epochs_fwd, epochs_bwd=sc.epochs_bwd)
            model = JEPAWorldModel.load(tr["checkpoint"])
            br = run_method_on_tasks(method_bridge(model, sc, dof), arm, tasks)
            ce = run_method_on_tasks(method_cem(model, sc, dof), arm, tasks)
            budget_curve["bridge"].append((b, M.summarize(br)["success_rate"]))
            budget_curve["cem_mpc"].append((b, M.summarize(ce)["success_rate"]))
            if b == max(sc.budgets):
                top_model, top_train = model, tr
                top_bridge, top_cem = br, ce

        # --- noW model (ABL-noW) at top budget -----------------------------------
        ds_top = subset_dataset(shards, max(sc.budgets), seed)
        cfg_noW = WMConfig(latent_dim=sc.latent_dim, hidden=sc.hidden, use_mode_latent=False)
        tr_noW = train_world_model(ds_top, cfg_noW, out_dir=str(out / f"model_noW_seed{seed}"),
                                   seed=seed, epochs_fwd=sc.epochs_fwd, epochs_bwd=sc.epochs_bwd)
        model_noW = JEPAWorldModel.load(tr_noW["checkpoint"])

        # --- value function ------------------------------------------------------
        value = train_value(top_model, ds_top, ValueConfig(), seed=seed)

        # --- full method set at top budget ---------------------------------------
        rrt = run_method_on_tasks(method_rrt(sc), arm, tasks)
        methods_eps = {
            "bridge": top_bridge,
            "cem_mpc": top_cem,                     # == ABL-noB
            "value_fn": run_method_on_tasks(method_value(top_model, value, sc, dof), arm, tasks),
            "rrt": rrt,
            "abl_noFV": run_method_on_tasks(method_bridge(top_model, sc, dof, forward_validate=False), arm, tasks),
            "abl_interp": run_method_on_tasks(method_bridge(top_model, sc, dof, interp=True), arm, tasks),
            "abl_noW": run_method_on_tasks(method_bridge(model_noW, sc, dof), arm, tasks),
        }
        summaries = {k: M.summarize(v) for k, v in methods_eps.items()}
        opt_gaps = {k: M.optimality_gap(v, rrt) for k, v in methods_eps.items() if k != "rrt"}

        # --- H3 distance bins (near/far) for bridge & cem ------------------------
        dists = np.array([t["start_goal_ee_dist"] for t in tasks])
        med = float(np.median(dists))
        def bin_sr(eps, far):
            sel = [e for e, d in zip(eps, dists) if (d >= med) == far]
            return float(np.mean([e["success"] for e in sel])) if sel else float("nan")
        h3 = {"median_dist": med,
              "bridge_near": bin_sr(top_bridge, False), "bridge_far": bin_sr(top_bridge, True),
              "cem_near": bin_sr(top_cem, False), "cem_far": bin_sr(top_cem, True)}

        # --- sim2sim robustness proxy (HONESTY.md §2): perturbed-dynamics eval ---
        from ..envs.arm_env import DynParams
        dyn = DynParams(mass_scale=1.2, damping_scale=1.5, gain_scale=0.8, ctrl_latency_steps=1)
        s2s = M.summarize(run_method_on_tasks(method_bridge(top_model, sc, dof), arm, tasks, dyn=dyn))

        # --- backward guards (§2.3.1, §2.3.2) ------------------------------------
        guards = backward_guards(arm, seed, shards, top_model, cfg, sc, out)

        pr = H.load_prereg()
        cell_res = {
            "seed": seed, "wall_s": round(time.time() - t0, 1),
            "action_sensitivity_gate": top_train["action_sensitivity_gate"],
            "action_sensitivity": top_train["action_sensitivity"],
            "backward_trained": top_train["backward_trained"],
            "budget_curve": budget_curve,
            "summaries": summaries, "optimality_gaps": opt_gaps,
            "h3_bins": h3, "sim2sim_robustness_bridge": s2s,
            "backward_guards": guards,
            "n_eval": sc.n_eval, "budgets": sc.budgets,
        }
        # provenance for this cell
        RunContext(run_id=f"per_arm-{arm}-seed{seed}", seed=seed, arm=arm, regime="per_arm",
                   method="all", behavior_policy="policy_A+policy_B",
                   safety_config_path=SAFETY, prereg_config_path=PREREG,
                   out_dir=str(out / f"prov_seed{seed}"),
                   seed_record=seed_everything(seed),
                   scale_notes=asdict(sc)).write()
        cell.write_text(json.dumps(cell_res, indent=2, default=float))
        result["seeds"][seed] = cell_res
        print(f"[per_arm {arm} seed{seed}] done in {cell_res['wall_s']}s "
              f"bridge_sr={summaries['bridge']['success_rate']:.2f} "
              f"cem_sr={summaries['cem_mpc']['success_rate']:.2f} "
              f"gate={cell_res['action_sensitivity_gate']}")
    (out / "arm_summary.json").write_text(json.dumps(result, indent=2, default=float))
    return result


def backward_guards(arm, seed, shards, model, cfg, sc: Scale, out: Path) -> dict:
    """Train separate backward heads on policy-A-only and policy-B-only data, then measure
    the policy-confounding shift and invertibility (§2.3.1, §2.3.2)."""
    from ..data.dataset import TransitionDataset
    pr = H.load_prereg()["backward_pathway"]
    shards_A = [s for s in shards if "policy_A" in s]
    shards_B = [s for s in shards if "policy_B" in s]
    if not shards_A or not shards_B:
        return {"skipped": "missing policy shard"}
    dsA = TransitionDataset(shards_A); dsB = TransitionDataset(shards_B)
    from ..models.world_model import CVAEPredictor
    dev = model.device

    def fit_backward(ds):
        b = CVAEPredictor(cfg).to(dev)
        opt = torch.optim.Adam(b.parameters(), lr=1e-3)
        loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=True, drop_last=True)
        for _ in range(sc.epochs_bwd):
            for o, a, o2, _ in loader:
                o, a, o2 = o.to(dev), a.to(dev), o2.to(dev)
                with torch.no_grad():
                    z = model.encode(o); z2 = model.encode(o2)
                _, recon, kl = b.elbo(z2, a, z)
                loss = recon + cfg.beta_kl * kl
                opt.zero_grad(); loss.backward(); opt.step()
        b.eval(); return b

    bA = fit_backward(dsA); bB = fit_backward(dsB)
    neutral = TransitionDataset(shards)
    rep = G.evaluate_backward_guards(model, bA, bB, neutral,
                                     tau_policy=pr["tau_policy"], K_invert=pr["invertibility_samples"])
    return rep


# ----------------------------------------------------------------------------- cross-embodiment
def cross_embodiment(sc: Scale, root: Path) -> dict:
    out = root / "cross_embodiment"
    out.mkdir(parents=True, exist_ok=True)
    cfg = WMConfig(latent_dim=sc.latent_dim, hidden=sc.hidden)
    result = {"regime": "cross_embodiment", "held_out": {}}
    for held in ALL_ARMS:
        train_arms = [a for a in ALL_ARMS if a != held]
        heldout_sr = []; floor_sr = []; per_seed = []
        for seed in sc.seeds:
            cell = out / f"{held}_seed{seed}.json"
            if cell.exists():
                c = json.loads(cell.read_text())
            else:
                seed_everything(seed)
                shards = []
                for a in train_arms:
                    d = str(out / f"data_{a}_seed{seed}")
                    shards += ensure_data(a, seed, max(sc.budgets), d)
                ds = TransitionDataset(shards)
                tr = train_world_model(ds, cfg, out_dir=str(out / f"model_holdout_{held}_seed{seed}"),
                                       seed=seed, epochs_fwd=sc.epochs_fwd, epochs_bwd=sc.epochs_bwd)
                model = JEPAWorldModel.load(tr["checkpoint"])
                dof = get(held).dof
                tasks = eval_tasks(held, seed, sc.n_eval)
                br = run_method_on_tasks(method_bridge(model, sc, dof), held, tasks)
                fl = run_method_on_tasks(kinematic_floor(dof, sc), held, tasks)
                c = {"held_out": held, "train_arms": train_arms, "seed": seed,
                     "gate": tr["action_sensitivity_gate"],
                     "heldout_bridge": M.summarize(br), "floor": M.summarize(fl)}
                RunContext(run_id=f"xembod-holdout{held}-seed{seed}", seed=seed, arm=held,
                           regime="cross_embodiment", method="bridge",
                           behavior_policy="policy_A+policy_B(train arms)",
                           safety_config_path=SAFETY, prereg_config_path=PREREG,
                           out_dir=str(out / f"prov_{held}_seed{seed}"),
                           seed_record=seed_everything(seed), scale_notes=asdict(sc),
                           extra={"train_arms": train_arms}).write()
                cell.write_text(json.dumps(c, indent=2, default=float))
                print(f"[xembod holdout={held} seed{seed}] bridge_sr="
                      f"{c['heldout_bridge']['success_rate']:.2f} floor_sr={c['floor']['success_rate']:.2f}")
            heldout_sr.append(c["heldout_bridge"]["success_rate"])
            floor_sr.append(c["floor"]["success_rate"])
            per_seed.append(c)
        result["held_out"][held] = {"train_arms": train_arms,
                                    "heldout_success": heldout_sr, "floor_success": floor_sr,
                                    "per_seed": per_seed}
    (out / "cross_summary.json").write_text(json.dumps(result, indent=2, default=float))
    return result


# ----------------------------------------------------------------------------- aggregation
def _itt(curve, target):
    return M.interactions_to_threshold(curve, target)


def aggregate(per_arm_results: dict, cross: dict, sc: Scale, root: Path) -> dict:
    pr = H.load_prereg()
    target = pr["success_rate_target"]

    # H1: interactions-to-threshold ratio, per arm across seeds
    h1_ratios = {}
    for arm, res in per_arm_results.items():
        ratios = []
        for seed, cell in res["seeds"].items():
            bc = cell["budget_curve"]
            itt_b = _itt([tuple(x) for x in bc["bridge"]], target)
            itt_c = _itt([tuple(x) for x in bc["cem_mpc"]], target)
            if np.isinf(itt_b) and np.isinf(itt_c):
                continue                         # neither reaches target -> inconclusive seed
            if np.isinf(itt_c):
                ratios.append(float("inf"))      # baseline fails, bridge reaches -> bridge win
            elif np.isinf(itt_b):
                ratios.append(2.0)               # bridge fails, baseline reaches -> bridge loss
            else:
                ratios.append(itt_b / itt_c)
        h1_ratios[arm] = [r for r in ratios if not np.isinf(r)] or ratios
    H1 = H.eval_H1(h1_ratios, pr)

    # H2: planning flops/wall ratios (bridge/cem) per arm+seed at top budget
    flops_r, wall_r = [], []
    for arm, res in per_arm_results.items():
        for seed, cell in res["seeds"].items():
            b = cell["summaries"]["bridge"]; c = cell["summaries"]["cem_mpc"]
            if c["mean_planning_flops"] > 0:
                flops_r.append(b["mean_planning_flops"] / c["mean_planning_flops"])
            if c["mean_planning_wall_s"] == c["mean_planning_wall_s"] and c["mean_planning_wall_s"] > 0:
                wall_r.append(b["mean_planning_wall_s"] / c["mean_planning_wall_s"])
    H2 = H.eval_H2(flops_r, wall_r, pr)

    # H3: aggregate near/far success across arms+seeds
    bn, bf, cn, cf = [], [], [], []
    for arm, res in per_arm_results.items():
        for seed, cell in res["seeds"].items():
            h = cell["h3_bins"]
            for lst, key in [(bn, "bridge_near"), (bf, "bridge_far"),
                             (cn, "cem_near"), (cf, "cem_far")]:
                if h[key] == h[key]:
                    lst.append(h[key])
    H3 = H.eval_H3(np.mean(cn), np.mean(cf), np.mean(bn), np.mean(bf), pr)

    # H4: cross-embodiment
    heldout = {a: cross["held_out"][a]["heldout_success"] for a in cross["held_out"]}
    floor = {a: cross["held_out"][a]["floor_success"] for a in cross["held_out"]}
    H4 = H.eval_H4(heldout, floor, pr)

    verdicts = {"H1": H1, "H2": H2, "H3": H3, "H4": H4}
    write_json(root / "hypotheses.json", verdicts)
    return verdicts


def main(sc: Scale | None = None, root_dir: str = "results/study") -> dict:
    sc = sc or Scale()
    root = Path(root_dir) / sc.tag
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "scale.json", asdict(sc))
    t0 = time.time()
    per_arm_results = {arm: per_arm(arm, sc, root) for arm in ALL_ARMS}
    cross = cross_embodiment(sc, root)
    verdicts = aggregate(per_arm_results, cross, sc, root)
    summary = {"tag": sc.tag, "wall_s": round(time.time() - t0, 1),
               "verdicts": {k: v["verdict"] for k, v in verdicts.items()}}
    write_json(root / "run_summary.json", summary)
    print("\n=== HYPOTHESIS VERDICTS ===")
    for k, v in verdicts.items():
        print(f"  {k}: {v['verdict']}")
    print(f"total wall {summary['wall_s']}s")
    return {"per_arm": per_arm_results, "cross": cross, "verdicts": verdicts, "summary": summary}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        sc = Scale(tag="smoke", seeds=[0, 1], budgets=[1000, 3000], n_eval=5,
                   epochs_fwd=10, epochs_bwd=8, mppi_samples=128, mppi_iters=2,
                   bridge_particles=64, bridge_cloud=64, max_env_steps=200)
    else:
        sc = Scale(tag=args.tag)
    main(sc)
