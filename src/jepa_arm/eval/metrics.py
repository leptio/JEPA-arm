"""Metric aggregation (directive §6.1, §6.2). Every number here traces to logged
per-episode records; nothing is computed that a raw log cannot reproduce (§7.2)."""
from __future__ import annotations
import numpy as np


def summarize(episodes: list[dict]) -> dict:
    """Aggregate per-episode records for one (method, arm, seed) cell."""
    n = len(episodes)
    succ = np.array([e["success"] for e in episodes], dtype=float)
    inter_succ = [e["interactions"] for e in episodes if e["success"]]
    halts = sum(int(e.get("safety_halt", False)) for e in episodes)
    clamps = sum(e.get("safety_summary", {}).get("n_clamped", 0) for e in episodes)
    rejects = sum(e.get("safety_summary", {}).get("n_rejected", 0) for e in episodes)
    return {
        "n_episodes": n,
        "success_rate": float(succ.mean()) if n else 0.0,
        "mean_interactions_success": float(np.mean(inter_succ)) if inter_succ else float("nan"),
        "mean_planning_flops": float(np.mean([e.get("planning_flops", 0) for e in episodes])),
        "mean_planning_wall_s": float(np.mean([e.get("planning_wall_s", float("nan")) for e in episodes])),
        "mean_energy": float(np.mean([e["energy"] for e in episodes])),
        "mean_jerk": float(np.mean([e["jerk"] for e in episodes])),
        "mean_ee_path_len": float(np.nanmean([e.get("ee_path_len") or np.nan for e in episodes])),
        "constraint_violations": int(halts),
        "safety_clamps": int(clamps),
        "safety_rejections": int(rejects),
        "waypoint_rejection_frac": float(np.mean(
            [e.get("waypoint_rejection_frac", np.nan) for e in episodes])),
    }


def optimality_gap(method_eps: list[dict], rrt_eps: list[dict]) -> dict:
    """Path optimality gap vs the classical RRT reference (§6.1), paired by task index.
    gap = method_path_len / rrt_path_len - 1, over tasks BOTH solved successfully."""
    gaps = []
    for m, r in zip(method_eps, rrt_eps):
        if m["success"] and r.get("success") and r.get("ee_path_len"):
            gaps.append(m["ee_path_len"] / r["ee_path_len"] - 1.0)
    return {
        "n_paired": len(gaps),
        "mean_optimality_gap": float(np.mean(gaps)) if gaps else float("nan"),
        "median_optimality_gap": float(np.median(gaps)) if gaps else float("nan"),
    }


def interactions_to_threshold(budget_points: list[tuple[int, float]], target: float) -> float:
    """Given [(interactions, success_rate), ...] sorted by interactions, return the
    (linearly interpolated) interaction count at which success_rate first reaches target.
    inf if the target is never reached (H1 falsification handles this)."""
    pts = sorted(budget_points)
    for i, (x, y) in enumerate(pts):
        if y >= target:
            if i == 0:
                return float(x)
            x0, y0 = pts[i - 1]
            if y == y0:
                return float(x)
            frac = (target - y0) / (y - y0)
            return float(x0 + frac * (x - x0))
    return float("inf")


def mean_std(values: list[float]) -> dict:
    v = np.array([x for x in values if x == x], dtype=float)   # drop NaN
    if len(v) == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1) if len(v) > 1 else 0.0),
            "n": int(len(v))}


def ci95_upper(values: list[float]) -> float:
    v = np.array([x for x in values if x == x], dtype=float)
    if len(v) < 2:
        return float("nan")
    return float(v.mean() + 1.96 * v.std(ddof=1) / np.sqrt(len(v)))
