# Pre-Registration v2 — Double-Ended JEPA Bridge (iteration 2)

**Status:** FROZEN before the v2 run. Builds on v1 (`PREREGISTRATION.md`) and the v1
results (`FINDINGS.md`). **SIMULATION ONLY** (see `HONESTY.md`).

**Integrity note (directive §5.4):** the H1–H4 numeric thresholds are **unchanged from
v1** — they are NOT retuned after seeing v1's results. v2 changes the *engineering* (fixing
representation/planner/model bugs that hurt all methods equally) and the *task scoping*
(reachable goals), documented below, not the bars a hypothesis must clear.

---

## 1. What v2 changes vs v1, and why (each is a fix to a real defect, not a thumb on the scale)

From `FINDINGS.md`, v1 had three defects. v2 addresses them as follows:

1. **Representation (v1 defect #2).** v1 encoded raw joint angles, so wrapping made the
   latent metric misleading. **v2 uses a sin/cos joint encoding** (`joint_encoding=sincos`,
   obs dim 17→24). Validated: it does not hurt, and it yields correct invertibility.

2. **UR5e/Gen3 near-zero success (v1's headline confound).** Re-diagnosed: the true cause
   was **unreachably far goals** — UR5e/Gen3 have ±6.28 rad / continuous joints, so v1's
   uniform goal sampling produced goals 3–6 rad away that a finite-horizon planner cannot
   reach in the step budget (RRT's 2× path length in v1 was the tell). **v2 bounds goal
   displacement to ≤ 2.0 rad/joint from the start**, so task difficulty is comparable across
   embodiments. Validated: forward-only success on UR5e/Gen3 went from ~0.0 (v1) to
   ~0.73–0.93 (v2). This is applied identically to every method.

3. **Diffuse / confounded backward pathway (v1 defect #3).** **v2 tightens the backward
   CVAE** (annealed KL to `beta_kl_backward=0.5`, smaller mode latent `w_dim=6`). Validated:
   invertibility spread dropped from ~4–5 (v1) to ~1.9–2.3 (v2), high-destruction fraction
   from ~0.9 to ~0.03–0.23.

Also fixed (bugs that unfairly hurt one method): the bridge's waypoint-advance tolerance was
mis-scaled for bounded goals (it got stuck on the first waypoint); v2 scales it by actual
inter-waypoint spacing with a floor plus a per-waypoint step timeout. And latent dim reverted
64 (dim 96 mis-scaled the MPPI temperature/cost).

**Attempted but dropped:** an obstacle-rich task (to defeat the trivial floor, v1 defect #1)
was implemented (`OBSTACLE_V2`, MjSpec injection) and tested. It defeats latent-distance MPPI
for **both** the bridge and forward-only (greedy latent cost → local minima at the obstacle;
fragile arm–obstacle contacts trip the safety watchdog at reset), giving all-zero learned
success. It needs a fundamentally different planner (obstacle-aware cost / much longer
horizon), out of scope for one iteration. v2 therefore keeps the **bounded free-space** task
and reports the strong-floor limitation honestly (H4 remains hard by construction).

## 2. Task and success criterion (unchanged criterion; scoped goals)

- Bounded free-space reaching: start = varied collision-free config; goal within ±2.0
  rad/joint of start, collision-free, EE above the base plane.
- **Success (unchanged): EE within 0.05 m of goal AND settled (joint speed < 0.10 rad/s).**

## 3. Hypotheses & thresholds — IDENTICAL to v1 (frozen)

- **H1** sample efficiency: N_bridge/N_forward ≤ 0.70, CI95↑ < 1.0, on ≥ 2/3 arms.
- **H2** planning compute: bridge FLOPs and wall ≤ 0.70× forward at matched quality.
- **H3** long-horizon: where forward degrades ≥ 20 pp with distance, bridge stays within 10 pp.
- **H4** cross-embodiment: held-out bridge beats the straight-line joint-interpolation floor
  by ≥ 15 pp.

Backward-pathway thresholds unchanged: `TAU_POLICY=0.30`, high-destruction eff-spread ≥ 3.
Determinism: 5 seeds, mean ± sample std. Verdicts emitted mechanically by
`eval/hypotheses.py`. **Any hypothesis may be disconfirmed; that is reported plainly.**

## 4. Scale executed (recorded in each provenance.json; HONESTY.md §3)

5 seeds × 3 arms, per-arm + cross-embodiment; sin/cos; bounded goals δ=2.0; latent 64 /
hidden 384; MPPI 320×18×3; bridge 160 particles / 160 cloud / 8 waypoints; data budgets
{3k, 9k, 18k}; 20 eval tasks/cell; tightened backward CVAE. Reduced from an idealized scale
by single-workstation compute; under-powered hypotheses are reported INCONCLUSIVE, not
confirmed.
