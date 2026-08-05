# Findings & Interpretation

Honest interpretation of the study in `results/study/full/REPORT.md` (all numbers there
trace to logged per-cell JSON + provenance). This file is prose; the report is the data.
Per the directive's prime directive and §5.4, the results are reported as they came out —
including that they are **largely negative for the double-ended bridge**.

## Headline

**In the directive's own deliberately-favorable domain (low-information-destruction
rigid-arm free-space reaching), the double-ended JEPA bridge provided no advantage over a
forward-only planner, and the backward pathway did not earn its keep.** Verdicts on the
frozen thresholds: **H1 inconclusive, H2 disconfirmed, H3 inconclusive, H4 disconfirmed.**
This is stated as evidence against the approach here, not hidden (§5.4).

## What each hypothesis actually showed

- **H1 (sample efficiency) — INCONCLUSIVE, trending negative.** Only FR3 produced a
  measurable interactions-to-threshold ratio, and there it was **1.13** — the bridge
  needed *more* environment interactions than forward-only CEM/MPC to reach the 70% target
  (the opposite of the hypothesized ≤0.70). UR5e and GEN3 never reached the target with
  either method, so no ratio exists. With <2 arms measurable, the pre-registered rule
  returns INCONCLUSIVE; the one measurable arm favors the baseline.

- **H2 (planning compute) — DISCONFIRMED, but directionally cheaper.** The bridge used
  **0.86× the FLOPs and 0.85× the wall-clock** of forward-only shooting (CI upper 0.96 /
  0.95, i.e. reliably below 1.0). So the bridge *is* modestly cheaper — but not by the
  pre-registered ≥30% margin, so the hypothesis as written is disconfirmed. Reporting the
  direction honestly: a ~14–15% compute saving, not the claimed ≥30%.

- **H3 (long-horizon) — INCONCLUSIVE (premise unmet).** Forward-only success degraded only
  **5.6 pp** from near to far goals — far short of the ≥20 pp compounding-error collapse the
  hypothesis presupposes. Receding-horizon replanning kept forward-only stable, so there is
  no long-horizon regime here in which to demonstrate a bridge advantage. Notably the bridge
  degraded *more* (8.9 pp) than forward-only over distance.

- **H4 (cross-embodiment) — DISCONFIRMED, decisively.** Zero-shot to a held-out embodiment,
  the shared-latent bridge reached **0–5%** while the trivial straight-line joint-
  interpolation floor reached **87–100%**. The bridge is ~80–98 pp *below* the floor. The
  learned latent model does not transfer across embodiments here at all.

## Why the results came out this way (honest mechanism analysis)

1. **The task is too easy for a trivial baseline.** Free-space reaching is solved ~100% by
   RRT and ~85–100% by straight-line joint interpolation. Where a naive baseline suffices,
   a learned world model — let alone a two-sided bridge — has no room to add value. This is
   the low-information-destruction domain the directive chose *on purpose*; the bridge’s
   failure to help here is the intended stress test, and it failed it.

2. **Raw joint-angle latents break on large-range / continuous joints.** UR5e (±6.28 rad
   joints) and GEN3 (continuous wrist joints) are near-0% for *all* learned methods, because
   the encoder ingests raw angles: +3 rad and −3 rad look far apart in latent even when
   physically close, so latent-distance MPPI drives large wrapping excursions and never
   settles. RRT, planning in joint space, is unaffected. **This is a representation
   limitation (raw angles vs sin/cos), it hits forward-only and the bridge equally, so the
   bridge-vs-forward comparison stays fair — but it caps absolute performance on 2 of 3
   arms.** A sin/cos joint encoding is the clear next fix.

3. **The backward pathway is diffuse and partly confounded.** The backward predictor’s
   predecessor distributions were flagged high-spread on **77–99%** of transitions even
   though the true dynamics are near-invertible — the CVAE over-expresses uncertainty via
   its mode latent. GEN3’s backward pathway exceeded the pre-registered policy-confounding
   threshold `TAU_POLICY=0.30` and was auto-flagged **CONFOUNDED** (§2.3.1, a §8.1
   review-required condition), triggering its trust down-weight. Forward-validation rejected
   ~12% of backward-proposed waypoints (§2.6). A proposal source this diffuse and this
   policy-sensitive cannot be expected to sharpen planning — consistent with the null result.

## What *did* work (and is worth keeping)

- **The two-sided bridge genuinely beats linear interpolation** (ABL-interp): FR3 63% vs
  **17%**. This confirms §2.4 empirically — the latent manifold is curved and a straight line
  through it is not feasible. The proper bridge is a real object; it just doesn’t beat a
  forward-only planner on this task.
- **Every honesty guard fired as designed:** the action-sensitivity gate passed on all
  models; the confounding guard caught GEN3; forward-validation rejected unreachable
  waypoints; the safety layer logged **141 watchdog halts and 14,853 pre-actuation clamps**
  with zero unsafe commands reaching the (simulated) arm.
- **The forward JEPA world model is sound** on the well-conditioned arm (FR3): latent-vs-
  joint distance correlation ≈0.98, 60–68% reach success with either planner.

## Threats to validity (stated, not buried)

- **Scale reduction (HONESTY.md §3):** run on one workstation — MPPI 256 samples × 15
  horizon × 2 iters, n_eval=12, 3 data budgets. Larger planner budgets would raise absolute
  success but there is no reason to expect them to reverse the bridge-vs-forward ordering.
- **Simulation only.** No real-hardware claim is made (HONESTY.md §1). The `sim2sim`
  robustness proxy is not a sim-to-real number.
- **The negative result is specific to this domain and this representation.** It is evidence
  the bridge does not help *here*; it is not proof it cannot help in high-information-
  destruction or obstacle-rich tasks. The directive’s point stands: since it fails in the
  favorable case, the burden is on the approach to show where it wins.

## Recommended next steps (if continued)

1. sin/cos joint encoding in the observation (fixes UR5e/GEN3 latent metric).
2. An **obstacle-rich** task where straight-line interpolation fails, so planning quality
   actually separates methods — the regime where a bridge could plausibly help.
3. Tighten the backward CVAE (β-schedule / mode-latent regularization) to stop it over-
   expressing predecessor uncertainty; re-test invertibility.
