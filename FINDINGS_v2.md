# Findings v2 — Iteration 2, with the v1 confounds removed

Honest interpretation of the v2 study (`results/study/v2/REPORT.md`), and a direct
comparison to v1 (`FINDINGS.md`). Simulation only. Thresholds were frozen before the run
and are identical to v1 (`PREREGISTRATION_v2.md`); verdicts are emitted mechanically.

## Headline

**v2 fixed the v1 methodology defects — and with a sound methodology the double-ended
bridge fails *more* decisively, not less.** All three arms now work, cross-embodiment
transfer is real, backward invertibility is measured correctly — and across every
pre-registered hypothesis the bridge still earns nothing. Verdicts: **H1 DISCONFIRMED
(v1: inconclusive), H2 DISCONFIRMED, H3 INCONCLUSIVE, H4 DISCONFIRMED. None confirmed.**

## Did the v1 fixes work? Yes — as engineering.

| v1 defect (FINDINGS.md) | v2 fix | Did it work? |
|---|---|---|
| Raw joint angles broke the latent metric | sin/cos encoding | Yes (representation sound) |
| UR5e/Gen3 ≈ 0% success | **bounded reachable goals** (δ≤2.0 rad) — the true cause was unreachably-far goals, not encoding | **Yes: forward-only UR5e/Gen3 0.0 → 0.25/0.65** |
| Backward pathway diffuse | tightened CVAE (annealed KL, w_dim 6) | Partial (isolated validation 4→2; at full data ~3.8) |
| Bridge unfairly broken on bounded goals | fixed waypoint-advance (spacing tol + timeout); latent 64 | Yes: bridge 0.13 → competitive |
| Cross-embodiment 0% | (above fixes) | Improved: held-out transfer 0.0 → up to 0.20 |

So the testbed is now genuinely better: **no arm is broken, and the comparison is fair.**

## The verdict on the bridge (the actual question)

Per-arm success, mean ± std over 5 seeds (RRT ref: 0.94–1.00 everywhere):

| Arm | Bridge | Forward-only CEM | Δ |
|-----|--------|------------------|----|
| FR3 | 0.80 ± 0.12 | 0.76 ± 0.10 | **+4 pp** (within noise) |
| UR5e | 0.33 ± 0.12 | 0.25 ± 0.10 | **+8 pp** |
| Gen3 | 0.44 ± 0.08 | 0.65 ± 0.08 | **−21 pp** |

The bridge edges forward-only on two arms (within/near noise) and is clearly **worse on
Gen3**. There is no consistent net advantage; where it loses, it loses bigger than where
it wins.

- **H1 (sample efficiency) — DISCONFIRMED.** Interactions-to-threshold ratio
  N_bridge/N_forward = **1.01 (FR3), 1.00 (UR5e), 1.96 (Gen3)** — the bridge needs *equal or
  more* environment interactions to reach the target, ~2× as many on Gen3. (v1 was
  inconclusive only because arms were broken; with them fixed the answer is a clean no.)
- **H2 (planning compute) — DISCONFIRMED, and worse than v1.** The bridge used **1.24×
  the FLOPs and 1.21× the wall-clock** of forward-only. The backward clouds + forward-
  validation + waypoint tracking are pure overhead here; they buy no success to justify it.
- **H3 (long-horizon) — INCONCLUSIVE.** Forward-only did not degrade with goal distance
  (−0.0 pp), so the compounding-error regime the hypothesis needs never appeared (bounded
  goals + receding-horizon replanning). Untestable here, as pre-registered.
- **H4 (cross-embodiment) — DISCONFIRMED.** Held-out bridge 0.02–0.20 vs the straight-line
  joint-interpolation floor 0.99–1.00. Transfer improved over v1 (esp. UR5e, 0→0.20) but
  the free-space floor is unbeatable.

## The ablations are the clincher

| Arm | Bridge (full) | Linear-interp bridge | No mode-latent w |
|-----|---------------|----------------------|------------------|
| FR3 | 0.80 | **0.79** | 0.72 |
| UR5e | 0.33 | **0.33** | **0.36** |
| Gen3 | 0.44 | **0.46** | **0.48** |

In v1, the proper Schrödinger bridge crushed linear interpolation (0.63 vs 0.17), which we
reported as confirming the curved-manifold argument. **In v2's bounded-goal regime that
gap vanishes: interpolation matches the full bridge, and removing the mode latent `w`
doesn't hurt (helps on UR5e/Gen3).** Interpretation: once goals are close enough for the
forward planner to solve directly, the bridge's entire apparatus — backward predecessor
clouds, two-sided IPF, mode latent, forward-validation — is redundant. The v1 interpolation
gap was an artifact of *far* goals (where a straight latent line overshoots), not evidence
the backward machinery adds planning power.

## Backward pathway (§6.2)

Invertibility spread stayed ~3.8–4.0 at full data (the isolated validation's ~2 did not
hold once trained on the full 18k-transition sets — the backward CVAE still expresses
more predecessor spread than the near-invertible dynamics warrant). Gen3's backward
pathway again exceeded the policy-confounding threshold `TAU_POLICY=0.30` and was
auto-flagged **CONFOUNDED** — and Gen3 is exactly the arm where the bridge hurt most. The
guard predicted the failure.

## Bottom line across two iterations

v1 asked whether the bridge helps and got a noisy "no" confounded by broken arms. v2
removed every confound we could identify — reachable goals, wrapping-robust encoding,
tightened backward model, fair bridge planner, all three arms working — and got a **clean,
better-powered "no."** The double-ended JEPA bridge does not improve sample efficiency,
planning compute, or long-horizon/cross-embodiment behavior over a forward-only planner in
the low-information-destruction reaching domain the directive chose as most favorable to
it; its sophisticated components are empirically redundant with linear interpolation here.

**Where the bridge could still matter is exactly where this planner can't go:** obstacle-
rich / high-information-destruction tasks. We implemented that task (`OBSTACLE_V2`) and found
latent-distance MPPI cannot solve it for *either* method (local minima at the obstacle;
fragile contacts). That — not another free-space iteration — is the real frontier: a
planner that can route around, on which a two-sided bridge finally has something to prove.
