# Pre-Registration — Double-Ended JEPA Bridge, Multi-Arm Sim Testbed

**Status:** FROZEN before any training/evaluation run. Git-tag `prereg-v1`.
**Scope of this study:** SIMULATION ONLY. See `HONESTY.md` for the hard boundary
between what is executed here and what the directive's §1/§3 hardware clauses require.
All thresholds below are set *now*, before results exist, to satisfy §0.3 and §5.4
(no post-hoc threshold tuning to manufacture confirmation).

---

## 0. Epistemic contract (directive §0.3)

Every claim in the final report carries a triple: **(baseline, seed-count+variance,
falsification test)**. A claim that cannot be paired with a concrete test that *would
have disproven it* is unfalsifiable and MUST NOT appear. This file defines those tests
up front. The backward pathway is a **hypothesis under test**, never an oracle
(directive prime directive, §2.3, §2.6).

Domain rationale (directive §5.4): rigid-arm reaching in free space is deliberately
**low information-destruction** — it is the *favorable* case for backward modeling.
If the bridge fails here, that is evidence **against** the approach and will be reported
as such, not hidden.

---

## 1. Task and success criterion

- **Task:** joint-space reaching. From a randomized feasible start configuration,
  drive the end-effector to a randomized feasible goal pose.
- **Primary success:** end-effector position within **`SUCC_POS_TOL = 0.05 m`** of the
  goal AND terminal joint velocity below **`0.10 rad/s`** (settled), within the horizon
  cap for the regime. Set before runs; frozen.
- **Environment interaction unit:** one `env.step()` (one control period) = one
  interaction, counted for sample-efficiency accounting (H1).

---

## 2. Pre-registered hypotheses and numeric thresholds

Each hypothesis states: the metric, the comparison, the **pre-set numeric threshold**,
and the **falsification test** (the observation that disconfirms it).

### H1 — Sample efficiency
- **Claim:** the double-ended bridge reaches `SR_TARGET = 0.70` success rate using
  **≤ 70%** of the environment interactions that forward-only CEM/MPC needs to reach the
  same `SR_TARGET`, in the same latent space.
- **Metric:** interactions-to-threshold, ratio `N_bridge / N_forward-only`.
- **Falsification:** ratio ≥ 1.0 (bridge needs as many or more interactions), or bridge
  never reaches `SR_TARGET` while the baseline does, on ≥ 2 of 3 arms.
- **Decision rule:** confirmed only if mean ratio ≤ 0.70 with the 95% CI upper bound
  < 1.0 across ≥ 5 seeds, on ≥ 2 of 3 arms.

### H2 — Planning compute
- **Claim:** for solutions of matched quality (path within `+15%` of the same
  optimality gap band), bridge inference uses **≤ 70%** of the wall-clock AND **≤ 70%**
  of the FLOPs-per-decision of forward-only shooting.
- **Metric:** wall-clock/decision and FLOPs/decision at matched success + optimality band.
- **Falsification:** either resource ratio ≥ 1.0 at matched quality.
- **Decision rule:** confirmed only if BOTH ratios' 95% CI upper bounds < 1.0 over ≥ 5 seeds.

### H3 — Long-horizon robustness
- **Claim:** at the horizon `H_long` where forward-only success rate has dropped by
  **≥ 20 percentage points** from its short-horizon value (compounding-error regime),
  the bridge retains success rate within **10 percentage points** of its own
  short-horizon value.
- **Metric:** success-rate vs horizon curves for both methods.
- **Falsification:** bridge degrades by > 10 pp at `H_long`, or forward-only does *not*
  degrade by ≥ 20 pp (in which case H3 is **untestable** on this task and MUST be
  reported as "premise not met — inconclusive", not as confirmation).

### H4 — Cross-embodiment transfer
- **Claim:** a shared-latent bridge trained on a subset of arms and evaluated on a
  **held-out** arm achieves success rate **≥ chance/kinematic floor + 15 pp**, where the
  floor is a straight-line joint-interpolation controller with the same safety clamps.
- **Metric:** held-out success rate minus floor.
- **Falsification:** held-out success ≤ floor + 15 pp, on the held-out arm, over ≥ 5 seeds.

**All four may be disconfirmed. That is an acceptable and reportable outcome (§5.4).**

---

## 3. Mandatory baselines (directive §5.2)

| id | baseline | role |
|----|----------|------|
| B-a | forward-only CEM/MPC in the same latent space | H1/H2/H3 primary comparator |
| B-b | forward model + learned value function (backward *value*, no state reconstruction) | isolates value- vs state-backprop |
| B-c | RRT-Connect classical sampling planner (joint space, MuJoCo collision) | non-learned optimality-gap reference (§6.1) |
| floor | straight-line joint interpolation + safety clamp | H4 kinematic floor |

## 4. Mandatory ablations (directive §5.3)

| id | ablation | expected effect |
|----|----------|-----------------|
| ABL-noB | remove backward predictor | recovers forward-only (sanity: ≈ B-a) |
| ABL-noFV | remove backward-frontier forward-validation (§2.6) | more unreachable waypoints executed → worse/unsafe |
| ABL-noW | remove mode latent `w` | backward loses multimodality → mode collapse |
| ABL-interp | replace bridge sampler with linear latent interpolation (§2.4) | **expected to fail** — confirms curved manifold |

## 5. Determinism & aggregation (directive §4.4)

- Seeds: `{0,1,2,3,4}` minimum (5). Every reported number is mean ± sample std over seeds.
- All seeds recorded per run in `provenance.json`. RNG: Python, NumPy, Torch (+cuda),
  MuJoCo reset seeded.
- A run whose safety config or seed is not logged is **invalid and discarded** (§3.5).

## 6. Backward-pathway pre-registered thresholds (directive §2.3.1, §6.2)

- **Policy-confounding threshold `TAU_POLICY = 0.30`** (normalized): if the mean shift of
  backward-predicted predecessors between behavior-policy A and a deliberately different
  policy B exceeds `0.30` (in units of the predecessor-latent std under A), the backward
  pathway is flagged **CONFOUNDED**, its planning authority is down-weighted (waypoint
  trust weight → 0.25), and this is reported as a §8.1 review-worthy condition.
- **Invertibility:** per-transition effective predecessor-mode count via participation
  ratio of `K=32` backward samples. Transitions with effective modes ≥ 3 are flagged
  high-destruction and excluded from point-estimate predecessor use (§2.3.2).
- **Forward-validation rejection rate** reported as a distribution (§6.2); no threshold —
  descriptive.

## 7. What would make the whole study invalid

- Any reported number not traceable to a logged run (`results/**/metrics.json` +
  `provenance.json`) → the claim is removed (§7.2).
- Any post-hoc change to §2 thresholds → study invalidated; must re-tag prereg.
- Presenting sim numbers as real-hardware numbers → prohibited (see `HONESTY.md`).
