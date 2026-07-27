# HONESTY & LIMITATIONS — read before trusting any number in this repo

This document exists because the governing directive demands, as its **prime
directive** and in §0.3, §5.4, §7.2, and §8.2, that no claim be made that the evidence
does not support, and that negative or impossible results be stated plainly. This file
is the plain statement.

## 1. This is a SIMULATION-ONLY study. No physical robot was involved.

The directive §1 requires **three physical arms** (FR3, UR5e, Kinova Gen3) and §3
requires a **physical safety envelope** (hardware e-stop wired to motor power, enforced
human-exclusion zone, force watchdog on real sensors). This study was run as a software
process with **no attached hardware, no lab, no motor circuit, and no force/presence
sensors**. Therefore:

- **NOT executed (physically impossible here):** §1.1–1.4 real-arm provisioning and
  real action–observation sync; §1.3 *real* sim-to-real gap; §3.1–3.4 physical e-stop,
  human-exclusion zone, real-force watchdog, gated real exploration.
- **What stands in for them:** MuJoCo simulation twins of all three arms (from
  `robot_descriptions` / MuJoCo Menagerie), a *software* safety layer that clamps/rejects
  and logs limit violations exactly as a hardware gate would (`src/jepa_arm/safety.py`),
  and a documented hardware-abstraction boundary (`docs/HARDWARE_INTERFACE.md`) that a
  lab would implement to go to real arms. These are honest simulacra, **not** evidence
  about real hardware.

Any table in the report labeled with a real arm name refers to that arm's **simulated
twin**. We never report a real-hardware success rate, because we never ran on real
hardware. Per directive §8.2 this substitution is stated, not hidden.

## 2. "Sim-to-real gap" (§1.3, §6.1) is reported as a PROXY, clearly labeled.

We cannot measure a real gap without the real side. We instead report a
**dynamics-robustness proxy**: performance degradation when the evaluation twin's
dynamics parameters (mass, friction, damping, actuator gain, control latency) are
perturbed away from the training twin's. This upper-bounds sensitivity to model
mismatch and is the honest, available surrogate. It is labeled `sim2sim_robustness`
everywhere and is **never** called a sim-to-real number.

## 3. Scale caveat.

The run attempts the directive's mandated scale (≥5 seeds, per-arm + cross-embodiment,
all baselines and ablations). Where compute/time forced a reduction (e.g., fewer
planning particles, shorter training), the reduction is recorded in that run's
`provenance.json` under `scale_notes`, and any hypothesis whose statistical power was
thereby weakened is reported as **inconclusive**, not confirmed.

## 4. The backward pathway is treated as a suspect, per directive.

`B` is never trusted as an oracle. Its proposals are (a) validated by forward
reachability under `F` before influencing any command (§2.6), (b) tested for
policy-confounding against a second data-collection policy (§2.3.1), and (c) scored for
per-transition invertibility (§2.3.2). If the confounding threshold `TAU_POLICY` is
exceeded, the code down-weights `B` automatically and the report flags it.

## 5. Standing halt conditions (directive §8.1) in the sim context.

The code halts and surfaces for review if: a safety limit is violated during evaluation
(logged, run flagged); the backward pathway is found confounded beyond `TAU_POLICY`; or
an arm's twin behaves outside its validated envelope (NaN dynamics, solver blow-up).
There is no physical e-stop to trigger; its software analogue is the watchdog in
`safety.py`.

## 6. If a hypothesis fails, the report says so.

There is no configuration of this repo that rewrites §2 thresholds after seeing results.
The evaluation reads thresholds from the frozen `PREREGISTRATION.md`-derived config
(`configs/experiment/prereg.yaml`) and emits `CONFIRMED` / `DISCONFIRMED` /
`INCONCLUSIVE` mechanically.
