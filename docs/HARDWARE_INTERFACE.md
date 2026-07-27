# Hardware Abstraction Boundary — the §1/§3 clauses NOT executed here

This repo is **simulation only** (see `HONESTY.md`). This document specifies the exact
seam a robotics lab would implement to take the trained models to the real FR3 / UR5e /
Kinova Gen3, and the directive clauses that become the lab's responsibility. Nothing in
this file was executed; it is a hand-off contract.

## 1. The seam: `ArmEnv` is the only component that touches "hardware"

Every learning/planning component (encoder, F, B, bridge, planners, baselines) talks to
the world exclusively through the `ArmEnv` API:

| method | sim meaning | real-hardware meaning (lab implements) |
|--------|-------------|----------------------------------------|
| `reset(start, goal_q)` | set MuJoCo state | move to a safe start under supervision; register goal |
| `step(action)` → obs, info | `mj_step` | send one control-period command to the arm driver; read encoders/FT sensor |
| `obs()` | proprio + FK EE pose | read joint encoders; FK or measured EE pose |
| `ee_force_proxy()` | `cfrc_ext` on EE body | **real** wrist force/torque sensor |
| `collision_free(q)` | MuJoCo contact test | offline collision model / planning-scene check |
| `safety.*` | software clamp+watchdog | software clamp **plus** the §3.1 hardware e-stop |

A `RealArmEnv` implementing this same interface (via `franky`/`libfranka` for FR3,
`ur_rtde` for UR5e, `kortex` for Gen3) is a drop-in replacement. **We did not write it**
because we have no hardware; writing untested motor-command code and calling it done would
violate the directive's honesty clauses.

## 2. Directive clauses that become the lab's responsibility

- **§1.1–1.2 embodiments.** Provision the three physical arms. Heterogeneity (7/6/7 DoF,
  position vs velocity control, distinct topology) is already modeled; match it on real HW.
- **§1.3 sim-to-real gap.** Run system identification against each real arm and report the
  gap. Our `sim2sim_robustness` proxy (perturbed-dynamics eval) is only a lower bound on
  the sensitivity you will see; it is **not** a sim-to-real number.
- **§1.4 synchronized logging.** `ArmEnv` already logs timestamped q, qvel, commanded
  action, `qfrc_actuator`, EE force proxy, and EE pose per step, with zero-by-construction
  action→obs sync. On real HW, measure the true sync error and confirm ≤ one control period.
- **§3.1 [HARD] e-stop.** Physically present, tested each session, wired to cut motor power
  independent of software. **Has no software analogue**; `safety.py`'s watchdog is not a
  substitute.
- **§3.2 [HARD] limits.** `configs/safety/default.yaml` already encodes per-arm
  joint/vel/accel/force caps below datasheet maxima, enforced+logged by `safety.py`. Verify
  the numbers against your specific units before energizing.
- **§3.3 [HARD] human-exclusion zone + speed-cap curriculum.** The speed-cap curriculum is
  implemented (`SafetyMonitor.maybe_raise_cap`, raised only after a clean session). The
  physical exclusion zone and presence sensing are the lab's to install and interlock.
- **§3.4 [HARD] sim-first + watchdog.** Exploration runs in sim here (`data/collect.py`).
  The force/velocity/NaN watchdog is implemented; wire it to the real FT sensor and to the
  e-stop.

## 3. Halt conditions (§8.1) already wired

`SafetyMonitor` raises `SafetyHalt` (caught by every executor, which flags the run) on:
anomalous joint velocity, EE-force-proxy exceedance, NaN/inf state, or solver blow-up
(twin outside validated envelope). On real hardware these same triggers must additionally
fire the hardware e-stop. The backward-pathway confounding flag (§2.3.1) is likewise
surfaced as a review-required condition in the guards report.
