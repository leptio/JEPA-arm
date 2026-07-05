# Double-Ended JEPA Bridging Testbed — Multi-Arm Optimal Motion (Simulation)

A reproducible testbed that trains and evaluates a **double-ended (forward + backward)
action-conditioned JEPA bridging** world model for motion generation across three
heterogeneous robotic arms, and tests — as *falsifiable hypotheses*, not assumptions —
whether the backward pathway and the two-sided bridge actually help.

> **Scope, stated plainly (read `HONESTY.md` first).** This is **simulation only**. The
> governing directive requires three *physical* arms and a *physical* safety envelope; this
> agent has no hardware. Every arm here is a MuJoCo twin. No real-hardware number is
> reported. The directive's own epistemic clauses (§0.3, §5.4, §7.2) forbid fabricating
> results, so we did not stand up a fake hardware run — we built the honest software
> testbed and documented the hardware seam in `docs/HARDWARE_INTERFACE.md`.

## What this implements (directive → code)

| Directive | Where |
|-----------|-------|
| §2.1 Encoder `E`, JEPA latent, **no pixel reconstruction** | `models/world_model.py` (VICReg-JEPA, stop-grad targets) |
| §2.2 Action-conditioned forward `F(z,a,w)` + intervention test | `models/world_model.py`, `eval/guards.py::action_sensitivity_test` |
| §2.3 Backward `B(z',a,w')` as **multi-modal proposal** | `models/world_model.py::CVAEPredictor` |
| §2.3.1 [HARD] policy-confounding guard | `eval/guards.py::policy_confounding_shift`, `configs/experiment/prereg.yaml` (`TAU_POLICY`) |
| §2.3.2 invertibility / info-destruction score | `eval/guards.py::invertibility_scores` |
| §2.4 Two-sided bridge (Schrödinger/IPF), **not interpolation** | `bridge/sampler.py::LatentBridge` (interp is the ABL-interp ablation only) |
| §2.5 Planning-as-inference (path-integral, `do(a)`) | `bridge/planner.py::PathIntegralController` |
| §2.6 [HARD] backward-frontier forward-validation | `bridge/planner.py::forward_reachable`, `BridgePlanner` |
| §2.7 heuristic honesty / optimality gap vs classical | `eval/metrics.py::optimality_gap` |
| §3 safety envelope (software analogue) | `safety.py`, `configs/safety/default.yaml` |
| §4 curriculum, provenance, determinism | `train/train_jepa.py`, `provenance.py`, `seeding.py` |
| §5 hypotheses, baselines, ablations | `configs/experiment/prereg.yaml`, `baselines/`, `experiments/run_experiment.py` |
| §6 metrics | `eval/metrics.py`, `eval/guards.py` |

## Arms (heterogeneous by requirement, §1.2)

| arm | DoF | control | topology note |
|-----|-----|---------|---------------|
| FR3 (Franka) | 7 | position/impedance (stiff) | Franka chain |
| UR5e (Universal) | 6 | **velocity** | industrial serial |
| Kinova Gen3 | 7 | position | continuous wrist joints |

Two control paradigms (position + velocity) span the fleet, per §1.1. Models are loaded
from MuJoCo Menagerie via `robot_descriptions`.

## Quickstart

```bash
# reproducible env (Python 3.12 pinned; torch cu126)
uv venv --python 3.12 .venv
uv pip install torch==2.13.0+cu126 --index-url https://download.pytorch.org/whl/cu126
uv pip install -r requirements.txt

# fast end-to-end smoke (2 seeds, tiny scale)
PYTHONPATH=src python -m jepa_arm.experiments.run_experiment --smoke

# full study (5 seeds x 3 arms, per-arm + cross-embodiment; resumable)
PYTHONPATH=src python -m jepa_arm.experiments.run_experiment --tag full

# build the report from logged results
PYTHONPATH=src python -m jepa_arm.experiments.make_report --tag full
```

Every run writes `provenance.json` (seeds, safety+prereg config hashes, dep versions,
scale) next to its results. A run without a recorded safety config is invalid by
construction (§3.5).

## Hypotheses (frozen in `PREREGISTRATION.md`, evaluated mechanically)

- **H1** sample efficiency (bridge vs forward-only CEM/MPC, interactions-to-threshold)
- **H2** planning compute (FLOPs + wall at matched quality)
- **H3** long-horizon robustness (compounding-error regime)
- **H4** cross-embodiment transfer (held-out arm vs kinematic floor)

Verdicts are emitted as `CONFIRMED / DISCONFIRMED / INCONCLUSIVE` by
`eval/hypotheses.py` reading the frozen thresholds — no post-hoc tuning (§5.4). **Any of
them may be disconfirmed; that is a valid, reported outcome.**

## Layout

```
PREREGISTRATION.md   frozen hypotheses + thresholds (§5.1)
HONESTY.md           sim-only boundary, proxies, caveats (§0.3, §5.4)
configs/             safety/ (versioned §3.5) + experiment/ (frozen thresholds)
src/jepa_arm/
  envs/              MuJoCo arm twins + embodiment registry
  models/            JEPA world model (E, F, B)
  bridge/            two-sided bridge + path-integral planner
  baselines/         CEM/MPC, value-fn, RRT-Connect
  train/             curriculum trainer
  eval/              metrics, guards, mechanical hypothesis eval, executor
  experiments/       orchestrator + report generator
results/             per-run JSON + provenance + report (generated)
```
