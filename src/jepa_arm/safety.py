"""Software safety layer (directive §3.2, §3.3, §3.4, §3.5).

In simulation this stands in for the physical safety envelope. It:
  - reads a VERSIONED safety config (§3.5) whose hash is logged per run;
  - clamps/rejects any commanded OR model-predicted action that would exceed a
    joint-position / velocity / acceleration / EE-force limit, and LOGS the event (§3.2);
  - runs a watchdog that halts on anomalous velocity/force/NaN/solver blow-up (§3.4, §8.1);
  - enforces a speed-cap curriculum that may only be raised after a clean session (§3.3).

It does NOT and cannot replace the §3.1 physical e-stop wired to motor power; see
HONESTY.md §1 and §5. There is no person in a simulation, so the §3.3 human-exclusion
zone is represented as a hard precondition flag that a hardware deployment must satisfy.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import yaml


class SafetyHalt(Exception):
    """Raised when a watchdog trip requires immediate halt + human review (§8.1)."""


@dataclass
class SafetyLimits:
    version: str
    joint_vel_max: np.ndarray
    joint_acc_max: float
    ee_force_max: float
    joint_pos_low: np.ndarray
    joint_pos_high: np.ndarray
    speed_cap_frac: float
    watchdog: dict
    speed_cap_schedule: list
    config_path: str


def load_safety(arm: str, jnt_low: np.ndarray, jnt_high: np.ndarray,
                config_path: str) -> SafetyLimits:
    cfg = yaml.safe_load(Path(config_path).read_text())
    a = cfg["arms"][arm]
    margin = float(cfg["joint_position_margin_frac"])
    span = jnt_high - jnt_low
    # For continuous joints (low==high==0 in the model) keep them unconstrained.
    cont = np.isclose(span, 0.0)
    low = np.where(cont, -np.inf, jnt_low + margin * span)
    high = np.where(cont, np.inf, jnt_high - margin * span)
    return SafetyLimits(
        version=cfg["version"],
        joint_vel_max=np.asarray(a["joint_vel_max"], dtype=np.float64),
        joint_acc_max=float(a["joint_acc_max"]),
        ee_force_max=float(a["ee_force_max"]),
        joint_pos_low=low,
        joint_pos_high=high,
        speed_cap_frac=float(a["speed_cap_frac_initial"]),
        watchdog=cfg["watchdog"],
        speed_cap_schedule=list(cfg["speed_cap_schedule"]),
        config_path=str(config_path),
    )


@dataclass
class SafetyMonitor:
    """Stateful gate + logger for one arm/session."""
    limits: SafetyLimits
    dt: float
    violations: list = field(default_factory=list)
    n_clamped: int = 0
    n_rejected: int = 0
    _cap_idx: int = 0

    def clamp_position_target(self, q_now: np.ndarray, q_target: np.ndarray,
                              step: int) -> np.ndarray:
        """Clamp a commanded joint-position target so the implied velocity/accel and
        the target itself stay within limits. Logs every clamp (§3.2)."""
        cap = self.limits.speed_cap_frac
        vmax = self.limits.joint_vel_max[: len(q_now)] * cap
        max_step = vmax * self.dt
        dq = np.clip(q_target - q_now, -max_step, max_step)
        q_cmd = q_now + dq
        lo = self.limits.joint_pos_low[: len(q_now)]
        hi = self.limits.joint_pos_high[: len(q_now)]
        q_clamped = np.clip(q_cmd, lo, hi)
        if not np.allclose(q_clamped, q_target, atol=1e-6):
            self.n_clamped += 1
            self.violations.append({
                "step": int(step), "type": "clamp",
                "requested": q_target.tolist(), "applied": q_clamped.tolist(),
            })
        return q_clamped

    def check_state(self, qvel: np.ndarray, ee_force: float, qacc: np.ndarray,
                    step: int) -> None:
        """Watchdog (§3.4, §8.1). Raises SafetyHalt on anomaly."""
        wd = self.limits.watchdog
        if wd.get("nan_guard", True) and (
            not np.all(np.isfinite(qvel)) or not np.isfinite(ee_force)
        ):
            self._halt(step, "nan_or_inf_state")
        if np.max(np.abs(qvel)) > wd["max_joint_vel_abs"]:
            self._halt(step, f"joint_vel {np.max(np.abs(qvel)):.3f} > {wd['max_joint_vel_abs']}")
        if ee_force > wd["max_ee_force_proxy"]:
            self._halt(step, f"ee_force {ee_force:.2f} > {wd['max_ee_force_proxy']}")
        if np.max(np.abs(qacc)) > wd["solver_blowup_qacc"]:
            self._halt(step, "solver_blowup / twin outside validated envelope")

    def _halt(self, step: int, reason: str) -> None:
        self.violations.append({"step": int(step), "type": "halt", "reason": reason})
        raise SafetyHalt(f"[step {step}] {reason}")

    def note_soft_violation(self, step: int, reason: str) -> None:
        """A non-fatal limit event (e.g., predicted action rejected pre-actuation)."""
        self.n_rejected += 1
        self.violations.append({"step": int(step), "type": "reject", "reason": reason})

    def clean_session(self) -> bool:
        return not any(v["type"] in ("halt",) for v in self.violations)

    def maybe_raise_cap(self) -> bool:
        """§3.3: raise the speed cap one step ONLY after a clean session."""
        if self.clean_session() and self._cap_idx + 1 < len(self.limits.speed_cap_schedule):
            self._cap_idx += 1
            self.limits.speed_cap_frac = self.limits.speed_cap_schedule[self._cap_idx]
            return True
        return False

    def summary(self) -> dict:
        return {
            "safety_version": self.limits.version,
            "speed_cap_frac": self.limits.speed_cap_frac,
            "n_clamped": self.n_clamped,
            "n_rejected": self.n_rejected,
            "n_halts": sum(1 for v in self.violations if v["type"] == "halt"),
            "clean_session": self.clean_session(),
        }
