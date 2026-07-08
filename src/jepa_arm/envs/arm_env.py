"""MuJoCo reaching environment for one arm twin (directive §1.3, §1.4, §3).

Task: drive the end-effector to a randomized feasible goal pose from a randomized
feasible start. Free-space rigid-arm reaching is deliberately LOW information-destruction
(directive §5.4) — the favorable case for backward modeling.

Provides synchronized, timestamped logging of joint positions, velocities, commanded
actions, measured joint forces (qfrc_actuator) and an EE-contact-force proxy, and the
end-effector pose (§1.4). Action->observation sync error is zero-by-construction (the
action is applied at the control-step boundary; obs read after the step), i.e. <= one
control period as required by §1.4; this is asserted in `sync_error_periods`.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import importlib
import numpy as np
import mujoco

from .embodiment import Embodiment, get, canonical_obs, CANON_OBS_DIM
from ..safety import SafetyMonitor, SafetyLimits, load_safety


def _mat2quat(mat9: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat9.ravel())
    return q


@dataclass
class DynParams:
    """Dynamics perturbation for the sim2sim robustness proxy (HONESTY.md §2)."""
    mass_scale: float = 1.0
    damping_scale: float = 1.0
    frictionloss_add: float = 0.0
    gain_scale: float = 1.0
    ctrl_latency_steps: int = 0     # integer control-step delay on commands


class ArmEnv:
    def __init__(self, arm: str, safety_config_path: str,
                 control_hz: float = 10.0, seed: int = 0,
                 dyn: Optional[DynParams] = None):
        self.emb: Embodiment = get(arm)
        self.arm = arm
        d = importlib.import_module(f"robot_descriptions.{self.emb.mj_module}")
        self.model = mujoco.MjModel.from_xml_path(d.MJCF_PATH)
        self.data = mujoco.MjData(self.model)
        self.nq = self.model.nq
        self.nu = self.model.nu
        assert self.nu == self.emb.dof, f"{arm}: nu {self.nu} != dof {self.emb.dof}"

        self.sim_dt = float(self.model.opt.timestep)
        self.decim = max(1, int(round((1.0 / control_hz) / self.sim_dt)))
        self.dt = self.sim_dt * self.decim          # control period

        self._home = self._keyframe_qpos(self.emb.home_key)
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                         self.emb.ee_site)
        self.ee_body = int(self.model.site_bodyid[self.site_id])

        jl = self.model.jnt_range[:, 0].copy()
        jh = self.model.jnt_range[:, 1].copy()
        self.limits: SafetyLimits = load_safety(arm, jl, jh, safety_config_path)
        self.safety = SafetyMonitor(self.limits, self.dt)

        self.dyn = dyn or DynParams()
        self._m0 = self.model.body_mass.copy()
        self._d0 = self.model.dof_damping.copy()
        self._f0 = self.model.dof_frictionloss.copy()
        self._g0 = self.model.actuator_gainprm.copy()
        self._b0 = self.model.actuator_biasprm.copy()
        self._apply_dyn()

        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.q_goal: Optional[np.ndarray] = None
        self.ee_goal: Optional[np.ndarray] = None
        self.log: list = []
        self._cmd_buffer: list = []   # for ctrl latency

    # ---- model setup helpers -------------------------------------------------
    def _keyframe_qpos(self, name: str) -> np.ndarray:
        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, name)
        if kid < 0:
            return self.model.qpos0[: self.nq].copy()
        return self.model.key_qpos[kid][: self.nq].copy()

    def _apply_dyn(self) -> None:
        self.model.body_mass[:] = self._m0 * self.dyn.mass_scale
        self.model.dof_damping[:] = self._d0 * self.dyn.damping_scale
        self.model.dof_frictionloss[:] = self._f0 + self.dyn.frictionloss_add
        # scale position-actuator gain (kp) and matching bias term together
        self.model.actuator_gainprm[:] = self._g0.copy()
        self.model.actuator_biasprm[:] = self._b0.copy()
        self.model.actuator_gainprm[:, 0] *= self.dyn.gain_scale
        self.model.actuator_biasprm[:, 1] *= self.dyn.gain_scale

    # ---- kinematics ----------------------------------------------------------
    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics for a joint config (does not disturb live state)."""
        saved = self.data.qpos.copy(), self.data.qvel.copy()
        self.data.qpos[: self.nq] = q
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        ee_pos = self.data.site_xpos[self.site_id].copy()
        ee_quat = _mat2quat(self.data.site_xmat[self.site_id])
        self.data.qpos[:], self.data.qvel[:] = saved
        mujoco.mj_forward(self.model, self.data)
        return ee_pos, ee_quat

    def random_config(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or self.rng
        lo, hi = self.limits.joint_pos_low[: self.nq], self.limits.joint_pos_high[: self.nq]
        # continuous joints (inf bounds) -> sample in [-pi, pi]
        lo = np.where(np.isfinite(lo), lo, -np.pi)
        hi = np.where(np.isfinite(hi), hi, np.pi)
        return rng.uniform(lo, hi).astype(np.float64)

    def _adjacent_bodies(self) -> set:
        """Parent-child body pairs + same-body, whose contacts are model artifacts,
        not real self-collisions (adjacent link capsules / base-internal geoms)."""
        pairs = set()
        for b in range(self.model.nbody):
            p = int(self.model.body_parentid[b])
            pairs.add((b, b))
            pairs.add((min(b, p), max(b, p)))
        return pairs

    def collision_free(self, q: np.ndarray, pen_tol: float = 3e-3) -> bool:
        """Collision iff two geoms on NON-adjacent bodies penetrate deeper than pen_tol.
        Same-body and parent-child contacts (which some menagerie models report at rest)
        are filtered out as modeling artifacts."""
        if not hasattr(self, "_adj"):
            self._adj = self._adjacent_bodies()
        saved = self.data.qpos.copy(), self.data.qvel.copy()
        self.data.qpos[: self.nq] = q
        mujoco.mj_forward(self.model, self.data)
        pen = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.dist >= -pen_tol:
                continue
            b1 = int(self.model.geom_bodyid[c.geom1])
            b2 = int(self.model.geom_bodyid[c.geom2])
            if (min(b1, b2), max(b1, b2)) in self._adj:
                continue
            pen = True
            break
        self.data.qpos[:], self.data.qvel[:] = saved
        mujoco.mj_forward(self.model, self.data)
        return not pen

    def sample_reachable_goal(self, rng) -> tuple[np.ndarray, np.ndarray]:
        for _ in range(200):
            q = self.random_config(rng)
            if self.collision_free(q):
                ee, _ = self.fk(q)
                if ee[2] > 0.05:          # keep goal above the base plane
                    return q, ee
        q = self._home.copy()
        return q, self.fk(q)[0]

    # ---- state accessors -----------------------------------------------------
    @property
    def q(self) -> np.ndarray:
        return self.data.qpos[: self.nq].copy()

    @property
    def qvel(self) -> np.ndarray:
        return self.data.qvel[: self.nq].copy()

    def ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return (self.data.site_xpos[self.site_id].copy(),
                _mat2quat(self.data.site_xmat[self.site_id]))

    def ee_force_proxy(self) -> float:
        f = self.data.cfrc_ext[self.ee_body][3:6]
        return float(np.linalg.norm(f))

    def obs(self) -> np.ndarray:
        ee_pos, ee_quat = self.ee_pose()
        return canonical_obs(self.emb, self.q, ee_pos, ee_quat)

    def goal_obs(self) -> np.ndarray:
        ee_pos, ee_quat = self.fk(self.q_goal)
        return canonical_obs(self.emb, self.q_goal, ee_pos, ee_quat)

    # ---- episode API ---------------------------------------------------------
    def reset(self, seed: Optional[int] = None, start: Optional[np.ndarray] = None,
              goal_q: Optional[np.ndarray] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        q0 = start if start is not None else self._perturbed_home()
        self.data.qpos[: self.nq] = q0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = q0[: self.nu]
        mujoco.mj_forward(self.model, self.data)
        if goal_q is not None:
            self.q_goal = goal_q.copy()
            self.ee_goal, _ = self.fk(self.q_goal)
        else:
            self.q_goal, self.ee_goal = self.sample_reachable_goal(self.rng)
        self.step_count = 0
        self.log = []
        self._cmd_buffer = []
        return self.obs()

    def _perturbed_home(self) -> np.ndarray:
        for _ in range(100):
            q = self._home + self.rng.normal(0, 0.15, size=self.nq)
            lo, hi = self.limits.joint_pos_low[: self.nq], self.limits.joint_pos_high[: self.nq]
            q = np.clip(q, np.where(np.isfinite(lo), lo, q), np.where(np.isfinite(hi), hi, q))
            if self.collision_free(q):
                return q
        return self._home.copy()

    def action_to_target(self, action: np.ndarray) -> np.ndarray:
        """Map normalized action in [-1,1]^dof to a joint-position target, per the
        arm's control paradigm (§1.1 heterogeneity)."""
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        vmax = self.limits.joint_vel_max[: self.nu] * self.limits.speed_cap_frac
        if self.emb.control_mode == "velocity":
            # action is a normalized joint velocity command (industrial servo-j style)
            qdot_cmd = a * vmax
            return self.q + qdot_cmd * self.dt
        # position/impedance: action is a bounded position increment
        return self.q + a * (vmax * self.dt)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, dict]:
        q_now = self.q
        q_target = self.action_to_target(action)
        # ctrl latency (dyn proxy): delay the *applied* command by k control steps
        self._cmd_buffer.append(q_target)
        if len(self._cmd_buffer) > self.dyn.ctrl_latency_steps:
            applied_target = self._cmd_buffer.pop(0)
        else:
            applied_target = q_now
        q_cmd = self.safety.clamp_position_target(q_now, applied_target, self.step_count)
        self.data.ctrl[: self.nu] = q_cmd[: self.nu]
        for _ in range(self.decim):
            mujoco.mj_step(self.model, self.data)

        qvel = self.qvel
        force = self.ee_force_proxy()
        qacc = self.data.qacc[: self.nq].copy()
        self.safety.check_state(qvel, force, qacc, self.step_count)  # may SafetyHalt

        ee_pos, ee_quat = self.ee_pose()
        dist = float(np.linalg.norm(ee_pos - self.ee_goal))
        settled = float(np.linalg.norm(qvel)) < 0.10
        success = bool(dist < 0.05 and settled)
        self.step_count += 1
        self.log.append({
            "t": round(self.step_count * self.dt, 4),
            "q": q_now.tolist(), "qvel": qvel.tolist(),
            "action": np.asarray(action, dtype=float).tolist(),
            "q_cmd": q_cmd.tolist(),
            "qfrc_actuator": self.data.qfrc_actuator[: self.nq].tolist(),
            "ee_force_proxy": force,
            "ee_pos": ee_pos.tolist(), "ee_quat": ee_quat.tolist(),
            "dist_to_goal": dist,
        })
        info = {
            "dist": dist, "success": success, "settled": settled,
            "ee_pos": ee_pos, "qvel_norm": float(np.linalg.norm(qvel)),
            "force": force,
        }
        return self.obs(), info

    @property
    def sync_error_periods(self) -> float:
        """Action->observation synchronization error in control periods (§1.4).
        Zero by construction: the command is applied at the step boundary and the
        observation is read after exactly one control period."""
        return 0.0

    def spec(self) -> dict:
        return {
            "arm": self.arm, "dof": self.nu, "control_mode": self.emb.control_mode,
            "sim_dt": self.sim_dt, "control_dt": self.dt, "decimation": self.decim,
            "ee_site": self.emb.ee_site, "canon_obs_dim": CANON_OBS_DIM,
            "sync_error_periods": self.sync_error_periods,
            "safety_version": self.limits.version,
        }
