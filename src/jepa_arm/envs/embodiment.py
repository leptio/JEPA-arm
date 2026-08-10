"""Embodiment registry and the canonical cross-embodiment obs/action encoding.

Heterogeneity is a REQUIREMENT (directive §1.2): the arms differ in DoF, control
interface, and kinematic topology, so cross-embodiment generalization (H4) is a
testable variable rather than an assumption.

    FR3   : 7-DoF, position/impedance (stiff), Franka topology
    UR5e  : 6-DoF, VELOCITY control, industrial serial topology
    Gen3  : 7-DoF, position control, continuous wrist joints (distinct topology)

The two control paradigms across the fleet (position + velocity) satisfy the
"at least two actuation/control paradigms" clause of §1.1.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

MAX_DOF = 7
EE_POSE_DIM = 7                     # ee_pos(3) + ee_quat(4)
N_ARMS = 3

# Canonical observation layout (shared latent space across embodiments, §4.3b).
# Joint velocity is deliberately EXCLUDED (quasi-static position control; velocity is
# transient nuisance that swamps the controllable position signal). It is still logged by
# the env for the settle-based success criterion and for safety (§1.4).
#
# Two joint encodings:
#   "raw"    : [ q_padded(7),            ee_pos(3), ee_quat(4), onehot(3) ]  = 17  (v1)
#   "sincos" : [ sin(q)(7), cos(q)(7),   ee_pos(3), ee_quat(4), onehot(3) ]  = 24  (v2)
# v2 uses sin/cos so the latent metric respects joint wrapping: FINDINGS.md showed raw
# angles make +pi and -pi (physically identical on continuous joints) look maximally far
# apart, which broke latent-distance planning on UR5e/Gen3 for ALL learned methods.
CANON_ACT_DIM = MAX_DOF                                       # = 7 (masked per arm)


def obs_dim(encoding: str = "raw") -> int:
    base = EE_POSE_DIM + N_ARMS
    return (2 * MAX_DOF + base) if encoding == "sincos" else (MAX_DOF + base)


CANON_OBS_DIM = obs_dim("raw")                               # = 17 (v1 default preserved)


@dataclass(frozen=True)
class Embodiment:
    name: str
    mj_module: str          # robot_descriptions module name
    dof: int                # active actuated joints
    control_mode: str       # "position" | "velocity"
    ee_site: str            # MuJoCo site used as the end-effector frame
    home_key: str           # keyframe name used as reset home
    index: int              # embodiment id for one-hot

    @property
    def onehot(self) -> np.ndarray:
        v = np.zeros(N_ARMS, dtype=np.float32)
        v[self.index] = 1.0
        return v

    @property
    def act_mask(self) -> np.ndarray:
        m = np.zeros(MAX_DOF, dtype=np.float32)
        m[: self.dof] = 1.0
        return m


REGISTRY: dict[str, Embodiment] = {
    "fr3": Embodiment("fr3", "fr3_mj_description", 7, "position",
                      "attachment_site", "home", 0),
    "ur5e": Embodiment("ur5e", "ur5e_mj_description", 6, "velocity",
                       "attachment_site", "home", 1),
    "gen3": Embodiment("gen3", "gen3_mj_description", 7, "position",
                       "pinch_site", "home", 2),
}

ALL_ARMS = list(REGISTRY.keys())


def pad(vec: np.ndarray, n: int = MAX_DOF) -> np.ndarray:
    """Zero-pad a per-arm joint vector up to n dims (canonical encoding)."""
    out = np.zeros(n, dtype=np.float32)
    out[: len(vec)] = vec
    return out


def canonical_obs(emb: Embodiment, q: np.ndarray,
                  ee_pos: np.ndarray, ee_quat: np.ndarray,
                  encoding: str = "raw") -> np.ndarray:
    if encoding == "sincos":
        joint = np.concatenate([pad(np.sin(q)), pad(np.cos(q))])
    else:
        joint = pad(q)
    return np.concatenate([
        joint,
        ee_pos.astype(np.float32), ee_quat.astype(np.float32),
        emb.onehot,
    ]).astype(np.float32)


def get(arm: str) -> Embodiment:
    return REGISTRY[arm]
