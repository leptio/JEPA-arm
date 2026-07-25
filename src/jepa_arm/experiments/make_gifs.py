"""Render GIF demonstrations of the simulated arms moving start->goal (directive §7.1e:
figures reproducible from logged runs). Physics is run by ArmEnv (robot MJCF); rendering
replays the captured joint trajectory on the menagerie `scene.xml` (floor + lights +
skybox) with a green goal marker and a caption banner. Purely a visualization layer — it
adds nothing to the physics or the reported metrics.
"""
from __future__ import annotations
from pathlib import Path
import os
import importlib
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw

from ..envs.arm_env import ArmEnv
from ..envs.embodiment import get
from ..models.world_model import JEPAWorldModel
from ..bridge.sampler import LatentBridge, BridgeConfig
from ..bridge.planner import BridgePlanner, BridgePlannerConfig, MPPIConfig
from ..baselines.rrt_connect import solve_rrt, RRTConfig

SAFETY = "configs/safety/default.yaml"
CAM = {  # per-arm free-camera framing (lookat, distance, azimuth, elevation)
    "fr3":  ([0.30, 0.0, 0.40], 2.2, 135, -20),
    "ur5e": ([0.0, 0.15, 0.40], 2.3, 150, -20),
    "gen3": ([0.0, 0.0, 0.55], 2.2, 140, -20),
}


def _scene_model(arm: str) -> mujoco.MjModel:
    d = importlib.import_module(f"robot_descriptions.{get(arm).mj_module}")
    scene = os.path.join(os.path.dirname(d.MJCF_PATH), "scene.xml")
    return mujoco.MjModel.from_xml_path(scene)


def _caption(frame: np.ndarray, text: str, ok: bool) -> np.ndarray:
    im = Image.fromarray(frame)
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, im.width, 22], fill=(11, 11, 11))
    dr.text((8, 5), text, fill=(255, 255, 255))
    tag = "SUCCESS" if ok else "reaching"
    col = (60, 200, 90) if ok else (200, 200, 200)
    dr.text((im.width - 78, 5), tag, fill=col)
    return np.asarray(im)


def render_gif(arm: str, qtraj, goal_ee, out: Path, label: str, success: bool,
               width=480, height=360, fps=18, max_frames=70):
    m = _scene_model(arm)
    data = mujoco.MjData(m)
    lookat, dist, az, el = CAM[arm]
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat; cam.distance = dist; cam.azimuth = az; cam.elevation = el
    r = mujoco.Renderer(m, height, width)
    # subsample trajectory to <= max_frames, always keep the last (settled) frame
    idx = np.unique(np.linspace(0, len(qtraj) - 1, min(max_frames, len(qtraj))).astype(int))
    frames = []
    for k in idx:
        data.qpos[: m.nq] = np.asarray(qtraj[k])[: m.nq]
        mujoco.mj_forward(m, data)
        r.update_scene(data, camera=cam)
        s = r.scene
        g = s.geoms[s.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.045, 0, 0]),
                            np.asarray(goal_ee, dtype=float), np.eye(3).ravel(),
                            np.array([0.1, 0.9, 0.2, 0.9], dtype=np.float32))
        s.ngeom += 1
        frames.append(_caption(r.render(), label, success and k == idx[-1]))
    # hold the final frame briefly
    frames += [frames[-1]] * int(fps * 0.8)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, duration=1.0 / fps, loop=0)
    return out


def _traj_from_env(env) -> list:
    return [step["q"] for step in env.log]


def demo_bridge(arm: str, model_path: str, seed: int, max_tries: int = 6):
    model = JEPAWorldModel.load(model_path)
    dof = get(arm).dof
    bcfg = BridgeConfig(n_waypoints=8, n_particles=128, n_action_samples=6, backward_cloud_size=128)
    pcfg = BridgePlannerConfig(mppi=MPPIConfig(horizon=15, n_samples=256, iters=2, sigma=0.6, lam=0.2),
                               max_env_steps=250, forward_validate=True)
    best = None
    for i in range(max_tries):
        env = ArmEnv(arm, SAFETY, seed=1000 + i)
        env.reset(seed=1000 + i)
        r = BridgePlanner(model, LatentBridge(model, bcfg), pcfg, active_dof=dof).solve(env)
        traj = _traj_from_env(env)
        if r["success"]:
            return traj, env.ee_goal, True
        if best is None or r["final_dist"] < best[2]:
            best = (traj, env.ee_goal, r["final_dist"])
    return best[0], best[1], False


def demo_rrt(arm: str, seed: int):
    env = ArmEnv(arm, SAFETY, seed=seed)
    env.reset(seed=seed)
    r = solve_rrt(env, seed=seed, cfg=RRTConfig(max_env_steps=400))
    return _traj_from_env(env), env.ee_goal, r["success"]


def main():
    root = Path("results/study/full")
    out = root / "figures" / "gifs"
    fr3_model = str(root / "per_arm/fr3/model_seed0_b9000/world_model.pt")
    jobs = []
    # method under test on the arm where it works
    tb, gb, okb = demo_bridge("fr3", fr3_model, seed=0)
    jobs.append(("fr3", tb, gb, okb, "FR3 - double-ended JEPA bridge",
                 out / "fr3_bridge.gif"))
    # classical reference reaching on all three embodiments
    for arm, sd in [("fr3", 3), ("ur5e", 5), ("gen3", 2)]:
        t, g, ok = demo_rrt(arm, sd)
        jobs.append((arm, t, g, ok, f"{arm.upper()} - RRT-Connect (reference)",
                     out / f"{arm}_rrt.gif"))
    for arm, traj, goal, ok, label, path in jobs:
        p = render_gif(arm, traj, goal, path, label, ok)
        print(f"wrote {p}  ({len(traj)} steps, success={ok})")


if __name__ == "__main__":
    main()
