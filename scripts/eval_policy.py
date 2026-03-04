#!/usr/bin/env python3
import os
import time
from pathlib import Path

# Try egl first; if it crashes, run with: MUJOCO_GL=glfw python3 scripts/eval_policy_viewer.py
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

from rl_examples.tasks.diff_drive_nav import DiffDriveNavTask
from rl_examples.mjx_gym_env import MJXGymEnv


def add_goal_marker(viewer, goal_xy, z=0.02, radius=0.03):
    """
    Add/update a goal marker in viewer.user_scn.
    We re-add it each call and keep only one marker.
    """
    # Clear previous user geoms (keep it simple: only goal marker)
    viewer.user_scn.ngeom = 0

    g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    viewer.user_scn.ngeom += 1

    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, 0, 0], dtype=np.float32),
        pos=np.array([goal_xy[0], goal_xy[1], z], dtype=np.float32),
        mat=np.eye(3, dtype=np.float32).flatten(),
        rgba=np.array([1.0, 0.1, 0.1, 0.9], dtype=np.float32),
    )

    # Make sure it renders
    g.category = mujoco.mjtCatBit.mjCAT_DECOR


def main():
    model_path = Path("runs/diff_drive_nav_ppo/best/best_model.zip")
    if not model_path.exists():
        model_path = Path("runs/diff_drive_nav_ppo/final_model.zip")
    assert model_path.exists(), f"Could not find model at {model_path}"

    # IMPORTANT: match what you trained with
    # If you trained with normalized actions:
    action_mode = "normalized"

    task = DiffDriveNavTask()
    env = MJXGymEnv(
        task,
        render_mode=None,          # manage viewer manually
        action_mode=action_mode,
    )

    model = PPO.load(str(model_path))

    obs, info = env.reset(seed=0)
    print("Loaded:", model_path)
    print("Initial obs:", obs)

    viewer = mujoco.viewer.launch_passive(env._model, env._data)
    # --- Select fixed tracking camera defined in XML ---
    # cam_id = mujoco.mj_name2id(
    #     env._model,
    #     mujoco.mjtObj.mjOBJ_CAMERA,
    #     "track_cam"   # must match XML name exactly
    # )

    # viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    # viewer.cam.fixedcamid = cam_id
    # draw initial goal marker
    add_goal_marker(viewer, task.goal, z=0.02, radius=0.03)

    ep_r = 0.0
    steps = 0

    try:
        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_r += float(reward)
            steps += 1

            # update marker occasionally (cheap, but keeps it correct if goal resets)
            if steps % 10 == 0:
                add_goal_marker(viewer, task.goal, z=0.02, radius=0.03)

            if steps % 100 == 0:
                print(f"step={steps:5d} dist={obs[0]:.3f} ang={obs[1]:+.3f} ep_r={ep_r:+.1f}")

            viewer.sync()
            # time.sleep(0.01)

            if terminated or truncated:
                print(f"Episode done. term={terminated} trunc={truncated} "
                      f"steps={steps} return={ep_r:.1f} final_dist={obs[0]:.3f}")

                obs, info = env.reset()
                ep_r = 0.0
                steps = 0

                # new goal marker
                add_goal_marker(viewer, task.goal, z=0.02, radius=0.03)

    finally:
        viewer.close()
        env.close()


if __name__ == "__main__":
    main()
