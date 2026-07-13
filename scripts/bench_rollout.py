#!/usr/bin/env python3
"""
Fixed-goal, deterministic rollout of a trained PPO diff-drive policy, for the
wmr-simulator benchmark. Emits a single BENCH_RESULT line and (optionally) writes
the executed trajectory (x, y, yaw) to a CSV.

Contract matches wmr-simulator's other runners (mpc / smag):
  success        = goal reached (xy within --xy_thr AND yaw within --yaw_thr)
                   within --max_steps
  cost_s         = executed time-to-goal = n_steps * sim_dt   (None if not reached)
  search_time_s  = accumulated policy inference time (model.predict), the RL
                   analogue of MPC solve time / smag search time

The policy is queried once per MuJoCo step (matching the training control rate).
"""
import os
import argparse
import pickle
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
from stable_baselines3 import PPO

from rl_examples.tasks.diff_drive_nav import DiffDriveNavTask
from rl_examples.mjx_gym_env import MJXGymEnv
from rl_examples.action_repeat import ActionRepeat, DEFAULT_REPEAT
from rl_examples.utils import quat_to_yaw


def yaw_to_quat(yaw: float) -> np.ndarray:
    """MuJoCo quat [w, x, y, z] for a rotation about +z."""
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=float)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                   help="start pose x y yaw")
    p.add_argument("--goal", type=float, nargs=3, required=True,
                   help="goal pose x y yaw")
    p.add_argument("--max_steps", type=int, default=6000)
    p.add_argument("--xy_thr", type=float, default=0.1)   # task-construction only
    p.add_argument("--yaw_thr", type=float, default=0.1)  # task-construction only
    # Success criterion (matches smag / MPC): reached when
    #   dist_xy + yaw_weight*|wrapped dyaw| < thr
    p.add_argument("--thr", type=float, default=0.2, help="goal_threshold")
    p.add_argument("--yaw_weight", type=float, default=0.01,
                   help="orientation weight = smag goal_error_tolerance")
    p.add_argument("--goal_range", type=float, default=3.5,
                   help="must match training goal_range (obs are goal-relative, "
                        "but the task is constructed with it)")
    p.add_argument("--action_repeat", type=int, default=DEFAULT_REPEAT,
                   help="physics steps per policy action; must match training")
    p.add_argument("--traj_out", type=str, default="")
    p.add_argument("--model", type=str, default="")
    args = p.parse_args()

    model_path = Path(args.model) if args.model else Path("runs/diff_drive_nav_ppo/best/best_model.zip")
    if not model_path.exists():
        model_path = Path("runs/diff_drive_nav_ppo/final_model.zip")
    assert model_path.exists(), f"Could not find trained model at {model_path}"

    task = DiffDriveNavTask(goal_xy_threshold=args.xy_thr, goal_yaw_threshold=args.yaw_thr,
                            max_steps=args.max_steps, goal_range=args.goal_range)
    base_env = MJXGymEnv(task, action_mode="normalized")
    env = ActionRepeat(base_env, repeat=args.action_repeat)
    model = PPO.load(str(model_path))

    # Load VecNormalize obs stats (must match training) and build a normalizer.
    vn_path = model_path.parent / "vecnormalize.pkl"
    obs_rms, clip_obs = None, 10.0
    if vn_path.exists():
        with open(vn_path, "rb") as f:
            vn = pickle.load(f)
        obs_rms, clip_obs = vn.obs_rms, float(vn.clip_obs)

    def norm(o):
        if obs_rms is None:
            return o
        return np.clip((o - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8),
                       -clip_obs, clip_obs).astype(np.float32)

    # Reset, then pin the start pose and the fixed goal.
    env.reset(seed=0)
    base_env._data.qpos[0] = args.start[0]
    base_env._data.qpos[1] = args.start[1]
    base_env._data.qpos[3:7] = yaw_to_quat(args.start[2])
    task.goal = np.array(args.goal, dtype=float)
    mujoco.mj_forward(base_env._model, base_env._data)
    obs = task.observation(base_env._get_state()).astype(np.float32)

    sim_dt = float(base_env._model.opt.timestep)
    policy_dt = sim_dt * args.action_repeat   # wall time advanced per policy step

    def pose():
        q = base_env._data.qpos
        return [float(q[0]), float(q[1]), float(quat_to_yaw(q[3:7]))]

    poses = [pose()]
    t_predict = 0.0
    reached = False
    n = 0
    # max policy steps so total physics steps stay within max_steps
    max_policy_steps = max(1, args.max_steps // args.action_repeat)
    for _ in range(max_policy_steps):
        t0 = time.perf_counter()
        action, _ = model.predict(norm(obs), deterministic=True)
        t_predict += time.perf_counter() - t0
        obs, _, term, trunc, _ = env.step(action)
        n += 1
        p = pose()
        poses.append(p)
        dxy_ = float(np.hypot(p[0] - args.goal[0], p[1] - args.goal[1]))
        dyaw_ = float(abs(np.arctan2(np.sin(p[2] - args.goal[2]), np.cos(p[2] - args.goal[2]))))
        if dxy_ + args.yaw_weight * dyaw_ < args.thr:
            reached = True
            break
        if trunc:
            break

    cost_s = n * policy_dt if reached else None
    if args.traj_out:
        Path(args.traj_out).parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(args.traj_out, np.array(poses), delimiter=",")

    cs = f"{cost_s:.6f}" if cost_s is not None else "nan"
    # Also report final distance/yaw error for debugging.
    fp = poses[-1]
    dxy = float(np.hypot(fp[0] - args.goal[0], fp[1] - args.goal[1]))
    dyaw = float(abs(np.arctan2(np.sin(fp[2] - args.goal[2]), np.cos(fp[2] - args.goal[2]))))
    print(f"BENCH_RESULT success={1 if reached else 0} cost_s={cs} "
          f"search_time_s={t_predict:.6f} final_dxy={dxy:.4f} final_dyaw={dyaw:.4f} steps={n}")

    env.close()


if __name__ == "__main__":
    main()
