#!/usr/bin/env python3
"""
Roll out a trained policy for N episodes and save trajectories to NPZ.

- If --dt is NOT provided, we keep the timestep from the XML model.
- If --dt is provided, we override MuJoCo timestep at rollout time.
- We run policy inference + logging at --target_hz (default 100 Hz) by holding the
  same action for action_repeat simulation steps.

Saved NPZ per episode contains:
  - x_traj: (T, nq+nv) where x = [qpos | qvel] (logged at ~target_hz)
  - u_traj: (T, nu) actions applied (ctrl) (logged at ~target_hz)
  - goal:   (3,) [x, y, yaw]
  - success: bool
  - ep_return: float
  - log_dt: float (seconds between samples in x_traj/u_traj)
  - sim_dt: float (MuJoCo timestep)
  - action_repeat: int
"""

import os
import argparse
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from stable_baselines3 import PPO

from rl_examples.tasks.diff_drive_nav import DiffDriveNavTask
from rl_examples.mjx_gym_env import MJXGymEnv


def compute_action_repeat(sim_dt: float, target_hz: float) -> tuple[int, float, float, float]:
    sim_hz = 1.0 / sim_dt
    action_repeat = max(1, int(round(sim_hz / target_hz)))
    log_dt = sim_dt * action_repeat
    log_hz = 1.0 / log_dt
    return action_repeat, log_dt, log_hz, sim_hz


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Override MuJoCo model timestep (seconds), e.g. 0.01 or 0.002. "
             "If omitted, uses timestep from XML.",
    )
    p.add_argument("--target_hz", type=float, default=100.0,
                   help="Policy inference/logging rate in Hz (default 100).")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="runs/eval_rollouts_npz")
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--goal_range", type=float, default=2.0)
    p.add_argument("--goal_xy_threshold", type=float, default=0.1)
    p.add_argument("--goal_yaw_threshold", type=float, default=0.1)

    args = p.parse_args()

    # Load model
    model_path = Path("runs/diff_drive_nav_ppo/best/best_model.zip")
    if not model_path.exists():
        model_path = Path("runs/diff_drive_nav_ppo/final_model.zip")
    assert model_path.exists(), f"Could not find model at {model_path}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Must match training
    action_mode = "normalized"

    task = DiffDriveNavTask(
        goal_xy_threshold=args.goal_xy_threshold,   
        goal_yaw_threshold=args.goal_yaw_threshold,       
        goal_range=args.goal_range,
        max_steps=args.max_steps,
    )

    env = MJXGymEnv(
        task,
        render_mode=None,
        action_mode=action_mode,
    )
    model = PPO.load(str(model_path))

    # -------------------------------
    # timestep (dt) handling
    # -------------------------------
    sim_dt_xml = float(env._model.opt.timestep)
    sim_dt = sim_dt_xml

    if args.dt is not None:
        assert args.dt > 0.0, "--dt must be > 0"
        print(f"Overriding model timestep: {sim_dt_xml:.6f} -> {args.dt:.6f}")
        env._model.opt.timestep = float(args.dt)
        sim_dt = float(env._model.opt.timestep)
    else:
        print(f"Using model timestep from XML: {sim_dt_xml:.6f}")

    # Compute action_repeat for target_hz using the *actual* sim_dt
    action_repeat, log_dt, log_hz, sim_hz = compute_action_repeat(sim_dt, args.target_hz)

    print("Loaded:", model_path)
    print("Saving rollouts to:", out_dir)
    print(
        f"sim_dt={sim_dt:.6f}s ({sim_hz:.1f} Hz)  target_hz={args.target_hz:.1f} "
        f"action_repeat={action_repeat}  log_dt={log_dt:.6f}s ({log_hz:.1f} Hz)"
    )
    if sim_hz < args.target_hz:
        print(f"WARNING: sim_hz ({sim_hz:.1f}) < target_hz ({args.target_hz:.1f}); "
              f"cannot reach target rate. Using action_repeat=1 -> log_hz={sim_hz:.1f}.")

    for ep in range(args.episodes):
        # After a dt override, a fresh reset is important
        obs, info = env.reset(seed=args.seed + ep)
        qpos0 = env._data.qpos.copy()
        qvel0 = env._data.qvel.copy()

        x_list = []
        u_list = []
        ep_r = 0.0
        success = False

        # Log initial state
        x0 = np.concatenate([env._data.qpos.copy(), env._data.qvel.copy()])
        x_list.append(x0)
        u_list.append(env._data.ctrl.copy())

        while True:
            # Policy inference at ~target_hz
            action, _ = model.predict(obs, deterministic=True)

            terminated = truncated = False
            reward_block = 0.0

            # Hold action for action_repeat sim steps
            for _ in range(action_repeat):
                obs, reward, terminated, truncated, info = env.step(action)
                reward_block += float(reward)
                if terminated or truncated:
                    break

            ep_r += reward_block

            # Log at ~target_hz
            x = np.concatenate([env._data.qpos.copy(), env._data.qvel.copy()])
            x_list.append(x)
            u_list.append(env._data.ctrl.copy())

            if terminated or truncated:
                print("terminated: ", terminated)
                print("truncated: ", truncated)
                print("yaw_error_goal: ", obs[2])
                success = bool(terminated)
                break

        x_traj = np.asarray(x_list, dtype=np.float32)
        u_traj = np.asarray(u_list, dtype=np.float32)
        goal = task.goal.copy().astype(np.float32)

        npz_path = out_dir / f"ep_{ep:03d}.npz"
        np.savez(
            npz_path,
            x_traj=x_traj,
            u_traj=u_traj,
            goal=goal,
            success=success,
            ep_return=float(ep_r),
            log_dt=float(log_dt),
            sim_dt=float(sim_dt),
            action_repeat=int(action_repeat),
            target_hz=float(args.target_hz),
            qpos0=qpos0.astype(np.float32),
            qvel0=qvel0.astype(np.float32),
        )

        print(
            f"[ep {ep:03d}] success={success} samples={len(u_traj)} "
            f"return={ep_r:.2f} goal=({goal[0]:+.2f},{goal[1]:+.2f},{goal[2]:+.2f}) "
            f"saved={npz_path.name}"
        )

    env.close()


if __name__ == "__main__":
    main()
