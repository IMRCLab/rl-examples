from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure

from rl_examples.tasks.diff_drive_nav import DiffDriveNavTask
from rl_examples.mjx_gym_env import MJXGymEnv


def make_env(seed: int, action_mode="ctrl"):
    def _thunk():
        task = DiffDriveNavTask(goal_xy_threshold=0.1, goal_yaw_threshold=0.1, max_steps=2500, goal_range=2.0)
        env = MJXGymEnv(
            task,
            action_mode=action_mode,
        )
        env.reset(seed=seed)
        return env
    return _thunk


def main():
    outdir = Path("runs/diff_drive_nav_ppo")
    outdir.mkdir(parents=True, exist_ok=True)
    
    action_mode = "normalized"   # change to "ctrl" if you want raw rad/s action space

    n_envs = 8
    train_env = DummyVecEnv([make_env(seed=1000 + i, action_mode=action_mode) for i in range(n_envs)])
    train_env = VecMonitor(train_env)

    eval_env = DummyVecEnv([make_env(seed=2000, action_mode=action_mode)])
    eval_env = VecMonitor(eval_env)

    # Tensorboard + SB3 logger
    logger = configure(str(outdir), ["stdout", "tensorboard"])

    # PPO defaults
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=str(outdir),
        device="auto",
    )
    model.set_logger(logger)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(outdir / "best"),
        log_path=str(outdir / "eval"),
        eval_freq=20_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    total_timesteps = 1_000_000
    model.learn(total_timesteps=total_timesteps, callback=eval_cb)

    model.save(str(outdir / "final_model"))
    print(f"Saved to: {outdir}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
