from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.logger import configure

from rl_examples.tasks.diff_drive_nav import DiffDriveNavTask
from rl_examples.mjx_gym_env import MJXGymEnv
from rl_examples.action_repeat import ActionRepeat, DEFAULT_REPEAT


def make_env(seed: int, action_mode="ctrl", action_repeat=DEFAULT_REPEAT):
    def _thunk():
        task = DiffDriveNavTask(goal_xy_threshold=0.1, goal_yaw_threshold=0.1, max_steps=4000, goal_range=3.5)
        env = MJXGymEnv(
            task,
            action_mode=action_mode,
        )
        env = ActionRepeat(env, repeat=action_repeat)
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
    # Normalize observations and rewards: the raw reward magnitude (~1e4) and
    # unnormalized obs (dist 0..5) otherwise make PPO's value learning flat.
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                             clip_obs=10.0, clip_reward=10.0, gamma=0.995)

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
        gamma=0.995,
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

    total_timesteps = 600_000   # policy steps; x10 physics steps via ActionRepeat
    model.learn(total_timesteps=total_timesteps)

    model.save(str(outdir / "final_model"))
    # Save normalization stats so eval/rollout can reproduce the obs scaling.
    train_env.save(str(outdir / "vecnormalize.pkl"))
    print(f"Saved to: {outdir}")

    train_env.close()


if __name__ == "__main__":
    main()
