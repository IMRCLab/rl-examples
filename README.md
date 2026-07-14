## RL Examples (MuJoCo + Gymnasium + SB3)

Small reinforcement learning examples built around a reusable MuJoCo task interface.

This repo currently includes a differential-drive navigation task trained with PPO.

## Installation

```bash
# Clone repo
git clone git@github.com:IMRCLab/rl-examples.git
cd rl-examples

# Create and activate a virtual env (Python >=3.11,<3.13)
python3 -m venv .venv
source .venv/bin/activate

# Install CPU-only torch first; otherwise pip pulls ~2 GB of CUDA wheels.
# The MLP policies here train fine (and fast) on CPU.
pip install --index-url https://download.pytorch.org/whl/cpu torch

# Install the package and remaining dependencies
pip install -e .
```

If you use VS Code, note that its Python environment tooling may detect a `uv`
project and silently recreate `.venv`, wiping the installed packages. If that
happens, create the venv outside the repo directory and point at it explicitly.

## What is in this repo

- `rl_examples/mjx_task.py`: abstract task interface (`MJXTask`) and state container (`MJXState`)
- `rl_examples/mjx_gym_env.py`: Gymnasium wrapper that runs MuJoCo and delegates task logic
- `rl_examples/tasks/diff_drive_nav.py`: example task (diff-drive robot navigation to XY + yaw goal)
- `rl_examples/action_repeat.py`: wrapper holding each action for N physics steps (see Notes)
- `models/pololu.xml`: MuJoCo robot model (diff-drive robot)
- `scripts/train_diff_drive_ppo.py`: PPO training script (Stable-Baselines3)
- `scripts/eval_policy.py`: interactive policy evaluation viewer
- `scripts/eval_policy_rollout_and_save.py`: rollout policy and save trajectories to `.npz`
- `scripts/bench_rollout.py`: deterministic fixed start/goal rollout; prints a `BENCH_RESULT`
  line and writes the trajectory to CSV (used by the wmr-simulator benchmark)
- `scripts/render_rollout_tracking_video.py`: render saved trajectories to `.mp4`
- `runs/`: saved models, tensorboard logs, eval outputs

## Code Hierarchy (How it fits together)

1. `models/*.xml`
   - MuJoCo robot/world definition (physics model, actuators, camera, etc.)
2. `rl_examples/mjx_task.py`
   - Defines the task API:
   - `observation(...)`, `reward(...)`, `is_terminated(...)`, `is_truncated(...)`, `reset_task(...)`
3. `rl_examples/tasks/*.py`
   - Concrete task implementations (goal sampling, reward shaping, termination logic)
4. `rl_examples/mjx_gym_env.py`
   - Wraps MuJoCo into Gymnasium `Env`
   - Calls task methods for observations/rewards/termination
5. `scripts/*.py`
   - Training, evaluation, rollout logging, video rendering

## Quick Start (Example Commands)

Run from the repo root with your virtual environment activated.

### Train PPO

```bash
python scripts/train_diff_drive_ppo.py
```

Outputs are written to `runs/diff_drive_nav_ppo/`.

### Evaluate a trained policy (interactive MuJoCo viewer)

```bash
python scripts/eval_policy.py
```

### Roll out policy and save episodes to NPZ

```bash
python scripts/eval_policy_rollout_and_save.py --episodes 5
```

Example with custom rollout rate and timestep override:

```bash
python scripts/eval_policy_rollout_and_save.py --episodes 10 --target_hz 100 --dt 0.01
```

### Deterministic rollout for a fixed start/goal (benchmarking)

```bash
python scripts/bench_rollout.py --start 0 0 0 --goal 2 2 0 --thr 0.2 --yaw_weight 0.01
```

Prints `BENCH_RESULT success=.. cost_s=.. search_time_s=..`. Success is
`dist_xy + yaw_weight * |dyaw| < thr`; `--traj_out` writes the trajectory to CSV.

### Render a saved rollout to video

Replay saved states:

```bash
python scripts/render_rollout_tracking_video.py runs/eval_rollouts_npz/ep_000.npz --mode states
```

Replay saved controls (re-simulate actions):

```bash
python scripts/render_rollout_tracking_video.py runs/eval_rollouts_npz/ep_000.npz --mode actions
```

## Creating a New Task

Create a new file under `rl_examples/tasks/`, for example:

- `rl_examples/tasks/my_task.py`

Implement a class that inherits from `MJXTask` and defines:

- `xml_path` (points to your MuJoCo XML model)
- `observation(state)`
- `reward(state, action, next_state)`
- `is_terminated(state)`
- `is_truncated(state, step_count)`
- optional `reset_task(rng)` for randomized goals/initial conditions
- optional `get_info(state)` for logging/debugging info

Minimal flow:

1. Build/load your MuJoCo XML model in `models/`
2. Implement task logic in `rl_examples/tasks/my_task.py`
3. Instantiate your task inside `MJXGymEnv(task, ...)`
4. Train with an SB3 algorithm (PPO, SAC, etc.)

## Creating a New Training Script

Start from `scripts/train_diff_drive_ppo.py` and replace the task:

1. Import your task class
2. Create a `make_env()` function that builds `MJXGymEnv(MyTask(...))`, wrapped in `ActionRepeat`
3. Wrap with `DummyVecEnv` / `VecMonitor`, then `VecNormalize`
4. Create the SB3 model (e.g. `PPO("MlpPolicy", env, ...)`)
5. Train and save the model *and* the `VecNormalize` stats under `runs/`

Suggested pattern:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from rl_examples.mjx_gym_env import MJXGymEnv
from rl_examples.action_repeat import ActionRepeat
from rl_examples.tasks.my_task import MyTask

def make_env(seed: int):
    def _thunk():
        env = MJXGymEnv(MyTask(), action_mode="normalized")
        env = ActionRepeat(env, repeat=10)
        env.reset(seed=seed)
        return env
    return _thunk

train_env = VecMonitor(DummyVecEnv([make_env(1000 + i) for i in range(8)]))
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)  # required
model = PPO("MlpPolicy", train_env, verbose=1)
model.learn(total_timesteps=1_000_000)
model.save("runs/my_task_ppo/final_model")
train_env.save("runs/my_task_ppo/vecnormalize.pkl")
```

## Notes

- `ffmpeg` is required for `scripts/render_rollout_tracking_video.py`.
- `VecNormalize` (observation + reward normalization) is required for training to
  converge. Without it the reward curve stays flat and PPO learns nothing: raw
  returns are on the order of 1e4 and the observations are unnormalized. The stats
  are saved to `runs/<run>/vecnormalize.pkl` and must be reapplied at rollout time.
- `pololu.xml` steps at 500 Hz. Querying the policy every physics step gives an
  effective horizon of only ~0.2 s at `gamma=0.99`, so `ActionRepeat` (repeat=10,
  i.e. 50 Hz control) is used during training. Rollouts must use the same repeat.
- The trained policy reaches the goal position but not the goal yaw: it parks
  facing backwards. Terminal orientation is not solved.
