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

# Install dependencies
# uv must be installed: https://docs.astral.sh/uv/getting-started/installation/
uv lock
uv sync
uv pip install -e .
```

## What is in this repo

- `rl_examples/mjx_task.py`: abstract task interface (`MJXTask`) and state container (`MJXState`)
- `rl_examples/mjx_gym_env.py`: Gymnasium wrapper that runs MuJoCo and delegates task logic
- `rl_examples/tasks/diff_drive_nav.py`: example task (diff-drive robot navigation to XY + yaw goal)
- `models/pololu.xml`: MuJoCo robot model (diff-drive robot)
- `scripts/train_diff_drive_ppo.py`: PPO training script (Stable-Baselines3)
- `scripts/eval_policy.py`: interactive policy evaluation viewer
- `scripts/eval_policy_rollout_and_save.py`: rollout policy and save trajectories to `.npz`
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
2. Create a `make_env()` function that builds `MJXGymEnv(MyTask(...))`
3. Wrap with `DummyVecEnv` / `VecMonitor`
4. Create the SB3 model (e.g. `PPO("MlpPolicy", env, ...)`)
5. Train and save to a new folder under `runs/`

Suggested pattern:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from rl_examples.mjx_gym_env import MJXGymEnv
from rl_examples.tasks.my_task import MyTask

def make_env(seed: int):
    def _thunk():
        env = MJXGymEnv(MyTask(), action_mode="normalized")
        env.reset(seed=seed)
        return env
    return _thunk

train_env = VecMonitor(DummyVecEnv([make_env(1000 + i) for i in range(8)]))
model = PPO("MlpPolicy", train_env, verbose=1)
model.learn(total_timesteps=1_000_000)
model.save("runs/my_task_ppo/final_model")
```

## Notes

- Commit `uv.lock` to keep dependency resolution reproducible across machines.
- `ffmpeg` is required for `scripts/render_rollout_tracking_video.py`.
