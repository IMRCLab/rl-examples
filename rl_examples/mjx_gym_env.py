"""
MJX Gymnasium Environment Wrapper

- Default action space = MuJoCo actuator ctrlrange (i.e., rad/s if you use <velocity> actuators).
- Optional normalized action mode [-1, 1] that maps to rad/s via action_max.
- Reset includes mj_forward() + optional "settle" steps to avoid first-step contact impulses.
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from typing import Any, Dict, Optional, Tuple, Literal

from .mjx_task import MJXTask, MJXState


ActionMode = Literal["ctrl", "normalized"]


class MJXGymEnv(gym.Env):
    """Gymnasium wrapper for MJX tasks."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        task: MJXTask,
        render_mode: Optional[str] = None,
        *,
        # Action handling
        action_mode: ActionMode = "ctrl",
        action_max: float = 150.0,  # used only if action_mode == "normalized" (rad/s)
        # Reset settling
    ):
        super().__init__()

        self.task = task
        self.render_mode = render_mode

        self._model = mujoco.MjModel.from_xml_path(task.xml_path)
        self._data = mujoco.MjData(self._model)
        self._rng = np.random.default_rng()
        self._step_count = 0

        # Action config
        self._action_mode: ActionMode = action_mode
        self._action_max: float = float(action_max)

        self._setup_action_space()
        self._setup_observation_space()

        self._viewer = None

    # ------------------------
    # Spaces
    # ------------------------
    def _setup_action_space(self) -> None:
        nu = int(self._model.nu)
        if nu == 0:
            self.action_space = spaces.Box(low=np.array([], dtype=np.float32),
                                           high=np.array([], dtype=np.float32),
                                           dtype=np.float32)
            return

        if self._action_mode == "ctrl":
            # Use MuJoCo actuator ctrlrange directly (rad/s for <velocity> actuators).
            low = self._model.actuator_ctrlrange[:, 0].astype(np.float32)
            high = self._model.actuator_ctrlrange[:, 1].astype(np.float32)
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        elif self._action_mode == "normalized":
            # Policy outputs [-1, 1], env maps to rad/s via action_max.
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(nu,), dtype=np.float32)
        else:
            raise ValueError(f"Unknown action_mode: {self._action_mode}")

    def _setup_observation_space(self) -> None:
        state = self._get_state()
        obs = self.task.observation(state)
        obs_dim = int(obs.shape[0])
        low = np.full(obs_dim, -np.inf, dtype=np.float32)
        high = np.full(obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # ------------------------
    # State helpers
    # ------------------------
    def _get_state(self) -> MJXState:
        return MJXState(
            qpos=self._data.qpos.copy(),
            qvel=self._data.qvel.copy(),
            time=float(self._data.time),
            ctrl=self._data.ctrl.copy(),
            raw=self._data,  # NOTE: reference (not a copy) for advanced debugging if needed
        )

    def _action_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        """Convert policy action -> MuJoCo ctrl (float64)."""
        if self._model.nu == 0:
            return np.array([], dtype=np.float64)

        if self._action_mode == "ctrl":
            # Clip to actuator ctrlrange.
            low = self._model.actuator_ctrlrange[:, 0]
            high = self._model.actuator_ctrlrange[:, 1]
            a = np.asarray(action, dtype=np.float64)
            return np.clip(a, low, high)
        else:
            # normalized -> rad/s
            a = np.asarray(action, dtype=np.float64)
            a = np.clip(a, -1.0, 1.0)
            return a * self._action_max

    # ------------------------
    # Gym API
    # ------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self._model, self._data)
        self.task.reset_task(self._rng)
        self._step_count = 0

        # Compute kinematics/contact Jacobians before first step
        mujoco.mj_forward(self._model, self._data)

        state = self._get_state()
        obs = self.task.observation(state)
        info = self.task.get_info(state)
        return obs.astype(np.float32), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        ctrl = self._action_to_ctrl(action)

        state_before = self._get_state()
        if self._model.nu > 0:
            self._data.ctrl[:] = ctrl

        mujoco.mj_step(self._model, self._data)
        self._step_count += 1
        state_after = self._get_state()

        reward = self.task.reward(state_before, ctrl, state_after)
        terminated = self.task.is_terminated(state_after)
        truncated = self.task.is_truncated(state_after, self._step_count)

        obs = self.task.observation(state_after)
        info = self.task.get_info(state_after)
        return obs.astype(np.float32), float(reward), bool(terminated), bool(truncated), info

    def render(self) -> None:
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
            self._viewer.sync()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
