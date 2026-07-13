"""
Differential Drive Navigation Task (XY + yaw goal)

Observation: [dist_xy, angle_to_goal, yaw_error_to_goal, forward_vel, wz]
Reward: -dist_xy + bonus_xy + bonus_yaw(only near goal)
Terminate: position only
"""

import numpy as np
from pathlib import Path
from ..mjx_task import MJXTask, MJXState
from ..utils import quat_to_yaw, wrap_angle


class DiffDriveNavTask(MJXTask):

    def __init__(
        self,
        goal_xy_threshold: float = 0.1,
        goal_yaw_threshold: float = 0.1,  # radians
        max_steps: int = 5000,
        goal_range: float = 2.0,
    ):
        self.goal_xy_threshold = float(goal_xy_threshold)
        self.goal_yaw_threshold = float(goal_yaw_threshold)
        self.max_steps = int(max_steps)
        self.goal_range = float(goal_range)

        # Goal pose: [x, y, yaw]
        self.goal = np.array([0.0, 0.0, 0.0], dtype=float)

    @property
    def xml_path(self) -> str:
        return str(Path(__file__).parent.parent.parent / "models" / "pololu.xml")

    def reset_task(self, rng: np.random.Generator) -> None:
        """Randomize goal position AND goal yaw."""
        goal_xy = rng.uniform(-self.goal_range, self.goal_range, size=2)
        while np.linalg.norm(goal_xy) < 0.5:
            goal_xy = rng.uniform(-self.goal_range, self.goal_range, size=2)

        goal_yaw = float(rng.uniform(-np.pi, np.pi))
        self.goal = np.array([float(goal_xy[0]), float(goal_xy[1]), goal_yaw], dtype=float)

    def observation(self, state: MJXState) -> np.ndarray:
        x, y = float(state.qpos[0]), float(state.qpos[1])
        quat = state.qpos[3:7]
        yaw = float(quat_to_yaw(quat))

        vx, vy = float(state.qvel[0]), float(state.qvel[1])
        wz = float(state.qvel[5])

        dx = float(self.goal[0] - x)
        dy = float(self.goal[1] - y)
        dist = float(np.sqrt(dx * dx + dy * dy))

        angle_to_goal = float(wrap_angle(np.arctan2(dy, dx) - yaw))
        yaw_error = float(wrap_angle(float(self.goal[2]) - yaw))
        forward_vel = float(vx * np.cos(yaw) + vy * np.sin(yaw))

        return np.array([dist, angle_to_goal, yaw_error, forward_vel, wz], dtype=np.float32)
    

    def reward(self, state: MJXState, action: np.ndarray, next_state: MJXState) -> float:
        x, y = float(next_state.qpos[0]), float(next_state.qpos[1])
        yaw = float(quat_to_yaw(next_state.qpos[3:7]))

        dx = float(self.goal[0] - x)
        dy = float(self.goal[1] - y)
        dist_xy = float(np.sqrt(dx * dx + dy * dy))
        yaw_error = abs(float(wrap_angle(float(self.goal[2]) - yaw)))

        # 1) Distance shaping — always active
        reward = -dist_xy
    
        # 2) Yaw shaping — active in a wider radius and much stronger, so the
        # robot aligns heading instead of arriving backwards (yaw ~ pi). Weight
        # ramps up as it nears the goal.
        if dist_xy < 0.6:
            near = 1.0 - dist_xy / 0.6          # 0 at r=0.6, 1 at the goal
            reward -= (1.0 + 5.0 * near) * yaw_error

        # 3) XY arrival bonus
        if dist_xy < self.goal_xy_threshold:
            reward += 300.0

            # 4) Yaw alignment bonus — graded credit for being close in heading,
            # plus a large bonus for nailing it, to make final alignment worth it.
            reward += 200.0 * max(0.0, 1.0 - yaw_error / np.pi)
            if yaw_error < self.goal_yaw_threshold:
                reward += 300.0

        # Distance at current state
        x0, y0 = float(state.qpos[0]), float(state.qpos[1])
        dx0 = float(self.goal[0] - x0)
        dy0 = float(self.goal[1] - y0)
        dist0 = float(np.sqrt(dx0 * dx0 + dy0 * dy0))

        # Distance at next state (you already compute dist_xy for next_state)
        progress = dist0 - dist_xy   # positive if we moved closer

        reward += 100.0 * progress     # weight to tune
        return float(reward)
    
    def is_terminated(self, state: MJXState) -> bool:
        """Terminate when position AND yaw are both reached."""
        x, y = float(state.qpos[0]), float(state.qpos[1])
        yaw = float(quat_to_yaw(state.qpos[3:7]))
        dist_xy = float(np.linalg.norm(self.goal[:2] - np.array([x, y], dtype=float)))
        yaw_error = abs(float(wrap_angle(float(self.goal[2]) - yaw)))
        return dist_xy < self.goal_xy_threshold and yaw_error < self.goal_yaw_threshold

    def is_truncated(self, state: MJXState, step_count: int) -> bool:
        if step_count >= self.max_steps:
            return True
        x, y = float(state.qpos[0]), float(state.qpos[1])
        if abs(x) > 10.0 or abs(y) > 10.0:
            return True
        return False
