"""
Action-repeat wrapper.

The MuJoCo model steps at a high physics rate (500 Hz for pololu.xml). Letting the
policy decide every physics step gives it an effective foresight of only
1/(1-gamma) steps ~ 0.2 s at gamma=0.99, and produces jittery control that cannot
settle on a goal. Repeating each action for `repeat` physics steps lowers the
control rate (500 Hz -> ~50 Hz at repeat=10), which stretches the effective
horizon and makes the XY+yaw parking task learnable.

Rewards over the repeated physics steps are summed, matching standard action-repeat
semantics.
"""
from __future__ import annotations

import gymnasium as gym


DEFAULT_REPEAT = 10


class ActionRepeat(gym.Wrapper):
    def __init__(self, env: gym.Env, repeat: int = DEFAULT_REPEAT):
        super().__init__(env)
        self.repeat = int(repeat)

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        obs = None
        info: dict = {}
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info
