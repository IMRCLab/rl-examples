"""
MJX Task Base Class

Users inherit from MJXTask and implement:
    - xml_path: path to robot model
    - observation: what the policy sees
    - reward: what to optimize
    - is_terminated: goal reached?
    - is_truncated: timeout?
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np


@dataclass
class MJXState:
    """Container for simulation state."""
    qpos: np.ndarray    # Position state [x, y, theta, ...]
    qvel: np.ndarray    # Velocity state [vx, vy, omega, ...]
    time: float         # Simulation time
    ctrl: np.ndarray    # Current control input
    raw: Any            # Full MJX data (for advanced use)


class MJXTask(ABC):
    """
    Base class for RL tasks.
    
    Inherit from this and implement the abstract methods.
    """
    
    @property
    @abstractmethod
    def xml_path(self) -> str:
        """Path to your MJCF robot model."""
        pass
    
    @abstractmethod
    def observation(self, state: MJXState) -> np.ndarray:
        """Extract observation from state. This is what the policy sees."""
        pass
    
    @abstractmethod
    def reward(self, state: MJXState, action: np.ndarray, next_state: MJXState) -> float:
        """Compute reward for a transition."""
        pass
    
    @abstractmethod
    def is_terminated(self, state: MJXState) -> bool:
        """Check if episode ended (goal reached or failure)."""
        pass
    
    @abstractmethod
    def is_truncated(self, state: MJXState, step_count: int) -> bool:
        """Check if episode should stop (timeout, out of bounds)."""
        pass
    
    def reset_task(self, rng: np.random.Generator) -> None:
        """Reset task state (e.g., randomize goal). Called at episode start."""
        pass
    
    def get_info(self, state: MJXState) -> Dict[str, Any]:
        """Extra info for logging/debugging."""
        return {}