import numpy as np


def quat_to_yaw(quat: np.ndarray) -> float:
    """
    Extract yaw (rotation about z-axis) from quaternion [w, x, y, z].
    """
    w, x, y, z = quat

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)

    return np.arctan2(siny_cosp, cosy_cosp)


def wrap_angle(angle: float) -> float:
    """
    Wrap angle to [-pi, pi].
    """
    return np.arctan2(np.sin(angle), np.cos(angle))
