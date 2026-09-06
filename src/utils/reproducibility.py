"""Reproducibility utilities for deterministic experiment execution.

This module provides functions to set random seeds across all relevant libraries
(Python, NumPy, PyTorch, Gymnasium) to ensure reproducible experiments.
"""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility.
    
    Sets seeds for:
    - Python's random module
    - NumPy's random number generator
    - PyTorch (CPU and CUDA)
    - PyTorch's cuDNN backend (deterministic mode)
    
    Args:
        seed: Random seed value to use across all libraries.
        
    Note:
        - Setting cudnn.deterministic=True may reduce performance
        - Some CUDA operations are still non-deterministic even with these settings
        - For exact reproducibility, use CPU device
        - Gymnasium environments should be reset with the same seed separately
        
    Example:
        >>> set_seed(42)
        >>> # All random operations are now deterministic
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the appropriate torch device (CUDA if available, otherwise CPU).
    
    Args:
        prefer_cuda: If True, use CUDA if available; if False, use CPU.
        
    Returns:
        PyTorch device object.
        
    Example:
        >>> device = get_device()
        >>> tensor = torch.zeros(10).to(device)
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_env(env: "gym.Env", seed: int) -> None:  # type: ignore # noqa: F821
    """Seed a Gymnasium environment for reproducibility.
    
    Args:
        env: Gymnasium environment instance.
        seed: Random seed value.
        
    Example:
        >>> import gymnasium as gym
        >>> env = gym.make("CartPole-v1")
        >>> seed_env(env, 42)
        >>> obs, info = env.reset()  # Deterministic reset
    """
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
