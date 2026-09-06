"""Gymnasium environment wrappers for observation normalization and logging.

Provides:
- NormalizeObservation: Scale observations to [0, 1] range
- LoggingWrapper: Track episode statistics (reward, length, success)
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper, Wrapper


class NormalizeObservation(ObservationWrapper):
    """Normalize discrete observations to [0, 1] range.
    
    For GridWorld-1D with 10 states:
        state 0 → 0.0
        state 5 → 0.5
        state 9 → 1.0
    
    This provides better input scaling for neural networks (Observer encoder).
    
    Usage:
        env = gym.make("GridWorld-1D")
        env = NormalizeObservation(env)
        obs, info = env.reset()  # obs is now float in [0, 1]
    """
    
    def __init__(self, env: gym.Env) -> None:
        """Initialize normalization wrapper.
        
        Args:
            env: Base environment with Discrete observation space.
        """
        super().__init__(env)
        
        # Verify observation space is Discrete
        if not isinstance(env.observation_space, gym.spaces.Discrete):
            raise ValueError(
                f"NormalizeObservation requires Discrete observation space, "
                f"got {type(env.observation_space)}"
            )
        
        self.num_states = env.observation_space.n
        
        # Update observation space to Box[0, 1]
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
    
    def observation(self, obs: int) -> np.ndarray:
        """Normalize discrete observation to [0, 1] range.
        
        Args:
            obs: Discrete state index.
            
        Returns:
            Normalized observation as float array.
        """
        normalized = float(obs) / (self.num_states - 1)
        return np.array([normalized], dtype=np.float32)


class LoggingWrapper(Wrapper):
    """Log episode statistics (cumulative reward, length, success).
    
    Automatically tracks metrics at episode completion and adds them to info dict.
    Useful for training loops and W&B logging.
    
    Usage:
        env = gym.make("GridWorld-1D")
        env = LoggingWrapper(env)
        obs, info = env.reset()
        
        # ... run episode ...
        
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"Episode reward: {info['episode']['reward']}")
            print(f"Success: {info['episode']['success']}")
    """
    
    def __init__(self, env: gym.Env) -> None:
        """Initialize logging wrapper.
        
        Args:
            env: Base environment to wrap.
        """
        super().__init__(env)
        
        self.episode_reward: float = 0.0
        self.episode_length: int = 0
        self.episode_success: bool = False
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reset environment and episode statistics.
        
        Args:
            seed: Random seed.
            options: Reset options.
            
        Returns:
            observation: Initial observation.
            info: Info dict (episode stats added on termination).
        """
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_success = False
        
        return self.env.reset(seed=seed, options=options)
    
    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Execute action and track episode statistics.
        
        Args:
            action: Action to execute.
            
        Returns:
            observation: Next observation.
            reward: Immediate reward.
            terminated: Episode terminated (success).
            truncated: Episode truncated (timeout).
            info: Info dict with episode stats if episode ended.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.episode_reward += reward
        self.episode_length += 1
        
        # Track success from environment info
        if "success" in info:
            self.episode_success = info["success"]
        
        # Add episode statistics to info on episode completion
        if terminated or truncated:
            info["episode"] = {
                "reward": self.episode_reward,
                "length": self.episode_length,
                "success": self.episode_success
            }
        
        return obs, reward, terminated, truncated, info


def make_wrapped_env(
    env_name: str = "GridWorld-1D",
    normalize: bool = True,
    logging: bool = True,
    render_mode: Optional[str] = None,
    **env_kwargs
) -> gym.Env:
    """Create environment with standard wrappers applied.
    
    Args:
        env_name: Gymnasium environment ID (e.g., "GridWorld-1D").
        normalize: If True, apply NormalizeObservation wrapper.
        logging: If True, apply LoggingWrapper.
        render_mode: Rendering mode to pass to environment.
        **env_kwargs: Additional keyword arguments for environment creation.
        
    Returns:
        Wrapped environment ready for training.
        
    Example:
        env = make_wrapped_env("GridWorld-1D", normalize=True, logging=True)
        obs, info = env.reset(seed=42)
    """
    env = gym.make(env_name, render_mode=render_mode, **env_kwargs)
    
    if logging:
        env = LoggingWrapper(env)
    
    if normalize:
        env = NormalizeObservation(env)
    
    return env
