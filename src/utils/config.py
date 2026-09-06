"""Configuration management for experiments.

This module defines dataclasses for experiment configuration and transition data.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

import numpy as np
import torch


@dataclass
class ExperimentConfig:
    """Complete training configuration for reproducibility and hyperparameter tracking."""
    
    # Environment
    env_name: str = "GridWorld-1D"
    max_episode_steps: int = 100
    
    # Observer
    latent_dim: int = 8
    observer_lr: float = 1e-3
    encoder_hidden: List[int] = field(default_factory=lambda: [32, 16])
    predictor_hidden: List[int] = field(default_factory=lambda: [32])
    
    # Actor
    actor_lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5000
    target_update_interval: int = 100
    policy_hidden: List[int] = field(default_factory=lambda: [64, 64])
    
    # Training
    n_episodes: int = 100
    batch_size: int = 64
    buffer_size: int = 10000
    update_frequency: int = 1
    
    # Planning
    planning_horizon: int = 0  # 0 = reactive (no planning)
    use_planning: bool = False
    
    # Reproducibility
    seed: int = 42
    device: str = "cpu"
    
    # Tracking
    wandb_project: str = "aixi-observer-actor"
    wandb_entity: Optional[str] = None
    save_frequency: int = 10  # Save checkpoint every N episodes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for W&B logging."""
        return asdict(self)
    
    def __post_init__(self) -> None:
        """Validate configuration values."""
        assert self.observer_lr > 0, "observer_lr must be positive"
        assert self.actor_lr > 0, "actor_lr must be positive"
        assert 0 <= self.gamma <= 1, "gamma must be in [0, 1]"
        assert 0 <= self.epsilon_start <= 1, "epsilon_start must be in [0, 1]"
        assert 0 <= self.epsilon_end <= 1, "epsilon_end must be in [0, 1]"
        assert self.n_episodes > 0, "n_episodes must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.buffer_size > 0, "buffer_size must be positive"
        assert self.planning_horizon >= 0, "planning_horizon must be non-negative"


@dataclass
class Transition:
    """Single environment interaction, core unit for training Observer and Actor."""
    
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool
    latent_z: Optional[np.ndarray] = None
    next_latent_z: Optional[np.ndarray] = None
    
    def to_observer_batch(self) -> Dict[str, torch.Tensor]:
        """Extract Observer training data (no rewards).
        
        Returns:
            Dictionary with obs, action, next_obs as tensors.
        """
        return {
            "obs": torch.from_numpy(self.obs).float(),
            "action": torch.tensor(self.action),
            "next_obs": torch.from_numpy(self.next_obs).float()
        }
    
    def to_actor_batch(self) -> Dict[str, torch.Tensor]:
        """Extract Actor training data (latent states only).
        
        Returns:
            Dictionary with latent_z, action, reward, next_latent_z, done as tensors.
            
        Raises:
            AssertionError: If latent_z or next_latent_z are not set.
        """
        assert self.latent_z is not None, "Must encode obs first"
        assert self.next_latent_z is not None, "Must encode next_obs first"
        
        return {
            "latent_z": torch.from_numpy(self.latent_z).float(),
            "action": torch.tensor(self.action),
            "reward": torch.tensor(self.reward, dtype=torch.float32),
            "next_latent_z": torch.from_numpy(self.next_latent_z).float(),
            "done": torch.tensor(float(self.done), dtype=torch.float32)
        }


class ReplayBuffer:
    """Fixed-size buffer for storing transitions for off-policy RL.
    
    Uses a deque for efficient FIFO (First-In-First-Out) behavior.
    When capacity is reached, oldest transitions are automatically evicted.
    """
    
    def __init__(self, capacity: int) -> None:
        """Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store.
        """
        from collections import deque
        assert capacity > 0, "Capacity must be positive"
        self.buffer: deque = deque(maxlen=capacity)
        self.capacity = capacity
    
    def add(self, transition: Transition) -> None:
        """Add transition to buffer (evicts oldest if full).
        
        Args:
            transition: Transition object to add.
        """
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> List[Transition]:
        """Sample random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample.
            
        Returns:
            List of sampled transitions.
            
        Raises:
            ValueError: If batch_size > current buffer size.
        """
        import random
        
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer "
                f"of size {len(self.buffer)}"
            )
        
        return random.sample(list(self.buffer), batch_size)
    
    def __len__(self) -> int:
        """Return current number of transitions in buffer."""
        return len(self.buffer)
