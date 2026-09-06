"""Metrics computation utilities for Observer and Actor evaluation."""

from typing import List, Dict

import numpy as np
import torch


def compute_rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute Root Mean Squared Error for Observer predictions.
    
    Args:
        predictions: Predicted observations, shape (batch, obs_dim) or (obs_dim,).
        targets: Target observations, same shape as predictions.
        
    Returns:
        RMSE value as float.
        
    Example:
        >>> pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> targ = torch.tensor([[1.1, 1.9], [3.2, 3.8]])
        >>> rmse = compute_rmse(pred, targ)
    """
    mse = torch.nn.functional.mse_loss(predictions, targets)
    return torch.sqrt(mse).item()


def compute_success_rate(episode_rewards: List[float], threshold: float = 0.5) -> float:
    """Compute success rate from episode rewards.
    
    Args:
        episode_rewards: List of total rewards per episode.
        threshold: Minimum reward to count as success.
        
    Returns:
        Success rate as percentage (0.0 to 1.0).
        
    Example:
        >>> rewards = [0.0, 1.0, 1.0, 0.0, 1.0]
        >>> success_rate = compute_success_rate(rewards)  # Returns 0.6
    """
    if len(episode_rewards) == 0:
        return 0.0
    
    successes = sum(1 for reward in episode_rewards if reward > threshold)
    return successes / len(episode_rewards)


def compute_episode_statistics(episode_rewards: List[float]) -> Dict[str, float]:
    """Compute comprehensive statistics for episode rewards.
    
    Args:
        episode_rewards: List of total rewards per episode.
        
    Returns:
        Dictionary with mean, std, min, max, median statistics.
        
    Example:
        >>> rewards = [0.0, 1.0, 1.0, 0.0, 1.0, 0.5]
        >>> stats = compute_episode_statistics(rewards)
        >>> print(stats['mean'], stats['std'])
    """
    if len(episode_rewards) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0
        }
    
    rewards_array = np.array(episode_rewards)
    return {
        "mean": float(np.mean(rewards_array)),
        "std": float(np.std(rewards_array)),
        "min": float(np.min(rewards_array)),
        "max": float(np.max(rewards_array)),
        "median": float(np.median(rewards_array))
    }


def compute_mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute Mean Absolute Error.
    
    Args:
        predictions: Predicted values.
        targets: Target values.
        
    Returns:
        MAE value as float.
    """
    mae = torch.nn.functional.l1_loss(predictions, targets)
    return mae.item()


def compute_per_dimension_mse(predictions: torch.Tensor, targets: torch.Tensor) -> List[float]:
    """Compute MSE for each dimension separately (useful for debugging).
    
    Args:
        predictions: Predicted observations, shape (batch, obs_dim).
        targets: Target observations, shape (batch, obs_dim).
        
    Returns:
        List of MSE values, one per dimension.
    """
    mse_per_dim = ((predictions - targets) ** 2).mean(dim=0)
    return mse_per_dim.tolist()
