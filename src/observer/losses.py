"""Observer loss functions.

Constitutional Constraint: Observer losses MUST NOT contain rewards.
Observer learns world dynamics via prediction error only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def prediction_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """Compute MSE prediction loss for Observer training.
    
    Constitutional Constraint: This is the ONLY loss for Observer.
    NO reward information allowed.
    
    Args:
        predictions: Predicted next observations, shape (batch, obs_dim).
        targets: Actual next observations, shape (batch, obs_dim).
        reduction: "mean", "sum", or "none" (default: "mean").
        
    Returns:
        loss: MSE prediction loss scalar (if reduction="mean" or "sum").
    """
    return F.mse_loss(predictions, targets, reduction=reduction)


def compute_prediction_error(
    observer: nn.Module,
    obs: torch.Tensor,
    action: torch.Tensor,
    next_obs: torch.Tensor
) -> torch.Tensor:
    """Convenience function to compute Observer prediction loss.
    
    Args:
        observer: Observer model.
        obs: Current observations, shape (batch, obs_dim).
        action: Actions taken, shape (batch,).
        next_obs: Actual next observations, shape (batch, obs_dim).
        
    Returns:
        loss: MSE prediction loss.
    """
    next_obs_pred = observer(obs, action)
    return prediction_loss(next_obs_pred, next_obs)
