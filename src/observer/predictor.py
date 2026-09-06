"""MLP Predictor: Predicts next observations from latent state + action.

Constitutional Constraint: Predictor only learns state transitions, never rewards.
Observer learns world dynamics via prediction error, not task performance.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class Predictor(nn.Module):
    """MLP predictor: (latent_z, action) → next_obs
    
    Models world dynamics by predicting next observations from current
    latent state and action. Used by Actor for model-based planning.
    
    For GridWorld-1D:
        Input: latent_z (8) + action_onehot (3) → 11 total
        Hidden: [32] → learns transition dynamics
        Output: next_obs (1) → predicted next state
    
    Constitutional Requirement:
        - NO reward prediction
        - Only (latent_z, action) → next_obs mapping
        - Used by Actor.plan() for imagination-based rollouts
    """
    
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        obs_dim: int,
        hidden_dims: List[int] = [32]
    ) -> None:
        """Initialize MLP predictor.
        
        Args:
            latent_dim: Dimensionality of latent state (e.g., 8).
            action_dim: Number of discrete actions (e.g., 3 for GridWorld).
            obs_dim: Dimensionality of observations (e.g., 1 for GridWorld-1D).
            hidden_dims: List of hidden layer sizes (default: [32]).
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.hidden_dims = hidden_dims
        
        # Build MLP: [latent_z, action_onehot] → hidden[0] → ... → next_obs
        layers: List[nn.Module] = []
        
        input_dim = latent_dim + action_dim  # Concatenate latent + action_onehot
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer (no activation - linear prediction)
        layers.append(nn.Linear(prev_dim, obs_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, latent_z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next observation from latent state and action.
        
        Args:
            latent_z: Latent state, shape (batch, latent_dim) or (latent_dim,).
            action: Action tensor, either:
                - Integer action index, shape (batch,) or scalar
                - One-hot encoded action, shape (batch, action_dim) or (action_dim,)
                
        Returns:
            next_obs_pred: Predicted next observation, shape (batch, obs_dim) or (obs_dim,).
        """
        # Handle unbatched inputs
        unbatched = False
        if latent_z.dim() == 1:
            latent_z = latent_z.unsqueeze(0)
            unbatched = True
        
        # Convert action to one-hot if integer
        if action.dtype in [torch.int32, torch.int64, torch.long]:
            if action.dim() == 0:  # Scalar action
                action = action.unsqueeze(0)
            action_onehot = F.one_hot(action, num_classes=self.action_dim).float()
        else:
            # Assume already one-hot encoded
            action_onehot = action
            if action_onehot.dim() == 1:
                action_onehot = action_onehot.unsqueeze(0)
        
        # Concatenate latent_z and action_onehot
        x = torch.cat([latent_z, action_onehot], dim=1)
        
        # Predict next observation
        next_obs_pred = self.network(x)
        
        # Remove batch dimension if input was unbatched
        if unbatched:
            next_obs_pred = next_obs_pred.squeeze(0)
        
        return next_obs_pred
