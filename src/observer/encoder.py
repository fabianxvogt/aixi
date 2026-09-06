"""MLP Encoder: Compresses observations to latent representations.

Constitutional Constraint: Encoder only processes observations, never rewards.
Observer learns world dynamics via prediction loss, not task performance.
"""

from typing import List

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """MLP encoder: obs → latent_z
    
    Compresses high-dimensional observations to low-dimensional latent
    representations that capture task-relevant state information.
    
    For GridWorld-1D:
        Input: 1D observation (single state index normalized to [0, 1])
        Hidden: [32, 16] → captures non-linear relationships
        Output: latent_dim=8 → compact representation
    
    Constitutional Requirement:
        - NO reward inputs
        - Only observation → latent_z mapping
        - Used by Actor for policy (Actor never sees raw obs)
    """
    
    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = [32, 16]
    ) -> None:
        """Initialize MLP encoder.
        
        Args:
            obs_dim: Dimensionality of observations (e.g., 1 for GridWorld-1D).
            latent_dim: Dimensionality of latent embedding (e.g., 8).
            hidden_dims: List of hidden layer sizes (default: [32, 16]).
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Build MLP: obs → hidden[0] → hidden[1] → latent
        layers: List[nn.Module] = []
        
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer (no activation - linear projection to latent space)
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to latent state.
        
        Args:
            obs: Observation tensor, shape (batch, obs_dim) or (obs_dim,).
            
        Returns:
            latent_z: Latent representation, shape (batch, latent_dim) or (latent_dim,).
        """
        # Handle both batched and unbatched inputs
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # Add batch dimension
            latent_z = self.network(obs)
            return latent_z.squeeze(0)  # Remove batch dimension
        
        return self.network(obs)
