"""Observer: World model combining Encoder and Predictor.

Constitutional Constraint: Observer learns via PREDICTION LOSS ONLY.
NO rewards in Observer training. Observer learns world dynamics,
Actor learns policy.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from src.observer.encoder import Encoder
from src.observer.predictor import Predictor


class Observer(nn.Module):
    """Observer world model: learns state dynamics via prediction.
    
    Two-stage architecture:
        1. Encoder: obs → latent_z (dimensionality reduction)
        2. Predictor: (latent_z, action) → next_obs (dynamics model)
    
    Training:
        - Loss: MSE between predicted and actual next observations
        - NO reward information (constitutional requirement)
        - Optimized independently from Actor
    
    Inference:
        - encode(obs): Provides latent states for Actor policy
        - predict(latent_z, action): Enables Actor planning (imagination)
    
    Example:
        observer = Observer(obs_dim=1, action_dim=3, latent_dim=8)
        latent_z = observer.encode(obs)
        next_obs_pred = observer.predict(latent_z, action)
        # Or combined:
        next_obs_pred = observer(obs, action)
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int,
        encoder_hidden: Optional[List[int]] = None,
        predictor_hidden: Optional[List[int]] = None
    ) -> None:
        """Initialize Observer world model.
        
        Args:
            obs_dim: Dimensionality of observations (e.g., 1 for GridWorld-1D).
            action_dim: Number of discrete actions (e.g., 3 for GridWorld).
            latent_dim: Dimensionality of latent embedding (e.g., 8).
            encoder_hidden: Hidden layer sizes for encoder (default: [32, 16]).
            predictor_hidden: Hidden layer sizes for predictor (default: [32]).
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        # Default hidden dimensions if not provided
        encoder_hidden = encoder_hidden or [32, 16]
        predictor_hidden = predictor_hidden or [32]
        
        # Build encoder and predictor
        self.encoder = Encoder(
            obs_dim=obs_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden
        )
        
        self.predictor = Predictor(
            latent_dim=latent_dim,
            action_dim=action_dim,
            obs_dim=obs_dim,
            hidden_dims=predictor_hidden
        )
    
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to latent state.
        
        Args:
            obs: Observation tensor, shape (batch, obs_dim) or (obs_dim,).
            
        Returns:
            latent_z: Latent representation, shape (batch, latent_dim) or (latent_dim,).
        """
        return self.encoder(obs)
    
    def predict(self, latent_z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next observation from latent state and action.
        
        Used by Actor for model-based planning (imagination).
        
        Args:
            latent_z: Latent state, shape (batch, latent_dim) or (latent_dim,).
            action: Action tensor (integer or one-hot).
            
        Returns:
            next_obs_pred: Predicted next observation, shape (batch, obs_dim) or (obs_dim,).
        """
        return self.predictor(latent_z, action)
    
    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """Combined encode + predict for efficiency during training.
        
        Args:
            obs: Current observation, shape (batch, obs_dim).
            action: Action taken, shape (batch,) as integer indices.
            
        Returns:
            next_obs_pred: Predicted next observation, shape (batch, obs_dim).
        """
        latent_z = self.encode(obs)
        next_obs_pred = self.predict(latent_z, action)
        return next_obs_pred
