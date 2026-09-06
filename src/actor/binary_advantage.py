"""Minimal reward-aware Actor head for bounded two-action control tasks."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BinaryAdvantageHead(nn.Module):
    """Learn the sign of ``reward(action=1) - reward(action=0)`` from latents."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        self.advantage = nn.Linear(latent_dim, 1)

    def forward(self, latent_z: torch.Tensor) -> torch.Tensor:
        """Return one action-1-versus-action-0 preference logit per finite latent."""
        if not bool(torch.isfinite(latent_z).all()):
            raise ValueError("latent_z must be finite")

        logits: torch.Tensor = self.advantage(latent_z)
        squeezed_logits: torch.Tensor = logits.squeeze(-1)
        return squeezed_logits

    def preference_loss(
        self,
        latent_z: torch.Tensor,
        action_rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Fit unique pairwise action preferences from Actor-owned rewards."""
        if action_rewards.ndim < 1 or action_rewards.shape[-1] != 2:
            raise ValueError("action_rewards must end with exactly two actions")
        if action_rewards.dtype == torch.bool or action_rewards.is_complex():
            raise TypeError("action_rewards must contain real numeric values")
        if not bool(torch.isfinite(action_rewards).all()):
            raise ValueError("action_rewards must be finite")

        logits = self(latent_z)
        margins = action_rewards[..., 1] - action_rewards[..., 0]
        if logits.shape != margins.shape:
            raise ValueError("latent and reward batch shapes do not align")
        if torch.any(margins == 0):
            raise ValueError("pairwise preference target requires a unique best action")

        targets = (margins > 0).to(dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, targets)

    def select_action(self, latent_z: torch.Tensor) -> torch.Tensor:
        """Choose action 1 exactly when its learned advantage is positive."""
        actions: torch.Tensor = (self(latent_z) > 0).to(dtype=torch.long)
        return actions
