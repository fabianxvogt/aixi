"""Model checkpointing utilities with git integration and metadata tracking.

Provides save/load functionality for Observer, Actor, optimizers, and training state.
Automatically captures git commit hash for reproducibility.
"""

import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.config import ExperimentConfig


def get_git_commit() -> str:
    """Get current git commit hash for reproducibility tracking.
    
    Returns:
        Short commit hash (7 characters), or "no-git" if not in git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return "no-git"


def save_checkpoint(
    path: Path,
    observer: Optional[nn.Module] = None,
    actor: Optional[nn.Module] = None,
    observer_optimizer: Optional[optim.Optimizer] = None,
    actor_optimizer: Optional[optim.Optimizer] = None,
    config: Optional[ExperimentConfig] = None,
    step: int = 0,
    episode: int = 0,
    metrics: Optional[Dict[str, float]] = None
) -> None:
    """Save training checkpoint with models, optimizers, and metadata.
    
    Args:
        path: Path to save checkpoint (e.g., checkpoints/observer_best.pt).
        observer: Observer model (optional if only saving Actor).
        actor: Actor model (optional if only saving Observer).
        observer_optimizer: Observer optimizer state.
        actor_optimizer: Actor optimizer state.
        config: Experiment configuration for reproducibility.
        step: Global training step count.
        episode: Episode count.
        metrics: Optional dict of metrics (e.g., {"rmse": 0.05, "success_rate": 0.85}).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint: Dict[str, Any] = {
        "step": step,
        "episode": episode,
        "git_commit": get_git_commit(),
        "metrics": metrics or {}
    }
    
    # Save Observer if provided
    if observer is not None:
        checkpoint["observer_state_dict"] = observer.state_dict()
    if observer_optimizer is not None:
        checkpoint["observer_optimizer_state_dict"] = observer_optimizer.state_dict()
    
    # Save Actor if provided
    if actor is not None:
        checkpoint["actor_state_dict"] = actor.state_dict()
    if actor_optimizer is not None:
        checkpoint["actor_optimizer_state_dict"] = actor_optimizer.state_dict()
    
    # Save config as dict
    if config is not None:
        checkpoint["config"] = {
            "env_name": config.env_name,
            "seed": config.seed,
            "latent_dim": config.latent_dim,
            "encoder_hidden": config.encoder_hidden,
            "predictor_hidden": config.predictor_hidden,
            "policy_hidden": config.policy_hidden,
            "observer_lr": config.observer_lr,
            "actor_lr": config.actor_lr,
            "gamma": config.gamma,
            "buffer_size": config.buffer_size,
            "batch_size": config.batch_size,
            "target_update_interval": config.target_update_interval,
            "planning_enabled": config.planning_enabled,
            "planning_horizon": config.planning_horizon
        }
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path} (step={step}, episode={episode}, commit={checkpoint['git_commit']})")


def load_checkpoint(
    path: Path,
    observer: Optional[nn.Module] = None,
    actor: Optional[nn.Module] = None,
    observer_optimizer: Optional[optim.Optimizer] = None,
    actor_optimizer: Optional[optim.Optimizer] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Load checkpoint and restore model/optimizer states.
    
    Args:
        path: Path to checkpoint file.
        observer: Observer model to load state into (if checkpoint contains observer).
        actor: Actor model to load state into (if checkpoint contains actor).
        observer_optimizer: Observer optimizer to load state into.
        actor_optimizer: Actor optimizer to load state into.
        device: Device to map tensors to ("cpu", "cuda").
        
    Returns:
        Dictionary containing checkpoint metadata (step, episode, metrics, config, git_commit).
        
    Example:
        checkpoint = load_checkpoint(
            path="checkpoints/observer_best.pt",
            observer=observer,
            observer_optimizer=observer_optimizer
        )
        print(f"Loaded checkpoint from step {checkpoint['step']}")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location=device)
    
    # Restore Observer
    if observer is not None and "observer_state_dict" in checkpoint:
        observer.load_state_dict(checkpoint["observer_state_dict"])
        print(f"Loaded Observer state from {path}")
    
    if observer_optimizer is not None and "observer_optimizer_state_dict" in checkpoint:
        observer_optimizer.load_state_dict(checkpoint["observer_optimizer_state_dict"])
    
    # Restore Actor
    if actor is not None and "actor_state_dict" in checkpoint:
        actor.load_state_dict(checkpoint["actor_state_dict"])
        print(f"Loaded Actor state from {path}")
    
    if actor_optimizer is not None and "actor_optimizer_state_dict" in checkpoint:
        actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
    
    print(f"Checkpoint loaded: step={checkpoint.get('step', 'unknown')}, "
          f"episode={checkpoint.get('episode', 'unknown')}, "
          f"commit={checkpoint.get('git_commit', 'unknown')}")
    
    return {
        "step": checkpoint.get("step", 0),
        "episode": checkpoint.get("episode", 0),
        "metrics": checkpoint.get("metrics", {}),
        "config": checkpoint.get("config", {}),
        "git_commit": checkpoint.get("git_commit", "unknown")
    }
