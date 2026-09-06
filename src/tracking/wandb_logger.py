"""Weights & Biases experiment tracking integration.

Provides unified interface for logging metrics, hyperparameters, and artifacts
across all experiment scenarios (ES1-ES4).
"""

from pathlib import Path
from typing import Any, Dict, Optional

import wandb

from src.utils.config import ExperimentConfig


class WandbLogger:
    """Wrapper for Weights & Biases logging with AIXI-specific conventions.
    
    Handles initialization, step-level metrics, episode-level metrics,
    and graceful shutdown. Automatically syncs with W&B cloud.
    
    Usage:
        logger = WandbLogger(project="aixi-observer-actor", name="ES2-observer", config=exp_config)
        logger.log_step({"observer/pred_loss": 0.15}, step=100)
        logger.log_episode({"reward": 10.0, "success": True}, episode=5)
        logger.finish()
    """
    
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[ExperimentConfig] = None,
        tags: Optional[list] = None,
        resume: bool = False
    ) -> None:
        """Initialize W&B logger.
        
        Args:
            project: W&B project name (e.g., "aixi-observer-actor").
            name: Run name (e.g., "ES2-observer-seed42"). Auto-generated if None.
            config: Experiment configuration to log as hyperparameters.
            tags: List of tags for filtering runs (e.g., ["ES2", "observer"]).
            resume: If True, resume previous run with same name.
        """
        self.project = project
        self.name = name
        
        # Convert ExperimentConfig to dict for W&B
        config_dict = None
        if config is not None:
            config_dict = {
                # Environment
                "env_name": config.env_name,
                "seed": config.seed,
                
                # Observer
                "latent_dim": config.latent_dim,
                "encoder_hidden": config.encoder_hidden,
                "predictor_hidden": config.predictor_hidden,
                "observer_lr": config.observer_lr,
                
                # Actor
                "policy_hidden": config.policy_hidden,
                "actor_lr": config.actor_lr,
                "gamma": config.gamma,
                "epsilon_start": config.epsilon_start,
                "epsilon_end": config.epsilon_end,
                "epsilon_decay_steps": config.epsilon_decay_steps,
                
                # Training
                "buffer_size": config.buffer_size,
                "batch_size": config.batch_size,
                "target_update_interval": config.target_update_interval,
                "max_episode_steps": config.max_episode_steps,
                "num_train_episodes": config.num_train_episodes,
                "num_eval_episodes": config.num_eval_episodes,
                "eval_interval": config.eval_interval,
                
                # Planning
                "planning_enabled": config.planning_enabled,
                "planning_horizon": config.planning_horizon,
                
                # Paths
                "checkpoint_dir": str(config.checkpoint_dir),
                "log_dir": str(config.log_dir),
                "plot_dir": str(config.plot_dir),
            }
        
        # Initialize W&B run
        self.run = wandb.init(
            project=project,
            name=name,
            config=config_dict,
            tags=tags,
            resume="allow" if resume else False
        )
        
        print(f"W&B run initialized: {self.run.name} (id: {self.run.id})")
    
    def log_step(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics at specific training step.
        
        Typical use cases:
        - Observer prediction loss: {"observer/pred_loss": 0.15}
        - Actor TD-error: {"actor/td_loss": 0.42, "actor/q_value": 5.3}
        - Planning metrics: {"planning/horizon": 3, "planning/time_ms": 25}
        
        Args:
            metrics: Dictionary of metric_name -> value.
            step: Global step counter (e.g., gradient steps, env steps).
        """
        wandb.log(metrics, step=step)
    
    def log_episode(self, metrics: Dict[str, Any], episode: int) -> None:
        """Log metrics at episode completion.
        
        Typical use cases:
        - Episode reward: {"episode/reward": 10.0}
        - Success indicator: {"episode/success": True}
        - Episode length: {"episode/length": 15}
        
        Args:
            metrics: Dictionary of metric_name -> value.
            episode: Episode counter.
        """
        wandb.log(metrics, step=episode)
    
    def log_artifact(self, artifact_path: Path, artifact_type: str = "model") -> None:
        """Log checkpoint or plot as W&B artifact.
        
        Args:
            artifact_path: Path to file to upload (e.g., checkpoint, plot).
            artifact_type: Artifact category ("model", "plot", "dataset").
        """
        artifact = wandb.Artifact(
            name=artifact_path.stem,
            type=artifact_type
        )
        artifact.add_file(str(artifact_path))
        self.run.log_artifact(artifact)
    
    def finish(self) -> None:
        """Gracefully finish W&B run and sync remaining data."""
        if self.run is not None:
            self.run.finish()
            print("W&B run finished and synced.")
