#!/usr/bin/env python3
"""ES2: Observer Training Experiment

Trains Observer world model to predict next observations from latent states + actions.
Uses MSE prediction loss (constitutional requirement: NO rewards in Observer training).

Target Performance:
- Prediction RMSE < 0.5 on test set (normalized observations [0, 1])
- RMSE < 0.05 indicates near-perfect world model

Usage:
    python experiments/scripts/train_observer.py --config experiments/configs/observer_train.yaml
    python experiments/scripts/train_observer.py --seed 123 --n_train_steps 2000
"""

import argparse
from pathlib import Path
from typing import Dict, List

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import yaml

# Import AIXI modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.observer.models import Observer
from src.observer.losses import compute_prediction_error
from src.observer.data_collection import (
    collect_random_transitions,
    split_train_test,
    save_transitions,
    load_transitions
)
from src.utils.config import Transition
from src.utils.reproducibility import set_seed, get_device
from src.utils.metrics import compute_rmse
from src.utils.visualization import plot_prediction_errors
from src.tracking.wandb_logger import WandbLogger
from src.tracking.checkpointing import save_checkpoint
from src.environments.wrappers import make_wrapped_env
from src.environments import gridworld


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def prepare_batch(
    transitions: List[Transition],
    batch_size: int,
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """Sample batch of transitions and convert to tensors.
    
    Args:
        transitions: List of Transition objects.
        batch_size: Number of transitions to sample.
        device: Device to place tensors on.
        
    Returns:
        Dictionary with tensors: {obs, action, next_obs}.
    """
    # Sample random batch
    batch_indices = np.random.choice(len(transitions), size=batch_size, replace=False)
    batch = [transitions[i] for i in batch_indices]
    
    # Convert to tensors
    obs = torch.tensor(
        np.array([t.obs for t in batch]),
        dtype=torch.float32,
        device=device
    )
    action = torch.tensor(
        [t.action for t in batch],
        dtype=torch.long,
        device=device
    )
    next_obs = torch.tensor(
        np.array([t.next_obs for t in batch]),
        dtype=torch.float32,
        device=device
    )
    
    return {"obs": obs, "action": action, "next_obs": next_obs}


def evaluate_observer(
    observer: Observer,
    test_transitions: List[Transition],
    device: str = "cpu"
) -> Dict[str, float]:
    """Evaluate Observer on test set.
    
    Args:
        observer: Observer model.
        test_transitions: List of test transitions.
        device: Device to run evaluation on.
        
    Returns:
        Dictionary of metrics: {rmse, mae, max_error}.
    """
    observer.eval()
    
    # Prepare all test data
    batch = prepare_batch(test_transitions, len(test_transitions), device)
    
    with torch.no_grad():
        # Predict next observations
        next_obs_pred = observer(batch["obs"], batch["action"])
        
        # Compute metrics
        rmse = compute_rmse(next_obs_pred, batch["next_obs"])
        mae = torch.mean(torch.abs(next_obs_pred - batch["next_obs"])).item()
        max_error = torch.max(torch.abs(next_obs_pred - batch["next_obs"])).item()
    
    observer.train()
    
    return {
        "rmse": rmse,
        "mae": mae,
        "max_error": max_error
    }


def train_observer(
    env_name: str,
    latent_dim: int,
    encoder_hidden: List[int],
    predictor_hidden: List[int],
    n_data_transitions: int,
    test_fraction: float,
    n_train_steps: int,
    batch_size: int,
    observer_lr: float,
    eval_interval: int,
    seed: int,
    wandb_project: str,
    wandb_name: str,
    wandb_tags: List[str],
    checkpoint_dir: str,
    plot_dir: str,
    data_dir: str,
    **kwargs
) -> None:
    """Train Observer world model.
    
    Args:
        env_name: Gymnasium environment ID.
        latent_dim: Latent dimensionality.
        encoder_hidden: Encoder hidden layer sizes.
        predictor_hidden: Predictor hidden layer sizes.
        n_data_transitions: Number of transitions to collect.
        test_fraction: Fraction of data for testing.
        n_train_steps: Number of gradient steps.
        batch_size: Batch size for training.
        observer_lr: Learning rate.
        eval_interval: Evaluate every N steps.
        seed: Random seed.
        wandb_project: W&B project name.
        wandb_name: W&B run name.
        wandb_tags: W&B tags.
        checkpoint_dir: Directory for checkpoints.
        plot_dir: Directory for plots.
        data_dir: Directory for saved transitions.
    """
    # Set random seed
    set_seed(seed)
    device = get_device()
    
    # Create output directories
    checkpoint_dir = Path(checkpoint_dir)
    plot_dir = Path(plot_dir)
    data_dir = Path(data_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create environment
    env = make_wrapped_env(env_name, normalize=True, logging=False)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Initialize W&B logger
    logger = WandbLogger(
        project=wandb_project,
        name=wandb_name,
        tags=wandb_tags,
        config=None
    )
    
    print(f"\n{'='*60}")
    print(f"ES2: Observer Training - {env_name}")
    print(f"Latent Dim: {latent_dim} | Obs Dim: {obs_dim} | Action Dim: {action_dim}")
    print(f"Train Steps: {n_train_steps} | Batch Size: {batch_size} | LR: {observer_lr}")
    print(f"Device: {device} | Seed: {seed}")
    print(f"{'='*60}\n")
    
    # Collect or load transitions
    transitions_path = data_dir / f"observer_transitions_seed{seed}.pt"
    
    if transitions_path.exists():
        print(f"Loading existing transitions from {transitions_path}")
        transitions = load_transitions(transitions_path)
    else:
        print(f"Collecting {n_data_transitions} transitions...")
        transitions = collect_random_transitions(env, n_data_transitions, seed)
        save_transitions(transitions, transitions_path)
    
    # Split train/test
    train_transitions, test_transitions = split_train_test(
        transitions, test_fraction, seed
    )
    
    # Initialize Observer
    observer = Observer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        encoder_hidden=encoder_hidden,
        predictor_hidden=predictor_hidden
    ).to(device)
    
    print(f"\nObserver Architecture:")
    print(f"  Encoder: obs({obs_dim}) → {encoder_hidden} → latent({latent_dim})")
    print(f"  Predictor: [latent({latent_dim}), action({action_dim})] → {predictor_hidden} → next_obs({obs_dim})")
    print(f"  Total Parameters: {sum(p.numel() for p in observer.parameters()):,}\n")
    
    # Initialize optimizer
    optimizer = optim.Adam(observer.parameters(), lr=observer_lr)
    
    # Training loop
    print("Starting training...")
    train_losses: List[float] = []
    test_rmses: List[float] = []
    best_test_rmse = float("inf")
    
    for step in range(n_train_steps):
        # Sample batch
        batch = prepare_batch(train_transitions, batch_size, device)
        
        # Forward pass
        optimizer.zero_grad()
        loss = compute_prediction_error(
            observer,
            batch["obs"],
            batch["action"],
            batch["next_obs"]
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track training loss
        train_losses.append(loss.item())
        
        # Log to W&B
        logger.log_step({"train/loss": loss.item()}, step=step)
        
        # Evaluate on test set
        if (step + 1) % eval_interval == 0 or step == n_train_steps - 1:
            metrics = evaluate_observer(observer, test_transitions, device)
            test_rmses.append(metrics["rmse"])
            
            # Log test metrics
            logger.log_step({
                "test/rmse": metrics["rmse"],
                "test/mae": metrics["mae"],
                "test/max_error": metrics["max_error"]
            }, step=step)
            
            print(f"Step {step+1:4d}/{n_train_steps} | "
                  f"Train Loss: {loss.item():.4f} | "
                  f"Test RMSE: {metrics['rmse']:.4f} | "
                  f"Test MAE: {metrics['mae']:.4f}")
            
            # Save best checkpoint
            if metrics["rmse"] < best_test_rmse:
                best_test_rmse = metrics["rmse"]
                save_checkpoint(
                    path=checkpoint_dir / "observer_best.pt",
                    observer=observer,
                    observer_optimizer=optimizer,
                    step=step,
                    metrics=metrics
                )
                print(f"  ✓ New best RMSE: {best_test_rmse:.4f} (checkpoint saved)")
    
    # Save final checkpoint
    final_metrics = evaluate_observer(observer, test_transitions, device)
    save_checkpoint(
        path=checkpoint_dir / "observer_final.pt",
        observer=observer,
        observer_optimizer=optimizer,
        step=n_train_steps,
        metrics=final_metrics
    )
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Best Test RMSE:   {best_test_rmse:.4f}")
    print(f"Final Test RMSE:  {final_metrics['rmse']:.4f}")
    print(f"Final Test MAE:   {final_metrics['mae']:.4f}")
    print(f"Final Max Error:  {final_metrics['max_error']:.4f}")
    print(f"{'='*60}\n")
    
    # Log final summary
    logger.log_step({
        "final/best_rmse": best_test_rmse,
        "final/test_rmse": final_metrics["rmse"],
        "final/test_mae": final_metrics["mae"]
    }, step=n_train_steps)
    
    # Generate visualization
    plot_path = plot_dir / "observer_training_loss.png"
    plot_prediction_errors(
        errors=train_losses,
        save_path=str(plot_path)
    )
    print(f"Training curve saved: {plot_path}")
    logger.log_artifact(plot_path, artifact_type="plot")
    
    # Finish W&B run
    logger.finish()
    
    # Close environment
    env.close()


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="ES2: Observer Training for GridWorld-1D"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/observer_train.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)"
    )
    parser.add_argument(
        "--n_train_steps",
        type=int,
        default=None,
        help="Number of training steps (overrides config)"
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=None,
        help="Latent dimensionality (overrides config)"
    )
    parser.add_argument(
        "--n_data_transitions",
        type=int,
        default=None,
        help="Number of transitions to collect (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.seed is not None:
        config["seed"] = args.seed
    if args.n_train_steps is not None:
        config["n_train_steps"] = args.n_train_steps
    if args.latent_dim is not None:
        config["latent_dim"] = args.latent_dim
    if args.n_data_transitions is not None:
        config["n_data_transitions"] = args.n_data_transitions
    
    # Train Observer
    train_observer(**config)


if __name__ == "__main__":
    main()
