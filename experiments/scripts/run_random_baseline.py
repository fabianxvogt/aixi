#!/usr/bin/env python3
"""ES1: Random Baseline Experiment

Validates environment, tracking infrastructure, and metrics pipeline
by running a random policy in GridWorld-1D for 100 episodes.

Expected Results:
- Success rate: ~5% (sparse reward, random exploration)
- Uniform action distribution: ~33% LEFT, ~33% STAY, ~33% RIGHT
- Mean episode length: ~100 steps (usually truncated)
- W&B logging: All metrics tracked correctly

Usage:
    python experiments/scripts/run_random_baseline.py --config experiments/configs/gridworld_random.yaml
    python experiments/scripts/run_random_baseline.py --seed 123 --n_episodes 50
"""

import argparse
from pathlib import Path
from typing import Dict, List

import gymnasium as gym
import numpy as np
import yaml

# Import AIXI utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.reproducibility import set_seed
from src.tracking.wandb_logger import WandbLogger
from src.utils.metrics import compute_success_rate, compute_episode_statistics
from src.utils.visualization import plot_learning_curve
from src.environments.wrappers import make_wrapped_env
from src.environments import gridworld  # Register GridWorld-1D environment


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file.
        
    Returns:
        Dictionary of configuration parameters.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def run_random_baseline(
    env_name: str,
    n_episodes: int,
    seed: int,
    wandb_project: str,
    wandb_name: str,
    wandb_tags: List[str],
    plot_dir: str,
    **kwargs
) -> None:
    """Run random policy baseline in GridWorld-1D.
    
    Args:
        env_name: Gymnasium environment ID.
        n_episodes: Number of episodes to run.
        seed: Random seed for reproducibility.
        wandb_project: W&B project name.
        wandb_name: W&B run name.
        wandb_tags: List of tags for W&B run.
        plot_dir: Directory to save plots.
    """
    # Set random seed for reproducibility
    set_seed(seed)
    
    # Create output directories
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Create environment with wrappers
    env = make_wrapped_env(env_name, normalize=False, logging=True)
    
    # Initialize W&B logger
    logger = WandbLogger(
        project=wandb_project,
        name=wandb_name,
        tags=wandb_tags,
        config=None  # No ExperimentConfig for random baseline
    )
    
    # Episode tracking
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    episode_successes: List[bool] = []
    
    # Global action distribution
    global_action_counts = [0, 0, 0]  # LEFT, STAY, RIGHT
    
    print(f"\n{'='*60}")
    print(f"ES1: Random Baseline - {env_name}")
    print(f"Episodes: {n_episodes} | Seed: {seed}")
    print(f"{'='*60}\n")
    
    # Run episodes
    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0.0
        episode_length = 0
        action_counts = [0, 0, 0]
        
        # Run episode until done
        while not done:
            # Random action
            action = env.action_space.sample()
            action_counts[action] += 1
            global_action_counts[action] += 1
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        # Extract episode info from LoggingWrapper
        episode_info = info.get("episode", {})
        success = episode_info.get("success", False)
        
        # Track episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_successes.append(success)
        
        # Log to W&B
        logger.log_episode({
            "episode/reward": episode_reward,
            "episode/length": episode_length,
            "episode/success": float(success),
            "episode/action_dist_left": action_counts[0] / episode_length,
            "episode/action_dist_stay": action_counts[1] / episode_length,
            "episode/action_dist_right": action_counts[2] / episode_length,
        }, episode=episode)
        
        # Print progress every 20 episodes
        if (episode + 1) % 20 == 0:
            recent_success_rate = np.mean(episode_successes[-20:])
            recent_mean_reward = np.mean(episode_rewards[-20:])
            print(f"Episode {episode+1:3d} | "
                  f"Success Rate: {recent_success_rate:.2%} | "
                  f"Mean Reward: {recent_mean_reward:.2f}")
    
    # Compute final statistics
    stats = compute_episode_statistics(episode_rewards)
    success_rate = compute_success_rate(episode_rewards)
    
    total_actions = sum(global_action_counts)
    action_dist = [count / total_actions for count in global_action_counts]
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Success Rate:         {success_rate:.2%}")
    print(f"Mean Reward:          {stats['mean']:.3f} ± {stats['std']:.3f}")
    print(f"Min/Max Reward:       {stats['min']:.1f} / {stats['max']:.1f}")
    print(f"Mean Episode Length:  {np.mean(episode_lengths):.1f}")
    print(f"\nAction Distribution:")
    print(f"  LEFT  (0): {action_dist[0]:.2%}")
    print(f"  STAY  (1): {action_dist[1]:.2%}")
    print(f"  RIGHT (2): {action_dist[2]:.2%}")
    print(f"{'='*60}\n")
    
    # Log final summary to W&B
    logger.log_step({
        "final/success_rate": success_rate,
        "final/mean_reward": stats["mean"],
        "final/std_reward": stats["std"],
        "final/min_reward": stats["min"],
        "final/max_reward": stats["max"],
        "final/mean_length": np.mean(episode_lengths),
        "final/action_dist_left": action_dist[0],
        "final/action_dist_stay": action_dist[1],
        "final/action_dist_right": action_dist[2],
    }, step=n_episodes)
    
    # Generate learning curve visualization
    plot_path = plot_dir / "random_baseline_rewards.png"
    plot_learning_curve(
        steps=list(range(n_episodes)),
        rewards=episode_rewards,
        save_path=str(plot_path)
    )
    print(f"Learning curve saved: {plot_path}")
    
    # Log plot to W&B
    logger.log_artifact(plot_path, artifact_type="plot")
    
    # Finish W&B run
    logger.finish()
    
    # Close environment
    env.close()


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="ES1: Random Baseline Experiment for GridWorld-1D"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/gridworld_random.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)"
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=None,
        help="Number of episodes (overrides config)"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project name (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.seed is not None:
        config["seed"] = args.seed
    if args.n_episodes is not None:
        config["n_episodes"] = args.n_episodes
    if args.wandb_project is not None:
        config["wandb_project"] = args.wandb_project
    
    # Run random baseline
    run_random_baseline(**config)


if __name__ == "__main__":
    main()
