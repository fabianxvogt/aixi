"""Data collection utilities for Observer training.

Collect transitions from random policy or trained Actor for Observer learning.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

from src.utils.config import Transition, ReplayBuffer
from src.utils.reproducibility import set_seed


def collect_random_transitions(
    env: gym.Env,
    n_transitions: int,
    seed: int = 42
) -> List[Transition]:
    """Collect transitions using random policy.
    
    Used for Observer training - collects diverse state-action-next_state data.
    
    Args:
        env: Gymnasium environment.
        n_transitions: Number of transitions to collect.
        seed: Random seed for reproducibility.
        
    Returns:
        List of Transition objects.
    """
    set_seed(seed)
    
    transitions: List[Transition] = []
    
    obs, info = env.reset(seed=seed)
    episode_count = 0
    
    print(f"Collecting {n_transitions} transitions via random policy...")
    
    while len(transitions) < n_transitions:
        # Random action
        action = env.action_space.sample()
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store transition (no latent_z yet)
        transition = Transition(
            obs=np.array(obs) if not isinstance(obs, np.ndarray) else obs,
            action=int(action),
            reward=float(reward),
            next_obs=np.array(next_obs) if not isinstance(next_obs, np.ndarray) else next_obs,
            done=done
        )
        transitions.append(transition)
        
        # Reset if episode ends
        if done:
            obs, info = env.reset(seed=seed + episode_count + 1)
            episode_count += 1
        else:
            obs = next_obs
        
        # Progress update
        if len(transitions) % 1000 == 0:
            print(f"  Collected {len(transitions)}/{n_transitions} transitions...")
    
    print(f"Data collection complete: {len(transitions)} transitions from {episode_count} episodes")
    
    return transitions


def split_train_test(
    transitions: List[Transition],
    test_fraction: float = 0.2,
    seed: int = 42
) -> Tuple[List[Transition], List[Transition]]:
    """Split transitions into train and test sets.
    
    Args:
        transitions: List of all transitions.
        test_fraction: Fraction to use for testing (default: 0.2).
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (train_transitions, test_transitions).
    """
    np.random.seed(seed)
    
    n_total = len(transitions)
    n_test = int(n_total * test_fraction)
    n_train = n_total - n_test
    
    # Shuffle indices
    indices = np.random.permutation(n_total)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_transitions = [transitions[i] for i in train_indices]
    test_transitions = [transitions[i] for i in test_indices]
    
    print(f"Dataset split: {n_train} train, {n_test} test ({test_fraction:.0%})")
    
    return train_transitions, test_transitions


def save_transitions(
    transitions: List[Transition],
    save_path: Path
) -> None:
    """Save transitions to disk.
    
    Args:
        transitions: List of transitions.
        save_path: Path to save file (.pt format).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict format
    data = {
        "transitions": transitions,
        "n_transitions": len(transitions)
    }
    
    torch.save(data, save_path)
    print(f"Transitions saved: {save_path} ({len(transitions)} transitions)")


def load_transitions(load_path: Path) -> List[Transition]:
    """Load transitions from disk.
    
    Args:
        load_path: Path to saved transitions file.
        
    Returns:
        List of Transition objects.
    """
    load_path = Path(load_path)
    
    if not load_path.exists():
        raise FileNotFoundError(f"Transitions file not found: {load_path}")
    
    data = torch.load(load_path)
    transitions = data["transitions"]
    
    print(f"Transitions loaded: {load_path} ({len(transitions)} transitions)")
    
    return transitions
