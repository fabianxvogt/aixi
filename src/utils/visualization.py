"""Visualization utilities for experiment results."""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(
    steps: List[int],
    rewards: List[float],
    save_path: str,
    title: str = "Learning Curve",
    xlabel: str = "Episode",
    ylabel: str = "Episode Reward",
    figsize: tuple = (10, 6)
) -> None:
    """Plot learning curve (reward vs episodes).
    
    Args:
        steps: List of step/episode numbers.
        rewards: List of rewards corresponding to steps.
        save_path: Path to save the plot.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size (width, height).
    """
    plt.figure(figsize=figsize)
    plt.plot(steps, rewards, linewidth=2, alpha=0.8)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_state_visitation_heatmap(
    states: np.ndarray,
    save_path: str,
    title: str = "State Visitation Heatmap",
    max_states: int = 10,
    figsize: tuple = (12, 2)
) -> None:
    """Plot heatmap of state visitation frequencies (for GridWorld).
    
    Args:
        states: Array of visited states (position indices).
        save_path: Path to save the plot.
        title: Plot title.
        max_states: Maximum number of states (GridWorld size).
        figsize: Figure size (width, height).
    """
    # Count visitation frequencies
    visitation = np.zeros(max_states)
    for state in states:
        if 0 <= state < max_states:
            visitation[int(state)] += 1
    
    # Normalize
    if visitation.sum() > 0:
        visitation = visitation / visitation.sum()
    
    # Plot heatmap
    plt.figure(figsize=figsize)
    plt.imshow(visitation.reshape(1, -1), cmap='YlOrRd', aspect='auto')
    plt.colorbar(label='Visitation Frequency')
    plt.xlabel('State Position', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(range(max_states))
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_prediction_errors(
    errors: List[float],
    save_path: str,
    title: str = "Prediction Errors Over Training",
    xlabel: str = "Training Step",
    ylabel: str = "RMSE",
    figsize: tuple = (10, 6),
    moving_avg_window: Optional[int] = 50
) -> None:
    """Plot Observer prediction errors over training.
    
    Args:
        errors: List of RMSE values.
        save_path: Path to save the plot.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size (width, height).
        moving_avg_window: Window size for moving average smoothing (None to disable).
    """
    plt.figure(figsize=figsize)
    steps = list(range(len(errors)))
    
    # Plot raw errors
    plt.plot(steps, errors, alpha=0.3, label='Raw RMSE')
    
    # Plot moving average if requested
    if moving_avg_window and len(errors) >= moving_avg_window:
        moving_avg = np.convolve(
            errors,
            np.ones(moving_avg_window) / moving_avg_window,
            mode='valid'
        )
        plt.plot(
            range(moving_avg_window - 1, len(errors)),
            moving_avg,
            linewidth=2,
            label=f'{moving_avg_window}-step Moving Average'
        )
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_planning_comparison(
    horizons: List[int],
    success_rates: List[float],
    planning_times: List[float],
    save_path: str,
    figsize: tuple = (12, 5)
) -> None:
    """Plot comparison of planning performance vs computational cost.
    
    Args:
        horizons: List of planning horizon values (K).
        success_rates: Success rates for each horizon.
        planning_times: Planning times in milliseconds for each horizon.
        save_path: Path to save the plot.
        figsize: Figure size (width, height).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot success rate vs horizon
    ax1.plot(horizons, success_rates, marker='o', linewidth=2, markersize=8)
    ax1.set_xlabel('Planning Horizon K', fontsize=12)
    ax1.set_ylabel('Success Rate', fontsize=12)
    ax1.set_title('Planning Performance', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Plot planning time vs horizon
    ax2.plot(horizons, planning_times, marker='s', linewidth=2, markersize=8, color='orange')
    ax2.axhline(y=100, color='r', linestyle='--', label='100ms Budget')
    ax2.set_xlabel('Planning Horizon K', fontsize=12)
    ax2.set_ylabel('Planning Time (ms)', fontsize=12)
    ax2.set_title('Computational Cost', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
