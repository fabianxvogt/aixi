"""GridWorld-1D: Simple 1-dimensional environment for Observer-Actor validation.

10 states in a line: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Agent starts at state 0, goal is state 9.
Actions: LEFT (0), STAY (1), RIGHT (2)
Deterministic transitions with wall boundaries.

Task success: Reaching state 9 gives +1 reward, episode terminates.
All other transitions give 0 reward. Max episode length: 100 steps.
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class GridWorld1D(gym.Env):
    """1-dimensional GridWorld environment with 10 states.
    
    Observation Space: Discrete(10) - integer state index [0-9]
    Action Space: Discrete(3) - 0=LEFT, 1=STAY, 2=RIGHT
    
    Dynamics:
        - LEFT (0): state = max(0, state - 1)
        - STAY (1): state = state
        - RIGHT (2): state = min(9, state + 1)
    
    Rewards:
        - Reach state 9: +1 (episode terminates)
        - All other transitions: 0
    
    Episode Termination:
        - Success: Agent reaches state 9
        - Truncation: 100 steps elapsed without reaching goal
    
    Constitutional Requirement:
        Observer learns world dynamics (transitions) via prediction.
        Actor learns policy via task rewards (reaching goal).
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        num_states: int = 10,
        goal_state: int = 9,
        max_episode_steps: int = 100,
        render_mode: Optional[str] = None
    ) -> None:
        """Initialize GridWorld-1D environment.
        
        Args:
            num_states: Number of states in the grid (default: 10).
            goal_state: Target state for success (default: 9, rightmost state).
            max_episode_steps: Max steps before truncation (default: 100).
            render_mode: Rendering mode ("human" for text, "rgb_array" for image).
        """
        super().__init__()
        
        assert num_states > 1, "Must have at least 2 states"
        assert 0 <= goal_state < num_states, "Goal must be within state range"
        
        self.num_states = num_states
        self.goal_state = goal_state
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        
        # Gymnasium spaces
        self.observation_space = spaces.Discrete(num_states)
        self.action_space = spaces.Discrete(3)  # LEFT, STAY, RIGHT
        
        # Internal state
        self.current_state: int = 0
        self.step_count: int = 0
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """Reset environment to initial state.
        
        Args:
            seed: Random seed for reproducibility (GridWorld is deterministic).
            options: Additional reset options (unused).
            
        Returns:
            observation: Initial state (always 0).
            info: Empty dict (no auxiliary information).
        """
        super().reset(seed=seed)
        
        self.current_state = 0  # Always start at leftmost state
        self.step_count = 0
        
        return self.current_state, {}
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """Execute action and transition to next state.
        
        Args:
            action: 0 (LEFT), 1 (STAY), or 2 (RIGHT).
            
        Returns:
            observation: Next state after action.
            reward: 1.0 if goal reached, 0.0 otherwise.
            terminated: True if goal reached (success).
            truncated: True if max_episode_steps reached without goal.
            info: Dictionary with {"success": bool, "state": int}.
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # Deterministic state transitions
        if action == 0:  # LEFT
            self.current_state = max(0, self.current_state - 1)
        elif action == 1:  # STAY
            pass  # State unchanged
        elif action == 2:  # RIGHT
            self.current_state = min(self.num_states - 1, self.current_state + 1)
        
        self.step_count += 1
        
        # Compute reward and termination
        terminated = (self.current_state == self.goal_state)
        truncated = (self.step_count >= self.max_episode_steps) and not terminated
        reward = 1.0 if terminated else 0.0
        
        info = {
            "success": terminated,
            "state": self.current_state,
            "step": self.step_count
        }
        
        return self.current_state, reward, terminated, truncated, info
    
    def render(self) -> Optional[str]:
        """Render current environment state.
        
        Returns:
            String representation of grid (if render_mode="human"), else None.
        """
        if self.render_mode == "human":
            # Create ASCII visualization: [A . . . . . . . . G]
            grid = ["."] * self.num_states
            grid[self.current_state] = "A"  # Agent
            grid[self.goal_state] = "G" if self.current_state != self.goal_state else "★"  # Goal or success
            
            render_str = f"[{' '.join(grid)}] Step: {self.step_count}/{self.max_episode_steps}"
            print(render_str)
            return render_str
        
        return None
    
    def close(self) -> None:
        """Cleanup environment resources (no-op for GridWorld)."""
        pass


# Register environment with Gymnasium
gym.register(
    id="GridWorld-1D",
    entry_point="src.environments.gridworld:GridWorld1D",
    max_episode_steps=100
)
