# Environment Contract

**Module**: `src/environments/`  
**Purpose**: Provide standardized RL environments following Gymnasium interface.  
**Version**: 1.0.0  
**Date**: 2026-02-25

---

## Overview

Environments provide the RL task specification: observation/action spaces, reward structure, and dynamics. All environments follow Gymnasium API for compatibility with existing RL tools.

**Design Principles**:
- Gymnasium-compatible interface (`reset()`, `step()`, `render()`)
- Deterministic dynamics for MVP (stochasticity is future extension)
- Simple observation spaces for fast iteration (discrete for GridWorld, low-dim continuous for CartPole)

---

## GridWorld-1D Specification

**Environment ID**: `"GridWorld-1D"`  
**Purpose**: Minimal goal-reaching task for validating Observer-Actor architecture.

### Spaces

**Observation Space**: `Box(low=0, high=9, shape=(1,), dtype=float32)`
- Single continuous value representing position in [0, 9]
- Position 0 = start, position 9 = goal

**Action Space**: `Discrete(3)`
- `0` = Move left
- `1` = Stay in place
- `2` = Move right

### Dynamics (Deterministic)

**Transition Function**:
```python
def step(action):
    if action == 0:  # Left
        position = max(0, position - 1)
    elif action == 1:  # Stay
        position = position
    elif action == 2:  # Right
        position = min(9, position + 1)
    
    # Reward structure
    reward = 1.0 if position == 9 else 0.0
    done = (position == 9)  # Episode ends at goal
    
    return obs, reward, done, False, {}
```

**Properties**:
- **Deterministic**: Same (state, action) always produces same next state
- **Fully observable**: Observation equals internal state (position)
- **Markov**: Current position fully determines future (no history needed)

### Reward Structure

**Sparse Reward**:
- `+1.0` if agent reaches goal (position 9)
- `0.0` otherwise

**Rationale**: Tests Actor's ability to learn long-term credit assignment.

### Episode Termination

**Terminated**:
- Position reaches 9 (goal)

**Truncated**:
- Max 100 steps per episode (prevents infinite loops)

**Initial State**:
- Always position 0 (deterministic start)

---

## Public Interface (Gymnasium-Compatible)

### GridWorldEnv Class

**Constructor**:
```python
GridWorldEnv(
    size: int = 10,           # Grid size (0 to size-1)
    max_steps: int = 100,     # Max steps before truncation
    goal_position: int = 9    # Goal location
)
```

---

### Core Methods

#### 1. `reset(seed: Optional[int] = None) -> Tuple[np.ndarray, dict]`

**Purpose**: Reset environment to initial state.

**Signature**:
```python
def reset(
    self,
    seed: Optional[int] = None,
    options: Optional[dict] = None
) -> Tuple[np.ndarray, dict]:
    """
    Reset environment to starting position.
    
    Args:
        seed: Random seed (for future stochastic extensions)
        options: Additional reset options (unused for MVP)
        
    Returns:
        obs: Initial observation, shape (1,), value [0.0] (start position)
        info: Empty dict (no auxiliary information)
    """
```

**Contract**:
- **Deterministic reset**: Always returns position 0 for MVP
- **Seed support**: Accepts seed parameter for reproducibility (Gymnasium requirement)
- **Info dict**: Empty for MVP, can include debugging info later

**Example**:
```python
env = GridWorldEnv()
obs, info = env.reset(seed=42)
assert obs == [0.0]  # Start position
```

---

#### 2. `step(action: int) -> Tuple[np.ndarray, float, bool, bool, dict]`

**Purpose**: Execute action and return transition.

**Signature**:
```python
def step(
    self,
    action: int
) -> Tuple[np.ndarray, float, bool, bool, dict]:
    """
    Execute action and return next state.
    
    Args:
        action: Integer in [0, 1, 2] (left, stay, right)
        
    Returns:
        obs: Next observation (new position), shape (1,)
        reward: Scalar reward (1.0 if goal, else 0.0)
        terminated: True if goal reached
        truncated: True if max_steps exceeded
        info: Dict with {"position": int}
        
    Raises:
        ValueError: If action not in [0, 1, 2]
    """
```

**Contract**:
- **Deterministic**: Same (state, action) produces same output
- **Bounded state**: Position always in [0, size-1]
- **Valid actions**: Rejects actions outside [0, 1, 2]

**Example**:
```python
obs, reward, terminated, truncated, info = env.step(action=2)  # Move right
# obs = [1.0], reward = 0.0, terminated = False
```

---

#### 3. `render() -> Optional[str]`

**Purpose**: Visualize current state (text-based for MVP).

**Signature**:
```python
def render(self) -> Optional[str]:
    """
    Render current state as text.
    
    Returns:
        text: ASCII visualization of grid
        
    Example:
        "[X][ ][ ][ ][ ][ ][ ][ ][ ][G]"  # Agent at 0, goal at 9
    """
```

**Example Output**:
```
Position: 3 / 9
[S][ ][ ][X][ ][ ][ ][ ][ ][G]
```
- `S` = Start position
- `X` = Current agent position
- `G` = Goal position

---

## Testing Contract

### Required Tests

1. **Determinism**: Same (state, action) produces same next state
2. **Bounds**: Position never goes below 0 or above size-1
3. **Termination**: Episode ends when position == goal
4. **Truncation**: Episode ends after max_steps
5. **Reward**: Correct reward (1.0 at goal, 0.0 elsewhere)

### Example Test

```python
def test_gridworld_determinism():
    env1 = GridWorldEnv()
    env2 = GridWorldEnv()
    
    env1.reset(seed=42)
    env2.reset(seed=42)
    
    for _ in range(10):
        action = 2  # Always move right
        obs1, r1, d1, t1, _ = env1.step(action)
        obs2, r2, d2, t2, _ = env2.step(action)
        
        assert np.array_equal(obs1, obs2), "Environments must be deterministic"
        assert r1 == r2
        assert d1 == d2
```

---

## CartPole-v1 Wrapper (Phase 2 Extension)

**Environment ID**: `"CartPole-v1"` (Gymnasium built-in)

**Why CartPole**:
- **4D continuous observations**: Tests Observer on continuous state space
- **Balancing task**: Requires reactive control (good Actor stress test)
- **Well-established benchmark**: Easy comparison to literature

**Wrapper Purpose**:
Normalize observations to [−1, 1] for stable Observer training.

**Usage**:
```python
import gymnasium as gym
env = gym.make("CartPole-v1")
env = NormalizeObservationWrapper(env)  # Custom wrapper in src/environments/wrappers.py
```

---

## Performance Expectations

| Operation | Latency |
|-----------|---------|
| `reset()` | <0.1ms |
| `step(action)` | <0.1ms |
| `render()` | <1ms (text-based) |

**Deterministic Guarantee**: Exact reproducibility with same seed.

---

## Summary

**GridWorld-1D**:
- Minimal 1D goal-reaching task
- Deterministic dynamics, sparse rewards
- Perfect for validating Observer-Actor architecture

**CartPole-v1** (Phase 2):
- Standard Gymnasium benchmark
- Continuous observations (4D)
- Tests scaling to higher dimensions

**Gymnasium Compliance**: ✅ Full compatibility with Gymnasium API

---

## Version History

- **1.0.0** (2026-02-25): Initial contract for GridWorld-1D
- Future: Stochastic GridWorld, custom 2D navigation environments
