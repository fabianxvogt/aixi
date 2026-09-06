# Data Model: Observer-Actor RL System

**Purpose**: Define core entities, their relationships, and validation rules for Observer-Actor architecture.  
**Date**: 2026-02-25  
**Status**: Phase 1 Design

## Overview

This data model supports modular Observer-Actor RL with strict architectural boundaries: Observer learns world dynamics via prediction, Actor learns policies via task rewards, and planning connects them via latent state queries.

---

## Core Entities

### 1. ObserverState

**Purpose**: Compressed world representation output by Observer's encoder.

**Fields**:
- `latent_z: Tensor` - Latent state vector, shape `(latent_dim,)` or `(batch, latent_dim)`
- `obs_history: Optional[Deque[Tensor]]` - Rolling history of recent observations (for recurrent extensions)
- `encoding_time: Optional[float]` - Time to compute encoding (for profiling)

**Validation Rules**:
- `latent_z.shape[-1] == latent_dim` (consistent dimensionality)
- `latent_z` must contain finite values (no NaN, no Inf)
- If `obs_history` provided: `len(obs_history) <= max_history_len`

**Relationships**:
- **Producer**: Observer.encode(obs) → ObserverState
- **Consumer**: Actor.select_action(observer_state.latent_z) → action

**Example**:
```python
@dataclass
class ObserverState:
    latent_z: torch.Tensor
    obs_history: Optional[Deque[torch.Tensor]] = None
    encoding_time: Optional[float] = None
    
    def __post_init__(self):
        assert torch.isfinite(self.latent_z).all(), "latent_z must be finite"
```

---

### 2. ActorState

**Purpose**: Actor's internal state for exploration scheduling and action selection.

**Fields**:
- `epsilon: float` - Current exploration rate for epsilon-greedy, range `[0, 1]`
- `step_count: int` - Total training steps taken (for epsilon decay)
- `last_action: Optional[int]` - Previous action taken (for logging/debugging)
- `target_update_counter: int` - Steps since last target network update (DQN)

**Validation Rules**:
- `0 <= epsilon <= 1` (valid probability)
- `step_count >= 0` (non-negative counter)
- `target_update_counter >= 0`

**State Transitions**:
```python
# Epsilon decay (linear)
epsilon_new = max(
    epsilon_end,
    epsilon_start - (epsilon_start - epsilon_end) * (step_count / epsilon_decay_steps)
)
```

**Relationships**:
- **Owner**: Actor module (internal state)
- **Triggers**: `step_count` increments trigger epsilon updates and target network syncs

**Example**:
```python
@dataclass
class ActorState:
    epsilon: float = 1.0
    step_count: int = 0
    last_action: Optional[int] = None
    target_update_counter: int = 0
    
    def decay_epsilon(self, epsilon_end: float, epsilon_decay_steps: int):
        epsilon_start = 1.0
        self.epsilon = max(
            epsilon_end,
            epsilon_start - (epsilon_start - epsilon_end) * (self.step_count / epsilon_decay_steps)
        )
```

---

### 3. Transition

**Purpose**: Single environment interaction, core unit for training Observer and Actor.

**Fields**:
- `obs: Tensor` - Current observation, shape `(obs_dim,)`
- `action: int` - Action taken, range `[0, action_dim-1]`
- `reward: float` - Task reward received from environment
- `next_obs: Tensor` - Next observation after action, shape `(obs_dim,)`
- `done: bool` - Episode termination flag
- `latent_z: Optional[Tensor]` - Observer's encoding of `obs` (cached for Actor training)
- `next_latent_z: Optional[Tensor]` - Observer's encoding of `next_obs` (cached for Actor training)

**Validation Rules**:
- `obs.shape == next_obs.shape` (consistent observation space)
- `reward` must be finite (no NaN, no Inf)
- `0 <= action < action_dim` (valid action for environment)
- If `latent_z` provided: `latent_z.shape[-1] == latent_dim`

**Relationships**:
- **Source**: Environment.step(action) → Transition
- **Consumer (Observer)**: Uses `(obs, action, next_obs)` for prediction loss
- **Consumer (Actor)**: Uses `(latent_z, action, reward, next_latent_z, done)` for RL loss
- **Storage**: Replay buffer stores T transitions for off-policy learning

**Example**:
```python
@dataclass
class Transition:
    obs: torch.Tensor
    action: int
    reward: float
    next_obs: torch.Tensor
    done: bool
    latent_z: Optional[torch.Tensor] = None
    next_latent_z: Optional[torch.Tensor] = None
    
    def to_observer_batch(self) -> Dict[str, torch.Tensor]:
        """Extract Observer training data (no rewards)."""
        return {
            "obs": self.obs,
            "action": torch.tensor(self.action),
            "next_obs": self.next_obs
        }
    
    def to_actor_batch(self) -> Dict[str, torch.Tensor]:
        """Extract Actor training data (latent states only)."""
        assert self.latent_z is not None, "Must encode obs first"
        return {
            "latent_z": self.latent_z,
            "action": torch.tensor(self.action),
            "reward": torch.tensor(self.reward),
            "next_latent_z": self.next_latent_z,
            "done": torch.tensor(self.done, dtype=torch.float32)
        }
```

---

### 4. ExperimentConfig

**Purpose**: Complete training configuration for reproducibility and hyperparameter tracking.

**Fields**:

**Environment**:
- `env_name: str` - Gymnasium environment ID (e.g., "GridWorld-1D", "CartPole-v1")
- `max_episode_steps: int` - Maximum steps per episode (prevents infinite loops)

**Observer**:
- `latent_dim: int` - Latent state dimensionality
- `observer_lr: float` - Observer learning rate
- `encoder_hidden: List[int]` - Encoder hidden layer sizes (e.g., [32, 16])
- `predictor_hidden: List[int]` - Predictor hidden layer sizes (e.g., [32])

**Actor**:
- `actor_lr: float` - Actor learning rate
- `gamma: float` - Discount factor for future rewards
- `epsilon_start: float` - Initial exploration rate
- `epsilon_end: float` - Final exploration rate
- `epsilon_decay_steps: int` - Steps for linear epsilon decay
- `target_update_interval: int` - Steps between target network updates (DQN)

**Training**:
- `n_episodes: int` - Total training episodes
- `batch_size: int` - Minibatch size for gradient descent
- `buffer_size: int` - Replay buffer capacity
- `update_frequency: int` - Gradient steps per environment step

**Planning**:
- `planning_horizon: int` - Rollout depth K (0 = reactive, >0 = planning)
- `use_planning: bool` - Enable/disable planning mode

**Reproducibility**:
- `seed: int` - Random seed for all RNGs
- `device: str` - PyTorch device ("cpu" or "cuda")

**Validation Rules**:
- Learning rates: `lr > 0`
- Discount factor: `0 <= gamma <= 1`
- Epsilon: `0 <= epsilon_start, epsilon_end <= 1`
- Positive integers: `n_episodes, batch_size, buffer_size > 0`
- Planning horizon: `planning_horizon >= 0`

**Example**:
```python
@dataclass
class ExperimentConfig:
    # Environment
    env_name: str = "GridWorld-1D"
    max_episode_steps: int = 100
    
    # Observer
    latent_dim: int = 8
    observer_lr: float = 1e-3
    encoder_hidden: List[int] = field(default_factory=lambda: [32, 16])
    predictor_hidden: List[int] = field(default_factory=lambda: [32])
    
    # Actor
    actor_lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5000
    target_update_interval: int = 100
    
    # Training
    n_episodes: int = 100
    batch_size: int = 64
    buffer_size: int = 10000
    update_frequency: int = 1
    
    # Planning
    planning_horizon: int = 0  # 0 = reactive
    use_planning: bool = False
    
    # Reproducibility
    seed: int = 42
    device: str = "cpu"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for W&B logging."""
        return asdict(self)
```

---

### 5. Checkpoint

**Purpose**: Saveable experiment state for resumption, evaluation, and reproducibility.

**Fields**:
- `observer_state_dict: Dict[str, Tensor]` - Observer model weights
- `actor_state_dict: Dict[str, Tensor]` - Actor model weights
- `observer_optimizer_state: Dict` - Observer optimizer state (Adam momentum, etc.)
- `actor_optimizer_state: Dict` - Actor optimizer state
- `config: ExperimentConfig` - Full hyperparameter configuration
- `step: int` - Global training step (for resumption)
- `episode: int` - Current episode number (for resumption)
- `metrics: Dict[str, float]` - Latest performance metrics (reward, RMSE, success rate)
- `git_commit: str` - Git commit hash for code version
- `timestamp: str` - ISO 8601 timestamp (e.g., "2026-02-25T21:00:00Z")

**Validation Rules**:
- State dicts must match model architectures (checked on load)
- `metrics` must be non-empty (at least one metric logged)
- `git_commit` must be 7-40 char hexadecimal (short or full hash)
- `step >= 0`, `episode >= 0`

**Relationships**:
- **Producer**: Training loop saves checkpoints periodically
- **Consumer**: Evaluation script loads checkpoints for testing
- **Storage**: Saved to `checkpoints/<experiment_name>_step<N>.pt`

**Example**:
```python
@dataclass
class Checkpoint:
    observer_state_dict: Dict[str, torch.Tensor]
    actor_state_dict: Dict[str, torch.Tensor]
    observer_optimizer_state: Dict
    actor_optimizer_state: Dict
    config: ExperimentConfig
    step: int
    episode: int
    metrics: Dict[str, float]
    git_commit: str
    timestamp: str
    
    def save(self, path: str):
        """Save checkpoint to disk."""
        torch.save(asdict(self), path)
    
    @classmethod
    def load(cls, path: str) -> "Checkpoint":
        """Load checkpoint from disk."""
        data = torch.load(path, map_location="cpu")
        data["config"] = ExperimentConfig(**data["config"])
        return cls(**data)
```

---

## Aggregate Entities

### ReplayBuffer

**Purpose**: Store transitions for off-policy training.

**Fields**:
- `buffer: Deque[Transition]` - Fixed-size FIFO queue
- `capacity: int` - Maximum buffer size
- `

_size: int` - Current number of transitions stored

**Operations**:
- `add(transition: Transition)` - Add new transition (evict oldest if full)
- `sample(batch_size: int) -> List[Transition]` - Random batch sampling
- `__len__() -> int` - Current size

**Validation**:
- `capacity > 0`
- Sampled batch_size <= buffer size

---

## Relationships Diagram

```text
Environment
    ↓ (obs, reward, done)
Transition ──────────────────┐
    ↓                        │
ReplayBuffer                 │
    │                        │
    ├─→ Observer Training    │ (obs, action, next_obs)
    │   (predict next_obs)   │
    │                        │
    └─→ Actor Training       │ (latent_z, action, reward, next_latent_z, done)
        (maximize reward)    │
                             │
Observer ←───────────────────┘
    ↓ (latent_z)
Actor
    ↓ (action)
Environment
```

**Key Architectural Boundaries**:
- Observer receives `(obs, action, next_obs)` - NO rewards
- Actor receives `(latent_z, action, reward, next_latent_z)` - NO raw observations
- Transition stores both for separation

---

## Summary

**Entity Count**: 5 core entities (ObserverState, ActorState, Transition, ExperimentConfig, Checkpoint)  
**Aggregate Count**: 1 (ReplayBuffer)

**Constitutional Compliance**:
- ✅ Modular Architecture: Observer and Actor states are separate
- ✅ Dual Optimization: Transition provides separate views (to_observer_batch vs. to_actor_batch)
- ✅ Reproducibility: ExperimentConfig + Checkpoint capture full experiment state

**Next Steps**: Define module contracts (Observer, Actor, Environment interfaces).
