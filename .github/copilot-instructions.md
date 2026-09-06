# AIXI Development Guidelines

Auto-generated from feature plans. Last updated: 2026-02-25

## Project Overview

AIXI is an RL research project implementing **Observer-Actor architecture** with model-based planning. Observer learns world dynamics via prediction, Actor learns policies via task rewards, and planning enables imagination-based decision making.

**Constitutional Principles** (`.specify/memory/constitution.md`):
1. **Modular Architecture**: Observer/Actor separation with `z_t` latent interface only
2. **Dual Optimization**: Prediction loss (Observer) vs. reward (Actor) - never mix
3. **Model-Based Planning**: Actor queries Observer.predict() for rollouts
4. **Reproducible Experiments**: Strict seed management, W&B tracking
5. **Incremental Complexity**: Baseline → Observer → Actor → Planning progression

## Active Technologies

### Core Stack
- **Python 3.11+**: Modern type hints, match statements, performance improvements
- **PyTorch 2.1+**: Neural networks (Observer encoder/predictor, Actor Q-network)
- **Gymnasium 0.29+**: RL environment interface (GridWorld-1D, CartPole-v1)
- **Weights & Biases 0.16+**: Experiment tracking, metrics visualization
- **pytest 7.4+**: Testing framework with coverage and hypothesis property-based testing

### Development Tools
- **Poetry** or **uv**: Dependency management (prefer uv for speed)
- **Black**: Code formatting (line length 88)
- **mypy**: Type checking (strict mode)
- **pytest-cov**: Code coverage reporting
- **W&B**: Cloud logging for experiments

## Project Structure

```text
aixi/
├── src/
│   ├── observer/          # World model (prediction)
│   │   ├── encoder.py     # obs → latent_z encoding
│   │   ├── predictor.py   # (z, action) → next_obs prediction
│   │   ├── losses.py      # MSE prediction loss (NO rewards)
│   │   └── models.py      # MLP architectures
│   ├── actor/             # Policy (decision-making)
│   │   ├── policy.py      # latent_z → action (NO raw obs)
│   │   ├── planning.py    # K-step rollouts using Observer
│   │   ├── losses.py      # DQN TD-error (task rewards only)
│   │   └── dqn.py         # DQN implementation
│   ├── environments/      # GridWorld, CartPole wrappers
│   │   ├── gridworld.py   # 1D GridWorld (10 states, goal at 9)
│   │   └── wrappers.py    # Normalization, logging
│   ├── utils/             # Config, metrics, reproducibility
│   │   ├── config.py      # Experiment config dataclasses
│   │   ├── metrics.py     # RMSE, success rate computation
│   │   ├── reproducibility.py  # set_seed() for determinism
│   │   └── visualization.py    # Plotting utilities
│   └── tracking/          # W&B integration, checkpoints
│       ├── wandb_logger.py
│       └── checkpointing.py
├── experiments/
│   ├── configs/           # YAML hyperparameter files
│   ├── scripts/           # Training entry points (train_observer.py, etc.)
│   └── baselines/         # Model-free DQN baseline
├── tests/                 # Unit + integration tests
└── specs/                 # Feature specifications
    └── 001-observer-actor-mvp/  # Current feature
        ├── spec.md        # Requirements (4 experiment scenarios)
        ├── plan.md        # Implementation plan
        ├── research.md    # Architectural decisions
        ├── data-model.md  # Entity schemas
        ├── contracts/     # Module interfaces
        └── quickstart.md  # Getting started guide
```

## Key Architectural Patterns

### Observer Interface (World Model)

```python
# contracts/observer.md - Constitutional constraints enforced

class Observer(nn.Module):
    def encode(self, obs: Tensor) -> Tensor:
        """Compress observation to latent state z_t.
        
        Args:
            obs: Raw environment observation
        Returns:
            latent_z: Compressed representation (latent_dim,)
        """
        # MLP: obs → 32 → 16 → latent_dim (8 for GridWorld)
        
    def predict(self, latent_z: Tensor, action: Tensor) -> Tensor:
        """Predict next observation from latent state + action.
        
        Args:
            latent_z: Current latent state
            action: Action to simulate
        Returns:
            next_obs_pred: Predicted next observation
        """
        # MLP: [latent_z, action] → 32 → obs_dim
        # Used by Actor.plan() for imagination
        
    def compute_loss(self, obs, action, next_obs) -> Tensor:
        """Compute prediction loss (MSE only).
        
        CONSTITUTIONAL CONSTRAINT: NO rewards in this loss.
        Observer learns via prediction error, not task performance.
        """
        next_obs_pred = self.predict(self.encode(obs), action)
        return F.mse_loss(next_obs_pred, next_obs)
```

### Actor Interface (Policy)

```python
# contracts/actor.md - Constitutional constraints enforced

class Actor(nn.Module):
    def select_action(self, latent_z: Tensor, training: bool) -> int:
        """Select action from latent state (epsilon-greedy).
        
        Args:
            latent_z: Encoded state from Observer (NOT raw obs)
            training: If True, use epsilon-greedy; else greedy
        Returns:
            action: Integer action
            
        CONSTITUTIONAL CONSTRAINT: NO raw observations.
        Actor only sees latent states from Observer.
        """
        # DQN Q-network: latent_z → 64 → 64 → action_dim
        
    def compute_loss(self, batch: Dict) -> Tensor:
        """Compute DQN TD-error loss.
        
        batch contains: {latent_z, action, reward, next_latent_z, done}
        Uses task rewards (NOT prediction error).
        """
        # Double DQN loss, target network updated every 100 steps
        
    def plan(self, latent_z: Tensor, observer: Observer, horizon: int) -> int:
        """Select action via K-step rollout planning.
        
        Args:
            latent_z: Current state
            observer: World model for imagination
            horizon: K-step lookahead depth
        Returns:
            best_action: Action with highest cumulative Q-value
            
        CONSTITUTIONAL REQUIREMENT: Planning queries Observer.predict()
        for imagination-based decision making.
        """
        # Depth-first rollout: O(action_dim^K) evaluations
```

### Key Entities (data-model.md)

```python
@dataclass
class ObserverState:
    latent_z: Tensor           # Current latent encoding
    obs_history: Optional[Deque]  # For recurrent Observer (future)
    encoding_time: Optional[float]  # Performance tracking

@dataclass
class ActorState:
    epsilon: float             # Current exploration rate
    step_count: int            # Global step counter
    last_action: Optional[int] # For debugging
    target_update_counter: int # Track target network updates

@dataclass
class Transition:
    """Dual-view transition for Observer and Actor."""
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool
    latent_z: np.ndarray       # Observer encoding
    next_latent_z: np.ndarray  # Next state encoding

@dataclass
class ExperimentConfig:
    # Environment
    env_name: str
    seed: int
    
    # Observer
    latent_dim: int = 8        # GridWorld: 8, CartPole: 32
    encoder_hidden: List[int] = field(default_factory=lambda: [32, 16])
    predictor_hidden: List[int] = field(default_factory=lambda: [32])
    observer_lr: float = 1e-3
    
    # Actor
    policy_hidden: List[int] = field(default_factory=lambda: [64, 64])
    actor_lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5000
    
    # Training
    buffer_size: int = 10000
    batch_size: int = 64
    target_update_interval: int = 100
    
    # Planning
    planning_horizon: int = 3  # K-step rollout depth
    
    # Tracking
    wandb_project: str = "aixi-observer-actor"
```

## Common Commands

### Setup
```bash
# Install dependencies
poetry install  # or: uv pip install -e ".[dev]"

# Activate environment
poetry shell

# Run tests
pytest tests/ -v --cov=src
```

### Running Experiments
```bash
# ES1: Random baseline (validate infrastructure)
python experiments/scripts/run_random_baseline.py --env GridWorld-1D --seed 42

# ES2: Train Observer (world model)
python experiments/scripts/train_observer.py --config experiments/configs/observer_train.yaml --seed 42

# ES3: Train Actor on latent states
python experiments/scripts/train_actor.py --observer_checkpoint checkpoints/observer_latest.pt --seed 42

# ES4: Enable model-based planning
python experiments/scripts/train_planning.py --planning_horizon 3 --seed 42

# Evaluate saved checkpoint
python experiments/scripts/evaluate.py --observer_checkpoint checkpoints/observer_best.pt --actor_checkpoint checkpoints/actor_best.pt --n_eval_episodes 100
```

### Development Workflow
```bash
# Type checking
mypy src/

# Code formatting
black src/ tests/

# Run specific test
pytest tests/test_observer/test_prediction.py -v

# Check constitutional compliance
pytest tests/test_constitution.py  # Verifies no reward leakage, no obs leakage
```

## Code Style

### Python Conventions
- **Type hints**: Required for all function signatures
- **Docstrings**: Google style for all public methods
- **Line length**: 88 characters (Black default)
- **Imports**: isort with Black-compatible settings
- **Constants**: UPPER_CASE for module-level constants

### RL-Specific Patterns

**Reproducibility** (CRITICAL):
```python
from src.utils.reproducibility import set_seed

# ALWAYS call before any randomness
set_seed(seed=42)  # Sets Python, NumPy, PyTorch, Gymnasium seeds

# For exact GPU reproducibility
torch.backends.cudnn.deterministic = True  # Already in set_seed()
```

**Experiment Tracking** (W&B integration):
```python
from src.tracking.wandb_logger import WandbLogger

logger = WandbLogger(project="aixi-observer-actor", config=exp_config)
logger.log({"observer/pred_loss": loss.item(), "step": step})
logger.log_episode({"reward": episode_reward, "success": success})
```

**Checkpointing** (with metadata):
```python
from src.tracking.checkpointing import save_checkpoint, load_checkpoint

# Save
save_checkpoint(
    path="checkpoints/observer_001.pt",
    observer=observer,
    optimizer=optimizer,
    config=config,
    step=step,
    metrics={"rmse": 0.15}
)

# Load
checkpoint = load_checkpoint("checkpoints/observer_001.pt")
observer.load_state_dict(checkpoint["observer_state_dict"])
```

## Testing Requirements

### Constitutional Tests (tests/test_constitution.py)
```python
def test_observer_no_reward_leakage():
    """Observer loss must not contain task rewards."""
    observer = Observer(latent_dim=8, obs_dim=10, action_dim=3)
    loss = observer.compute_loss(obs, action, next_obs)  # No reward argument
    # If this function signature accepts rewards → FAIL

def test_actor_no_raw_obs():
    """Actor must only see latent states, not raw observations."""
    actor = Actor(latent_dim=8, action_dim=3)
    action = actor.select_action(latent_z=z)  # No obs argument
    # If this function signature accepts raw obs → FAIL
    
def test_planning_uses_observer():
    """Planning must query Observer.predict(), not ground truth."""
    actor = Actor(latent_dim=8, action_dim=3)
    observer = Observer(latent_dim=8, obs_dim=10, action_dim=3)
    
    action = actor.plan(latent_z=z, observer=observer, horizon=3)
    # Verify Observer.predict() was called during planning
```

### Property-Based Tests (hypothesis)
```python
from hypothesis import given, strategies as st

@given(obs=st.tensors(shape=(1, 10), dtype=torch.float32))
def test_encoder_determinism(obs):
    """Encoder must be deterministic: same input → same output."""
    observer = Observer(latent_dim=8, obs_dim=10, action_dim=3)
    z1 = observer.encode(obs)
    z2 = observer.encode(obs)
    assert torch.allclose(z1, z2), "Encoder must be deterministic"
```

## Performance Expectations

| Operation | Target Latency | Notes |
|-----------|----------------|-------|
| Observer.encode() | <1ms | MLP forward pass |
| Observer.predict() | <1ms | MLP forward pass |
| Actor.select_action() | <1ms | Q-network forward |
| Actor.plan(K=3) | <30ms | 27 rollouts (3^3) |
| Episode (GridWorld) | <100ms | 100 steps max |
| Observer training | <10s | 1000 gradient steps |
| Actor training | <5min | 10,000 env steps |

## Recent Features

### 001-observer-actor-mvp (2026-02-25)
**Added**:
- Observer module (encode, predict, compute_loss)
- Actor module (select_action, compute_loss, plan)
- GridWorld-1D environment (10 states, deterministic)
- DQN baseline for comparison
- W&B experiment tracking
- Reproducibility utilities (set_seed, deterministic PyTorch)
- 4 experiment scenarios: Random → Observer → Actor → Planning

**Key Decisions** (research.md):
- Observer: MLP encoder/predictor (not linear tabular)
- Actor: DQN with epsilon-greedy (not PPO)
- Planning: Depth-first K-step rollout (O(actions^K))
- Tracking: W&B (not MLflow or TensorBoard)
- Baselines: Stable-Baselines3 DQN on raw obs

<!-- MANUAL ADDITIONS START -->
<!-- Add project-specific conventions, team preferences, or custom guidelines here -->
<!-- MANUAL ADDITIONS END -->
