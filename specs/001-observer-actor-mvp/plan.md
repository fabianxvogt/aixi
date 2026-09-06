# Implementation Plan: Observer-Actor RL with Model-Based Planning

**Branch**: `001-observer-actor-mvp` | **Date**: 2026-02-25 | **Spec**: [spec.md](spec.md)  
**Input**: Feature specification from `/specs/001-observer-actor-mvp/spec.md`

**Note**: This plan follows AIXI Constitution v1.0.0 principles. All design decisions enforce modular architecture, dual optimization, and reproducible experiments.

## Summary

Build an AIXI-inspired reinforcement learning system with strict architectural separation between world modeling (Observer) and decision-making (Actor). Observer learns environment dynamics via prediction error minimization, Actor learns task policies via reward maximization, and planning enables imagination-based decision making. Start with simple GridWorld environment, establish baselines, validate architecture, then enable model-based planning with rollout simulation.

**Technical Approach**: PyTorch neural networks for Observer encoder/predictor and Actor policy, Gymnasium for environment interface, Weights & Biases for experiment tracking, modular architecture enables independent testing and comparison against model-free baselines (DQN).

## Technical Context

**Language/Version**: Python 3.11+ (for match statements, improved type hints, performance)  
**Primary Dependencies**: PyTorch 2.1+, Gymnasium 0.29+, NumPy 1.24+, Weights & Biases (wandb) 0.16+  
**Storage**: Local filesystem for checkpoints, W&B cloud for experiment logs and metrics  
**Testing**: pytest 7.4+ with pytest-cov for coverage, hypothesis for property-based testing of RL components  
**Target Platform**: Linux/macOS development machines with CUDA-capable GPU (optional, CPU fallback supported)  
**Project Type**: Research library + experiment scripts (not production deployment)  
**Performance Goals**: 
- Observer training: <10 seconds for 1000 gradient steps on GridWorld (CPU)
- Actor training: <5 minutes for 10,000 environment steps (CPU)
- Planning overhead: <100ms per action decision with K=3 rollouts
- Experiment throughput: 100 episodes in <2 minutes for GridWorld

**Constraints**:
- Modular architecture: Observer and Actor must be independently loadable/testable
- No shared parameters between Observer and Actor (separate optimizers, state dicts)
- Prediction loss must never leak into Actor, task rewards must never leak into Observer
- All experiments logged with reproducible seeds (Python, NumPy, PyTorch)
- Checkpoint size <50MB per experiment (efficient for git LFS or local storage)

**Scale/Scope**: 
- Single-agent, single-environment for MVP
- GridWorld: 10-state 1D environment (goal-reaching task)
- CartPole extension: 4D continuous observations, 2 discrete actions (Phase 2 validation)
- Observer latent dimension: 8-16 for GridWorld, 32-64 for CartPole
- Training budget: ~20,000 environment interactions per experiment
- 3-5 experiment runs per configuration for statistical significance

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Constitutional Gates (AIXI v1.0.0):**

- [x] **Modular Architecture**: Observer and Actor are separate modules with clear interface (`z_t` only)
  - Observer: `src/observer/{encoder.py, predictor.py, losses.py}`
  - Actor: `src/actor/{policy.py, planning.py, losses.py}`
  - Interface: `z_t = observer.encode(obs)`, `action = actor.policy(z_t)`
  
- [x] **Dual Optimization**: Observer uses prediction loss, Actor uses task reward (no mixing)
  - Observer optimizer: Adam(observer.parameters(), lr_obs)
  - Actor optimizer: Adam(actor.parameters(), lr_actor)
  - Verified via separate training loops and metric logging
  
- [x] **Model-Based Planning**: Actor can query Observer for state predictions
  - `z_next_pred = observer.predict(z_t, action)` interface implemented
  - Planning module simulates K-step rollouts for action selection
  - Configurable planning horizon K ∈ {0, 1, 3, 5}
  
- [x] **Reproducible Experiments**: Experiment tracking configured (W&B), logging spec defined
  - W&B integration for all training runs with hyperparameter logging
  - Checkpoints saved with metadata: {step, timestamp, metrics, config, git commit}
  - Random seed management: `set_seed(seed)` function for Python/NumPy/PyTorch
  
- [x] **Incremental Complexity**: Baseline established before adding complexity
  - Phase 0: Random baseline (no learning)
  - Phase 1: Observer-only (world model)
  - Phase 2: Actor on latent states (policy learning)
  - Phase 3: Model-based planning (imagination)
  
- [x] **Baselines Defined**: List of baseline comparisons (random, model-free, oracle where applicable)
  - Random agent: uniform action sampling
  - Model-free DQN: standard RL baseline on raw observations
  - Oracle Observer: perfect world model (deterministic lookup for GridWorld)
  
- [x] **Metrics Defined**: Observer metrics (RMSE, NLL) and Actor metrics (reward, sample efficiency)
  - Observer: prediction RMSE, per-dimension MSE, calibration (if probabilistic)
  - Actor: cumulative episode reward, success rate, sample efficiency (reward vs. steps)
  - Planning: rollout accuracy, planning horizon, wall-clock time per decision

**Gate Status**: ✅ **PASSED** - All constitutional requirements satisfied in design

## Project Structure

### Documentation (this feature)

```text
specs/001-observer-actor-mvp/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (technology decisions, best practices)
├── data-model.md        # Phase 1 output (entity schemas, state representations)
├── quickstart.md        # Phase 1 output (getting started, running experiments)
├── contracts/           # Phase 1 output (module interfaces, API contracts)
│   ├── observer.md      # Observer interface specification
│   ├── actor.md         # Actor interface specification
│   └── environment.md   # Environment wrapper specification
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
aixi/                    # Repository root
├── src/
│   ├── observer/        # World model (prediction)
│   │   ├── __init__.py
│   │   ├── encoder.py       # obs → latent_z mapping
│   │   ├── predictor.py     # (z, action) → next_obs prediction
│   │   ├── losses.py        # prediction losses (MSE, MDL)
│   │   └── models.py        # specific architectures (MLP, linear)
│   ├── actor/           # Policy (decision-making)
│   │   ├── __init__.py
│   │   ├── policy.py        # latent_z → action mapping
│   │   ├── planning.py      # rollout-based planning
│   │   ├── losses.py        # RL losses (policy gradient, Q-learning)
│   │   └── dqn.py           # DQN baseline implementation
│   ├── environments/    # Custom environments + wrappers
│   │   ├── __init__.py
│   │   ├── gridworld.py     # 1D GridWorld implementation
│   │   └── wrappers.py      # Gymnasium wrappers (logging, normalization)
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py        # configuration management (dataclasses)
│   │   ├── metrics.py       # metric computation (RMSE, success rate)
│   │   ├── reproducibility.py  # seed setting, determinism
│   │   └── visualization.py # plotting (learning curves, heatmaps)
│   └── tracking/
│       ├── __init__.py
│       ├── wandb_logger.py  # W&B integration
│       └── checkpointing.py # save/load with metadata
├── experiments/
│   ├── configs/         # Hyperparameter configs (YAML or Python)
│   │   ├── gridworld_random.yaml
│   │   ├── observer_train.yaml
│   │   ├── actor_train.yaml
│   │   └── planning.yaml
│   ├── scripts/         # Experiment entry points
│   │   ├── run_random_baseline.py
│   │   ├── train_observer.py
│   │   ├── train_actor.py
│   │   ├── train_planning.py
│   │   └── evaluate.py
│   └── baselines/       # Baseline implementations
│       └── dqn_baseline.py
├── tests/
│   ├── test_observer/   # Observer prediction tests
│   │   ├── test_encoder.py
│   │   ├── test_predictor.py
│   │   └── test_losses.py
│   ├── test_actor/      # Actor policy tests
│   │   ├── test_policy.py
│   │   ├── test_planning.py
│   │   └── test_dqn.py
│   ├── test_integration/ # Full agent tests
│   │   ├── test_observer_actor_integration.py
│   │   └── test_planning_integration.py
│   └── test_environments/ # Environment validity tests
│       └── test_gridworld.py
├── pyproject.toml       # Poetry/uv dependencies
├── README.md            # Project overview, installation
└── .gitignore           # Ignore checkpoints/, wandb/, __pycache__
```

**Structure Decision**: Selected RL/ML project structure with Observer-Actor modularity. Rationale:
- Clear separation of concerns enables independent development and testing
- `src/observer/` and `src/actor/` enforce architectural boundaries
- `experiments/` directory separates research scripts from library code
- `tests/` mirror source structure for comprehensive coverage
- Configuration management via `experiments/configs/` supports reproducibility

## Complexity Tracking

> **Not Applicable** - No constitutional violations to justify. All design decisions align with principles.

## Phase 0: Outline & Research

**Purpose**: Resolve all unknowns in Technical Context and establish best practices for Observer-Actor RL implementation.

### Research Tasks

1. **Observer Architecture Decisions**
   - **Unknown**: Optimal latent dimension for GridWorld (8 vs 16 vs 32)?
   - **Unknown**: Encoder architecture (linear projection vs. MLP) for discrete observations?
   - **Unknown**: Predictor architecture (predict raw obs vs. predict latent z)?
   - **Research**: Survey World Models paper (Ha & Schmidhuber), model-based RL best practices
   - **Decision Point**: Separate encoder/predictor vs. joint architecture

2. **Actor Training Algorithm**
   - **Unknown**: DQN vs. Policy Gradient (REINFORCE) vs. PPO for discrete actions?
   - **Unknown**: On-policy vs. off-policy learning on latent states?
   - **Unknown**: Replay buffer management when Observer updates (non-stationary latent space)?
   - **Research**: Stable-Baselines3 implementations, PyTorch RL tutorials
   - **Decision Point**: Choose one RL algorithm for MVP, defer others to ablations

3. **Planning Implementation**
   - **Unknown**: Rollout strategy (breadth-first vs. depth-first vs. MCTS)?
   - **Unknown**: Action selection from rollouts (max Q-value vs. UCB vs. policy-weighted)?
   - **Unknown**: How to handle stochasticity in planning (deterministic GridWorld only for MVP)?
   - **Research**: AlphaZero planning, Dyna-style model-based RL, MuZero architecture
   - **Decision Point**: Start with simple depth-first rollouts, defer MCTS

4. **Experiment Tracking Best Practices**
   - **Unknown**: W&B vs. MLflow vs. TensorBoard for RL experiments?
   - **Unknown**: What hyperparameters to log (all config vs. critical subset)?
   - **Unknown**: Checkpoint frequency (every N episodes vs. best model only)?
   - **Research**: RL Baselines Zoo configuration management, ML experiment tracking guides
   - **Decision Point**: Select W&B (recommended) or MLflow, define logging schema

5. **Reproducibility Implementation**
   - **Unknown**: How to handle PyTorch non-determinism (CUDA, cudnn.benchmark)?
   - **Unknown**: What seeds to set (Python random, NumPy, PyTorch, environment)?
   - **Unknown**: How to validate reproducibility (re-run verification)?
   - **Research**: PyTorch reproducibility docs, RL reproducibility crisis papers
   - **Decision Point**: Implement strict seed setting, document known non-deterministic operations

6. **Baseline Implementation Strategy**
   - **Unknown**: Use Stable-Baselines3 DQN or roll custom implementation?
   - **Unknown**: Baseline hyperparameters (literature defaults vs. GridWorld-specific tuning)?
   - **Unknown**: How much tuning effort for baselines (minimal vs. fair comparison)?
   - **Research**: Stable-Baselines3 API, DQN hyperparameter sensitivity studies
   - **Decision Point**: Use SB3 for baselines, minimal tuning (literature defaults)

### Research Output

**Deliverable**: `research.md` containing:

```markdown
# Research: Observer-Actor RL Implementation Decisions

## 1. Observer Architecture

**Decision**: MLP Encoder (obs → latent_z) + MLP Predictor ((z, action) → next_obs)

**Rationale**:
- Predicting next observation enables direct validation against ground truth
- Separate encoder allows flexible latent dimension independent of obs dimension
- MLP handles discrete/continuous observations uniformly (GridWorld → CartPole extension)

**Alternatives Considered**:
- Linear encoder: Too simple, may not capture non-linear dynamics (rejected for CartPole extension)
- Predict next latent: Requires additional decoder, harder to validate (deferred to future)

**Architecture Specs**:
- Encoder: [obs_dim → 32 → 16 → latent_dim] with ReLU
- Predictor: [latent_dim + action_dim → 32 → obs_dim] with ReLU → Linear output
- Latent dim: 8 for GridWorld, 32 for CartPole

## 2. Actor Training Algorithm

**Decision**: DQN (Deep Q-Network) with experience replay on latent states

**Rationale**:
- Off-policy learning tolerates non-stationary latent space during joint training
- Experience replay improves sample efficiency
- Q-values directly comparable to model-free baseline
- Well-established, easier to debug than policy gradients

**Alternatives Considered**:
- REINFORCE: On-policy, lower sample efficiency (rejected)
- PPO: More complex, requires value function (deferred to Phase 2)

**Hyperparameters** (from DQN paper + SB3 defaults):
- Replay buffer: 10,000 transitions
- Batch size: 64
- Learning rate: 1e-3
- Gamma (discount): 0.99
- Epsilon (exploration): 1.0 → 0.05 (linear decay over 5000 steps)
- Target network update: every 100 steps

## 3. Planning Implementation

**Decision**: Depth-first K-step rollout with Q-value aggregation

**Rationale**:
- Simple to implement and debug
- Computational cost O(n_actions^K) acceptable for discrete actions and small K
- Q-value aggregation naturally extends DQN

**Algorithm**:
```
For each candidate action a:
  1. Predict next latent: z' = observer.predict(z, a)
  2. Recursively rollout K-1 steps (greedy or sampled)
  3. Aggregate Q-values: Q_rollout(z, a) = r + γ * max_a' Q(z', a')
Choose action: argmax_a Q_rollout(z, a)
```

**Alternatives Considered**:
- MCTS: More sophisticated but complex, requires tuning (deferred)
- Breadth-first: Same complexity, no clear advantage (deferred)

## 4. Experiment Tracking

**Decision**: Weights & Biases (wandb)

**Rationale**:
- Excellent RL support (videos, histograms, custom charts)
- Free tier sufficient for MVP
- Clean API, minimal boilerplate
- Automatic hyperparameter tracking

**Logging Schema**:
```python
wandb.init(project="aixi-observer-actor", config={
    "env": "GridWorld-1D",
    "observer_latent_dim": 8,
    "observer_lr": 1e-3,
    "actor_lr": 1e-3,
    "planning_horizon": 3,
    "seed": 42,
    # ... all hyperparameters
})

# Per-step metrics
wandb.log({"observer/pred_rmse": rmse, "step": global_step})
wandb.log({"actor/episode_reward": reward, "episode": episode_num})
```

**Alternatives Considered**:
- MLflow: More complex setup, less RL-specific tooling (rejected)
- TensorBoard: Local only, less collaborative (rejected for multi-run comparison)

## 5. Reproducibility

**Decision**: Strict seed setting with documentation of non-deterministic ops

**Implementation**:
```python
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Known Non-Determinism** (documented):
- Multi-threaded DataLoader (use num_workers=0 for reproducibility)
- Some PyTorch ops on GPU (document in README)

**Validation**: Re-run same config with same seed must produce identical results (±1e-6)

## 6. Baseline Implementation

**Decision**: Stable-Baselines3 DQN for model-free baseline

**Rationale**:
- Reference implementation, well-tested
- Apples-to-apples comparison (same hyperparameters except raw obs vs. latent)
- Minimal implementation effort

**Usage**:
```python
from stable_baselines3 import DQN
baseline = DQN("MlpPolicy", env, learning_rate=1e-3, buffer_size=10000, ...)
```

**Alternatives Considered**:
- Custom DQN: More control but risk of bugs (rejected for baseline)
- CleanRL: Good reference but no API, copy implementation (deferred)

## Summary

All NEEDS CLARIFICATION items resolved. Ready for Phase 1 design.
```

## Phase 1: Design & Contracts

**Purpose**: Define data models, module interfaces, and public contracts for Observer-Actor system.

### 1.1 Data Model (`data-model.md`)

**Entities**:

**ObserverState**
- **Fields**: `latent_z: Tensor(latent_dim)`, `obs_history: Optional[Deque[Tensor]]`
- **Description**: Compressed world state representation output by Observer
- **Validation**: `latent_z` must have shape (latent_dim,), finite values
- **Relationships**: Input to Actor

**ActorState**
- **Fields**: `epsilon: float`, `step_count: int`, `last_action: Optional[int]`
- **Description**: Actor's decision-making state (exploration schedule, history)
- **Validation**: `epsilon ∈ [0,1]`, `step_count >= 0`
- **State Transitions**: `epsilon` decays linearly over training steps

**Transition**
- **Fields**: `obs: Tensor`, `action: int`, `reward: float`, `next_obs: Tensor`, `done: bool`, `latent_z: Optional[Tensor]`, `next_latent_z: Optional[Tensor]`
- **Description**: Single environment interaction, stored in replay buffer
- **Validation**: `obs` and `next_obs` same shape, `reward` finite, `action` valid for environment
- **Relationships**: Aggregated into replay buffer, used for Observer and Actor training

**ExperimentConfig**
- **Fields**: 
  - Environment: `env_name: str`, `max_episode_steps: int`
  - Observer: `latent_dim: int`, `observer_lr: float`, `encoder_hidden: List[int]`, `predictor_hidden: List[int]`
  - Actor: `actor_lr: float`, `gamma: float`, `epsilon_start: float`, `epsilon_end: float`, `epsilon_decay_steps: int`
  - Training: `n_episodes: int`, `batch_size: int`, `buffer_size: int`, `update_frequency: int`
  - Planning: `planning_horizon: int`, `use_planning: bool`
  - Reproducibility: `seed: int`, `device: str`
- **Description**: Complete training configuration for reproducibility
- **Validation**: All hyperparameters within reasonable ranges (e.g., `lr > 0`, `0 <= gamma <= 1`)

**Checkpoint**
- **Fields**: `observer_state_dict: Dict`, `actor_state_dict: Dict`, `optimizer_states: Dict`, `config: ExperimentConfig`, `step: int`, `metrics: Dict`, `git_commit: str`, `timestamp: str`
- **Description**: Saveable experiment state for resumption and evaluation
- **Validation**: State dicts match model architectures, metrics non-empty

### 1.2 Module Contracts (`contracts/`)

**contracts/observer.md**:

```markdown
# Observer Module Contract

## Purpose
Learn world model via next-observation prediction. Compress observations into latent states.

## Public Interface

### Observer Class

**Constructor**:
```python
Observer(obs_dim: int, action_dim: int, latent_dim: int, 
         encoder_hidden: List[int], predictor_hidden: List[int])
```

**Methods**:

`encode(obs: Tensor) -> Tensor`:
- **Input**: Observation tensor, shape (batch, obs_dim) or (obs_dim,)
- **Output**: Latent state tensor, shape (batch, latent_dim) or (latent_dim,)
- **Contract**: Deterministic encoding, no side effects

`predict(latent_z: Tensor, action: int) -> Tensor`:
- **Input**: Latent state (batch, latent_dim), action (batch,) or scalar
- **Output**: Predicted next observation (batch, obs_dim)
- **Contract**: Differentiable, used for planning rollouts

`compute_loss(obs: Tensor, action: Tensor, next_obs: Tensor) -> Tensor`:
- **Input**: Batch of transitions
- **Output**: Scalar prediction loss (MSE)
- **Contract**: Only prediction error, NO task rewards

`save(path: str) -> None`:
- **Contract**: Save state_dict, can be loaded independently of Actor

`load(path: str) -> None`:
- **Contract**: Load state_dict, restore exact encoding function

## Constraints (Constitutional)
- MUST NOT access task rewards during training
- MUST be independently testable (no Actor dependency)
- Latent state z_t is ONLY interface to Actor
```

**contracts/actor.md**:

```markdown
# Actor Module Contract

## Purpose
Learn policy to maximize task rewards using Observer's latent states. Support planning via Observer's world model.

## Public Interface

### Actor Class

**Constructor**:
```python
Actor(latent_dim: int, action_dim: int, hidden_dims: List[int],
      gamma: float, epsilon_start: float, epsilon_end: float, epsilon_decay_steps: int)
```

**Methods**:

`select_action(latent_z: Tensor, training: bool = True) -> int`:
- **Input**: Latent state (latent_dim,), training mode flag
- **Output**: Action integer
- **Contract**: Uses epsilon-greedy if training=True, greedy if training=False

`compute_loss(batch: Dict[str, Tensor]) -> Tensor`:
- **Input**: Batch with keys {latent_z, action, reward, next_latent_z, done}
- **Output**: Scalar DQN loss (TD error)
- **Contract**: Only task rewards, NO prediction errors

`update_target_network() -> None`:
- **Contract**: Copy Q-network weights to target network (DQN)

`save(path: str) -> None`:
- **Contract**: Save state_dict, can be loaded independently of Observer

`load(path: str) -> None`:
- **Contract**: Load state_dict, restore exact policy

## Planning Interface (Optional)

`plan(latent_z: Tensor, observer: Observer, horizon: int) -> int`:
- **Input**: Current latent state, Observer reference, planning horizon K
- **Output**: Best action from K-step rollout
- **Contract**: Queries observer.predict() for imagination, returns action

## Constraints (Constitutional)
- MUST NOT access raw observations (only latent states)
- MUST be independently testable (Observer can be mocked)
- Planning MUST use Observer.predict() interface only
```

**contracts/environment.md**:

```markdown
# Environment Contract

## Purpose
Provide standardized interface for RL environments (Gymnasium-compatible).

## GridWorld-1D Specification

**Observation Space**: Discrete(10) - position in [0, 9]  
**Action Space**: Discrete(3) - {left=0, stay=1, right=2}  
**Reward**: +1.0 if position == goal (position 9), else 0.0  
**Termination**: Episode ends when goal reached or max_steps (100) exceeded

**Dynamics** (deterministic):
- left: position = max(0, position - 1)
- stay: position = position
- right: position = min(9, position + 1)

**Initial State**: position = 0 (start)

## Public Interface

`reset() -> Tuple[Tensor, Dict]`:
- **Output**: Initial observation, info dict
- **Contract**: Resets to position 0

`step(action: int) -> Tuple[Tensor, float, bool, bool, Dict]`:
- **Input**: Action in [0, 1, 2]
- **Output**: (next_obs, reward, terminated, truncated, info)
- **Contract**: Deterministic transitions, terminated=True iff goal reached

`render() -> Optional[ndarray]`:
- **Contract**: Text-based visualization of 1D grid
```

### 1.3 Quickstart Guide (`quickstart.md`)

```markdown
# Quickstart: Observer-Actor RL MVP

## Installation

**Requirements**: Python 3.11+, CUDA (optional)

```bash
# Clone repository
git clone <repo-url> aixi
cd aixi

# Install dependencies (using Poetry)
poetry install

# Or using uv
uv pip install -e .
```

## Running Experiments

### 1. Random Baseline (validate environment)

```bash
python experiments/scripts/run_random_baseline.py \
  --env GridWorld-1D \
  --n_episodes 100 \
  --seed 42
```

**Expected Output**: Average reward ~0.05 (random agent rarely reaches goal)

### 2. Train Observer (world model)

```bash
python experiments/scripts/train_observer.py \
  --config experiments/configs/observer_train.yaml \
  --seed 42
```

**Expected Output**: 
- Prediction RMSE < 0.5 after 1000 steps
- W&B dashboard shows decreasing loss curve
- Checkpoint saved to `checkpoints/observer_<timestamp>.pt`

### 3. Train Actor on Latent States

```bash
python experiments/scripts/train_actor.py \
  --observer_checkpoint checkpoints/observer_latest.pt \
  --config experiments/configs/actor_train.yaml \
  --seed 42
```

**Expected Output**:
- Success rate > 80% within 50 episodes
- W&B dashboard shows increasing reward curve
- Checkpoint saved to `checkpoints/actor_<timestamp>.pt`

### 4. Enable Planning (model-based)

```bash
python experiments/scripts/train_planning.py \
  --observer_checkpoint checkpoints/observer_latest.pt \
  --actor_checkpoint checkpoints/actor_latest.pt \
  --planning_horizon 3 \
  --seed 42
```

**Expected Output**:
- Success rate > 95% (20% improvement over reactive)
- Planning overhead <100ms per action

## Evaluation

```bash
python experiments/scripts/evaluate.py \
  --observer_checkpoint checkpoints/observer_best.pt \
  --actor_checkpoint checkpoints/actor_best.pt \
  --n_eval_episodes 100 \
  --render
```

## Troubleshooting

**Issue**: Observer not learning (RMSE not decreasing)
- Check learning rate (try 1e-3 to 1e-4)
- Verify data collection (enough diverse transitions)
- Check gradients: `wandb` logs should show non-zero gradients

**Issue**: Actor gets stuck in local optima
- Increase epsilon (exploration) decay steps
- Check replay buffer diversity
- Verify Observer latent states are informative (visualize with PCA)

**Issue**: Non-reproducible results
- Verify all seeds set: `set_seed(42)` called before environment creation
- Check CUDA non-determinism: use CPU for exact reproducibility
- Disable cudnn.benchmark: already done in `set_seed()`

## Next Steps

- Extend to CartPole: modify configs to use Gymnasium's CartPole-v1
- Implement probabilistic Observer: predict distributions instead of point estimates
- Add intrinsic curiosity: reward=prediction_error for exploration
```

### 1.4 Agent Context Update

**Action**: Run `.specify/scripts/bash/update-agent-context.sh copilot`

**Expected Additions to `.github/copilot-instructions.md`** (between markers):

```markdown
<!-- BEGIN SPECIFY CONTEXT -->

## Project: AIXI Observer-Actor RL

**Tech Stack**: Python 3.11+, PyTorch 2.1+, Gymnasium, Weights & Biases  
**Architecture**: Modular Observer (world model) + Actor (policy) with model-based planning  
**Key Principles** (AIXI Constitution v1.0.0):
- Observer and Actor are separate modules (no shared parameters)
- Observer optimizes prediction loss, Actor optimizes task rewards (strict separation)
- Planning via Observer.predict() interface (imagination/rollouts)
- All experiments tracked with W&B (reproducible seeds, checkpoints)

**Active Feature**: 001-observer-actor-mvp (GridWorld environment, DQN Actor, K-step planning)

**Code Patterns**:
- Observer: `z = observer.encode(obs)`, `next_obs_pred = observer.predict(z, action)`
- Actor: `action = actor.select_action(z)`, loss computed on latent states only
- Never mix: Observer sees prediction errors, Actor sees task rewards (constitutional requirement)

**Testing**: pytest with separate test_observer/, test_actor/, test_integration/ directories

<!-- END SPECIFY CONTEXT -->
```

## Re-Evaluation: Constitution Check (Post-Design)

**Status**: ✅ **PASSED** - All constitutional requirements remain satisfied after design phase.

**Verification**:

- [x] **Modular Architecture**: Contracts define clean Observer ↔ latent_z ↔ Actor interface
- [x] **Dual Optimization**: Observer.compute_loss() uses only prediction error, Actor.compute_loss() uses only task rewards
- [x] **Model-Based Planning**: Actor.plan() interface uses Observer.predict() for rollouts
- [x] **Reproducible Experiments**: ExperimentConfig + W&B + set_seed() ensures full reproducibility
- [x] **Incremental Complexity**: Research.md documents simple-to-complex progression (linear → MLP Observer, random → DQN Actor)
- [x] **Baselines**: DQN baseline via Stable-Baselines3 documented in research.md
- [x] **Metrics**: Observer (RMSE), Actor (reward, success rate), Planning (rollout accuracy) all defined

**No constitutional violations introduced during design.**

## Next Steps

1. **Commit this plan**: 
   ```bash
   git add specs/001-observer-actor-mvp/plan.md
   git commit -m "feat(001): add implementation plan for Observer-Actor RL MVP"
   ```

2. **Generate research.md** (if any NEEDS CLARIFICATION remain - none in this case, but research.md provides value):
   - Research.md already outlined in Phase 0
   - Should be created to document architectural decisions before implementation
   - Run: `echo "Create research.md from Phase 0 outline"`

3. **Generate data-model.md, contracts/, quickstart.md** (Phase 1 outputs already drafted above):
   - Create files from Phase 1.1, 1.2, 1.3 sections
   - Validate against spec.md requirements

4. **Update agent context**:
   ```bash
   .specify/scripts/bash/update-agent-context.sh copilot
   ```

5. **Generate tasks.md** (NOT done by /speckit.plan, requires /speckit.tasks):
   ```bash
   /speckit.tasks  # Generates actionable task list from this plan
   ```

6. **Begin implementation** once tasks.md is approved.

## Phase 2 Planning: Not Yet (Requires /speckit.tasks)

**Note**: This plan concludes after Phase 1 (design). Implementation tasks will be generated by the `/speckit.tasks` command, which creates:
- Phase 1: Setup (project structure, dependencies)
- Phase 2: Foundational (environment, baselines, experiment infrastructure)
- Phase 3-6: Experiment scenarios (ES1 random → ES2 Observer → ES3 Actor → ES4 Planning)
- Phase 7: Evaluation and documentation

**Status**: Plan complete. Ready for task generation.
