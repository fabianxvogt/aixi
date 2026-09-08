<!-- portfolio
{
  "title": "Aixi",
  "topic": "Artificial intelligence/Learning systems",
  "type": "research",
  "description": "AIXI-inspired observer-actor reinforcement learning experiments"
}
-->

# AIXI: Observer-Actor RL with Model-Based Planning

**An AIXI-inspired reinforcement learning system implementing modular Observer-Actor architecture with imagination-based planning.**

## Overview

This project implements a novel RL architecture that strictly separates world modeling (Observer) from decision-making (Actor):

- **Observer** learns environment dynamics via next-observation prediction (no access to task rewards)
- **Actor** learns optimal policies via task reward maximization (no access to raw observations)  
- **Planning** enables imagination-based decision making through K-step mental rollouts

This architectural separation enforces clean interfaces and enables powerful model-based RL capabilities.

## Features

✅ **Modular Architecture**: Observer and Actor are independent modules communicating only through latent states  
✅ **Dual Optimization**: Separate loss functions (prediction error vs. task reward) prevent information leakage  
✅ **Model-Based Planning**: Actor queries Observer to simulate future trajectories  
✅ **Reproducible Experiments**: Strict seed management and W&B experiment tracking  
✅ **Constitutional Compliance**: All design decisions validated against AIXI principles  

## Quick Start

### Installation

**Requirements**: Python 3.11+, CUDA optional

```bash
# Clone repository
git clone <repo-url> aixi
cd aixi

# Install dependencies using Poetry
poetry install

# Activate virtual environment
poetry shell

# Verify installation
pytest tests/ -v
```

**Alternative installation (using uv)**:
```bash
uv pip install -e ".[dev]"
```

### Running Experiments

See [quickstart.md](specs/001-observer-actor-mvp/quickstart.md) for detailed instructions.

```bash
# ES1: Random baseline (validate the environment)
python experiments/scripts/run_random_baseline.py \
  --config experiments/configs/gridworld_random.yaml \
  --seed 42

# ES2: Train the Observer world model
python experiments/scripts/train_observer.py \
  --config experiments/configs/observer_train.yaml \
  --seed 42
```

The experiment scripts currently present provide the random baseline and Observer
training only. Actor training, planning, and checkpoint evaluation are described in
the research plan but do not have entry-point scripts in the current tree.

### Bounded architecture result

A deterministic two-state regression now proves a narrow interface boundary:
perfect action-conditioned next-observation prediction can still provide the
same prediction signature in states with opposite reward-optimal actions.
Planning therefore needs actor-owned reward or value information in addition to
Observer dynamics. A bounded one-logit Actor head now demonstrates the minimal
two-action repair on the existing latent, including a flipped-reward control; it
is not yet the planned general Actor or rollout planner. See
[`docs/agent-wave-2026-08-25-prediction-control-sufficiency.md`](docs/agent-wave-2026-08-25-prediction-control-sufficiency.md).

## Project Structure

```
aixi/
├── src/                    # Core library code
│   ├── observer/          # World model (prediction)
│   ├── actor/             # Bounded binary advantage head; general Actor pending
│   ├── environments/      # GridWorld, CartPole wrappers
│   ├── utils/             # Config, metrics, reproducibility
│   └── tracking/          # W&B integration, checkpoints
├── experiments/
│   ├── configs/           # YAML hyperparameter files
│   ├── scripts/           # Training entry points
│   └── baselines/         # Model-free DQN baseline
├── tests/                 # Unit + integration tests
└── specs/                 # Feature specifications
    └── 001-observer-actor-mvp/
        ├── spec.md        # Requirements
        ├── plan.md        # Implementation plan
        ├── research.md    # Architectural decisions
        ├── data-model.md  # Entity schemas
        ├── contracts/     # Module interfaces
        ├── tasks.md       # Implementation tasks
        └── quickstart.md  # Getting started guide
```

## Architecture

### Observer Interface

```python
class Observer(nn.Module):
    def encode(self, obs: Tensor) -> Tensor:
        """Compress observation to latent state z_t."""
        # MLP: obs → 32 → 16 → latent_dim
        
    def predict(self, latent_z: Tensor, action: Tensor) -> Tensor:
        """Predict next observation from latent state + action."""
        # MLP: [latent_z, action] → 32 → obs_dim
        
    def compute_loss(self, obs, action, next_obs) -> Tensor:
        """Compute prediction loss (MSE only, NO rewards)."""
        # Constitutional constraint enforced
```

### Actor Interface

```python
class Actor(nn.Module):
    def select_action(self, latent_z: Tensor, training: bool) -> int:
        """Select action from latent state (epsilon-greedy)."""
        # DQN Q-network: latent_z → 64 → 64 → action_dim
        # Constitutional constraint: NO raw observations
        
    def plan(self, latent_z: Tensor, observer: Observer, horizon: int) -> int:
        """Select action via K-step rollout planning."""
        # Queries Observer.predict() for imagination
```

## Development

### Running Tests

```bash
# All tests with coverage
pytest tests/ -v --cov=src

# Specific test suite
pytest tests/test_observer/ -v

# Constitutional compliance tests
pytest tests/test_integration/test_constitution.py -v
```

### Code Quality

```bash
# Format code
black src/ tests/ experiments/

# Type checking
mypy src/ --strict

# Sort imports
isort src/ tests/ experiments/
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

## Results

**Experiment targets (not yet verified results)**:

- **ES1** (Random Baseline): ~5% success rate validates environment
- **ES2** (Observer Training): <5% prediction RMSE on test set
- **ES3** (Actor Training): >80% success rate on latent states
- **ES4** (Model-Based Planning): ≥20% improvement over reactive policy

See W&B dashboard for detailed metrics and visualizations.

## Constitutional Principles

All design decisions validated against [AIXI Constitution v1.0.0](.specify/memory/constitution.md):

1. **Modular Architecture**: Observer/Actor separation with latent interface
2. **Dual Optimization**: Separate losses (prediction vs. reward)
3. **Model-Based Planning**: Imagination via Observer.predict()
4. **Reproducible Experiments**: Seed management, W&B tracking
5. **Incremental Complexity**: Baseline → Observer → Actor → Planning

## Documentation

- **Specification**: [spec.md](specs/001-observer-actor-mvp/spec.md) - Feature requirements and acceptance criteria
- **Implementation Plan**: [plan.md](specs/001-observer-actor-mvp/plan.md) - Technical decisions and architecture
- **Research Decisions**: [research.md](specs/001-observer-actor-mvp/research.md) - Why MLP, DQN, W&B, etc.
- **Data Model**: [data-model.md](specs/001-observer-actor-mvp/data-model.md) - Entity schemas
- **Contracts**: [contracts/](specs/001-observer-actor-mvp/contracts/) - Module interface specifications
- **Tasks**: [tasks.md](specs/001-observer-actor-mvp/tasks.md) - Implementation task breakdown
- **Quickstart**: [quickstart.md](specs/001-observer-actor-mvp/quickstart.md) - Getting started guide

## References

- **AIXI**: Hutter (2005) - Universal Artificial Intelligence
- **World Models**: Ha & Schmidhuber (2018) - Learning world dynamics for planning
- **DQN**: Mnih et al. (2015) - Deep Q-Networks for Atari
- **Dream to Control**: Hafner et al. (2019) - Model-based RL with latent dynamics

## License

[Your License Here]

## Contributing

See [Contributing Guide](CONTRIBUTING.md) for development workflow and code style guidelines.

---

**Status**: 🚧 Under active development (MVP Phase)  
**Latest**: Feature 001-observer-actor-mvp in implementation
