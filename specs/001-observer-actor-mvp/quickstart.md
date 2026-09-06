# Quickstart: Observer-Actor RL MVP

**Purpose**: Get started with training and evaluating Observer-Actor RL agents.  
**Prerequisites**: Python 3.11+, CUDA optional  
**Date**: 2026-02-25

---

## Installation

### 1. Clone Repository

```bash
git clone <repo-url> aixi
cd aixi
```

### 2. Install Dependencies

**Option A: Using Poetry** (recommended):
```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install

# Activate virtual environment
poetry shell
```

**Option B: Using uv** (faster alternative):
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip install -e ".[dev]"
```

**Option C: Using pip**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 3. Verify Installation

```bash
pytest tests/ -v
```

Expected output: All tests pass ✅

---

## Project Structure Overview

```
aixi/
├── src/
│   ├── observer/          # World model (prediction)
│   ├── actor/             # Policy (decision-making)
│   ├── environments/      # GridWorld, wrappers
│   ├── utils/             # Config, metrics, reproducibility
│   └── tracking/          # W&B logging, checkpoints
├── experiments/
│   ├── scripts/           # Training scripts (run these!)
│   └── configs/           # Hyperparameter YAML files
└── tests/                 # Unit + integration tests
```

---

## Running Experiments

### Experiment 1: Random Baseline (Environment Validation)

**Purpose**: Validate environment setup and experiment tracking.

```bash
python experiments/scripts/run_random_baseline.py \
  --env GridWorld-1D \
  --n_episodes 100 \
  --seed 42 \
  --wandb_project aixi-observer-actor
```

**Expected Output**:
```
Episode 50/100: avg_reward=0.04, success_rate=4%
Episode 100/100: avg_reward=0.05, success_rate=5%

📊 W&B dashboard: https://wandb.ai/<username>/aixi-observer-actor/runs/...
```

**Interpretation**:
- Random agent rarely reaches goal (~5% success)
- Establishes baseline performance
- Verifies W&B logging works

---

### Experiment 2: Train Observer (World Model)

**Purpose**: Learn environment dynamics via prediction.

```bash
python experiments/scripts/train_observer.py \
  --config experiments/configs/observer_train.yaml \
  --seed 42
```

**Config File** (`observer_train.yaml`):
```yaml
env:
  name: GridWorld-1D
  max_steps: 100

observer:
  latent_dim: 8
  encoder_hidden: [32, 16]
  predictor_hidden: [32]
  learning_rate: 1e-3

training:
  n_episodes: 100  # Collect 100 episodes of data
  batch_size: 64
  gradient_steps: 1000
```

**Expected Output**:
```
Step 500/1000: pred_loss=0.12, rmse=0.35
Step 1000/1000: pred_loss=0.03, rmse=0.17

✅ Observer trained successfully
📁 Checkpoint saved: checkpoints/observer_2026-02-25_21-00-00.pt
```

**Success Criteria**:
- Prediction RMSE < 0.5 (ideally < 0.2)
- Loss decreases steadily
- W&B shows learning curve

**Troubleshooting**:
- If RMSE not decreasing: Check learning rate (try 1e-4 instead of 1e-3)
- If loss exploding: Check for NaN in gradients (verify data normalization)

---

### Experiment 3: Train Actor on Latent States

**Purpose**: Learn goal-reaching policy using Observer's latent states.

```bash
python experiments/scripts/train_actor.py \
  --observer_checkpoint checkpoints/observer_latest.pt \
  --config experiments/configs/actor_train.yaml \
  --seed 42
```

**Config File** (`actor_train.yaml`):
```yaml
env:
  name: GridWorld-1D
  max_steps: 100

observer:
  checkpoint: checkpoints/observer_latest.pt  # Use pre-trained Observer

actor:
  hidden_dims: [64, 64]
  learning_rate: 1e-3
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay_steps: 5000

training:
  n_episodes: 100
  batch_size: 64
  buffer_size: 10000
  target_update_interval: 100
```

**Expected Output**:
```
Episode 20/100: reward=0.0, success=0%, epsilon=0.80
Episode 50/100: reward=1.0, success=50%, epsilon=0.50
Episode 100/100: reward=1.0, success=85%, epsilon=0.05

✅ Actor trained successfully
📁 Checkpoint saved: checkpoints/actor_2026-02-25_21-30-00.pt
```

**Success Criteria**:
- Success rate > 80% by episode 100
- Reward increases from ~0.0 to ~1.0
- Epsilon decays to 0.05

**Comparison to Baseline** (automatic):
```
Observer-Actor: 85% success (10,000 steps)
Model-Free DQN: 80% success (10,000 steps)
Random Baseline: 5% success

✅ Latent-state learning matches model-free performance
```

---

### Experiment 4: Enable Model-Based Planning

**Purpose**: Improve performance using Observer's world model for imagination.

```bash
python experiments/scripts/train_planning.py \
  --observer_checkpoint checkpoints/observer_latest.pt \
  --actor_checkpoint checkpoints/actor_latest.pt \
  --planning_horizon 3 \
  --seed 42
```

**Expected Output**:
```
Reactive Policy (K=0): 85% success
Planning K=1: 90% success (+5%)
Planning K=3: 97% success (+12%)

✅ Planning improves performance by 12 percentage points
⏱️ Planning overhead: 25ms per action (well under 100ms budget)
```

**Success Criteria**:
- Planning K=3 achieves ≥20% improvement over reactive (target: 85% → 102%, capped at 100%)
- Planning latency < 100ms per action
- Rollout accuracy > 70% (predicted states match actual states)

---

## Evaluation

### Run Evaluation on Saved Checkpoint

```bash
python experiments/scripts/evaluate.py \
  --observer_checkpoint checkpoints/observer_best.pt \
  --actor_checkpoint checkpoints/actor_best.pt \
  --n_eval_episodes 100 \
  --render \
  --seed 42
```

**Output**:
```
Evaluating on 100 episodes (seed=42)...

Episode 1: [X][X][X][X][X][X][X][X][X][G] ✅ (9 steps)
Episode 2: [X][X][X][X][X][X][X][X][X][G] ✅ (9 steps)
...
Episode 100: [X][X][X][X][X][X][X][X][X][G] ✅ (9 steps)

📊 Results:
  Success Rate: 98% (98/100)
  Mean Episode Length: 9.2 ± 0.5 steps
  Mean Reward: 0.98
```

**Optimal Performance**:
- GridWorld-1D optimal policy: 9 steps (straight path to goal)
- Success rate should approach 100% with good training

---

## Visualizing Results

### W&B Dashboard

All experiments automatically log to Weights & Biases:

1. **Navigate to**: `https://wandb.ai/<username>/aixi-observer-actor`
2. **View Metrics**:
   - Observer: Prediction RMSE, loss curves
   - Actor: Episode reward, success rate, epsilon decay
   - Planning: Rollout accuracy, planning overhead
3. **Compare Runs**: Use W&B's comparison tool to see Observer-Actor vs. baselines

### Local Plots

Generate plots from saved checkpoints:

```bash
python experiments/scripts/plot_results.py \
  --checkpoints checkpoints/actor_*.pt \
  --output plots/learning_curves.png
```

**Generated Plots**:
- `learning_curves.png`: Reward vs. episodes
- `state_visitation.png`: Heatmap of visited states
- `planning_comparison.png`: Reactive vs. K=1 vs. K=3

---

## Troubleshooting

### Issue 1: Observer Not Learning (RMSE not decreasing)

**Symptoms**: Prediction loss stays high (>1.0 after 1000 steps)

**Solutions**:
1. Check learning rate: Try reducing to 1e-4
2. Verify data diversity: Print `replay_buffer` statistics, ensure all states visited
3. Check gradients: `wandb` should show non-zero gradients (check "observer/grad_norm")
4. Inspect predictions: Visualize `predicted_obs` vs. `actual_obs` (are they correlated?)

**Debug Command**:
```bash
python experiments/scripts/debug_observer.py --checkpoint checkpoints/observer_latest.pt
```

---

### Issue 2: Actor Gets Stuck in Local Optima

**Symptoms**: Success rate plateaus at <50%, doesn't improve

**Solutions**:
1. Increase exploration: Set `epsilon_end=0.1` (more exploration)
2. Extend epsilon decay: Set `epsilon_decay_steps=10000` (slower decay)
3. Check replay buffer: Ensure successful episodes are stored (not just failures)
4. Verify Observer quality: RMSE should be <0.5 before training Actor

**Debug Command**:
```bash
python experiments/scripts/visualize_latent_space.py \
  --observer_checkpoint checkpoints/observer_latest.pt
```
This plots latent space (PCA projection) to verify states are distinguishable.

---

### Issue 3: Non-Reproducible Results

**Symptoms**: Same seed produces different results across runs

**Solutions**:
1. Verify `set_seed(42)` is called before environment creation
2. Check CUDA: Use `--device cpu` for exact reproducibility (GPU has non-determinism)
3. Disable cudnn.benchmark: Already done in `set_seed()` function
4. Check environment seed: Ensure `env.reset(seed=42)` is called

**Test Reproducibility**:
```bash
# Run twice with same seed
python experiments/scripts/train_actor.py --seed 42 --output run1.log
python experiments/scripts/train_actor.py --seed 42 --output run2.log

# Compare final metrics
diff run1.log run2.log  # Should be identical
```

---

## Next Steps

### Extend to CartPole

**Modify Config**:
```yaml
env:
  name: CartPole-v1  # 4D continuous observations
  max_steps: 500

observer:
  latent_dim: 32  # Increase for higher-dimensional obs
  encoder_hidden: [64, 32]
```

**Run**:
```bash
python experiments/scripts/train_observer.py --config experiments/configs/cartpole_observer.yaml
python experiments/scripts/train_actor.py --config experiments/configs/cartpole_actor.yaml
```

---

### Try Advanced Features (Future)

1. **Probabilistic Observer**: Predict distributions instead of point estimates
   ```yaml
   observer:
     predictor_type: gaussian  # Output mean + variance
   ```

2. **Intrinsic Curiosity**: Add prediction error as exploration bonus
   ```yaml
   actor:
     intrinsic_reward: true
     curiosity_weight: 0.1
   ```

3. **Joint Observer-Actor Training**: Update both simultaneously
   ```yaml
   training:
     mode: joint  # Alternate Observer and Actor updates
   ```

---

## Resources

**Documentation**:
- [Specification](spec.md): Feature requirements and success criteria
- [Research Decisions](research.md): Architecture choices and rationale
- [Data Model](data-model.md): Entity schemas and relationships
- [Contracts](contracts/): Module interface specifications

**Troubleshooting**:
- W&B Dashboard: Check logged metrics and visualizations
- GitHub Issues: Report bugs or ask questions
- Constitution: [.specify/memory/constitution.md](.specify/memory/constitution.md) - Core principles

**Related Papers**:
- Ha & Schmidhuber (2018): [World Models](https://arxiv.org/abs/1803.10122)
- Mnih et al. (2015): [DQN](https://www.nature.com/articles/nature14236)
- Hafner et al. (2019): [Dream to Control](https://arxiv.org/abs/1912.01603)

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `python experiments/scripts/run_random_baseline.py` | Validate environment setup |
| `python experiments/scripts/train_observer.py` | Train world model |
| `python experiments/scripts/train_actor.py` | Train policy on latent states |
| `python experiments/scripts/train_planning.py` | Enable model-based planning |
| `python experiments/scripts/evaluate.py` | Test saved checkpoints |
| `pytest tests/ -v` | Run all tests |

**Happy experimenting!** 🚀
