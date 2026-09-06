# Research: Observer-Actor RL Implementation Decisions

**Purpose**: Resolve all unknowns from Technical Context and establish best practices for AIXI-inspired RL implementation.  
**Date**: 2026-02-25  
**Status**: Complete - All decisions finalized

## Overview

This document records research outcomes for technology choices, architectural patterns, and best practices that guide the Observer-Actor MVP implementation. All decisions align with AIXI Constitution v1.0.0 principles (modular architecture, dual optimization, reproducible experiments).

## 1. Observer Architecture

### Decision: MLP Encoder + MLP Predictor

**Architecture Choice**:
- **Encoder**: `obs → [32] → [16] → latent_z` (MLP with ReLU activations)
- **Predictor**: `[latent_z, action] → [32] → next_obs` (MLP with ReLU → Linear output)
- **Latent Dimension**: 8 for GridWorld, 32 for CartPole (rule of thumb: 2-4x state abstraction)

**Rationale**:
1. **Predicting next observation** (not next latent) enables direct validation against ground truth—no decoder needed
2. **Separate encoder** allows flexible latent dimension independent of observation dimension
3. **MLP universality** handles both discrete (GridWorld) and continuous (CartPole) observations
4. **Non-linear activation** (ReLU) captures complex dynamics while remaining debuggable

**Alternatives Considered**:
- **Linear Encoder**: Too restrictive, cannot capture non-linearities in CartPole dynamics (rejected)
- **Predict Next Latent**: Requires additional decoder for validation, harder to debug prediction errors (deferred to future work)
- **Variational Encoder (VAE)**: Added complexity for uncertainty modeling, not needed for deterministic GridWorld MVP (deferred)

**Supporting Literature**:
- Ha & Schmidhuber (2018) "World Models": Demonstrates effectiveness of learned latent representations for RL
- Hafner et al. (2019) "Dream to Control": Shows RSSM architecture predicting latent states, but our MVP predicts observations for simplicity

**Implementation Notes**:
```python
class Observer(nn.Module):
    def __init__(self, obs_dim, action_dim, latent_dim=8):
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 32), nn.ReLU(),
            nn.Linear(32, obs_dim)  # No activation - predict raw observations
        )
```

---

## 2. Actor Training Algorithm

### Decision: DQN (Deep Q-Network) with Experience Replay

**Algorithm Choice**: Off-policy Q-learning with epsilon-greedy exploration

**Rationale**:
1. **Off-policy learning** tolerates non-stationary latent space if Observer updates during joint training
2. **Experience replay** improves sample efficiency—critical for expensive environment interactions
3. **Q-values** are directly interpretable and comparable to model-free baseline
4. **Well-established algorithm** with known hyperparameters (DQN paper + Stable-Baselines3 defaults)
5. **Simpler than policy gradients** for MVP—no advantage estimation, no baseline variance reduction

**Alternatives Considered**:
- **REINFORCE (vanilla policy gradient)**: On-policy, lower sample efficiency, high variance (rejected)
- **PPO (Proximal Policy Optimization)**: More robust but complex (value function, clipping, GAE), overkill for discrete GridWorld (deferred)
- **A3C**: Requires multi-threading, added complexity (rejected for MVP)

**Hyperparameters** (from Mnih et al. 2015 + SB3 tuning):
| Parameter | Value | Source |
|-----------|-------|--------|
| Learning Rate | 1e-3 | SB3 default for small envs |
| Replay Buffer Size | 10,000 | Sufficient for GridWorld (100 steps/episode × 100 episodes) |
| Batch Size | 64 | Standard for DQN |
| Gamma (discount) | 0.99 | Standard for episodic tasks |
| Epsilon Start | 1.0 | Full exploration initially |
| Epsilon End | 0.05 | Retain 5% exploration |
| Epsilon Decay Steps | 5,000 | Linear decay over ~50 episodes |
| Target Network Update | Every 100 steps | Stabilizes learning |

**Supporting Literature**:
- Mnih et al. (2015) "Human-level control through deep RL": Original DQN paper
- Van Hasselt et al. (2016) "Deep Reinforcement Learning with Double Q-learning": Addresses overestimation (consider for Phase 2)

**Implementation Notes**:
```python
class Actor(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden=64):
        self.q_network = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
        self.target_network = copy.deepcopy(self.q_network)
        
    def select_action(self, latent_z, epsilon):
        if random.random() < epsilon:
            return random.randint(0, action_dim - 1)
        else:
            with torch.no_grad():
                q_values = self.q_network(latent_z)
                return q_values.argmax().item()
```

---

## 3. Planning Implementation

### Decision: Depth-First K-Step Rollout with Q-Value Aggregation

**Algorithm**: Recursive rollout using Observer's world model to simulate K steps ahead

**Pseudocode**:
```python
def plan(latent_z, observer, actor, horizon=3):
    """Rollout-based planning for action selection."""
    best_action = None
    best_value = -inf
    
    for action in range(action_dim):
        # Simulate K steps with greedy policy
        total_value = rollout(latent_z, action, observer, actor, horizon)
        if total_value > best_value:
            best_value = total_value
            best_action = action
    
    return best_action

def rollout(latent_z, action, observer, actor, depth):
    """Recursive rollout with Observer's predict()."""
    if depth == 0:
        return actor.q_network(latent_z)[action].item()
    
    # Predict next latent state using Observer
    next_z_pred = observer.predict(latent_z, action)
    
    # Get immediate Q-value + discounted future value
    immediate_q = actor.q_network(latent_z)[action].item()
    future_q = max([rollout(next_z_pred, a, observer, actor, depth-1) 
                    for a in range(action_dim)])
    
    return immediate_q + gamma * future_q
```

**Rationale**:
1. **Depth-first simplicity**: Easier to implement and debug than MCTS
2. **Q-value aggregation**: Naturally extends DQN's value function
3. **Computational cost**: O(n_actions^K) acceptable for discrete actions (3^3 = 27 evaluations) and small K
4. **Greedy rollouts**: Assumes Actor's policy is reasonable (true after training Phase 3)

**Alternatives Considered**:
- **MCTS (Monte Carlo Tree Search)**: More sophisticated but requires UCB balancing, tree management, compute budget tuning—overkill for GridWorld (deferred to Phase 2)
- **Breadth-first search**: Same computational complexity, no clear advantage (deferred)
- **Stochastic rollouts**: Sample actions from policy distribution—higher variance, requires more rollouts (deferred)

**Computational Analysis**:
- GridWorld: 3 actions, K=3 horizon → 27 forward passes through Observer.predict()
- Forward pass latency: <1ms on CPU for MLP predictor
- Total planning overhead: ~30ms << 100ms budget ✅

**Supporting Literature**:
- Silver et al. (2017) "Mastering Chess and Shogi by Self-Play with a General RL Algorithm" (AlphaZero): MCTS inspiration
- Sutton (1991) "Dyna: An integrated architecture for learning, planning, and reacting": Model-based planning integration

---

## 4. Experiment Tracking

### Decision: Weights & Biases (W&B)

**Tool Choice**: W&B for cloud-based experiment tracking with RL specializations

**Rationale**:
1. **Excellent RL support**: Native video logging, histogram visualizations, custom charts
2. **Free tier sufficient**: 100GB storage, unlimited runs for academic/personal projects
3. **Clean Python API**: Minimal boilerplate, automatic hyperparameter capture
4. **Collaboration features**: Shared dashboards, report generation, run comparison
5. **Integration ecosystem**: Works with PyTorch, Gymnasium, Hugging Face

**Alternatives Considered**:
- **MLflow**: More complex setup (requires server), less RL-specific tooling (e.g., no video logging), better for production ML (deferred)
- **TensorBoard**: Local-only, less convenient for multi-run comparison, no cloud backup (rejected)
- **Custom logging**: Reinventing the wheel, not worth effort (rejected)

**Logging Schema**:
```python
import wandb

# Initialize experiment
wandb.init(
    project="aixi-observer-actor",
    name=f"observer-train-{seed}",
    config={
        "env": "GridWorld-1D",
        "observer_latent_dim": 8,
        "observer_lr": 1e-3,
        "actor_lr": 1e-3,
        "planning_horizon": 3,
        "seed": 42,
        "git_commit": get_git_commit(),
        # ... all hyperparameters
    }
)

# Log metrics per step
wandb.log({
    "observer/pred_rmse": rmse,
    "observer/pred_loss": loss.item(),
    "step": global_step
})

# Log metrics per episode
wandb.log({
    "actor/episode_reward": total_reward,
    "actor/episode_length": steps,
    "actor/success": reached_goal,
    "episode": episode_num
})

# Log hyperparameters and model architecture
wandb.config.update({"model_params": count_parameters(observer)})
```

**Tracked Artifacts**:
- Model checkpoints (via `wandb.save("checkpoints/*.pt")`)
- Learning curves (automatic from logged metrics)
- Configuration files (YAML configs uploaded as artifacts)
- Code snapshots (automatic git integration)

---

## 5. Reproducibility Implementation

### Decision: Strict Seed Setting with Non-Determinism Documentation

**Implementation**:
```python
def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic mode (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Environment seed (Gymnasium)
    # Must be set separately: env.reset(seed=seed)
```

**Usage Pattern**:
```python
# At script start
set_seed(args.seed)

# For each environment
env = gym.make("GridWorld-1D")
obs, info = env.reset(seed=args.seed)
```

**Known Non-Deterministic Operations** (documented in README):
1. **Multi-threaded DataLoaders**: Use `num_workers=0` for exact reproducibility
2. **CUDA operations**: Some PyTorch GPU kernels are non-deterministic even with flags—use CPU for bit-exact reproducibility
3. **Python hash randomization**: Generally not an issue for RL, but set `PYTHONHASHSEED=0` if needed

**Validation Process**:
1. Run experiment with seed=42
2. Record final metrics (reward, RMSE, success rate)
3. Re-run with same seed=42
4. Verify metrics match within ±1e-6 (floating point tolerance)
5. If mismatch: investigate and document non-deterministic operation

**Supporting Literature**:
- PyTorch Reproducibility Guide: https://pytorch.org/docs/stable/notes/randomness.html
- Henderson et al. (2018) "Deep Reinforcement Learning that Matters": Highlights RL reproducibility crisis

---

## 6. Baseline Implementation Strategy

### Decision: Stable-Baselines3 (SB3) DQN for Model-Free Baseline

**Tool Choice**: Use SB3 library for reference DQN implementation

**Rationale**:
1. **Reference quality**: Well-tested, widely used implementation
2. **Apples-to-apples comparison**: Same DQN algorithm, only difference is raw obs vs. latent states
3. **Minimal effort**: Focus implementation time on Observer-Actor architecture, not baseline debugging
4. **Hyperparameter defaults**: SB3 provides sensible defaults tuned on Atari/control tasks

**Usage Example**:
```python
from stable_baselines3 import DQN

# Model-free baseline on raw observations
baseline = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    exploration_fraction=0.5,
    exploration_final_eps=0.05,
    target_update_interval=100,
    verbose=1,
    tensorboard_log="./logs"
)
baseline.learn(total_timesteps=20000)
```

**Comparison Protocol**:
1. Train SB3 DQN on GridWorld with raw observations (no latent encoding)
2. Train our Actor-DQN on GridWorld with Observer's latent states
3. Compare: sample efficiency (timesteps to 80% success), final performance, training time
4. Expected outcome: Similar performance (validates architecture), possible sample efficiency gain (validates world model utility)

**Hyperparameter Matching**:
- Use **identical hyperparameters** between SB3 baseline and our Actor
- Only difference: input to Q-network (raw obs vs. latent_z)
- Document any deviations and justify

**Alternatives Considered**:
- **Custom DQN from scratch**: More control but higher bug risk, not worth it for baseline (rejected)
- **CleanRL implementations**: Good reference code but no API, would require copying/adapting (deferred for learning purposes)

---

## Summary: Research Phase Complete

### Decisions Finalized

| Component | Decision | Rationale |
|-----------|----------|-----------|
| Observer Architecture | MLP Encoder + MLP Predictor | Validates predictions directly against obs |
| Actor Algorithm | DQN with Experience Replay | Off-policy, sample efficient, well-established |
| Planning | Depth-first K-step rollouts | Simple, debuggable, O(actions^K) acceptable |
| Experiment Tracking | Weights & Biases | RL-specific features, free tier, cloud collaboration |
| Reproducibility | Strict seed setting | set_seed() covers Python/NumPy/PyTorch/Gym |
| Baseline | Stable-Baselines3 DQN | Reference implementation, fair comparison |

### Technology Stack Confirmed

- **Python**: 3.11+ (type hints, performance)
- **PyTorch**: 2.1+ (latest stable, improved performance)
- **Gymnasium**: 0.29+ (modern Gym replacement)
- **Weights & Biases**: 0.16+ (experiment tracking)
- **Stable-Baselines3**: 2.2+ (baselines)
- **pytest**: 7.4+ (testing)

### Next Steps

1. ✅ Research complete—all unknowns resolved
2. → Proceed to Phase 1: Data model, contracts, quickstart
3. → Validate architecture via contract specification
4. → Update agent context with technology decisions

**No blockers remaining for implementation.**
