# Feature Specification: Observer-Actor RL with Model-Based Planning

**Feature Branch**: `001-observer-actor-mvp`  
**Created**: 2026-02-25  
**Status**: Draft  
**Input**: User description: "Build Observer-Actor RL system with model-based planning where Observer learns world dynamics via prediction and Actor learns policy via task rewards"

## User Scenarios & Testing *(mandatory)*

### Experiment Scenario 1 - Random Baseline in Simple Environment (Priority: P1) 🎯 MVP

A random agent acts in a simple discrete environment (1D GridWorld) to establish baseline performance and validate the environment-agent interface.

**Why this priority**: Establishes the foundational infrastructure (environment, agent interface, metrics logging) without any learning complexity. Everything else builds on this.

**Independent Test**: Run random agent for 1000 steps in GridWorld, measure average reward, confirm experiment tracking logs all data (actions, rewards, states, metrics).

**Acceptance Scenarios**:

1. **Given** a 1D GridWorld with 10 positions and goal at position 9, **When** random agent takes actions for 100 episodes, **Then** experiment tracker logs all episodes with timestamped metrics
2. **Given** the random baseline run, **When** querying metrics, **Then** average episode reward, episode length, and action distribution are recorded
3. **Given** experiment completed, **When** visualizing results, **Then** plots show reward per episode and state visitation heatmap

---

### Experiment Scenario 2 - Observer Learns World Dynamics (Priority: P2)

Observer module learns to predict next observations given current observation and action, trained purely on prediction error without access to task rewards.

**Why this priority**: Validates the core world modeling capability. Observer must learn environment dynamics independently before Actor can use it for planning.

**Independent Test**: Train Observer on collected trajectories (from random agent or simple policy), measure prediction RMSE on held-out test set. Observer should achieve <10% prediction error on deterministic GridWorld.

**Acceptance Scenarios**:

1. **Given** 10,000 timesteps of (observation, action, next_observation) transitions, **When** Observer trains for 1000 gradient steps, **Then** prediction RMSE decreases below 10% of observation range
2. **Given** a trained Observer, **When** presented with test transitions, **Then** predicted next states match actual next states with >90% accuracy
3. **Given** Observer training, **When** architecture has separate encoder (obs → latent_z) and predictor (latent_z, action → next_obs), **Then** both components exist as independent modules
4. **Given** Observer training logs, **When** examining metrics, **Then** only prediction loss appears (NO task reward leakage)

---

### Experiment Scenario 3 - Actor Policy Learning with Latent States (Priority: P3)

Actor learns to maximize task rewards using only Observer's latent state representation (z_t), never directly accessing raw observations. Demonstrates separation of world modeling and decision-making.

**Why this priority**: Core architectural validation of Observer-Actor separation. Actor must succeed using compressed latent space, proving the architecture works beyond naive end-to-end RL.

**Independent Test**: Train Actor using frozen Observer's latent states. Compare performance against model-free baseline (DQN/PPO on raw observations). Actor should match or exceed baseline while training on latent space only.

**Acceptance Scenarios**:

1. **Given** a frozen pre-trained Observer, **When** Actor trains for 10,000 steps receiving only latent states z_t, **Then** average episode reward increases by >3x from random baseline
2. **Given** Actor training, **When** examining code, **Then** Actor never receives raw observations, only Observer.encode(observation) outputs
3. **Given** Actor training logs, **When** examining metrics, **Then** only task rewards appear in Actor's loss function (NO prediction error)
4. **Given** trained Actor and model-free baseline (DQN on raw obs), **When** evaluating both on 100 test episodes, **Then** Actor achieves ≥80% of baseline's performance
5. **Given** interface validation, **When** checking data flow, **Then** Observer outputs z_t, Actor inputs z_t, returns action a_t (clean modular boundary)

---

### Experiment Scenario 4 - Model-Based Planning via Imagination (Priority: P4)

Actor queries Observer to simulate future trajectories ("What if I take action X?"), then chooses actions based on imagined rollouts. Demonstrates model-based RL capabilities beyond model-free policy learning.

**Why this priority**: The ultimate goal of Observer-Actor architecture. Planning via imagination distinguishes this from standard model-free RL and connects to AIXI-style reasoning.

**Independent Test**: Implement rollout-based planning where Actor simulates N-step futures using Observer's dynamics model. Measure planning horizon achieved and performance gain over reactive policy. Planning should improve task success rate by ≥20%.

**Acceptance Scenarios**:

1. **Given** a trained Observer and Actor, **When** implementing planning module, **Then** Actor can query Observer.predict(z_t, action) to get predicted next latent state z_{t+1}
2. **Given** planning enabled, **When** Actor chooses action, **Then** it evaluates multiple candidate actions by simulating K=3 step rollouts
3. **Given** planning vs. reactive comparison, **When** both policies tested on 100 episodes, **Then** planning policy achieves ≥20% higher success rate
4. **Given** computational budget, **When** planning with K=3 step rollouts, **Then** decision time remains <100ms per action (real-time capable)
5. **Given** planning visualization, **When** inspecting decision trace, **Then** imagined trajectories and their predicted rewards are logged

---

### Edge Cases

- **Non-stationary Observer**: What happens when Observer updates during Actor training? Does Actor's performance degrade due to shifting latent space?
- **Model exploitation**: Can Actor find adversarial actions that break Observer's predictions? How to detect and mitigate?
- **Stochastic environments**: How does Observer handle randomness? Should it predict distributions rather than point estimates?
- **Sparse rewards**: How does Actor learn when task rewards are very sparse (e.g., only at goal state)? Does intrinsic curiosity help?
- **High-dimensional latent space**: What happens if latent dimension is too large (overfitting) or too small (information bottleneck)?

## Requirements *(mandatory)*

### Functional Requirements

**Architecture & Modularity**

- **FR-001**: System MUST implement Observer as independent module with interface: `encode(observation) → latent_state` and `predict(latent_state, action) → next_observation`
- **FR-002**: System MUST implement Actor as independent module with interface: `policy(latent_state) → action`
- **FR-003**: Observer MUST NOT access task rewards during training (only prediction errors allowed)
- **FR-004**: Actor MUST NOT access raw observations during training (only latent states z_t allowed)
- **FR-005**: System MUST enforce modular boundary: Observer ↔ latent state z_t ↔ Actor

**Dual Optimization**

- **FR-006**: Observer MUST optimize prediction loss: `L_obs = MSE(obs_next, obs_next_predicted)`
- **FR-007**: Actor MUST optimize task reward: `L_actor = -E[cumulative_reward]` (e.g., policy gradient or Q-learning)
- **FR-008**: Observer and Actor MUST have separate optimizers with independent learning rates
- **FR-009**: System MUST support training modes: (1) Observer-only, (2) Actor-only, (3) Joint (alternating updates)

**Model-Based Planning**

- **FR-010**: Observer MUST support forward prediction queries: given (z_t, a_t), predict z_{t+1}
- **FR-011**: Actor MUST support planning mode where it simulates K-step rollouts before acting
- **FR-012**: Planning module MUST evaluate multiple candidate actions via imagined futures
- **FR-013**: System MUST support configurable planning horizon K ∈ {0, 1, 3, 5} (K=0 means reactive/model-free)

**Experiment Infrastructure**

- **FR-014**: System MUST log all training metrics: Observer prediction RMSE, Actor cumulative reward, episode length, sample efficiency
- **FR-015**: System MUST support experiment tracking via Weights & Biases or MLflow with automatic hyperparameter logging
- **FR-016**: System MUST save model checkpoints at regular intervals (every N episodes) with timestamped metadata
- **FR-017**: System MUST support evaluation mode: load checkpoint, run M test episodes, report mean ± std performance
- **FR-018**: System MUST log random seeds (Python, NumPy, PyTorch) for reproducibility

**Baseline Comparisons**

- **FR-019**: System MUST implement random baseline agent for environment validation
- **FR-020**: System MUST support model-free comparison baseline (DQN or PPO on raw observations)
- **FR-021**: System MUST generate comparison plots: Observer-Actor vs. baselines over training steps

**Environment Support**

- **FR-022**: System MUST support Gymnasium/Gym environments with discrete action spaces
- **FR-023**: System MUST start with simple deterministic environment (1D GridWorld with goal)
- **FR-024**: System MUST support extensibility to CartPole after GridWorld validation

### Key Entities *(include if feature involves data)*

**Observer (World Model)**

- **Encoder**: Maps raw observation `o_t` to latent state `z_t` (e.g., neural network or linear projection)
- **Predictor**: Maps `(z_t, a_t)` to predicted next observation `o_hat_{t+1}` or next latent `z_hat_{t+1}`
- **Prediction Loss**: MSE between predicted and actual next observations
- **Optimizer**: Separate optimizer (e.g., Adam) for Observer parameters only
- **Architecture**: Configurable hidden dimensions, activation functions, latent dimension size

**Actor (Policy)**

- **Policy Network**: Maps latent state `z_t` to action distribution or Q-values
- **Value Function** (optional): Estimates expected return from state z_t (for policy gradient methods)
- **Task Reward Buffer**: Stores (z_t, a_t, r_t, z_{t+1}) transitions for RL training
- **Optimizer**: Separate optimizer (e.g., Adam) for Actor parameters only
- **Architecture**: Configurable hidden dimensions, action selection strategy (epsilon-greedy, softmax)

**Planning Module**

- **Rollout Generator**: Simulates K-step trajectories using Observer's predict() function
- **Action Evaluator**: Scores candidate actions based on imagined cumulative rewards
- **Planning Horizon**: Configurable depth K for rollout simulations
- **Computational Budget**: Max time or simulations allowed per action decision

**Environment Wrapper**

- **Gymnasium Interface**: Standard `reset()`, `step(action)`, `render()` methods
- **State Space**: Observation dimensionality and type (discrete, continuous)
- **Action Space**: Action dimensionality and type
- **Reward Function**: Maps states/actions to scalar rewards
- **Termination Conditions**: When episode ends (goal reached, max steps, failure)

**Experiment Tracker**

- **Run Metadata**: Experiment name, timestamp, git commit hash, hyperparameters
- **Metrics Logger**: Records scalars (reward, loss) and histograms (action distribution) per step/episode
- **Checkpoint Manager**: Saves/loads model state_dicts with metadata
- **Visualizer**: Generates plots and metrics dashboards

## Success Criteria *(mandatory)*

**Observer Performance**

- Observer achieves <5% prediction RMSE on held-out test set for deterministic GridWorld environment
- Observer's latent state dimension is compact (e.g., 8-16 dims for GridWorld) while preserving predictive accuracy
- Observer training converges within 5,000 gradient steps on 10,000 collected transitions

**Actor Performance**

- Actor achieves >80% success rate (reaching goal) within 50 episodes when trained on Observer's latent states
- Actor matches or exceeds 80% of model-free baseline's performance (DQN/PPO on raw observations)
- Sample efficiency: Actor reaches 80% success rate using ≤20,000 environment interactions

**Model-Based Planning**

- Planning with K=3 step rollouts improves success rate by ≥20% over reactive (K=0) policy
- Planning decisions complete within <100ms per action (real-time constraint)
- Imagined trajectories correlate with actual outcomes (rollout accuracy >70%)

**Architecture Validation**

- Observer and Actor are independently loadable modules (can save/load separately)
- Observer training never accesses task rewards (verified via code inspection and logging)
- Actor training never accesses raw observations (verified via interface checks)
- Changing Observer architecture does not require Actor code changes (modular interface)

**Reproducibility & Tracking**

- Every experiment run is logged to W&B/MLflow with unique run ID
- All runs include: hyperparameters, random seeds, git commit, environment config
- Checkpoints can be loaded and reproduce exact evaluation performance (±2% variance)
- Learning curves (reward vs. steps) are generated automatically for all runs

**Baseline Comparisons**

- Random baseline establishes environment's minimum performance (≤10% success rate expected)
- Model-free baseline (DQN/PPO) achieves ≥80% success rate within 100 episodes (validation of environment learnability)
- Observer-Actor system reaches same performance as model-free within 50% more samples (acceptable overhead for model-based benefits)

## Assumptions *(mandatory)*

1. **Environment Characteristics**: Starting with deterministic, fully observable environments (GridWorld). Stochasticity and partial observability are future extensions.

2. **Discrete Action Space**: Initial implementation uses discrete actions (e.g., {left, right, stay}). Continuous actions are out of scope for MVP.

3. **Tabular/Compact State Space**: GridWorld has small state space (10 positions) for fast iteration. High-dimensional observations (images) are future work.

4. **Computational Resources**: Training assumes single GPU available (e.g., NVIDIA RTX 3060 or equivalent). No distributed training required for MVP.

5. **Hyperparameter Tuning**: Reasonable defaults will be chosen based on RL literature. Extensive hyperparameter search is out of scope unless performance critically depends on it.

6. **Observer Update Frequency**: Observer is pre-trained, then frozen during Actor training for MVP. Joint/alternating training is optional extension.

7. **Reward Scale**: Task rewards are normalized to [0, 1] range for stable Actor training.

8. **Episode Length**: Episodes terminate after max 100 steps to prevent infinite loops and ensure training throughput.

9. **Latent Space Interpretation**: Latent states (z_t) need not be human-interpretable; only predictive accuracy matters for MVP.

10. **Planning Computational Cost**: K=3 step planning is acceptable overhead. Deeper planning (K>5) may require search algorithms (MCTS) beyond MVP scope.

## Out of Scope *(mandatory)*

**Future Extensions (Explicitly Deferred)**

- **High-Dimensional Observations**: Image-based environments (Atari, MuJoCo) - requires convolutional encoders
- **Continuous Action Spaces**: Robotic control tasks - requires different Actor architecture (Gaussian policies)
- **Stochastic Dynamics**: Probabilistic Observer predicting distributions p(s_{t+1}|s_t,a_t) instead of point estimates
- **Partial Observability (POMDPs)**: Requires recurrent Observer (LSTM/GRU) to maintain belief states
- **Multi-Task Learning**: Single Observer learning dynamics across multiple environments
- **Curiosity-Driven Exploration**: Intrinsic motivation based on prediction error
- **Model-Based Value Iteration**: Full Dyna-style planning with value backups
- **Adversarial Robustness**: Detecting and mitigating Actor's exploitation of Observer model errors
- **Meta-Learning**: Observer adapting to new environments with few samples
- **Theoretical Analysis**: Formal sample complexity bounds, PAC guarantees

**Non-Goals**

- Real-time robotics applications (simulation only for MVP)
- Production deployment infrastructure
- Human-in-the-loop interactive learning
- Multi-agent scenarios (single agent only)
- Safety guarantees or formal verification
