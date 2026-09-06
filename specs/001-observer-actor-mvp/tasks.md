# Tasks: Observer-Actor RL with Model-Based Planning

**Branch**: `001-observer-actor-mvp`  
**Input**: Design documents from `/specs/001-observer-actor-mvp/`  
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Organization**: Tasks grouped by experiment scenario (ES1-ES4) to enable independent implementation and testing.

## Format: `[ID] [P?] [ES#] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[ES#]**: Experiment scenario this task supports (ES1, ES2, ES3, ES4)
- All tasks include exact file paths from plan.md structure

---

## Phase 1: Setup (Project Infrastructure)

**Purpose**: Initialize project structure, dependencies, and development tooling.

**⚠️ CRITICAL**: Must complete before any implementation work.

- [X] T001 Create project directory structure: `src/{observer,actor,environments,utils,tracking}`, `experiments/{configs,scripts,baselines}`, `tests/{test_observer,test_actor,test_integration,test_environments}`
- [X] T002 Initialize Python project with Poetry: create `pyproject.toml` with dependencies (PyTorch 2.1+, Gymnasium 0.29+, NumPy 1.24+, wandb 0.16+, pytest 7.4+, black, mypy)
- [X] T003 [P] Create `.gitignore` file: ignore `__pycache__/`, `*.pyc`, `wandb/`, `checkpoints/`, `.venv/`, `.pytest_cache/`, `.mypy_cache/`
- [X] T004 [P] Create `README.md` with project overview, installation instructions (Poetry setup), and quick start guide linking to specs/001-observer-actor-mvp/quickstart.md
- [X] T005 [P] Create empty `__init__.py` files in all Python package directories: `src/__init__.py`, `src/observer/__init__.py`, `src/actor/__init__.py`, `src/environments/__init__.py`, `src/utils/__init__.py`, `src/tracking/__init__.py`
- [X] T006 Install dependencies via Poetry: run `poetry install` to create virtual environment and install all packages
- [X] T007 Configure development tools: create `.pre-commit-config.yaml` with Black (line length 88), isort (Black-compatible), mypy (strict mode) hooks

**Checkpoint**: ✅ Project structure complete, dependencies installed, ready for foundational utilities

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core utilities that ALL experiment scenarios depend on—must be complete before ANY scenario implementation.

**⚠️ CRITICAL**: No experiment work can begin until this phase is complete.

### Core Utilities

- [X] T008 [P] Implement reproducibility module in `src/utils/reproducibility.py`: create `set_seed(seed: int)` function that sets Python random.seed(), np.random.seed(), torch.manual_seed(), torch.cuda.manual_seed_all(), torch.backends.cudnn.deterministic=True, torch.backends.cudnn.benchmark=False
- [X] T009 [P] Implement config management in `src/utils/config.py`: create `ExperimentConfig` dataclass with fields for env (env_name, max_episode_steps), observer (latent_dim, observer_lr, encoder_hidden, predictor_hidden), actor (actor_lr, gamma, epsilon_start/end/decay_steps, target_update_interval), training (n_episodes, batch_size, buffer_size, update_frequency), planning (planning_horizon, use_planning), reproducibility (seed, device)
- [X] T010 [P] Implement metrics computation in `src/utils/metrics.py`: create functions `compute_rmse(predictions: Tensor, targets: Tensor) -> float` for Observer evaluation, `compute_success_rate(rewards: List[float]) -> float` for Actor evaluation (percentage of episodes with reward > 0.5), `compute_episode_statistics(episode_rewards: List[float]) -> Dict[str, float]` returning mean, std, min, max
- [X] T011 Implement visualization utilities in `src/utils/visualization.py`: create `plot_learning_curve(steps: List[int], rewards: List[float], save_path: str)` using matplotlib, `plot_state_visitation_heatmap(states: np.ndarray, save_path: str)` for GridWorld, `plot_prediction_errors(errors: List[float], save_path: str)` for Observer validation

### Experiment Tracking (W&B)

- [X] T012 Implement W&B logger in `src/tracking/wandb_logger.py`: create `WandbLogger` class with methods `__init__(project: str, name: str, config: ExperimentConfig)` initializing wandb.init(), `log_step(metrics: Dict[str, float], step: int)` for per-step metrics, `log_episode(metrics: Dict[str, float], episode: int)` for per-episode metrics, `finish()` calling wandb.finish()
- [X] T013 Implement checkpointing in `src/tracking/checkpointing.py`: create `save_checkpoint(path: str, observer: nn.Module, actor: nn.Module, observer_opt: Optimizer, actor_opt: Optimizer, config: ExperimentConfig, step: int, episode: int, metrics: Dict[str, float])` saving torch.save() with git commit hash (via subprocess), and `load_checkpoint(path: str) -> Dict` returning all saved components
- [X] T014 Add git integration to checkpointing: implement `get_git_commit() -> str` function in `src/tracking/checkpointing.py` that runs `git rev-parse --short HEAD` via subprocess, returns "unknown" if git not available

### Environment Implementation

- [X] T015 Implement GridWorld-1D environment in `src/environments/gridworld.py`: create `GridWorldEnv` class inheriting `gym.Env` with constructor parameters (size=10, goal_position=9, max_steps=100), observation_space=Box(low=0, high=size-1, shape=(1,), dtype=float32), action_space=Discrete(3) for {left=0, stay=1, right=2}
- [X] T016 Implement GridWorld reset method in `src/environments/gridworld.py`: `reset(seed: Optional[int] = None) -> Tuple[np.ndarray, dict]` that sets self.position=0, self.step_count=0, calls super().reset(seed=seed) for Gymnasium compatibility, returns observation=np.array([self.position], dtype=np.float32), info={}
- [X] T017 Implement GridWorld step method in `src/environments/gridworld.py`: `step(action: int) -> Tuple[np.ndarray, float, bool, bool, dict]` with deterministic transitions (action 0: position=max(0, position-1), action 1: no change, action 2: position=min(size-1, position+1)), reward=1.0 if position==goal_position else 0.0, terminated=(position==goal_position), truncated=(step_count>=max_steps), info={"position": position}
- [X] T018 Implement GridWorld render method in `src/environments/gridworld.py`: `render() -> str` returning ASCII visualization like "[S][ ][ ][X][ ][ ][ ][ ][ ][G]" where S=start(0), X=current position, G=goal(9)
- [X] T019 Register GridWorld environment in `src/environments/__init__.py`: import GridWorldEnv and call `gym.register(id="GridWorld-1D", entry_point="src.environments.gridworld:GridWorldEnv", max_episode_steps=100)`
- [X] T020 [P] Implement environment wrappers in `src/environments/wrappers.py`: create `NormalizeObservationWrapper` class for CartPole (divides observations by range to [-1, 1]), create `LoggingWrapper` class that logs every step to W&B (optional, for debugging)

### Data Structures

- [X] T021 Implement Transition dataclass in `src/utils/config.py`: create `@dataclass Transition` with fields (obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool, latent_z: Optional[np.ndarray] = None, next_latent_z: Optional[np.ndarray] = None), methods `to_observer_batch() -> Dict[str, Tensor]` returning {obs, action, next_obs}, `to_actor_batch() -> Dict[str, Tensor]` returning {latent_z, action, reward, next_latent_z, done}
- [X] T022 Implement ReplayBuffer in `src/utils/config.py`: create `ReplayBuffer` class with `__init__(capacity: int)` creating deque, `add(transition: Transition)` appending and auto-evicting if full, `sample(batch_size: int) -> List[Transition]` returning random.sample(), `__len__() -> int` returning buffer size, raises ValueError if sample(batch_size > len(buffer))

**Checkpoint**: ✅ Foundation complete - all utilities available, environment works, tracking configured

---

## Phase 3: ES1 - Random Baseline (Priority: P1) 🎯 MVP

**Goal**: Validate environment, experiment tracking, and metrics pipeline with random agent.

**Independent Test**: Run random agent for 100 episodes in GridWorld-1D, verify W&B logs all metrics (episode reward, action distribution, state visitation), confirm success rate is ~5% (random exploratory baseline).

### Implementation for ES1

- [X] T023 [P] [ES1] Create random baseline config in `experiments/configs/gridworld_random.yaml`: YAML file with env_name="GridWorld-1D", n_episodes=100, seed=42, max_episode_steps=100, log all actions/states for debugging
- [X] T024 [ES1] Implement random agent script in `experiments/scripts/run_random_baseline.py`: main function that calls set_seed(config.seed), creates GridWorld-1D env, initializes WandbLogger(project="aixi-observer-actor", name=f"random-baseline-seed{seed}"), runs episode loop: for episode in range(n_episodes): obs, info = env.reset(seed=seed+episode), total_reward = 0, action_counts = [0, 0, 0], then inner step loop until done: action = env.action_space.sample(), obs, reward, terminated, truncated, info = env.step(action), total_reward += reward, action_counts[action] += 1, done = terminated or truncated, logs episode metrics to W&B: logger.log_episode({"reward": total_reward, "success": total_reward > 0.5, "action_dist_left": action_counts[0], "action_dist_stay": action_counts[1], "action_dist_right": action_counts[2]}, episode=episode)
- [X] T025 [ES1] Add command-line argument parsing to `experiments/scripts/run_random_baseline.py`: use argparse to accept --config (path to YAML), --seed (override config seed), --n_episodes (override config episodes), --wandb_project (override project name), load config from YAML using PyYAML or OmegaConf
- [X] T026 [ES1] Add evaluation summary at end of `experiments/scripts/run_random_baseline.py`: after all episodes, compute final metrics: mean_reward = np.mean(episode_rewards), success_rate = compute_success_rate(episode_rewards), print summary to console, log final summary to W&B: logger.log_step({"final/mean_reward": mean_reward, "final/success_rate": success_rate}, step=config.n_episodes)
- [X] T027 [ES1] Generate visualization in `experiments/scripts/run_random_baseline.py`: call plot_learning_curve(range(n_episodes), episode_rewards, save_path="plots/random_baseline_rewards.png"), save plot and log to W&B as wandb.log({"charts/learning_curve": wandb.Image("plots/random_baseline_rewards.png")})

**Checkpoint**: At this point, ES1 is fully testable - run `python experiments/scripts/run_random_baseline.py --config experiments/configs/gridworld_random.yaml` and verify W&B dashboard shows metrics, ~5% success rate confirms environment works

---

## Phase 4: ES2 - Observer Training (Priority: P2)

**Goal**: Train Observer to predict next observations from latent states and actions, achieving <5% prediction RMSE on held-out test set.

**Independent Test**: Collect 10,000 transitions via random agent, train Observer for 1000 gradient steps, evaluate RMSE on 1000 held-out transitions. Observer should achieve RMSE < 0.5 (normalized by observation range [0, 9]).

### Observer Module Implementation

- [X] T028 [P] [ES2] Implement MLP Encoder in `src/observer/encoder.py`: create `Encoder` class (nn.Module) with `__init__(obs_dim: int, latent_dim: int, hidden_dims: List[int] = [32, 16])` building sequential MLP: Linear(obs_dim, hidden_dims[0]), ReLU(), Linear(hidden_dims[0], hidden_dims[1]), ReLU(), Linear(hidden_dims[1], latent_dim), method `forward(obs: Tensor) -> Tensor` returning latent_z shape (batch, latent_dim) or (latent_dim,)
- [X] T029 [P] [ES2] Implement MLP Predictor in `src/observer/predictor.py`: create `Predictor` class (nn.Module) with `__init__(latent_dim: int, action_dim: int, obs_dim: int, hidden_dims: List[int] = [32])` building sequential MLP: Linear(latent_dim + action_dim, hidden_dims[0]), ReLU(), Linear(hidden_dims[0], obs_dim), method `forward(latent_z: Tensor, action: Tensor) -> Tensor` concatenating [latent_z, action_onehot] → next_obs_pred, handle action as integer or one-hot via F.one_hot(action, num_classes=action_dim)
- [X] T030 [ES2] Implement Observer module in `src/observer/models.py`: create `Observer` class (nn.Module) with `__init__(obs_dim, action_dim, latent_dim, encoder_hidden, predictor_hidden)` instantiating Encoder and Predictor, method `encode(obs: Tensor) -> Tensor` calling self.encoder(obs), method `predict(latent_z: Tensor, action: Tensor) -> Tensor` calling self.predictor(latent_z, action), method `forward(obs: Tensor, action: Tensor) -> Tensor` combining encode + predict in one pass for efficiency
- [X] T031 [P] [ES2] Implement Observer losses in `src/observer/losses.py`: create `prediction_loss(obs_next_pred: Tensor, obs_next_actual: Tensor) -> Tensor` computing F.mse_loss(obs_next_pred, obs_next_actual), create `compute_rmse(obs_next_pred: Tensor, obs_next_actual: Tensor) -> float` returning torch.sqrt(F.mse_loss(...)).item() for logging

### Data Collection for Observer

- [X] T032 [ES2] Implement data collection script in `experiments/scripts/collect_transitions.py`: main function that creates env, runs random policy or simple heuristic (always move right) for n_steps=10000, stores each transition in list: transitions.append(Transition(obs, action, reward, next_obs, done)), splits into train (80%) and test (20%), saves via pickle.dump() to `data/gridworld_transitions_train.pkl` and `data/gridworld_transitions_test.pkl`
- [X] T033 [ES2] Add reproducibility to data collection in `experiments/scripts/collect_transitions.py`: call set_seed(args.seed) before env creation, use env.reset(seed=seed) for consistent trajectories, log data collection metadata to console: total transitions, train/test split sizes, average episode length, state visitation distribution

### Observer Training Loop

- [X] T034 [P] [ES2] Create Observer training config in `experiments/configs/observer_train.yaml`: YAML with env_name="GridWorld-1D", latent_dim=8, encoder_hidden=[32, 16], predictor_hidden=[32], observer_lr=1e-3, batch_size=64, n_gradient_steps=1000, seed=42, data_path="data/gridworld_transitions_train.pkl", test_data_path="data/gridworld_transitions_test.pkl"
- [X] T035 [ES2] Implement Observer training script in `experiments/scripts/train_observer.py`: load config from YAML, call set_seed(config.seed), load train/test transitions from pickle, create Observer(obs_dim=1, action_dim=3, latent_dim=config.latent_dim, ...), create optimizer=torch.optim.Adam(observer.parameters(), lr=config.observer_lr), initialize WandbLogger(project="aixi-observer-actor", name=f"observer-train-seed{seed}", config=config.to_dict())
- [X] T036 [ES2] Implement training loop in `experiments/scripts/train_observer.py`: for step in range(n_gradient_steps): sample batch from train_transitions via random.sample(transitions, batch_size), convert to tensors: obs_batch, action_batch, next_obs_batch, compute predictions: next_obs_pred = observer(obs_batch, action_batch), compute loss=prediction_loss(next_obs_pred, next_obs_batch), backprop: optimizer.zero_grad(), loss.backward(), optimizer.step(), log every 50 steps: compute train_rmse=compute_rmse(next_obs_pred, next_obs_batch), logger.log_step({"observer/train_loss": loss.item(), "observer/train_rmse": train_rmse}, step=step)
- [X] T037 [ES2] Add test set evaluation to `experiments/scripts/train_observer.py`: every 200 steps, evaluate on test_transitions: observer.eval(), with torch.no_grad(): sample test batch, compute test_obs_pred = observer(test_obs, test_action), compute test_rmse = compute_rmse(test_obs_pred, test_next_obs), logger.log_step({"observer/test_rmse": test_rmse}, step=step), observer.train()
- [X] T038 [ES2] Add checkpointing to `experiments/scripts/train_observer.py`: every 200 steps and at end, save checkpoint: save_checkpoint(path=f"checkpoints/observer_step{step}.pt", observer=observer, actor=None, observer_opt=optimizer, actor_opt=None, config=config, step=step, episode=0, metrics={"train_rmse": train_rmse, "test_rmse": test_rmse}), keep best checkpoint based on lowest test_rmse: if test_rmse < best_rmse: shutil.copy(f"checkpoints/observer_step{step}.pt", "checkpoints/observer_best.pt")
- [X] T039 [ES2] Add final evaluation report to `experiments/scripts/train_observer.py`: after training, load best checkpoint, evaluate on full test set (all test_transitions, no sampling), compute comprehensive metrics: final_test_rmse, per-dimension MSE (for debugging multi-dim obs in CartPole), prediction accuracy at different action types (left vs right vs stay), print report to console and log to W&B as table

**Checkpoint**: At this point, ES2 is independently testable - run `python experiments/scripts/collect_transitions.py`, then `python experiments/scripts/train_observer.py --config experiments/configs/observer_train.yaml`, verify test RMSE < 0.5 on GridWorld

---

## Phase 5: ES3 - Actor Training on Latent States (Priority: P3)

**Goal**: Train Actor to maximize task rewards using only Observer's latent states (never raw observations), achieving >80% success rate within 100 episodes.

**Independent Test**: Load pre-trained Observer checkpoint, train Actor on latent states for 10,000 env steps, compare to model-free DQN baseline on raw observations. Actor should achieve ≥80% of baseline's performance.

### Actor Module Implementation

- [ ] T040 [P] [ES3] Implement DQN Q-network in `src/actor/dqn.py`: create `QNetwork` class (nn.Module) with `__init__(latent_dim: int, action_dim: int, hidden_dims: List[int] = [64, 64])` building MLP: Linear(latent_dim, hidden_dims[0]), ReLU(), Linear(hidden_dims[0], hidden_dims[1]), ReLU(), Linear(hidden_dims[1], action_dim), method `forward(latent_z: Tensor) -> Tensor` returning Q-values shape (batch, action_dim) or (action_dim,)
- [ ] T041 [ES3] Implement Actor policy in `src/actor/policy.py`: create `Actor` class (nn.Module) with `__init__(latent_dim, action_dim, hidden_dims)` creating self.q_network = QNetwork(...) and self.target_network = copy.deepcopy(self.q_network), method `select_action(latent_z: Tensor, epsilon: float) -> int` implementing epsilon-greedy: if random.random() < epsilon: return random action, else: return argmax Q-value, method `update_target_network()` copying q_network weights to target_network via target_network.load_state_dict(q_network.state_dict())
- [ ] T042 [P] [ES3] Implement DQN loss in `src/actor/losses.py`: create `dqn_loss(q_network: nn.Module, target_network: nn.Module, batch: Dict[str, Tensor], gamma: float) -> Tensor` computing: current_q = q_network(batch["latent_z"]).gather(1, batch["action"].unsqueeze(1)).squeeze(), next_q_target = target_network(batch["next_latent_z"]).max(1)[0], expected_q = batch["reward"] + gamma * next_q_target * (1 - batch["done"]), return F.mse_loss(current_q, expected_q.detach())
- [ ] T043 [P] [ES3] Implement epsilon decay in `src/actor/policy.py`: add method `compute_epsilon(step: int, epsilon_start: float, epsilon_end: float, epsilon_decay_steps: int) -> float` returning max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (step / epsilon_decay_steps)) for linear decay

### Model-Free Baseline (for comparison)

- [ ] T044 [P] [ES3] Implement Stable-Baselines3 DQN baseline in `experiments/baselines/dqn_baseline.py`: import from stable_baselines3 import DQN, create train_baseline() function that creates env=gym.make("GridWorld-1D"), model=DQN("MlpPolicy", env, learning_rate=1e-3, buffer_size=10000, batch_size=64, gamma=0.99, exploration_fraction=0.5, exploration_final_eps=0.05, verbose=1), trains via model.learn(total_timesteps=10000), returns trained model
- [ ] T045 [ES3] Add evaluation function to `experiments/baselines/dqn_baseline.py`: create evaluate_baseline(model, env, n_episodes=100) -> Dict[str, float] that runs model.predict() in loop, computes mean reward, success rate, episode length, returns metrics dict for comparison

### Actor Training Loop

- [ ] T046 [P] [ES3] Create Actor training config in `experiments/configs/actor_train.yaml`: YAML with env_name="GridWorld-1D", observer_checkpoint="checkpoints/observer_best.pt", latent_dim=8, actor_hidden=[64, 64], actor_lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=5000, target_update_interval=100, buffer_size=10000, batch_size=64, n_episodes=100, seed=42
- [ ] T047 [ES3] Implement Actor training script in `experiments/scripts/train_actor.py`: load config, call set_seed(seed), load Observer checkpoint: checkpoint = load_checkpoint(config.observer_checkpoint), observer = Observer(...), observer.load_state_dict(checkpoint["observer_state_dict"]), observer.eval(), freeze Observer parameters: for param in observer.parameters(): param.requires_grad = False
- [ ] T048 [ES3] Initialize Actor and training infrastructure in `experiments/scripts/train_actor.py`: create actor = Actor(latent_dim=config.latent_dim, action_dim=3, hidden_dims=config.actor_hidden), optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr), replay_buffer = ReplayBuffer(capacity=config.buffer_size), initialize WandbLogger(project="aixi-observer-actor", name=f"actor-train-seed{seed}", config), create env, global_step = 0
- [ ] T049 [ES3] Implement Actor episode loop in `experiments/scripts/train_actor.py`: for episode in range(n_episodes): obs, info = env.reset(seed=seed+episode), episode_reward = 0, episode_steps = 0, while not done: encode obs to latent: latent_z = observer.encode(torch.from_numpy(obs).float()), compute epsilon: epsilon = actor.compute_epsilon(global_step, ...), select action: action = actor.select_action(latent_z, epsilon), execute: next_obs, reward, terminated, truncated, info = env.step(action), done = terminated or truncated, encode next_obs: next_latent_z = observer.encode(torch.from_numpy(next_obs).float()), store transition: replay_buffer.add(Transition(obs, action, reward, next_obs, done, latent_z.numpy(), next_latent_z.numpy())), update counters: episode_reward += reward, episode_steps += 1, global_step += 1
- [ ] T050 [ES3] Implement Actor gradient updates in `experiments/scripts/train_actor.py`: inside episode loop after each step, if len(replay_buffer) >= batch_size: sample batch from buffer, convert to tensors: batch_dict = {"latent_z": torch.stack([t.latent_z for t in batch]), "action": torch.tensor([t.action for t in batch]), "reward": torch.tensor([t.reward for t in batch]), "next_latent_z": torch.stack([t.next_latent_z for t in batch]), "done": torch.tensor([t.done for t in batch], dtype=torch.float32)}, compute loss: loss = dqn_loss(actor.q_network, actor.target_network, batch_dict, gamma=config.gamma), backprop: optimizer.zero_grad(), loss.backward(), optimizer.step(), log: logger.log_step({"actor/loss": loss.item(), "actor/epsilon": epsilon}, step=global_step), update target network: if global_step % config.target_update_interval == 0: actor.update_target_network()
- [ ] T051 [ES3] Add episode-level logging to `experiments/scripts/train_actor.py`: after each episode ends, compute success = (episode_reward > 0.5), logger.log_episode({"actor/episode_reward": episode_reward, "actor/episode_steps": episode_steps, "actor/success": success}, episode=episode), print progress every 10 episodes: print(f"Episode {episode}/{n_episodes}: reward={episode_reward:.2f}, success={success}, epsilon={epsilon:.3f}")
- [ ] T052 [ES3] Add Actor checkpointing to `experiments/scripts/train_actor.py`: every 10 episodes and at end, save checkpoint including both Observer and Actor: save_checkpoint(path=f"checkpoints/actor_episode{episode}.pt", observer=observer, actor=actor, observer_opt=None, actor_opt=optimizer, config=config, step=global_step, episode=episode, metrics={"episode_reward": episode_reward, "success_rate": compute_success_rate(all_episode_rewards[-10:])}), save best checkpoint based on highest success rate over last 20 episodes
- [ ] T053 [ES3] Add baseline comparison to `experiments/scripts/train_actor.py`: after Actor training completes, train SB3 DQN baseline by calling train_baseline() from dqn_baseline.py, evaluate both: actor_metrics = evaluate_actor_on_latent_states(observer, actor, env, n_episodes=100), baseline_metrics = evaluate_baseline(baseline_model, env, n_episodes=100), log comparison: logger.log_step({"comparison/actor_success_rate": actor_metrics["success_rate"], "comparison/baseline_success_rate": baseline_metrics["success_rate"], "comparison/actor_relative_performance": actor_metrics["success_rate"] / baseline_metrics["success_rate"]}, step=global_step), print comparison table to console

**Checkpoint**: At this point, ES3 is independently testable - run `python experiments/scripts/train_actor.py --config experiments/configs/actor_train.yaml`, verify Actor achieves >80% success rate and ≥80% of baseline performance

---

## Phase 6: ES4 - Model-Based Planning (Priority: P4)

**Goal**: Enable Actor to query Observer for K-step rollout planning, improving success rate by ≥20% over reactive (K=0) policy while maintaining <100ms planning overhead per decision.

**Independent Test**: Load trained Observer and Actor checkpoints, implement planning module, compare reactive (K=0) vs. planning (K=3) on 100 test episodes. Planning should achieve ≥20% higher success rate.

### Planning Module Implementation

- [ ] T054 [P] [ES4] Implement rollout planning in `src/actor/planning.py`: create `rollout(latent_z: Tensor, action: int, observer: Observer, actor: Actor, gamma: float, depth: int) -> float` function that recursively computes: if depth == 0: return actor.q_network(latent_z)[action].item(), else: get immediate Q-value, predict next latent: next_latent_z_pred = observer.predict(latent_z, torch.tensor([action])), compute future Q-values recursively for all actions: future_q_values = [rollout(next_latent_z_pred, a, observer, actor, gamma, depth-1) for a in range(action_dim)], return immediate_q + gamma * max(future_q_values)
- [ ] T055 [ES4] Implement planning action selection in `src/actor/planning.py`: create `plan(latent_z: Tensor, observer: Observer, actor: Actor, gamma: float, horizon: int, action_dim: int) -> int` that evaluates all actions: action_values = [(action, rollout(latent_z, action, observer, actor, gamma, horizon)) for action in range(action_dim)], returns best_action = max(action_values, key=lambda x: x[1])[0], tracks planning_time for logging: start_time = time.time(), ..., planning_time = time.time() - start_time
- [ ] T056 [ES4] Add planning mode to Actor in `src/actor/policy.py`: modify Actor class to accept `planning_horizon: int = 0` in constructor, modify select_action() to conditionally call planning: if self.planning_horizon > 0 and not training: return plan(latent_z, self.observer_ref, self, gamma, self.planning_horizon, action_dim), else: use epsilon-greedy Q-network as before, store observer reference: self.observer_ref = observer (set externally before planning)

### Planning Evaluation

- [ ] T057 [P] [ES4] Create planning evaluation config in `experiments/configs/planning_eval.yaml`: YAML with observer_checkpoint="checkpoints/observer_best.pt", actor_checkpoint="checkpoints/actor_best.pt", planning_horizons=[0, 1, 3, 5], n_eval_episodes=100, seed=42
- [ ] T058 [ES4] Implement planning evaluation script in `experiments/scripts/evaluate_planning.py`: load config, set_seed(seed), load Observer and Actor checkpoints, create env, initialize WandbLogger(project="aixi-observer-actor", name=f"planning-eval-seed{seed}"), for each planning horizon K in config.planning_horizons: run evaluation episodes
- [ ] T059 [ES4] Implement planning evaluation loop in `experiments/scripts/evaluate_planning.py`: for horizon in planning_horizons: actor.planning_horizon = horizon, actor.observer_ref = observer, episode_metrics = [], total_planning_time = 0, for episode in range(n_eval_episodes): obs, info = env.reset(seed=seed+episode), episode_reward = 0, episode_steps = 0, planning_times = [], while not done: latent_z = observer.encode(obs_tensor), start = time.time(), action = actor.select_action(latent_z, epsilon=0.0), planning_time = time.time() - start, planning_times.append(planning_time), next_obs, reward, terminated, truncated, info = env.step(action), episode_reward += reward, episode_steps += 1, done = terminated or truncated, obs = next_obs, episode_metrics.append({"reward": episode_reward, "success": episode_reward > 0.5, "mean_planning_time": np.mean(planning_times)}), log horizon results: logger.log_step({f"planning_k{horizon}/mean_reward": np.mean([m["reward"] for m in episode_metrics]), f"planning_k{horizon}/success_rate": np.mean([m["success"] for m in episode_metrics]), f"planning_k{horizon}/mean_planning_time_ms": np.mean([m["mean_planning_time"] for m in episode_metrics]) * 1000}, step=horizon)
- [ ] T060 [ES4] Generate planning comparison report in `experiments/scripts/evaluate_planning.py`: after evaluating all horizons, create comparison dict: reactive_success = results[K=0]["success_rate"], planning_success = results[K=3]["success_rate"], improvement = (planning_success - reactive_success) / reactive_success * 100, print report: f"Reactive (K=0): {reactive_success:.1%} success\nPlanning (K=3): {planning_success:.1%} success\nImprovement: +{improvement:.1f}%\nPlanning overhead: {results[K=3]['mean_planning_time_ms']:.1f}ms", log final summary to W&B as table, save comparison plot: plot_planning_comparison(horizons, success_rates, planning_times, save_path="plots/planning_comparison.png")

### Planning Training (Optional)

- [ ] T061 [P] [ES4] Create planning training config in `experiments/configs/planning_train.yaml`: YAML extending actor_train.yaml with planning_horizon=3, n_episodes=50 (fine-tuning only), actor_checkpoint="checkpoints/actor_best.pt" (warm start), planning_enabled_from_episode=0
- [ ] T062 [ES4] Implement planning training script in `experiments/scripts/train_planning.py`: similar to train_actor.py but loads pre-trained Actor checkpoint, enables planning during training: actor.planning_horizon = config.planning_horizon, actor.observer_ref = observer, trains for additional episodes with planning enabled, logs comparison: logger.log_step({"planning_train/episode_reward": episode_reward, "planning_train/improvement_over_reactive": (episode_reward - baseline_reactive_reward) / baseline_reactive_reward}, episode=episode)

**Checkpoint**: At this point, ES4 is independently testable - run `python experiments/scripts/evaluate_planning.py --config experiments/configs/planning_eval.yaml`, verify planning K=3 achieves ≥20% improvement over K=0 and planning time <100ms

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, evaluation utilities, code quality, and final integration.

### Evaluation & Analysis

- [ ] T063 [P] Implement comprehensive evaluation script in `experiments/scripts/evaluate.py`: accept arguments --observer_checkpoint, --actor_checkpoint, --planning_horizon, --n_eval_episodes, --env_name, --seed, --render, runs evaluation loop similar to planning eval, generates detailed report with: mean ± std reward, success rate, episode length distribution, action distribution histogram, state visitation heatmap (for GridWorld), saves all plots to plots/ directory, logs to W&B
- [ ] T064 [P] Add video recording to evaluation in `experiments/scripts/evaluate.py`: wrap env with gym.wrappers.RecordVideo(env, video_folder="videos/", episode_trigger=lambda ep: ep % 10 == 0), log videos to W&B: wandb.log({"video": wandb.Video(video_path)}), useful for debugging and presentations
- [ ] T065 Implement comparative analysis script in `experiments/scripts/compare_experiments.py`: accept W&B run IDs or checkpoint paths for multiple experiments, loads metrics from W&B API or checkpoint files, generates comparison plots: learning curves (overlay multiple runs), success rate vs. training steps, Observer RMSE vs. episodes, planning overhead vs. horizon depth, saves comparison table as CSV and PDF report

### Testing & Validation

- [ ] T066 [P] Implement Observer unit tests in `tests/test_observer/test_encoder.py`: test encoder output shape (batch input, single input), test encoder determinism (same input → same output), test encoder handles edge cases (zeros, max values), test gradient flow (non-zero gradients after backward pass), use pytest fixtures for Observer creation
- [ ] T067 [P] Implement Observer prediction tests in `tests/test_observer/test_predictor.py`: test predictor output shape matches obs_dim, test predictor handles all action values (0, 1, 2 for GridWorld), test predictor with batched and single inputs, test gradient flow through predictor
- [ ] T068 [P] Implement Actor unit tests in `tests/test_actor/test_policy.py`: test select_action epsilon-greedy behavior (verify randomness when epsilon=1.0, verify greedy when epsilon=0.0), test action output is valid integer in [0, action_dim-1], test target network update (weights change after update_target_network()), test epsilon decay function (starts at 1.0, ends at epsilon_end, linear decay)
- [ ] T069 [P] Implement DQN loss tests in `tests/test_actor/test_dqn.py`: test loss computation with known Q-values (hand-calculate expected loss), test loss is non-negative, test gradient flow through Q-network, test target network is detached (no gradients)
- [ ] T070 [P] Implement environment tests in `tests/test_environments/test_gridworld.py`: test reset returns position 0, test step transitions deterministically (action 0 → left, action 2 → right), test boundary conditions (position stays at 0 when action=left at start, stays at 9 when action=right at goal), test reward structure (1.0 at goal, 0.0 elsewhere), test episode termination (terminated=True when goal reached, truncated=True after max_steps), test Gymnasium compliance (observation_space.contains(obs), action_space.contains(action))
- [ ] T071 Implement integration tests in `tests/test_integration/test_observer_actor_integration.py`: test full training loop (Observer trains, Actor trains on Observer's latent states), test checkpoint save/load (save checkpoint, load in new process, verify same performance), test planning integration (Observer.predict() called during Actor.plan()), test reproducibility (same seed produces same final metrics ±1e-4), use pytest fixtures to share Observer/Actor across tests
- [ ] T072 Implement constitutional compliance tests in `tests/test_integration/test_constitution.py`: test Observer loss never receives rewards (inspect function signature, mock reward argument should raise TypeError), test Actor never receives raw observations (inspect function signature, mock obs argument should raise TypeError), test planning queries Observer.predict() (use unittest.mock to verify Observer.predict called during planning), test separate optimizers (Observer and Actor have different optimizer instances), test latent interface (Actor input must be Observer output, verify via shape and data flow)

### Documentation & Code Quality

- [ ] T073 [P] Add docstrings to all modules: Observer (encoder.py, predictor.py, models.py), Actor (policy.py, dqn.py, planning.py), utils (config.py, metrics.py, reproducibility.py, visualization.py), tracking (wandb_logger.py, checkpointing.py), environments (gridworld.py, wrappers.py), use Google-style docstrings with Args, Returns, Raises, Examples sections
- [ ] T074 [P] Add type hints to all functions: use Python 3.11+ syntax (List[int], Dict[str, float], Optional[Tensor]), run mypy in strict mode: `mypy src/ --strict`, fix all typing errors, add `# type: ignore` only where truly necessary (e.g., third-party library issues) with explanatory comment
- [ ] T075 Run code formatting: `black src/ tests/ experiments/` with line length 88, `isort src/ tests/ experiments/` with Black-compatible profile, verify all files pass: `black --check src/`
- [ ] T076 Update README.md with comprehensive documentation: add Project Overview (links to AIXI paper, explains Observer-Actor architecture), add Installation section (Poetry setup, dependency installation, environment creation), add Quick Start (link to quickstart.md, show commands for ES1-ES4), add Repository Structure (explain src/, experiments/, tests/ directories), add Results section (show example learning curves, success rates, link to W&B dashboard), add Contributing section (code style, testing requirements), add References (AIXI, World Models, DQN papers)
- [ ] T077 Create experiment documentation in `experiments/README.md`: document all config files (gridworld_random.yaml, observer_train.yaml, actor_train.yaml, planning_eval.yaml), document all scripts (what they do, required arguments, expected outputs), provide usage examples for each experiment scenario, document hyperparameter tuning guidelines (what to change for different environments, sensitivity analysis)
- [ ] T078 Generate API documentation: set up Sphinx or mkdocs, configure autodoc to extract docstrings, generate HTML documentation: `sphinx-build -b html docs/source/ docs/build/html`, publish to GitHub Pages or ReadTheDocs (optional), ensure all public APIs documented (Observer.encode, Observer.predict, Actor.select_action, Actor.plan)

### Final Integration & Validation

- [ ] T079 Run full experiment pipeline end-to-end: execute ES1 random baseline → ES2 Observer training → ES3 Actor training → ES4 planning evaluation, verify all scripts run without errors, verify metrics logged to W&B, verify checkpoints saved correctly, verify final success rates meet acceptance criteria (ES1: ~5%, ES2: RMSE <0.5, ES3: >80%, ES4: ≥20% improvement)
- [ ] T080 Run test suite: `pytest tests/ -v --cov=src --cov-report=html`, verify all tests pass (≥95% coverage target), review coverage report for untested code paths, add tests for any critical uncovered code (especially Observer/Actor core logic), verify constitutional compliance tests pass
- [ ] T081 Run reproducibility validation: execute same experiment configuration with seed=42 three times, compare final metrics (mean reward, success rate, RMSE), verify variance is <2% (acceptable floating-point noise), document any non-reproducible operations in README (e.g., CUDA non-determinism), add reproducibility badge to README
- [ ] T082 Performance profiling: use cProfile or PyTorch profiler on training scripts, identify bottlenecks (likely environment step time for simple GridWorld), verify planning overhead <100ms for K=3, verify Observer encode/predict <1ms each, document performance characteristics in experiments/README.md, add optimization notes for future work (e.g., batched planning, compiled Observer)
- [ ] T083 Create final experiment summary report: generate PDF report with: project overview, architectural diagram (Observer-Actor with latent interface), experiment results (tables and plots for ES1-ES4), baseline comparison (Actor vs. model-free DQN), planning analysis (success rate vs. horizon, planning time vs. horizon), lessons learned, future work (curiosity, multi-task, stochastic dynamics), save to `docs/experiment_report.pdf`, link from README

---

## Dependency Graph

### Story Completion Order

**Sequential Dependencies**:
1. **Phase 1 (Setup)** → blocks all other phases
2. **Phase 2 (Foundational)** → blocks ES1, ES2, ES3, ES4
3. **ES1 (Random Baseline)** → optional validation, doesn't block others
4. **ES2 (Observer Training)** → blocks ES3, ES4 (Actor needs trained Observer)
5. **ES3 (Actor Training)** → blocks ES4 (Planning needs trained Actor)
6. **ES4 (Planning)** → independent after ES3 complete
7. **Phase 7 (Polish)** → can start anytime, integrates all phases

**Parallel Opportunities Within Phases**:
- Phase 1: T003, T004, T005 (gitignore, README, __init__ files) can run parallel after T001-T002
- Phase 2: T008, T009, T010 (utils modules) fully parallel, T012-T014 (tracking) parallel, T015-T020 (environment) parallel, T021-T022 (data structures) parallel
- ES2: T028, T029, T031, T034 (Observer components, config) parallel, T032-T033 (data collection) parallel with implementation
- ES3: T040-T043 (Actor components), T044-T045 (baseline) parallel
- ES4: T054-T056 (planning module), T057, T061 (configs) parallel
- Phase 7: All testing (T066-T072) fully parallel, all documentation (T073-T078) parallel

**Critical Path** (longest dependency chain):
T001 (structure) → T002 (dependencies) → T008-T022 (foundational utils) → T032 (data collection) → T035-T039 (Observer training) → T047-T053 (Actor training) → T058-T060 (planning eval) → T079 (full pipeline) = ~50-60 hours of sequential work

---

## Parallel Execution Examples

### Phase 2 Foundational (Maximum Parallelization)

**Batch 1** (after T001-T007 setup complete):
```bash
# Terminal 1: Core utilities
python -m tasks T008  # reproducibility.py
python -m tasks T009  # config.py
python -m tasks T010  # metrics.py

# Terminal 2: Tracking
python -m tasks T012  # wandb_logger.py
python -m tasks T013  # checkpointing.py

# Terminal 3: Environment
python -m tasks T015  # gridworld.py class
python -m tasks T020  # wrappers.py

# Terminal 4: Data structures
python -m tasks T021  # Transition dataclass
python -m tasks T022  # ReplayBuffer
```

**Batch 2** (after Batch 1 environment ready):
```bash
# Complete environment implementation sequentially
python -m tasks T016  # GridWorld reset
python -m tasks T017  # GridWorld step
python -m tasks T018  # GridWorld render
python -m tasks T019  # Gymnasium registration
```

### ES2 Observer Training (Parallel Components)

**Before training loop** (parallel implementation):
```bash
# Terminal 1: Observer components
python -m tasks T028  # encoder.py
python -m tasks T029  # predictor.py
python -m tasks T031  # losses.py

# Terminal 2: Data collection
python -m tasks T032  # collect_transitions.py
python -m tasks T033  # add reproducibility to collection

# Terminal 3: Config
python -m tasks T034  # observer_train.yaml
```

**After components ready** (sequential training tasks):
```bash
python -m tasks T030  # Observer model integration
python -m tasks T035  # Training script setup
python -m tasks T036  # Training loop
python -m tasks T037  # Test evaluation
python -m tasks T038  # Checkpointing
python -m tasks T039  # Final evaluation report
```

### Phase 7 Testing (Fully Parallel)

```bash
# All testing can run simultaneously
python -m tasks T066 &  # test_encoder.py
python -m tasks T067 &  # test_predictor.py
python -m tasks T068 &  # test_policy.py
python -m tasks T069 &  # test_dqn.py
python -m tasks T070 &  # test_gridworld.py
python -m tasks T071 &  # integration tests
python -m tasks T072 &  # constitution tests
wait

# All documentation can run simultaneously
python -m tasks T073 &  # Add docstrings
python -m tasks T074 &  # Add type hints
python -m tasks T076 &  # Update README
python -m tasks T077 &  # Experiment docs
python -m tasks T078 &  # API docs
wait
```

---

## Implementation Strategy

### MVP Scope (Minimum Viable Product)

**Recommended MVP**: Complete through **ES3 (Actor Training)** to validate core architecture before planning.

**Rationale**:
- ES1-ES3 proves Observer-Actor separation works
- ES3 validates that latent state learning matches model-free baseline
- ES4 (planning) is valuable but not architecturally critical
- MVP = ~60% of tasks (T001-T053), can deliver in 2-3 weeks

**MVP Deliverables**:
- ✅ Working GridWorld-1D environment
- ✅ Trained Observer with <5% prediction RMSE
- ✅ Trained Actor achieving >80% success rate on latent states
- ✅ Baseline comparison showing Observer-Actor matches model-free DQN
- ✅ W&B experiment tracking for all runs
- ✅ Constitutional compliance verified (no reward leakage, no observation leakage)

### Post-MVP Extensions

**Phase 1 Extensions** (after ES3 MVP):
1. Complete ES4 (planning) - adds imagination-based decision making
2. Add comprehensive testing (Phase 7 tasks T066-T072)
3. Polish documentation (Phase 7 tasks T073-T078)
4. Performance optimization (profiling, batched planning)

**Phase 2 Extensions** (future work):
1. CartPole environment (continuous observations, test scaling)
2. Joint Observer-Actor training (alternating updates)
3. Probabilistic Observer (predict distributions, uncertainty)
4. Curiosity-driven exploration (intrinsic rewards from prediction error)

---

## Summary

**Total Tasks**: 83  
**Task Breakdown**:
- Phase 1 (Setup): 7 tasks
- Phase 2 (Foundational): 15 tasks
- ES1 (Random Baseline): 5 tasks
- ES2 (Observer Training): 12 tasks
- ES3 (Actor Training): 14 tasks
- ES4 (Planning): 9 tasks
- Phase 7 (Polish): 21 tasks

**Parallelization Opportunities**: 45+ tasks can run in parallel (marked with [P])  
**Critical Path Length**: ~50-60 hours of sequential work  
**Estimated Total Effort**: 120-150 hours (with parallelization: ~80-100 hours)

**Independent Test Criteria**:
- **ES1**: Random agent achieves ~5% success, W&B logs all data
- **ES2**: Observer achieves <5% prediction RMSE on test set
- **ES3**: Actor achieves >80% success rate, ≥80% of baseline performance
- **ES4**: Planning K=3 achieves ≥20% improvement over K=0, <100ms overhead

**MVP Recommendation**: Complete tasks T001-T053 (ES1-ES3), defer ES4 and polish until core architecture validated.
