# Actor Module Contract

**Module**: `src/actor/`  
**Purpose**: Learn policy to maximize task rewards using Observer's latent states. Support model-based planning.  
**Version**: 1.0.0  
**Date**: 2026-02-25

---

## Overview

The Actor is responsible for **decision-making**—learning policies that maximize cumulative task rewards. It operates exclusively on Observer's latent states (never raw observations) and can leverage the Observer's world model for planning.

**Constitutional Requirements** (AIXI v1.0.0):
- ✅ MUST NOT access raw observations (only latent states `z_t`)
- ✅ MUST be independently testable (Observer can be mocked)
- ✅ Planning MUST use Observer.predict() interface only

---

## Public Interface

### Actor Class

**Constructor**:
```python
Actor(
    latent_dim: int,                  # Latent state dimension (from Observer)
    action_dim: int,                  # Number of discrete actions
    hidden_dims: List[int] = [64, 64],  # Q-network MLP layers
    gamma: float = 0.99,              # Discount factor
    epsilon_start: float = 1.0,       # Initial exploration rate
    epsilon_end: float = 0.05,        # Final exploration rate
    epsilon_decay_steps: int = 5000   # Linear decay duration
)
```

**Properties**:
- `epsilon: float` - Current exploration rate (read-only, managed internally)
- `step_count: int` - Total training steps (read-only)
- `device: torch.device` - Current device (CPU/CUDA)

---

### Core Methods

#### 1. `select_action(latent_z: Tensor, training: bool = True) -> int`

**Purpose**: Choose action based on latent state using epsilon-greedy policy.

**Signature**:
```python
def select_action(
    self,
    latent_z: torch.Tensor,  # shape: (latent_dim,)
    training: bool = True    # If True, use epsilon-greedy; if False, greedy
) -> int:
    """
    Select action from latent state using epsilon-greedy exploration.
    
    Args:
        latent_z: Latent state from Observer.encode(obs)
        training: Enable exploration (epsilon-greedy) if True
        
    Returns:
        action: Integer action in [0, action_dim-1]
        
    Constitutional Requirement:
        Input MUST be latent_z (not raw observation).
    """
```

**Contract**:
- **Constitutional compliance**: Input is `latent_z` ONLY - never raw observations
- **Epsilon-greedy**: Explores with probability `epsilon` if `training=True`
- **Greedy evaluation**: Always selects `argmax Q(z, a)` if `training=False`
- **Device handling**: Automatically moves `latent_z` to correct device

**Example**:
```python
actor = Actor(latent_dim=8, action_dim=3)
z = observer.encode(obs)  # Get latent state (constitutional requirement)
action = actor.select_action(z, training=True)  # Epsilon-greedy
```

---

#### 2. `compute_loss(batch: Dict[str, Tensor]) -> Tensor`

**Purpose**: Compute DQN loss (TD error) for training.

**Signature**:
```python
def compute_loss(
    self,
    batch: Dict[str, torch.Tensor]  # Keys: latent_z, action, reward, next_latent_z, done
) -> torch.Tensor:
    """
    Compute DQN loss (temporal difference error).
    
    Args:
        batch: Dictionary with keys:
            - latent_z: (batch, latent_dim) - current latent states
            - action: (batch,) - actions taken
            - reward: (batch,) - task rewards
            - next_latent_z: (batch, latent_dim) - next latent states
            - done: (batch,) - episode termination flags
            
    Returns:
        loss: Scalar TD error loss
        
    Constitutional Requirement:
        Loss MUST use only latent states and task rewards—NO raw observations.
    """
```

**Loss Function**: DQN Temporal Difference Error
```python
# Current Q-values
q_values = self.q_network(latent_z)
q_values = q_values.gather(1, action.unsqueeze(1)).squeeze()

# Target Q-values (using target network)
with torch.no_grad():
    next_q_values = self.target_network(next_latent_z)
    max_next_q = next_q_values.max(1)[0]
    q_targets = reward + gamma * max_next_q * (1 - done)

# TD error
loss = F.mse_loss(q_values, q_targets)
```

**Contract**:
- **Constitutional compliance**: Operates on `latent_z`, not raw observations
- **Uses task rewards**: This is the ONLY module that accesses task rewards
- **Target network**: Uses frozen target_network for stability (updated periodically)
- **Batch processing**: Efficient training on batched experiences

---

#### 3. `update_target_network() -> None`

**Purpose**: Synchronize target network with Q-network (hard update).

**Signature**:
```python
def update_target_network(self) -> None:
    """
    Copy Q-network weights to target network (DQN stabilization).
    
    Called every `target_update_interval` steps during training.
    """
```

**Contract**:
- **Hard update**: `target_network.load_state_dict(q_network.state_dict())`
- **Frequency**: Called by training loop, not automatically
- **Constitutional compliance**: Separate update from Observer (independent modules)

---

#### 4. `save(path: str) -> None`

**Purpose**: Save Actor model to disk.

**Signature**:
```python
def save(self, path: str) -> None:
    """
    Save Actor state_dict to file.
    
    Args:
        path: File path (e.g., "checkpoints/actor_step1000.pt")
        
    Saves:
        - Q-network state_dict
        - Target network state_dict
        - Actor state (epsilon, step_count)
        - Latent dimension (for architecture verification)
    """
```

**Contract**:
- **Independent saving**: Can save Actor without Observer
- **Full DQN state**: Includes both Q-network and target network
- **Resumable training**: Saves epsilon and step_count for continuation

---

#### 5. `load(path: str) -> None`

**Purpose**: Load Actor model from disk.

**Signature**:
```python
def load(self, path: str) -> None:
    """
    Load Actor state_dict from file.
    
    Args:
        path: File path to saved checkpoint
        
    Raises:
        ValueError: If loaded latent_dim doesn't match current architecture
    """
```

**Contract**:
- **Architecture matching**: Current Actor must have same latent_dim as saved model
- **Exact restoration**: Loaded model produces identical Q-values as saved model

---

## Planning Interface

#### `plan(latent_z: Tensor, observer: Observer, horizon: int = 3) -> int`

**Purpose**: Select action using model-based planning (rollout-based imagination).

**Signature**:
```python
def plan(
    self,
    latent_z: torch.Tensor,   # Current latent state
    observer: Observer,        # Observer reference for predict()
    horizon: int = 3           # Rollout depth K
) -> int:
    """
    Select action using K-step rollout planning.
    
    Args:
        latent_z: Current latent state
        observer: Observer instance for predict() queries
        horizon: Planning depth (0 = reactive, >0 = planning)
        
    Returns:
        action: Best action from rollout evaluation
        
    Algorithm:
        For each candidate action:
            Simulate K-step trajectory using observer.predict()
            Aggregate Q-values along trajectory
        Return action with highest aggregated value
    """
```

**Planning Algorithm**:
```python
def plan(self, latent_z, observer, horizon=3):
    best_action = None
    best_value = -float('inf')
    
    for action in range(self.action_dim):
        # Depth-first rollout
        value = self._rollout(latent_z, action, observer, horizon)
        if value > best_value:
            best_value = value
            best_action = action
    
    return best_action

def _rollout(self, latent_z, action, observer, depth):
    """Recursive K-step rollout."""
    if depth == 0:
        # Base case: Q-value at current state-action
        q_values = self.q_network(latent_z)
        return q_values[action].item()
    
    # Predict next latent state using Observer (constitutional requirement)
    next_z_pred = observer.predict(latent_z, torch.tensor(action))
    
    # Greedy action for future steps
    next_action = self.q_network(next_z_pred).argmax().item()
    
    # Recursive value
    immediate_q = self.q_network(latent_z)[action].item()
    future_q = self._rollout(next_z_pred, next_action, observer, depth - 1)
    
    return immediate_q + self.gamma * future_q
```

**Contract**:
- **Constitutional compliance**: Uses ONLY `observer.predict()` - no direct environment access
- **Planning overhead**: ~O(action_dim^K) forward passes through Observer
- **Differentiable rollouts**: Could support gradient-based planning (future extension)

---

## Usage Examples

### Training Actor (Reactive Mode)

```python
# Initialize
actor = Actor(latent_dim=8, action_dim=3)
optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)

# Training loop
for step in range(10000):
    # Sample batch from replay buffer
    batch = replay_buffer.sample(batch_size=64)
    
    # Encode observations to latent states (Observer does this)
    batch["latent_z"] = observer.encode(batch["obs"])
    batch["next_latent_z"] = observer.encode(batch["next_obs"])
    
    # Compute loss (only uses latent states + rewards - constitutional requirement)
    loss = actor.compute_loss(batch)
    
    # Update Actor
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Update target network periodically
    if step % 100 == 0:
        actor.update_target_network()
    
    # Log metrics
    wandb.log({"actor/loss": loss.item(), "actor/epsilon": actor.epsilon})
```

### Planning Mode

```python
# Enable planning (model-based RL)
z = observer.encode(obs)

if use_planning:
    # K-step rollout using Observer's world model
    action = actor.plan(z, observer, horizon=3)
else:
    # Reactive policy (no planning)
    action = actor.select_action(z, training=False)
```

---

## Testing Contract

### Required Tests

1. **Latent-Only Input**: Verify `select_action()` rejects raw observations, accepts only `latent_z`
2. **Epsilon Decay**: Epsilon decreases linearly over `epsilon_decay_steps`
3. **Target Network Update**: Target network copies Q-network weights correctly
4. **Constitutional Compliance**: `compute_loss()` never accesses raw observations
5. **Planning Correctness**: Planning selects better actions than reactive policy (on simple task)

### Example Test

```python
def test_actor_constitutional_compliance():
    """Verify Actor never accesses raw observations."""
    actor = Actor(latent_dim=8, action_dim=3)
    observer = Observer(obs_dim=10, action_dim=3, latent_dim=8)
    
    raw_obs = torch.randn(10)
    latent_z = observer.encode(raw_obs)
    
    # This should work (latent state)
    action = actor.select_action(latent_z)
    assert action in range(3)
    
    # This should fail (raw observation) - but Python typing won't enforce
    # Instead, document in tests that Actor receives only latent_z
```

---

## Constraints & Invariants

### Constitutional Constraints (NON-NEGOTIABLE)

1. **Latent-Only Access**: Actor MUST NOT receive raw observations - only `latent_z`
2. **Independent Testability**: Actor MUST work with mocked Observer
3. **Planning Interface**: Planning MUST use `observer.predict()` only

### Architectural Invariants

1. **Epsilon Bounds**: `0 <= epsilon <= 1` always
2. **Action Range**: `select_action()` returns `action ∈ [0, action_dim-1]`
3. **Non-Negative Steps**: `step_count >= 0` always

---

## Performance Expectations

| Operation | Latency (CPU) | Latency (GPU) |
|-----------|---------------|---------------|
| `select_action(z)` | <1ms | <0.1ms |
| `compute_loss(batch=64)` | <5ms | <1ms |
| `plan(z, observer, K=3)` | <30ms | <10ms |

**Planning Overhead Analysis**:
- GridWorld: 3 actions, K=3 → 27 rollouts
- Each rollout: 1 forward pass through Observer.predict() (~1ms)
- Total: ~30ms (well under 100ms budget ✅)

---

## Version History

- **1.0.0** (2026-02-25): Initial contract for Observer-Actor MVP
- Future: PPO algorithm, distributional Q-learning, MCTS planning
