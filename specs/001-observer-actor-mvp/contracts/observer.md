# Observer Module Contract

**Module**: `src/observer/`  
**Purpose**: Learn world model via next-observation prediction. Compress observations into latent states.  
**Version**: 1.0.0  
**Date**: 2026-02-25

---

## Overview

The Observer is responsible for **world modeling**—understanding environment dynamics through prediction. It learns to compress high-dimensional observations into compact latent representations and predict future observations given current state and action.

**Constitutional Requirements** (AIXI v1.0.0):
- ✅ MUST NOT access task rewards during training (only prediction errors)
- ✅ MUST be independently testable (no Actor dependency)
- ✅ Latent state `z_t` is the ONLY interface to Actor

---

## Public Interface

### Observer Class

**Constructor**:
```python
Observer(
    obs_dim: int,           # Observation dimensionality
    action_dim: int,        # Action space size
    latent_dim: int = 8,    # Latent state dimension
    encoder_hidden: List[int] = [32, 16],   # Encoder MLP layers
    predictor_hidden: List[int] = [32]      # Predictor MLP layers
)
```

**Properties**:
- `latent_dim: int` - Latent space dimensionality (read-only)
- `device: torch.device` - Current device (CPU/CUDA)

---

### Core Methods

#### 1. `encode(obs: Tensor) -> Tensor`

**Purpose**: Compress observation into latent representation.

**Signature**:
```python
def encode(self, obs: torch.Tensor) -> torch.Tensor:
    """
    Encode observation into latent state.
    
    Args:
        obs: Observation tensor, shape (batch, obs_dim) or (obs_dim,)
        
    Returns:
        latent_z: Latent state tensor, shape (batch, latent_dim) or (latent_dim,)
        
    Raises:
        ValueError: If obs.shape[-1] != obs_dim
    """
```

**Contract**:
- **Deterministic**: Same obs always produces same latent_z (no dropout randomness)
- **No side effects**: Does not modify model parameters or buffers
- **Shape preservation**: Output batch dimension matches input
- **Range**: No explicit range constraint (latent_z can be any real values)

**Example**:
```python
observer = Observer(obs_dim=10, action_dim=3, latent_dim=8)
obs = torch.tensor([3.0])  # GridWorld position
z = observer.encode(obs)   # shape: (8,)
```

---

#### 2. `predict(latent_z: Tensor, action: Tensor) -> Tensor`

**Purpose**: Predict next observation given current latent state and action.

**Signature**:
```python
def predict(
    self,
    latent_z: torch.Tensor,  # shape: (batch, latent_dim) or (latent_dim,)
    action: torch.Tensor     # shape: (batch,) or scalar
) -> torch.Tensor:
    """
    Predict next observation from latent state and action.
    
    Args:
        latent_z: Latent state tensor
        action: Action tensor (integers for discrete actions)
        
    Returns:
        next_obs_pred: Predicted next observation, shape (batch, obs_dim) or (obs_dim,)
        
    Raises:
        ValueError: If latent_z.shape[-1] != latent_dim
    """
```

**Contract**:
- **Differentiable**: Supports backpropagation through prediction (for planning)
- **Action encoding**: Discrete actions are one-hot encoded internally
- **Used for planning**: Actor queries this method for imagination-based rollouts
- **No gradient updates**: Calling predict() does not update Observer weights (use train mode separately)

**Example**:
```python
z = torch.randn(8)                      # Current latent state
action = torch.tensor(2)                # Action: move right
next_obs_pred = observer.predict(z, action)  # shape: (10,)
```

---

#### 3. `compute_loss(obs: Tensor, action: Tensor, next_obs: Tensor) -> Tensor`

**Purpose**: Compute prediction loss for training.

**Signature**:
```python
def compute_loss(
    self,
    obs: torch.Tensor,       # shape: (batch, obs_dim)
    action: torch.Tensor,    # shape: (batch,)
    next_obs: torch.Tensor   # shape: (batch, obs_dim)
) -> torch.Tensor:
    """
    Compute prediction loss (MSE) for Observer training.
    
    Args:
        obs: Current observations
        action: Actions taken
        next_obs: Actual next observations (ground truth)
        
    Returns:
        loss: Scalar prediction loss (mean squared error)
        
    Constitutional Requirement:
        Loss MUST use only prediction error—NO task rewards allowed.
    """
```

**Loss Function**: Mean Squared Error (MSE)
```python
z = self.encode(obs)
next_obs_pred = self.predict(z, action)
loss = F.mse_loss(next_obs_pred, next_obs)
```

**Contract**:
- **Constitutional compliance**: Loss uses ONLY `(obs, action, next_obs)` - NO rewards
- **Scalar output**: Returns single scalar for optimizer.step()
- **Batch processing**: Operates on batches for efficiency

---

#### 4. `save(path: str) -> None`

**Purpose**: Save Observer model to disk.

**Signature**:
```python
def save(self, path: str) -> None:
    """
    Save Observer state_dict to file.
    
    Args:
        path: File path (e.g., "checkpoints/observer_step1000.pt")
        
    Saves:
        - Model state_dict (encoder + predictor weights)
        - Latent dimension (for architecture verification on load)
    """
```

**Contract**:
- **Independent saving**: Can save Observer without Actor
- **State dict only**: Saves weights, not optimizer state (use Checkpoint for full state)
- **Architecture metadata**: Includes latent_dim for load validation

---

#### 5. `load(path: str) -> None`

**Purpose**: Load Observer model from disk.

**Signature**:
```python
def load(self, path: str) -> None:
    """
    Load Observer state_dict from file.
    
    Args:
        path: File path to saved checkpoint
        
    Raises:
        ValueError: If loaded latent_dim doesn't match current architecture
    """
```

**Contract**:
- **Architecture matching**: Current Observer must have same latent_dim as saved model
- **Exact restoration**: Loaded model produces identical encode() outputs as saved model
- **Device handling**: Automatically maps to self.device

---

## Optional Methods

#### `get_prediction_metrics(obs, action, next_obs) -> Dict[str, float]`

**Purpose**: Compute detailed prediction metrics for evaluation.

**Returns**:
```python
{
    "rmse": float,           # Root mean squared error
    "mae": float,            # Mean absolute error
    "per_dim_mse": List[float],  # MSE per observation dimension
}
```

---

## Usage Examples

### Training Observer

```python
# Initialize
observer = Observer(obs_dim=10, action_dim=3, latent_dim=8)
optimizer = torch.optim.Adam(observer.parameters(), lr=1e-3)

# Training loop
for batch in replay_buffer.sample(batch_size=64):
    obs = batch["obs"]
    action = batch["action"]
    next_obs = batch["next_obs"]
    
    # Compute loss (only prediction error - constitutional requirement)
    loss = observer.compute_loss(obs, action, next_obs)
    
    # Update Observer
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Log metrics
    wandb.log({"observer/pred_loss": loss.item()})
```

### Using Observer for Actor

```python
# Encode observation to latent state
obs = env.reset()
z = observer.encode(obs)

# Actor selects action using latent state (not raw obs - constitutional requirement)
action = actor.select_action(z)
```

### Planning with Observer

```python
# Simulate K-step rollout
def plan(observer, actor, z_current, horizon=3):
    best_action = None
    best_value = -float('inf')
    
    for candidate_action in range(action_dim):
        # Predict next state using Observer
        next_z = observer.predict(z_current, torch.tensor(candidate_action))
        
        # Recursive rollout...
        value = rollout(observer, actor, next_z, horizon - 1)
        if value > best_value:
            best_value = value
            best_action = candidate_action
    
    return best_action
```

---

## Testing Contract

### Required Tests

1. **Encoding Determinism**: Same obs → same latent_z across multiple calls
2. **Prediction Differentiability**: `predict()` supports backprop for planning
3. **Loss Constitutional Compliance**: `compute_loss()` never accesses rewards
4. **Save/Load Fidelity**: Loaded model produces identical outputs
5. **Batch Processing**: Works with batched inputs (batch_size > 1)

### Example Test

```python
def test_observer_encoding_determinism():
    observer = Observer(obs_dim=10, action_dim=3, latent_dim=8)
    obs = torch.randn(10)
    
    z1 = observer.encode(obs)
    z2 = observer.encode(obs)
    
    assert torch.allclose(z1, z2), "Encoding must be deterministic"
```

---

## Constraints & Invariants

### Constitutional Constraints (NON-NEGOTIABLE)

1. **No Reward Access**: Observer training MUST NOT use task rewards - only prediction error
2. **Independent Testability**: Observer MUST work without Actor (no dependency)
3. **Latent Interface Only**: Actor receives ONLY `latent_z`, never raw observations

### Architectural Invariants

1. **Dimensionality**: `encode()` always outputs `latent_dim` dimensions
2. **Prediction Shape**: `predict()` output matches `obs_dim`
3. **Differentiability**: All forward passes support autograd (for planning)

---

## Performance Expectations

| Operation | Latency (CPU) | Latency (GPU) |
|-----------|---------------|---------------|
| `encode(obs)` | <1ms | <0.1ms |
| `predict(z, action)` | <1ms | <0.1ms |
| `compute_loss(batch=64)` | <5ms | <1ms |

**Note**: GridWorld/CartPole have small obs_dim (1-4), so MLPs are very fast. Image observations would require CNN encoder.

---

## Version History

- **1.0.0** (2026-02-25): Initial contract for Observer-Actor MVP
- Future: Probabilistic predictions, recurrent encoder for POMDPs
