<!--
SYNC IMPACT REPORT - Constitution v1.0.0
========================================
Version Change: template → 1.0.0 (INITIAL)
Ratification: 2026-02-25

Modified/Added Principles:
- ✅ PRINCIPLE 1: Modular Architecture (Observer/Actor Separation)
- ✅ PRINCIPLE 2: Dual Optimization (Predictive vs Task Rewards)
- ✅ PRINCIPLE 3: Model-Based Planning
- ✅ PRINCIPLE 4: Reproducible Experiments  
- ✅ PRINCIPLE 5: Incremental Complexity

Added Sections:
- ✅ Research Standards
- ✅ Technology Stack

Templates Status:
- ⚠ plan-template.md: Review needed (add RL-specific gates)
- ⚠ spec-template.md: Review needed (add experiment scenarios)
- ⚠ tasks-template.md: Review needed (add training/eval phases)

Follow-up TODOs:
- None - all placeholders resolved

Next Steps:
- Update plan-template.md with RL-specific constitution checks
- Update spec-template.md to support experiment specifications
- Adapt tasks-template.md for training pipeline phases
-->

# AIXI Constitution

## Core Principles

### I. Modular Architecture (Observer/Actor Separation)

The system MUST maintain strict architectural separation between world modeling (Observer) and decision-making (Actor):

- **Observer** and **Actor** MUST be separate, independently instantiable modules
- Observer MUST NOT access task rewards; Actor MUST NOT directly access raw observations
- Interface contract: Observer outputs latent state `z_t`, Actor inputs `z_t` and outputs action `a_t`
- Each module MUST be testable in isolation (Observer via prediction metrics, Actor via reward metrics)
- Breaking this separation requires constitutional amendment

**Rationale**: Mirrors the epistemological separation between "what is the world?" and "what should I do?"—enabling independent development, testing, and theoretical analysis of world modeling vs. policy learning.

### II. Dual Optimization

Observer and Actor MUST optimize different, well-defined objectives:

- **Observer reward**: Minimize prediction error `L = ||o_{t+1} - o_hat_{t+1}||²` (or more sophisticated predictive losses)
- **Actor reward**: Maximize task-specific reward `r_actor` from environment
- Observer training MUST NOT use task rewards; Actor training MUST NOT modify Observer's predictive loss
- Optional: Observer may include complexity penalty (MDL principle) to prevent overfitting
- Mixing these objectives requires explicit justification and constitutional review

**Rationale**: Reflects the core insight that world modeling (prediction) and goal achievement (reward) are fundamentally different learning signals. Observer learns "physics," Actor learns "strategy."

### III. Model-Based Planning

Actor MUST leverage Observer's world model for imagination and planning:

- Actor MUST be capable of querying Observer for predicted next states (even if not always used)
- Implementation MUST support rollout-based planning: "What if I take action X?"
- Minimum requirement: Actor can simulate at least 1 step ahead using Observer's model
- Aspirational: Multi-step Monte Carlo tree search or trajectory optimization
- Purely reactive (model-free) policies are permitted only as baselines or ablations

**Rationale**: The entire architecture exists to enable model-based RL. Without imagination/planning, the Observer-Actor split loses its primary advantage over end-to-end model-free RL.

### IV. Reproducible Experiments (NON-NEGOTIABLE)

All experiments MUST be tracked, versioned, and reproducible:

- Every training run MUST log: hyperparameters, random seeds, environment config, git commit hash
- Experiment tracking via MLflow, Weights & Biases, or equivalent (select one standard tool)
- Checkpoints MUST be saved at regular intervals with metadata (step number, timestamp, metrics)
- Results MUST include: learning curves, final performance, compute time, hardware used
- Code changes affecting results MUST reference experiment IDs in commit messages
- No "one-off experiments" without logging—if not worth tracking, not worth running

**Rationale**: Research is only valid if reproducible. RL experiments are notoriously sensitive to hyperparameters and randomness; systematic tracking is the only defense against wasted effort.

### V. Incremental Complexity

Start simple, add complexity only when justified by experimental evidence:

- **Iteration 1**: Tabular or linear models, simple environments (GridWorld, CartPole)
- **Iteration 2**: Neural networks, moderately complex environments (LunarLander, Atari subset)
- **Iteration 3**: Advanced architectures (transformers, RNNs, attention), hard environments
- Each complexity increase MUST be motivated by: (a) baseline established, (b) clear research question, (c) failure mode identified
- When adding complexity, keep simplified version as ablation baseline
- Performance regressions require rollback or investigation—never ignore

**Rationale**: AIXI theory is elegant; implementation is messy. Premature optimization and architecture complexity destroy debuggability. Build understanding incrementally.

## Research Standards

### Baselines & Comparisons

Every novel component MUST be compared against:

- **Observer baselines**: Perfect model (oracle), random model, learned single-step predictor
- **Actor baselines**: Random policy, model-free DQN/PPO, oracle policy (if computable)
- Report relative performance, not just absolute numbers

### Metrics & Evaluation

Standard metrics MUST be reported for every experiment:

- **Observer**: Prediction RMSE, calibration, negative log-likelihood
- **Actor**: Cumulative reward, sample efficiency (reward vs. timesteps), wall-clock time
- **Planning**: Rollout accuracy, planning horizon achieved, compute cost per action

### Environment Standardization

Use standardized RL benchmarks from Gymnasium/Gym unless custom environment required:

- Document environment details: observation space, action space, reward structure, termination
- Custom environments MUST include: specification, test suite, visualization

## Technology Stack

### Required Stack

- **Language**: Python 3.10+
- **Deep Learning**: PyTorch 2.0+ (primary framework)
- **RL Environments**: Gymnasium (OpenAI Gym successor)
- **Experiment Tracking**: Weights & Biases (W&B) or MLflow
- **Testing**: pytest + hypothesis (property-based testing for RL edge cases)
- **Dependency Management**: Poetry or uv

### Permitted Libraries

- RL utilities: Stable-Baselines3 (for baselines), Tianshou, CleanRL patterns
- Visualization: matplotlib, plotly, tensorboard
- Model architectures: torchvision, timm (for visual encoders)

### Prohibited Without Justification

- Proprietary/closed-source RL frameworks
- Deprecated libraries (old gym, tf1.x)
- Heavy frameworks that obscure agent logic (unless comparing as baseline)

## Governance

### Amendment Process

1. Proposed change documented in GitHub issue with rationale
2. Constitutional review: Does this change principles or just clarify?
3. If principle change: Requires demonstration via experiment or migration plan
4. Amendment merged only after version number update and affected templates reviewed

### Version Semantics

- **MAJOR**: Backward-incompatible principle removals/redefinitions (e.g., removing Observer/Actor separation)
- **MINOR**: New principles added, material guidance expansion (e.g., adding principle VI)
- **PATCH**: Clarifications, wording improvements, non-semantic fixes

### Compliance

- Constitution supersedes ad-hoc decisions
- Violations MUST be documented as "technical debt" with remediation plan
- Code reviews MUST verify: modular separation, dual optimization, experiment logging

### Guidance File

For detailed development guidance, implementation patterns, and best practices, refer to:  
`.specify/memory/guidance.md` (to be created)

**Version**: 1.0.0 | **Ratified**: 2026-02-25 | **Last Amended**: 2026-02-25
