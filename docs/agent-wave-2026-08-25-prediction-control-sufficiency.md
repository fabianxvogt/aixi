# Prediction is not control sufficiency

## Result

**FORMAL:** A perfect action-conditioned next-observation predictor is not by
itself sufficient to select reward-optimal actions. Consider two fully observed
states and two actions. Every state-action pair transitions to state 0, while
the immediate reward is 1 exactly when the state and action indices match.
Both states therefore have the same predicted next-observation signature
`(0, 0)`, but their unique optimal actions are respectively 0 and 1.

For any discount `0 <= gamma < 1`, the two actions from a given state share the
same continuation value, so their Q-value difference is exactly their immediate
reward difference. A deterministic selector receiving only the identical
prediction signature cannot return both required actions.

## Repository regression

**EMPIRICAL:**
`tests/test_integration/test_prediction_control_sufficiency.py` configures the
current `Observer` as an exact model of this construction. It verifies zero
next-observation error, distinct encoded states, identical action-indexed
prediction signatures, strict reward margins selecting actions `[0, 1]`, and
the exact batch boundary. Observer batches contain only the constructed raw
observation, action, and next observation values. Actor batches contain only
the constructed latent states, action, reward, and terminal flag values; their
four reward tensors reproduce the construction's reward table exactly.

The result does not claim that the proposed Observer/Actor architecture must
fail. A reward-trained Actor can use the distinct latent states. It establishes
the narrower design requirement that planning must combine predicted dynamics
with actor-owned reward or value information.

The paired bounded repair uses `BinaryAdvantageHead`, a one-logit Actor readout
of the existing latent. Pairwise Actor-owned rewards train the sign of
`reward(action=1) - reward(action=0)`. It selects `[0, 1]` for the original
table and `[1, 0]` when only that table is flipped, while the Observer and its
perfect dynamics remain unchanged. This is an exact two-state/two-action
control probe, not the planned multi-action Actor or rollout implementation.
Its public forward, loss, and action-selection paths reject non-finite latents;
the loss path also rejects non-finite rewards and tied preferences.

Classification: **INCREMENTAL**. Evidence labels: **FORMAL** for the bounded
counterexample and **EMPIRICAL** for its realization through the current APIs.

## Minimality and next falsifiable step

For this bounded task, the coarsest control-sufficient representation is one bit:
the unique optimal-action class. The existing latent already contains that bit;
the scalar reward-trained readout supplies its task-specific orientation. Without
a reward-structure prior, one pairwise preference label per state is necessary to
identify all four possible deterministic two-state control maps. Larger reward or
Q models are sufficient but not minimal for this construction.

Next, implement the general Actor's reward/value-aware planning path, explicitly
convert each predicted next observation back to a latent state, and extend the
paired regression beyond two actions and one-step common-successor dynamics.
