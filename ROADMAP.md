# aixi Roadmap

AIXI-inspired RL with a strict Observer/Actor split: the Observer learns environment
dynamics from next-observation prediction only (never sees rewards); the Actor learns
policy from task reward only (never sees raw observations), communicating purely through
latent states, plus K-step imagination-based planning. Design is spec-driven
(`specs/001-observer-actor-mvp/`) with a constitution of 5 principles enforced by tests.

## Now

1. Finish the MVP task list: `specs/001-observer-actor-mvp/tasks.md` has roughly 39 of
   ~83 tasks checked. The gap concentrates in ES3 (Actor training) and ES4 (planning) —
   per the README, Actor training and planning have **no entry-point scripts in the
   current tree**.
2. Add `experiments/scripts/train_actor.py` and a planning-evaluation script so all four
   experiment scenarios (ES1–ES4) are runnable end to end.
3. Make planning combine Observer dynamics with actor-owned reward/value estimates;
   prediction-only signatures are formally insufficient for control. The bounded
   binary advantage head proves the interface repair; the general Actor/planning
   implementation is still absent.
4. Run ES2 (Observer training) and confirm the <5% prediction-RMSE target; record actual
   numbers instead of README targets.

## Next

- Validate the core hypothesis: Actor-on-latents matches or beats an observation-level
  DQN baseline (`experiments/baselines/`). SPECULATIVE that the information bottleneck
  costs little on GridWorld — this is the experiment that decides it.
- Measure planning uplift: ES4 target is ≥20% improvement over the reactive policy.
- Constitutional compliance in CI: run `tests/test_integration/test_constitution.py`
  automatically, not just locally.
- Replace the placeholder license ("[Your License Here]") — MIT per portfolio convention.
- Decide CartPole's role: wrappers exist but no experiment uses them; cut or schedule.

## Later

- Scale beyond MLP world models: recurrent or Transformer Observer for partial
  observability (the AIXI-relevant setting).
- Second spec cycle (002+): whatever ES3/ES4 results suggest — e.g. latent-drift issues
  between separately trained Observer and Actor. SPECULATIVE direction pending data.
- W&B dashboard made public as part of a publishable repo.

## Done

- 2026-08-25: added a bounded two-state regression proving that perfect
  action-conditioned next-observation prediction is not control-sufficient when
  reward preferences differ. Current APIs realize zero prediction error and
  opposite optimal actions while preserving the Observer/Actor reward boundary.
  A scalar Actor-owned advantage head learns both the original `[0, 1]` policy
  and its reward-flipped `[1, 0]` control without changing Observer dynamics.
  [INCREMENTAL / FORMAL counterexample and minimal repair + EMPIRICAL API regression]
- 2026-08-22 / 2026-08-21: Observer quickstart aligned, AI-owned docs boundary defined,
  project entrypoint established.
- Full specification suite written: spec, plan, research decisions, data model, module
  contracts, tasks, quickstart (`specs/001-observer-actor-mvp/`).
- Infrastructure: Poetry project, seed management, W&B tracking/checkpointing utilities,
  black/mypy/isort tooling, test scaffolding (~39 setup + ES1/ES2 tasks complete).
- ES1 random-baseline and ES2 Observer-training scripts are present and documented.
