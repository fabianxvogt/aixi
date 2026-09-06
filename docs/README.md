# AI-owned project documentation

Use this folder for concise, reviewable notes maintained by coding agents: verified setup, architecture, validation commands, and bounded cleanup plans.

The project README and any document marked `human-owned` remain authoritative. Do not overwrite human-owned material or make unsupported claims.

Never store credentials, private data, generated output, logs, datasets, or build artifacts here. Preserve unrelated local work and keep each change focused.

## Evidence notes

- [`agent-wave-2026-08-25-prediction-control-sufficiency.md`](agent-wave-2026-08-25-prediction-control-sufficiency.md)
  — bounded formal counterexample and current-API regression separating perfect
  next-observation prediction from reward-optimal control, plus the minimal
  scalar Actor-side repair and flipped-reward control.
