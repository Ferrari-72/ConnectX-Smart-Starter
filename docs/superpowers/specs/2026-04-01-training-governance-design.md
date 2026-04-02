# Training Governance Design

**Goal:** Upgrade the ConnectX training workflow so it can save the strongest checkpoint, train against a stronger opponent pool, and evaluate both DQN and Minimax-DQN during training without making the default workflow too heavy.

## Architecture

Keep `connectx_rl.dqn_trainer.DQNTrainer` focused on episode generation, replay updates, and checkpoint serialization. Add a higher-level training orchestration layer that trains in chunks, runs periodic evaluations, scores progress, and manages `latest` plus `best` checkpoints.

## Components

- `connectx_rl.dqn_trainer`: stays responsible for DQN optimization and raw checkpoint save/load.
- `connectx_rl.training_campaign`: new orchestration module for chunked training, evaluation scheduling, score comparison, and checkpoint policy.
- `connectx_rl.training_presets`: expanded to describe train opponents, eval opponents, eval interval, hybrid search depth, and separate best/latest checkpoint targets.
- `scripts.train_dqn`: becomes a thin CLI wrapper that resolves a preset, builds opponent pools, launches the campaign, and prints the evaluation timeline.

## Data Flow

1. CLI resolves a training preset.
2. CLI builds a training opponent pool and an evaluation opponent pool.
3. Campaign trains the DQN in chunks of `eval_interval`.
4. After each chunk it evaluates:
   - `DQNAgent` with greedy action selection
   - `MinimaxDQNAgent` with configurable search depth
5. Campaign computes weighted scores that value stronger opponents more heavily.
6. Campaign always saves `latest`; it overwrites `best` only when the weighted hybrid-first score improves.

## Error Handling

- Invalid opponent pool names fail fast with `ValueError`.
- `best` checkpoint logic must work even when no previous best exists.
- Final chunk must handle `episodes % eval_interval != 0`.
- Checkpoint directories continue to be created automatically.

## Testing

- Add orchestration tests for interval chunking, periodic evaluation, and best-checkpoint replacement.
- Extend preset tests to cover new scheduling and checkpoint fields.
- Re-run focused pytest targets, then the full suite if the focused checks pass.
