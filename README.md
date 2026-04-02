# ConnectX Smart Starter

A beginner-friendly ConnectX bot with smart tactical play and efficient hybrid search. Easy to try, fast to run, and stronger than a typical starter submission.

This repository is a small learning-first ConnectX project that grows in stages:

1. `Minimax` baseline with a hand-written heuristic.
2. `DQN` baseline that learns Q-values from self-play against a simple opponent.
3. `Minimax-DQN` hybrid that uses the learned network to score search leaves.

The code is intentionally compact so you can read the full pipeline and map each file back to the RL concepts.

## Current Structure

```text
connectx_rl/
  board_utils.py        # Board representation, legal moves, winner detection
  heuristics.py         # Hand-written evaluation for pure minimax
  minimax_agent.py      # Search baseline
  dqn_agent.py          # State encoding and epsilon-greedy action selection
  q_network.py          # Small fully-connected Q-network
  replay_buffer.py      # Experience replay storage
  dqn_trainer.py        # Minimal DQN training loop with checkpoint support
  minimax_dqn_agent.py  # Hybrid search that uses DQN at leaf nodes
  evaluation.py         # Shared evaluation helper

scripts/
  play_demo.py          # Pure minimax vs random
  train_dqn.py          # Train DQN and optionally save/load checkpoints
  play_hybrid_demo.py   # Small hybrid smoke test
  evaluate_agents.py    # Compare agent families against common opponents

tests/
  test_minimax_baseline.py
  test_dqn_baseline.py
  test_minimax_dqn.py
  test_project_upgrade.py
```

## Quick Start

Run the pure minimax baseline:

```bash
python -m scripts.play_demo
```

Train a small DQN model with periodic DQN and hybrid evaluation:

```bash
python -m scripts.train_dqn --preset quick
```

Recommended first serious run on your `RTX 4060` using the existing `pt` environment:

```powershell
& "E:\Users\ASUS\miniconda3\envs\pt\python.exe" -m scripts.train_dqn --preset midpack --device auto
```

Overnight run:

```bash
python -m scripts.train_dqn --preset overnight --device auto
```

Evaluate a saved hybrid agent:

```bash
python -m scripts.evaluate_agents --agent hybrid --checkpoint checkpoints/midpack_best.pt --games 10
```

Run the full test suite:

```bash
python -m pytest
```

## Learning Path

If you are studying the project from scratch, follow this order:

1. Read `board_utils.py` to understand how ConnectX states are represented.
2. Read `heuristics.py` and `minimax_agent.py` to see classical search.
3. Read `dqn_agent.py`, `q_network.py`, and `replay_buffer.py` to understand the DQN building blocks.
4. Read `dqn_trainer.py` to see how transitions are generated and optimized.
5. Read `minimax_dqn_agent.py` to see how search and learned value estimates are combined.

## Current Limitations

- If PyTorch is installed with CUDA support, training will auto-select `cuda`. Otherwise it falls back to `cpu`.
- The DQN trainer is intentionally minimal and not yet tuned for strong Kaggle performance.
- The hybrid agent uses `max Q` as a leaf score, which is a good teaching step but not the final form of a strong search evaluator.

## Training Presets

- `quick`: fast pipeline check with periodic evaluation and `latest`/`best` checkpoints.
- `midpack`: the best first balanced run on a 4060, using stronger opponents and hybrid evaluation.
- `overnight`: a longer run with the strongest built-in opponent ladder for comparing checkpoints the next day.

## Training Governance

- Training now runs in chunks and evaluates both `DQN` and `Minimax-DQN` after each interval.
- Every evaluation saves a `latest` checkpoint and may replace a `best` checkpoint if the hybrid-first weighted score improves.
- Opponent pools:
  - `random`: random-only smoke training
  - `mixed`: `random`, `minimax(depth=1)`, `minimax(depth=2)`
  - `ladder`: `random`, `minimax(depth=1)`, `minimax(depth=2)`, `minimax(depth=3)`

## Good Next Steps

- Install CUDA-enabled PyTorch so training can use the GPU.
- Add stronger opponents and periodic checkpoint evaluation.
- Replace the leaf `max Q` estimate with a more stable value-style target.
- Export a single-file Kaggle submission agent when you are ready to submit.
