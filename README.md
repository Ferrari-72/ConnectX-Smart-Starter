# ConnectX Smart Starter

A beginner-friendly ConnectX bot with smart tactical play and efficient hybrid search.

![Tests](https://img.shields.io/badge/tests-38%20passed-brightgreen)
![Python](https://img.shields.io/badge/python-3.x-blue)
![Project](https://img.shields.io/badge/focus-ConnectX%20hybrid%20agent-orange)

Build, train, evaluate, and export a practical ConnectX starter bot from one compact codebase.

This project turns a compact ConnectX reinforcement learning sandbox into a clean, approachable starter repository: readable code, exportable submissions, and a practical hybrid agent that combines tactical checks with search-guided play.

## Highlights

- Hybrid `Minimax-DQN` agent with tactical pre-checks for immediate wins and blocks
- Single-file submission exports ready for Kaggle-style ConnectX environments
- Compact codebase designed for learning, iteration, and experimentation
- Automated test coverage with `38` passing tests in the current public release

## Recommended Submission

If you want the best public-facing submission candidate from this repository, start with:

- `submissions/midpack_best_tactical_submission.py`

Conservative fallback:

- `submissions/midpack_best_submission.py`

## Why This Repo

`ConnectX Smart Starter` is built for people who want more than a toy baseline, but do not want a giant, hard-to-follow training stack.

You can use it to:

- study how a ConnectX board, search agent, replay buffer, and DQN trainer fit together
- run small or medium training presets locally
- evaluate saved checkpoints against common opponents
- export a standalone submission bot without dragging the full project into production

## Quick Start

Run the pure minimax baseline:

```bash
python -m scripts.play_demo
```

Train a quick pipeline check:

```bash
python -m scripts.train_dqn --preset quick
```

Recommended first serious run on an RTX 4060 with the existing `pt` environment:

```powershell
& "E:\Users\ASUS\miniconda3\envs\pt\python.exe" -m scripts.train_dqn --preset midpack --device auto
```

Evaluate a saved hybrid checkpoint:

```bash
python -m scripts.evaluate_agents --agent hybrid --checkpoint checkpoints/midpack_best.pt --games 10
```

Run the full test suite:

```bash
python -m pytest
```

## Current Results

- The repository contains both `DQN` and `Minimax-DQN` hybrid agents.
- Historical experiments showed the hybrid pipeline can produce strong intermediate checkpoints against built-in minimax opponents.
- The best public recommendation today is still the older proven `midpack_best` tactical submission lineage.
- Reaching stable strength against deeper minimax opponents remains future work rather than a solved result.

## Project Structure

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
  kaggle_export.py      # Standalone submission export

scripts/
  play_demo.py          # Pure minimax vs random
  train_dqn.py          # Train DQN and save/load checkpoints
  play_hybrid_demo.py   # Small hybrid smoke test
  evaluate_agents.py    # Compare agent families against common opponents

submissions/
  midpack_best_submission.py
  midpack_best_tactical_submission.py
  midpack_latest_submission.py
  midpack_latest_tactical_submission.py

tests/
  test_minimax_baseline.py
  test_dqn_baseline.py
  test_minimax_dqn.py
  test_kaggle_export.py
```

## Learning Path

If you are reading the code from scratch, this order works well:

1. `connectx_rl/board_utils.py`
2. `connectx_rl/minimax_agent.py`
3. `connectx_rl/dqn_agent.py`
4. `connectx_rl/dqn_trainer.py`
5. `connectx_rl/minimax_dqn_agent.py`
6. `connectx_rl/kaggle_export.py`

## Training Presets

- `quick`: fast smoke run with periodic evaluation and `latest`/`best` checkpoints
- `midpack`: balanced first serious run with stronger opponents and hybrid evaluation
- `overnight`: longer ladder run for comparing checkpoints over a longer window

## Current Limitations

- CUDA is only used if your PyTorch install supports it; otherwise training falls back to `cpu`
- The DQN trainer is intentionally small and not yet tuned for top-tier Kaggle strength
- Hybrid strength can vary across checkpoints, so checkpoint selection still matters a lot
- The current public repo excludes large checkpoint artifacts on purpose

## Tests

The current public release was verified with:

```bash
python -m pytest
```

Result:

```text
38 passed
```
