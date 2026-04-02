# Summary Report

## Repository

- Name: `ConnectX Smart Starter`
- Description: beginner-friendly ConnectX bot with smart tactical play and efficient hybrid search

## What This Project Includes

- A compact ConnectX training and evaluation pipeline
- A `DQN` baseline and a `Minimax-DQN` hybrid agent
- Exported single-file submissions for Kaggle-style use
- Tactical submission variants that add immediate win/block checks before search

## Today’s Outcome

- The strongest historically proven checkpoint line remains the older `midpack_best` lineage.
- The newer isolated `midpack` run finished, but it did not reproduce the earlier `hybrid` peak against `minimax_depth_2`.
- For submission use today, the preferred candidate is `submissions/midpack_best_tactical_submission.py`.
- The conservative fallback is `submissions/midpack_best_submission.py`.

## Evidence Used

- `checkpoints/midpack_run.log` shows a historical peak at episode `1000` where the hybrid agent achieved `12W/0L/0D` against `minimax_depth_2`.
- `checkpoints/runs/midpack-20260402-105710/best.pt` and `latest.pt` did not match that earlier result.

## Recommended Use

- Use `submissions/midpack_best_tactical_submission.py` as the main public-facing starter bot.
- Keep the non-tactical `best` submission as a fallback for comparison.

## Known Limits

- The project has shown unstable progress across checkpoints.
- The current training pipeline has not demonstrated a reliable path to beating `minimax_depth_3`.
- Large training checkpoints are intentionally excluded from the public repository.
