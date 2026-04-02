from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from connectx_rl.dqn_trainer import DQNTrainer
from connectx_rl.local_game import MinimaxAgent, RandomAgent
from connectx_rl.training_campaign import run_training_campaign
from connectx_rl.training_presets import PRESETS, resolve_training_args


def build_opponent_pool(mode: str):
    if mode == "random":
        return [("random", lambda seed: RandomAgent(seed=seed))]
    if mode == "mixed":
        return [
            ("random", lambda seed: RandomAgent(seed=seed)),
            ("minimax_depth_1", lambda seed: MinimaxAgent(depth=1)),
            ("minimax_depth_2", lambda seed: MinimaxAgent(depth=2)),
        ]
    if mode == "ladder":
        return [
            ("random", lambda seed: RandomAgent(seed=seed)),
            ("minimax_depth_1", lambda seed: MinimaxAgent(depth=1)),
            ("minimax_depth_2", lambda seed: MinimaxAgent(depth=2)),
            ("minimax_depth_3", lambda seed: MinimaxAgent(depth=3)),
        ]
    raise ValueError(f"Unsupported opponent mode: {mode}")


def derive_best_checkpoint_path(latest_checkpoint_path: Path) -> Path:
    return latest_checkpoint_path.with_name(f"{latest_checkpoint_path.stem}_best{latest_checkpoint_path.suffix}")


def resolve_run_checkpoint_paths(
    preset_name: str,
    save_checkpoint: Path | None,
    now: datetime | None = None,
) -> tuple[Path, Path]:
    if save_checkpoint is not None:
        return save_checkpoint, derive_best_checkpoint_path(save_checkpoint)

    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    run_directory = Path("checkpoints") / "runs" / f"{preset_name}-{timestamp}"
    return run_directory / "latest.pt", run_directory / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a minimal DQN agent for ConnectX.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="quick",
        help="Training preset tuned for a specific runtime budget.",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Override preset episode count.")
    parser.add_argument("--eval-games", type=int, default=None, help="Override preset evaluation game count.")
    parser.add_argument(
        "--opponents",
        choices=["random", "mixed", "ladder"],
        default=None,
        help="Override preset opponent pool.",
    )
    parser.add_argument("--save-checkpoint", type=Path, default=None, help="Optional checkpoint output path.")
    parser.add_argument("--load-checkpoint", type=Path, default=None, help="Optional checkpoint input path.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device selection for PyTorch.",
    )
    args = parser.parse_args()

    resolved = resolve_training_args(
        preset_name=args.preset,
        episodes=args.episodes,
        eval_games=args.eval_games,
        opponents=args.opponents,
    )
    train_opponent_pool = build_opponent_pool(resolved.opponents)
    eval_opponent_pool = build_opponent_pool(resolved.eval_opponents)
    trainer = DQNTrainer(opponent_pool=train_opponent_pool, device=args.device)
    if args.load_checkpoint is not None:
        trainer.load_checkpoint(args.load_checkpoint)
    latest_checkpoint_path, best_checkpoint_path = resolve_run_checkpoint_paths(
        preset_name=resolved.name,
        save_checkpoint=args.save_checkpoint,
    )
    report = run_training_campaign(
        trainer=trainer,
        total_episodes=resolved.episodes,
        eval_interval=resolved.eval_interval,
        eval_games=resolved.eval_games,
        eval_opponents=eval_opponent_pool,
        hybrid_depth=resolved.hybrid_depth,
        latest_checkpoint_path=latest_checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
    )
    stats = report.training_stats

    print("Training summary")
    print("================")
    print(f"Preset: {resolved.name}")
    print(f"Description: {resolved.description}")
    print(f"Episodes: {stats.episodes}")
    print(f"Wins: {stats.wins}")
    print(f"Losses: {stats.losses}")
    print(f"Draws: {stats.draws}")
    print(f"Gradient updates: {stats.updates}")
    print(f"Final epsilon: {trainer.epsilon:.3f}")
    print(f"Device: {trainer.device}")
    print(f"Train opponents: {resolved.opponents}")
    print(f"Eval opponents: {resolved.eval_opponents}")
    print(f"Eval interval: {resolved.eval_interval}")
    print(f"Latest checkpoint: {latest_checkpoint_path}")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print()
    print("Periodic evaluation")
    print("===================")
    for snapshot in report.evaluations:
        print(f"After episode {snapshot.episode}")
        print(f"  DQN score: {snapshot.dqn_score:.3f}")
        print(f"  Hybrid score: {snapshot.hybrid_score:.3f}")
        print("  DQN matchups:")
        for name, summary in snapshot.dqn_summary.items():
            print(
                f"    {name}: {summary['wins']}W/{summary['losses']}L/{summary['draws']}D"
                f" over {summary['games']} games"
            )
        print("  Hybrid matchups:")
        for name, summary in snapshot.hybrid_summary.items():
            print(
                f"    {name}: {summary['wins']}W/{summary['losses']}L/{summary['draws']}D"
                f" over {summary['games']} games"
            )
        print()
    print(f"Best checkpoint chosen at episode: {report.best_episode}")


if __name__ == "__main__":
    main()
