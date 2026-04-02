from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from connectx_rl.dqn_agent import DQNAgent
from connectx_rl.dqn_trainer import TrainingStats
from connectx_rl.evaluation import OpponentFactory, evaluate_agent
from connectx_rl.minimax_dqn_agent import MinimaxDQNAgent


EvaluationSummary = dict[str, dict[str, int]]


@dataclass(frozen=True)
class EvaluationSnapshot:
    episode: int
    dqn_summary: EvaluationSummary
    hybrid_summary: EvaluationSummary
    dqn_score: float
    hybrid_score: float


@dataclass(frozen=True)
class TrainingCampaignReport:
    training_stats: TrainingStats
    evaluations: list[EvaluationSnapshot]
    best_episode: int | None
    latest_checkpoint_path: Path
    best_checkpoint_path: Path


def _opponent_weight(name: str) -> float:
    if name == "random":
        return 1.0
    if name.startswith("minimax_depth_"):
        depth = int(name.removeprefix("minimax_depth_"))
        return float(depth + 1)
    return 1.0


def score_evaluation_summary(summary: EvaluationSummary) -> float:
    total = 0.0
    for name, result in summary.items():
        games = max(1, result["games"])
        normalized_margin = (result["wins"] - result["losses"]) / games
        total += _opponent_weight(name) * normalized_margin
    return total


def _combine_training_stats(stats: list[TrainingStats]) -> TrainingStats:
    return TrainingStats(
        episodes=sum(chunk.episodes for chunk in stats),
        wins=sum(chunk.wins for chunk in stats),
        losses=sum(chunk.losses for chunk in stats),
        draws=sum(chunk.draws for chunk in stats),
        updates=sum(chunk.updates for chunk in stats),
    )


def run_training_campaign(
    trainer: Any,
    total_episodes: int,
    eval_interval: int,
    eval_games: int,
    eval_opponents: list[tuple[str, OpponentFactory]],
    hybrid_depth: int,
    latest_checkpoint_path: str | Path,
    best_checkpoint_path: str | Path,
) -> TrainingCampaignReport:
    if eval_interval <= 0:
        raise ValueError("eval_interval must be positive")

    latest_checkpoint_path = Path(latest_checkpoint_path)
    best_checkpoint_path = Path(best_checkpoint_path)

    chunks: list[TrainingStats] = []
    evaluations: list[EvaluationSnapshot] = []
    best_episode: int | None = None
    best_key: tuple[float, float] | None = None
    completed_episodes = 0

    while completed_episodes < total_episodes:
        chunk_size = min(eval_interval, total_episodes - completed_episodes)
        chunk_stats = trainer.train(chunk_size)
        chunks.append(chunk_stats)
        completed_episodes += chunk_size

        dqn_agent = DQNAgent(q_network=trainer.q_network, epsilon=0.0)
        hybrid_agent = MinimaxDQNAgent(q_network=trainer.q_network, depth=hybrid_depth)
        dqn_summary = evaluate_agent(dqn_agent, opponents=eval_opponents, games_per_opponent=eval_games)
        hybrid_summary = evaluate_agent(hybrid_agent, opponents=eval_opponents, games_per_opponent=eval_games)

        snapshot = EvaluationSnapshot(
            episode=completed_episodes,
            dqn_summary=dqn_summary,
            hybrid_summary=hybrid_summary,
            dqn_score=score_evaluation_summary(dqn_summary),
            hybrid_score=score_evaluation_summary(hybrid_summary),
        )
        evaluations.append(snapshot)

        metadata = {
            "episode": completed_episodes,
            "dqn_score": snapshot.dqn_score,
            "hybrid_score": snapshot.hybrid_score,
            "dqn_summary": dqn_summary,
            "hybrid_summary": hybrid_summary,
        }
        trainer.save_checkpoint(latest_checkpoint_path, metadata={**metadata, "role": "latest"})

        current_key = (snapshot.hybrid_score, snapshot.dqn_score)
        if best_key is None or current_key > best_key:
            best_key = current_key
            best_episode = completed_episodes
            trainer.save_checkpoint(best_checkpoint_path, metadata={**metadata, "role": "best"})

    return TrainingCampaignReport(
        training_stats=_combine_training_stats(chunks),
        evaluations=evaluations,
        best_episode=best_episode,
        latest_checkpoint_path=latest_checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
    )
