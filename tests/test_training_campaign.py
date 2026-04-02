from pathlib import Path

import torch

from connectx_rl.dqn_trainer import TrainingStats
from connectx_rl.training_campaign import run_training_campaign, score_evaluation_summary


class FakeTrainer:
    def __init__(self) -> None:
        self.q_network = torch.nn.Linear(1, 1)
        self.train_calls: list[int] = []
        self.saved: list[tuple[Path, dict | None]] = []

    def train(self, episodes: int) -> TrainingStats:
        self.train_calls.append(episodes)
        return TrainingStats(
            episodes=episodes,
            wins=episodes,
            losses=0,
            draws=0,
            updates=episodes * 2,
        )

    def save_checkpoint(self, path: str | Path, metadata: dict | None = None) -> None:
        self.saved.append((Path(path), metadata))


def test_score_evaluation_summary_values_stronger_opponents_more():
    easy_only = {
        "random": {"games": 4, "wins": 4, "losses": 0, "draws": 0},
        "minimax_depth_2": {"games": 4, "wins": 0, "losses": 4, "draws": 0},
    }
    stronger = {
        "random": {"games": 4, "wins": 2, "losses": 2, "draws": 0},
        "minimax_depth_2": {"games": 4, "wins": 3, "losses": 1, "draws": 0},
    }

    assert score_evaluation_summary(stronger) > score_evaluation_summary(easy_only)


def test_campaign_trains_in_chunks_and_tracks_best_checkpoint(tmp_path: Path, monkeypatch):
    trainer = FakeTrainer()
    dqn_results = iter(
        [
            {
                "random": {"games": 2, "wins": 1, "losses": 1, "draws": 0},
                "minimax_depth_2": {"games": 2, "wins": 0, "losses": 2, "draws": 0},
            },
            {
                "random": {"games": 2, "wins": 2, "losses": 0, "draws": 0},
                "minimax_depth_2": {"games": 2, "wins": 0, "losses": 2, "draws": 0},
            },
            {
                "random": {"games": 2, "wins": 2, "losses": 0, "draws": 0},
                "minimax_depth_2": {"games": 2, "wins": 1, "losses": 1, "draws": 0},
            },
        ]
    )
    hybrid_results = iter(
        [
            {
                "random": {"games": 2, "wins": 2, "losses": 0, "draws": 0},
                "minimax_depth_2": {"games": 2, "wins": 0, "losses": 2, "draws": 0},
            },
            {
                "random": {"games": 2, "wins": 2, "losses": 0, "draws": 0},
                "minimax_depth_2": {"games": 2, "wins": 1, "losses": 1, "draws": 0},
            },
            {
                "random": {"games": 2, "wins": 1, "losses": 1, "draws": 0},
                "minimax_depth_2": {"games": 2, "wins": 0, "losses": 2, "draws": 0},
            },
        ]
    )

    monkeypatch.setattr(
        "connectx_rl.training_campaign.DQNAgent",
        lambda q_network, epsilon: {"kind": "dqn", "epsilon": epsilon},
    )
    monkeypatch.setattr(
        "connectx_rl.training_campaign.MinimaxDQNAgent",
        lambda q_network, depth: {"kind": "hybrid", "depth": depth},
    )

    def fake_evaluate(agent, opponents, games_per_opponent):
        assert games_per_opponent == 2
        assert len(opponents) == 2
        if agent["kind"] == "dqn":
            return next(dqn_results)
        return next(hybrid_results)

    monkeypatch.setattr("connectx_rl.training_campaign.evaluate_agent", fake_evaluate)

    report = run_training_campaign(
        trainer=trainer,
        total_episodes=10,
        eval_interval=4,
        eval_games=2,
        eval_opponents=[
            ("random", lambda seed: None),
            ("minimax_depth_2", lambda seed: None),
        ],
        hybrid_depth=2,
        latest_checkpoint_path=tmp_path / "latest.pt",
        best_checkpoint_path=tmp_path / "best.pt",
    )

    assert trainer.train_calls == [4, 4, 2]
    assert report.training_stats.episodes == 10
    assert len(report.evaluations) == 3
    assert report.best_episode == 8

    latest_saves = [path for path, _metadata in trainer.saved if path.name == "latest.pt"]
    best_saves = [path for path, _metadata in trainer.saved if path.name == "best.pt"]
    assert len(latest_saves) == 3
    assert len(best_saves) == 2
