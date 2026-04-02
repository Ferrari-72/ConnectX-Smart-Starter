from pathlib import Path

import torch

from connectx_rl.checkpoints import load_checkpoint, save_checkpoint
from connectx_rl.dqn_trainer import DQNTrainer
from connectx_rl.evaluation import summarize_matchups
from connectx_rl.local_game import MinimaxAgent, RandomAgent


def test_checkpoint_round_trip_preserves_weights(tmp_path: Path):
    trainer = DQNTrainer(batch_size=8)
    trainer.train(episodes=8)

    checkpoint_path = tmp_path / "dqn_checkpoint.pt"
    trainer.save_checkpoint(checkpoint_path)

    restored = DQNTrainer(batch_size=8)
    restored.load_checkpoint(checkpoint_path)

    for original_param, restored_param in zip(trainer.q_network.parameters(), restored.q_network.parameters()):
        assert torch.allclose(original_param, restored_param)


def test_trainer_accepts_opponent_pool():
    trainer = DQNTrainer(batch_size=8, opponents=[RandomAgent(seed=1), MinimaxAgent(depth=1)])
    stats = trainer.train(episodes=4)

    assert stats.episodes == 4
    assert stats.updates >= 0


def test_summarize_matchups_reports_named_results():
    summary = summarize_matchups(
        agent=MinimaxAgent(depth=2),
        opponents={
            "random": RandomAgent(seed=0),
            "minimax_d1": MinimaxAgent(depth=1),
        },
        games_per_matchup=2,
    )

    assert "random" in summary
    assert "minimax_d1" in summary
    assert set(summary["random"].keys()) == {"wins", "losses", "draws"}
