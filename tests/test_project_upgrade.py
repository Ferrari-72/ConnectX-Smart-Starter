from pathlib import Path

import torch

from connectx_rl.dqn_agent import DQNAgent
from connectx_rl.dqn_trainer import DQNTrainer
from connectx_rl.evaluation import evaluate_agent
from connectx_rl.local_game import RandomAgent
from connectx_rl.replay_buffer import Transition


def test_trainer_checkpoint_round_trip(tmp_path: Path):
    trainer = DQNTrainer(batch_size=8, seed=123)
    trainer.train(episodes=4)
    trainer.epsilon = 0.42

    checkpoint_path = tmp_path / "dqn_checkpoint.pt"
    trainer.save_checkpoint(checkpoint_path)

    restored = DQNTrainer(batch_size=8, seed=999)
    restored.load_checkpoint(checkpoint_path)

    assert restored.epsilon == 0.42
    original_weight = next(trainer.q_network.parameters()).detach().clone()
    restored_weight = next(restored.q_network.parameters()).detach().clone()
    assert torch.allclose(original_weight, restored_weight)


def test_trainer_accepts_single_opponent_pool_entry():
    trainer = DQNTrainer(
        batch_size=8,
        seed=0,
        opponent_pool=[("random", lambda seed: RandomAgent(seed=seed))],
    )

    stats = trainer.train(episodes=3)

    assert stats.episodes == 3


def test_evaluate_agent_returns_named_summary():
    trainer = DQNTrainer(batch_size=8, seed=0)
    trainer.train(episodes=3)
    agent = DQNAgent(q_network=trainer.q_network, epsilon=0.0)

    summary = evaluate_agent(
        agent,
        opponents=[("random", lambda seed: RandomAgent(seed=seed))],
        games_per_opponent=2,
    )

    assert "random" in summary
    assert summary["random"]["games"] == 2
    assert summary["random"]["wins"] >= 0


def test_save_checkpoint_creates_parent_directory(tmp_path: Path):
    trainer = DQNTrainer(batch_size=8, seed=123)
    nested_path = tmp_path / "nested" / "checkpoints" / "dqn.pt"

    trainer.save_checkpoint(nested_path)

    assert nested_path.exists()


class FixedActionValueNet(torch.nn.Module):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bias.unsqueeze(0).repeat(x.shape[0], 1)


def test_optimize_step_uses_online_argmax_with_target_values():
    trainer = DQNTrainer(batch_size=1, seed=0, gamma=0.5, device="cpu")
    trainer.q_network = FixedActionValueNet([1.0, 9.0, 3.0, 2.0, 0.0, -1.0, -2.0])
    trainer.target_network = FixedActionValueNet([4.0, 5.0, 100.0, 1.0, 0.0, -1.0, -2.0])
    trainer.optimizer = torch.optim.SGD(trainer.q_network.parameters(), lr=0.0)

    state = torch.zeros((2, 6, 7), dtype=torch.float32)
    trainer.replay_buffer.push(
        Transition(
            state=state,
            action=0,
            reward=1.0,
            next_state=state,
            done=False,
        )
    )

    targets = trainer._compute_targets(
        rewards=torch.tensor([1.0]),
        next_states=state.unsqueeze(0),
        dones=torch.tensor([0.0]),
    )

    assert targets.shape == (1,)
    assert targets.item() == 1.0 + 0.5 * 5.0
