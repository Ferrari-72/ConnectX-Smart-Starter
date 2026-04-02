import pytest
import torch

from connectx_rl.dqn_agent import DQNAgent
from connectx_rl.dqn_trainer import DQNTrainer
from connectx_rl.board_utils import empty_board
from connectx_rl.runtime import resolve_device


def test_resolve_device_prefers_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    assert resolve_device("auto") == "cpu"


def test_resolve_device_uses_cuda_when_available(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    assert resolve_device("auto") == "cuda"


def test_resolve_device_rejects_unavailable_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    with pytest.raises(ValueError):
        resolve_device("cuda")


def test_trainer_uses_resolved_device(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    trainer = DQNTrainer(device="auto", batch_size=8)

    assert trainer.device == "cpu"


def test_dqn_agent_accepts_auto_device_with_cpu_network():
    q_network = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(84, 7))
    agent = DQNAgent(q_network=q_network, epsilon=0.0, device="auto")

    action = agent.choose_action(empty_board(), mark=1)

    assert 0 <= action <= 6
