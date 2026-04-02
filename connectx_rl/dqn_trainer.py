from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Callable

import torch
from torch import nn

from connectx_rl.board_utils import drop_piece, empty_board, is_full, valid_moves, winner
from connectx_rl.checkpoints import load_checkpoint as load_checkpoint_file
from connectx_rl.checkpoints import save_checkpoint as save_checkpoint_file
from connectx_rl.dqn_agent import DQNAgent, encode_board
from connectx_rl.local_game import AgentProtocol, RandomAgent
from connectx_rl.q_network import QNetwork
from connectx_rl.replay_buffer import ReplayBuffer, Transition
from connectx_rl.runtime import resolve_device

OpponentFactory = Callable[[int], AgentProtocol]


@dataclass
class TrainingStats:
    episodes: int
    wins: int
    losses: int
    draws: int
    updates: int


class DQNTrainer:
    def __init__(
        self,
        rows: int = 6,
        columns: int = 7,
        inarow: int = 4,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        buffer_capacity: int = 10_000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.995,
        target_sync_interval: int = 20,
        double_dqn: bool = True,
        seed: int = 0,
        device: str = "auto",
        opponents: list[AgentProtocol] | None = None,
        opponent_pool: list[tuple[str, OpponentFactory]] | None = None,
    ) -> None:
        self.rows = rows
        self.columns = columns
        self.inarow = inarow
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_sync_interval = target_sync_interval
        self.double_dqn = double_dqn
        self.device = resolve_device(device)
        self.seed = seed
        self.rng = random.Random(seed)
        self.opponent_pool = self._normalize_opponents(opponents=opponents, opponent_pool=opponent_pool)

        self.q_network = QNetwork(rows=rows, columns=columns).to(self.device)
        self.target_network = QNetwork(rows=rows, columns=columns).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    def _normalize_opponents(
        self,
        opponents: list[AgentProtocol] | None,
        opponent_pool: list[tuple[str, OpponentFactory]] | None,
    ) -> list[tuple[str, OpponentFactory]]:
        if opponents:
            normalized = []
            for index, opponent in enumerate(opponents):
                normalized.append((f"opponent_{index}", lambda _seed, opponent=opponent: deepcopy(opponent)))
            return normalized
        if opponent_pool:
            return opponent_pool
        return [("random", lambda episode_seed: RandomAgent(seed=episode_seed))]

    def _choose_training_action(self, board, mark: int) -> int:
        agent = DQNAgent(
            q_network=self.q_network,
            epsilon=self.epsilon,
            seed=self.rng.randint(0, 10_000_000),
            device=self.device,
        )
        return agent.choose_action(board, mark=mark)

    def _sample_opponent(self) -> AgentProtocol:
        _name, factory = self.rng.choice(self.opponent_pool)
        return factory(self.rng.randint(0, 10_000_000))

    def _play_training_episode(self) -> tuple[int, int]:
        board = empty_board(rows=self.rows, columns=self.columns)
        updates = 0
        opponent = self._sample_opponent()

        while True:
            state = encode_board(board, my_mark=1)
            action = self._choose_training_action(board, mark=1)
            board_after_agent = drop_piece(board, action, 1)

            if winner(board_after_agent, n=self.inarow) == 1:
                self.replay_buffer.push(
                    Transition(
                        state=state,
                        action=action,
                        reward=1.0,
                        next_state=encode_board(board_after_agent, my_mark=1),
                        done=True,
                    )
                )
                return 1, updates

            if is_full(board_after_agent):
                self.replay_buffer.push(
                    Transition(
                        state=state,
                        action=action,
                        reward=0.0,
                        next_state=encode_board(board_after_agent, my_mark=1),
                        done=True,
                    )
                )
                return 0, updates

            opponent_action = opponent.choose_action(board_after_agent, 2)
            board_after_opponent = drop_piece(board_after_agent, opponent_action, 2)

            if winner(board_after_opponent, n=self.inarow) == 2:
                self.replay_buffer.push(
                    Transition(
                        state=state,
                        action=action,
                        reward=-1.0,
                        next_state=encode_board(board_after_opponent, my_mark=1),
                        done=True,
                    )
                )
                return 2, updates

            if is_full(board_after_opponent):
                self.replay_buffer.push(
                    Transition(
                        state=state,
                        action=action,
                        reward=0.0,
                        next_state=encode_board(board_after_opponent, my_mark=1),
                        done=True,
                    )
                )
                return 0, updates

            self.replay_buffer.push(
                Transition(
                    state=state,
                    action=action,
                    reward=0.0,
                    next_state=encode_board(board_after_opponent, my_mark=1),
                    done=False,
                )
            )
            board = board_after_opponent

            if len(self.replay_buffer) >= self.batch_size:
                self._optimize_step()
                updates += 1

    def _optimize_step(self) -> float:
        self.q_network.train()
        batch = self.replay_buffer.sample(self.batch_size)

        states = batch.states.to(self.device)
        actions = batch.actions.to(self.device)
        rewards = batch.rewards.to(self.device)
        next_states = batch.next_states.to(self.device)
        dones = batch.dones.to(self.device)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        targets = self._compute_targets(rewards=rewards, next_states=next_states, dones=dones)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.q_network.eval()
        return float(loss.item())

    def _compute_targets(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q_values = self.target_network(next_states).max(dim=1).values
            return rewards + (1.0 - dones) * self.gamma * next_q_values

    def save_checkpoint(self, path: str | Path, metadata: dict | None = None) -> None:
        checkpoint = {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "config": {
                "rows": self.rows,
                "columns": self.columns,
                "inarow": self.inarow,
                "gamma": self.gamma,
                "batch_size": self.batch_size,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay": self.epsilon_decay,
                "target_sync_interval": self.target_sync_interval,
                "double_dqn": self.double_dqn,
                "seed": self.seed,
                "device": self.device,
            },
        }
        if metadata is not None:
            checkpoint["metadata"] = metadata
        save_checkpoint_file(path, checkpoint)

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = load_checkpoint_file(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = float(checkpoint["epsilon"])
        self.q_network.eval()
        self.target_network.eval()

    def train(self, episodes: int) -> TrainingStats:
        wins = losses = draws = updates = 0

        for episode in range(1, episodes + 1):
            result, episode_updates = self._play_training_episode()
            updates += episode_updates

            if result == 1:
                wins += 1
            elif result == 2:
                losses += 1
            else:
                draws += 1

            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            if episode % self.target_sync_interval == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

        return TrainingStats(
            episodes=episodes,
            wins=wins,
            losses=losses,
            draws=draws,
            updates=updates,
        )
