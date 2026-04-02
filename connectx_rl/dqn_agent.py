from __future__ import annotations

from dataclasses import dataclass
from random import Random

import torch

from connectx_rl.board_utils import Board, opponent_mark, valid_moves
from connectx_rl.runtime import resolve_module_device


def encode_board(board: Board, my_mark: int) -> torch.Tensor:
    opp_mark = opponent_mark(my_mark)
    rows = len(board)
    columns = len(board[0])
    encoded = torch.zeros((2, rows, columns), dtype=torch.float32)

    for row in range(rows):
        for column in range(columns):
            if board[row][column] == my_mark:
                encoded[0, row, column] = 1.0
            elif board[row][column] == opp_mark:
                encoded[1, row, column] = 1.0

    return encoded


@dataclass
class DQNAgent:
    q_network: torch.nn.Module
    epsilon: float = 0.1
    seed: int | None = None
    device: str = "auto"

    def __post_init__(self) -> None:
        self._rng = Random(self.seed)
        self.device = resolve_module_device(self.q_network, self.device)
        self.q_network.eval()

    def choose_action(self, board: Board, mark: int) -> int:
        legal_moves = valid_moves(board)
        if not legal_moves:
            raise ValueError("No legal moves available")

        if self._rng.random() < self.epsilon:
            return self._rng.choice(legal_moves)

        state = encode_board(board, my_mark=mark).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state).squeeze(0).detach().cpu()

        masked_q_values = torch.full_like(q_values, float("-inf"))
        masked_q_values[legal_moves] = q_values[legal_moves]
        return int(torch.argmax(masked_q_values).item())
