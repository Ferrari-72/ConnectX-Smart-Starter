from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Protocol

from connectx_rl.board_utils import (
    Board,
    board_to_string,
    drop_piece,
    empty_board,
    is_full,
    valid_moves,
    winner,
)
from connectx_rl.minimax_agent import choose_minimax_action


class AgentProtocol(Protocol):
    def choose_action(self, board: Board, mark: int) -> int: ...


@dataclass
class RandomAgent:
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = Random(self.seed)

    def choose_action(self, board: Board, mark: int) -> int:
        del mark
        moves = valid_moves(board)
        return self._rng.choice(moves)


@dataclass
class MinimaxAgent:
    depth: int = 3

    def choose_action(self, board: Board, mark: int) -> int:
        return choose_minimax_action(board, my_mark=mark, depth=self.depth)


def play_game(
    player_one: AgentProtocol,
    player_two: AgentProtocol,
    rows: int = 6,
    columns: int = 7,
    inarow: int = 4,
) -> tuple[int, list[Board]]:
    board = empty_board(rows=rows, columns=columns)
    history = [board]
    current_mark = 1
    players = {1: player_one, 2: player_two}

    while True:
        agent = players[current_mark]
        action = agent.choose_action(board, current_mark)
        if action not in valid_moves(board):
            raise ValueError(f"Agent {current_mark} selected illegal move {action}")

        board = drop_piece(board, action, current_mark)
        history.append(board)

        game_winner = winner(board, n=inarow)
        if game_winner != 0:
            return game_winner, history
        if is_full(board):
            return 0, history

        current_mark = 2 if current_mark == 1 else 1


def format_history(history: list[Board]) -> str:
    frames = []
    for index, board in enumerate(history):
        frames.append(f"Turn {index}\n{board_to_string(board)}")
    return "\n\n".join(frames)
