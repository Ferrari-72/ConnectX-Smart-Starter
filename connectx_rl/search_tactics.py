from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from connectx_rl.board_utils import Board, drop_piece, has_connect_n, opponent_mark, valid_moves

LeafEvaluator = Callable[[Board], float]
TranspositionFlag = Literal["exact", "lower", "upper"]
TranspositionKey = tuple[Board, int, int, int]


def _center_first_key(board: Board, action: int) -> tuple[int, int]:
    center = len(board[0]) // 2
    return (abs(action - center), action)


def choose_preferred_action(board: Board, actions: list[int]) -> int:
    return min(actions, key=lambda action: _center_first_key(board, action))


def immediate_winning_moves(board: Board, mark: int) -> list[int]:
    wins = []
    for action in valid_moves(board):
        if has_connect_n(drop_piece(board, action, mark), mark):
            wins.append(action)
    return wins


def forced_tactical_action(board: Board, mark: int) -> int | None:
    winning_moves = immediate_winning_moves(board, mark)
    if winning_moves:
        return choose_preferred_action(board, winning_moves)

    opponent_wins = immediate_winning_moves(board, opponent_mark(mark))
    if len(opponent_wins) == 1:
        return opponent_wins[0]
    return None


def tactical_move_order(
    board: Board,
    current_mark: int,
    root_mark: int,
    leaf_evaluator: LeafEvaluator,
) -> list[int]:
    legal_moves = valid_moves(board)
    current_wins = set(immediate_winning_moves(board, current_mark))
    opponent_wins = set(immediate_winning_moves(board, opponent_mark(current_mark)))
    center = len(board[0]) // 2
    scored_moves = []

    for action in legal_moves:
        child = drop_piece(board, action, current_mark)
        opponent_reply_wins = immediate_winning_moves(child, opponent_mark(current_mark))
        next_turn_wins = immediate_winning_moves(child, current_mark)
        leaf_score = leaf_evaluator(child)
        score_for_current = leaf_score if current_mark == root_mark else -leaf_score

        scored_moves.append(
            (
                action,
                (
                    1 if action in current_wins else 0,
                    1 if action in opponent_wins else 0,
                    0 if opponent_reply_wins else 1,
                    len(next_turn_wins),
                    score_for_current,
                    -abs(action - center),
                    -action,
                ),
            )
        )

    scored_moves.sort(key=lambda item: item[1], reverse=True)
    return [action for action, _score in scored_moves]


@dataclass(frozen=True)
class TranspositionEntry:
    depth: int
    score: float
    action: int | None
    flag: TranspositionFlag


def probe_transposition(
    cache: dict[TranspositionKey, TranspositionEntry],
    key: TranspositionKey,
    depth: int,
    alpha: float,
    beta: float,
) -> tuple[float, float, tuple[float, int | None] | None]:
    entry = cache.get(key)
    if entry is None or entry.depth < depth:
        return alpha, beta, None

    if entry.flag == "exact":
        return alpha, beta, (entry.score, entry.action)
    if entry.flag == "lower":
        alpha = max(alpha, entry.score)
    else:
        beta = min(beta, entry.score)
    if alpha >= beta:
        return alpha, beta, (entry.score, entry.action)
    return alpha, beta, None


def store_transposition(
    cache: dict[TranspositionKey, TranspositionEntry],
    key: TranspositionKey,
    depth: int,
    score: float,
    action: int | None,
    original_alpha: float,
    original_beta: float,
) -> None:
    if score <= original_alpha:
        flag: TranspositionFlag = "upper"
    elif score >= original_beta:
        flag = "lower"
    else:
        flag = "exact"
    cache[key] = TranspositionEntry(depth=depth, score=score, action=action, flag=flag)
