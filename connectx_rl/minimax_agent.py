from __future__ import annotations

from math import inf

from connectx_rl.board_utils import Board, drop_piece, has_connect_n, is_full, opponent_mark, valid_moves
from connectx_rl.heuristics import WIN_SCORE, evaluate_board
from connectx_rl.search_tactics import (
    TranspositionKey,
    forced_tactical_action,
    probe_transposition,
    store_transposition,
    tactical_move_order,
)


def _ordered_moves(board: Board, current_mark: int, root_mark: int) -> list[int]:
    return tactical_move_order(
        board,
        current_mark=current_mark,
        root_mark=root_mark,
        leaf_evaluator=lambda child: float(evaluate_board(child, my_mark=root_mark)),
    )


def _is_terminal(board: Board) -> bool:
    return has_connect_n(board, 1) or has_connect_n(board, 2) or is_full(board)


def _minimax(
    board: Board,
    depth: int,
    alpha: float,
    beta: float,
    current_mark: int,
    root_mark: int,
    cache: dict[TranspositionKey, object],
) -> tuple[float, int | None]:
    if depth == 0 or _is_terminal(board):
        return evaluate_board(board, my_mark=root_mark), None

    original_alpha = alpha
    original_beta = beta
    cache_key = (board, depth, current_mark, root_mark)
    alpha, beta, cached = probe_transposition(cache, cache_key, depth, alpha, beta)
    if cached is not None:
        return cached

    maximizing = current_mark == root_mark
    best_action = None
    ordered_moves = _ordered_moves(board, current_mark=current_mark, root_mark=root_mark)

    if maximizing:
        value = -inf
        for action in ordered_moves:
            child = drop_piece(board, action, current_mark)
            child_value, _ = _minimax(
                child,
                depth=depth - 1,
                alpha=alpha,
                beta=beta,
                current_mark=opponent_mark(current_mark),
                root_mark=root_mark,
                cache=cache,
            )
            if child_value > value:
                value = child_value
                best_action = action
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        store_transposition(cache, cache_key, depth, value, best_action, original_alpha, original_beta)
        return value, best_action

    value = inf
    for action in ordered_moves:
        child = drop_piece(board, action, current_mark)
        child_value, _ = _minimax(
            child,
            depth=depth - 1,
            alpha=alpha,
            beta=beta,
            current_mark=opponent_mark(current_mark),
            root_mark=root_mark,
            cache=cache,
        )
        if child_value < value:
            value = child_value
            best_action = action
        beta = min(beta, value)
        if alpha >= beta:
            break
    store_transposition(cache, cache_key, depth, value, best_action, original_alpha, original_beta)
    return value, best_action


def choose_minimax_action(board: Board, my_mark: int, depth: int = 3) -> int:
    if not valid_moves(board):
        raise ValueError("No legal moves available")

    tactical_action = forced_tactical_action(board, my_mark)
    if tactical_action is not None:
        return tactical_action

    score, action = _minimax(
        board,
        depth=depth,
        alpha=-inf,
        beta=inf,
        current_mark=my_mark,
        root_mark=my_mark,
        cache={},
    )
    if action is None:
        # This happens only when the search starts on a terminal board.
        raise ValueError(f"Cannot choose an action from terminal position (score={score})")
    return action
