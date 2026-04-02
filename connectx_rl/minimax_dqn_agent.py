from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import inf
from time import perf_counter

import torch

from connectx_rl.board_utils import Board, drop_piece, has_connect_n, is_full, opponent_mark, valid_moves
from connectx_rl.dqn_agent import encode_board
from connectx_rl.heuristics import WIN_SCORE, evaluate_board
from connectx_rl.runtime import resolve_module_device
from connectx_rl.search_tactics import (
    TranspositionKey,
    choose_preferred_action,
    forced_tactical_action,
    probe_transposition,
    store_transposition,
    tactical_move_order,
)


def dqn_leaf_score(board: Board, root_mark: int, q_network: torch.nn.Module, device: str = "auto") -> float:
    legal_moves = valid_moves(board)
    if not legal_moves:
        return 0.0

    device = resolve_module_device(q_network, device)
    state = encode_board(board, my_mark=root_mark).unsqueeze(0).to(device)
    q_network.eval()
    with torch.no_grad():
        q_values = q_network(state).squeeze(0).detach().cpu()

    masked_q_values = torch.full_like(q_values, float("-inf"))
    masked_q_values[legal_moves] = q_values[legal_moves]
    return float(torch.max(masked_q_values).item())


def hybrid_leaf_evaluate(board: Board, root_mark: int, q_network: torch.nn.Module, device: str = "auto") -> float:
    opp_mark = opponent_mark(root_mark)

    if has_connect_n(board, root_mark):
        return float(WIN_SCORE)
    if has_connect_n(board, opp_mark):
        return float(-WIN_SCORE)
    if is_full(board):
        return 0.0
    heuristic_score = float(evaluate_board(board, my_mark=root_mark))
    learned_score = dqn_leaf_score(board, root_mark=root_mark, q_network=q_network, device=device)
    return heuristic_score + learned_score


def _ordered_moves(
    board: Board,
    current_mark: int,
    root_mark: int,
    q_network: torch.nn.Module,
    device: str,
) -> list[int]:
    return tactical_move_order(
        board,
        current_mark=current_mark,
        root_mark=root_mark,
        leaf_evaluator=lambda child: hybrid_leaf_evaluate(
            child,
            root_mark=root_mark,
            q_network=q_network,
            device=device,
        ),
    )


def _is_terminal(board: Board) -> bool:
    return has_connect_n(board, 1) or has_connect_n(board, 2) or is_full(board)


class SearchTimeout(RuntimeError):
    pass


def _minimax_dqn(
    board: Board,
    depth: int,
    alpha: float,
    beta: float,
    current_mark: int,
    root_mark: int,
    q_network: torch.nn.Module,
    device: str,
    cache: dict[TranspositionKey, object],
    deadline: float | None = None,
    time_source: Callable[[], float] | None = None,
) -> tuple[float, int | None]:
    if deadline is not None and time_source is not None and time_source() >= deadline:
        raise SearchTimeout
    if depth == 0 or _is_terminal(board):
        return hybrid_leaf_evaluate(board, root_mark=root_mark, q_network=q_network, device=device), None

    original_alpha = alpha
    original_beta = beta
    cache_key = (board, depth, current_mark, root_mark)
    alpha, beta, cached = probe_transposition(cache, cache_key, depth, alpha, beta)
    if cached is not None:
        return cached

    maximizing = current_mark == root_mark
    best_action = None
    ordered_moves = _ordered_moves(
        board,
        current_mark=current_mark,
        root_mark=root_mark,
        q_network=q_network,
        device=device,
    )

    if maximizing:
        value = -inf
        for action in ordered_moves:
            child = drop_piece(board, action, current_mark)
            child_value, _ = _minimax_dqn(
                child,
                depth=depth - 1,
                alpha=alpha,
                beta=beta,
                current_mark=opponent_mark(current_mark),
                root_mark=root_mark,
                q_network=q_network,
                device=device,
                cache=cache,
                deadline=deadline,
                time_source=time_source,
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
        child_value, _ = _minimax_dqn(
            child,
            depth=depth - 1,
            alpha=alpha,
            beta=beta,
            current_mark=opponent_mark(current_mark),
            root_mark=root_mark,
            q_network=q_network,
            device=device,
            cache=cache,
            deadline=deadline,
            time_source=time_source,
        )
        if child_value < value:
            value = child_value
            best_action = action
        beta = min(beta, value)
        if alpha >= beta:
            break
    store_transposition(cache, cache_key, depth, value, best_action, original_alpha, original_beta)
    return value, best_action


@dataclass
class MinimaxDQNAgent:
    q_network: torch.nn.Module
    depth: int = 2
    device: str = "auto"
    time_limit_s: float | None = None
    time_source: Callable[[], float] | None = None

    def __post_init__(self) -> None:
        self.device = resolve_module_device(self.q_network, self.device)
        if self.time_source is None:
            self.time_source = perf_counter

    def choose_action(self, board: Board, mark: int) -> int:
        legal_moves = valid_moves(board)
        if not legal_moves:
            raise ValueError("No legal moves available")

        tactical_action = forced_tactical_action(board, mark)
        if tactical_action is not None:
            return tactical_action

        action: int | None = None
        if self.time_limit_s is None:
            _score, action = _minimax_dqn(
                board,
                depth=self.depth,
                alpha=-inf,
                beta=inf,
                current_mark=mark,
                root_mark=mark,
                q_network=self.q_network,
                device=self.device,
                cache={},
                deadline=None,
                time_source=self.time_source,
            )
        else:
            deadline = self.time_source() + max(0.0, self.time_limit_s)
            best_action = choose_preferred_action(board, legal_moves)
            for current_depth in range(1, self.depth + 1):
                try:
                    _score, candidate_action = _minimax_dqn(
                        board,
                        depth=current_depth,
                        alpha=-inf,
                        beta=inf,
                        current_mark=mark,
                        root_mark=mark,
                        q_network=self.q_network,
                        device=self.device,
                        cache={},
                        deadline=deadline,
                        time_source=self.time_source,
                    )
                except SearchTimeout:
                    break
                if candidate_action is not None:
                    best_action = candidate_action
            action = best_action

        if action is None:
            raise ValueError("Cannot choose an action from a terminal position")
        return action
