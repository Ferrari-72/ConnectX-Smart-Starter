from __future__ import annotations

from dataclasses import dataclass
from math import inf

import torch

from connectx_rl.board_utils import Board, drop_piece, has_connect_n, is_full, opponent_mark, valid_moves
from connectx_rl.dqn_agent import encode_board
from connectx_rl.heuristics import WIN_SCORE
from connectx_rl.runtime import resolve_module_device
from connectx_rl.search_tactics import (
    TranspositionKey,
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
    return dqn_leaf_score(board, root_mark=root_mark, q_network=q_network, device=device)


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
) -> tuple[float, int | None]:
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

    def __post_init__(self) -> None:
        self.device = resolve_module_device(self.q_network, self.device)

    def choose_action(self, board: Board, mark: int) -> int:
        if not valid_moves(board):
            raise ValueError("No legal moves available")

        tactical_action = forced_tactical_action(board, mark)
        if tactical_action is not None:
            return tactical_action

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
        )
        if action is None:
            raise ValueError("Cannot choose an action from a terminal position")
        return action
