import torch

from connectx_rl.board_utils import drop_piece, empty_board
from connectx_rl.minimax_dqn_agent import MinimaxDQNAgent, dqn_leaf_score, hybrid_leaf_evaluate


def board_from_moves(moves, rows=6, columns=7):
    board = empty_board(rows=rows, columns=columns)
    mark = 1
    for column in moves:
        board = drop_piece(board, column, mark)
        mark = 2 if mark == 1 else 1
    return board


class FixedQNet(torch.nn.Module):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.values = torch.tensor([values], dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.values.repeat(x.shape[0], 1)


def test_dqn_leaf_score_uses_best_legal_q_value():
    board = empty_board()
    for mark in [1, 2, 1, 2, 1, 2]:
        board = drop_piece(board, 3, mark)

    score = dqn_leaf_score(board, root_mark=1, q_network=FixedQNet([1, 2, 3, 100, 4, 5, 6]))

    assert score == 6.0


def test_minimax_dqn_chooses_immediate_win():
    board = board_from_moves([0, 6, 1, 6, 2, 6])
    agent = MinimaxDQNAgent(q_network=FixedQNet([0, 0, 0, -10, 0, 0, 0]), depth=2)

    action = agent.choose_action(board, mark=1)

    assert action == 3


def test_minimax_dqn_blocks_immediate_loss():
    board = board_from_moves([6, 0, 6, 1, 5, 2])
    agent = MinimaxDQNAgent(q_network=FixedQNet([10, 9, 8, -99, 7, 6, 5]), depth=2)

    action = agent.choose_action(board, mark=1)

    assert action == 3


def test_minimax_dqn_blocks_immediate_loss_even_at_depth_one():
    board = board_from_moves([6, 0, 6, 1, 5, 2])
    agent = MinimaxDQNAgent(q_network=FixedQNet([10, 9, 8, -99, 7, 6, 5]), depth=1)

    action = agent.choose_action(board, mark=1)

    assert action == 3


def test_hybrid_leaf_evaluate_prefers_center_control_when_q_values_are_neutral():
    center_board = board_from_moves([3, 6])
    edge_board = board_from_moves([0, 6])

    center_score = hybrid_leaf_evaluate(center_board, root_mark=1, q_network=FixedQNet([0, 0, 0, 0, 0, 0, 0]))
    edge_score = hybrid_leaf_evaluate(edge_board, root_mark=1, q_network=FixedQNet([0, 0, 0, 0, 0, 0, 0]))

    assert center_score > edge_score


def test_hybrid_leaf_evaluate_penalizes_opponent_three_in_a_row_when_q_values_are_neutral():
    quiet_board = board_from_moves([6, 5, 6, 4])
    threatened_board = board_from_moves([6, 0, 6, 1, 5, 2])

    quiet_score = hybrid_leaf_evaluate(quiet_board, root_mark=1, q_network=FixedQNet([0, 0, 0, 0, 0, 0, 0]))
    threatened_score = hybrid_leaf_evaluate(
        threatened_board,
        root_mark=1,
        q_network=FixedQNet([0, 0, 0, 0, 0, 0, 0]),
    )

    assert threatened_score < quiet_score


def test_minimax_dqn_iterative_deepening_matches_deeper_search_when_time_allows():
    board = board_from_moves([6, 3, 6, 3, 0, 2])
    agent = MinimaxDQNAgent(
        q_network=FixedQNet([0, 0, 0, 0, 0, 0, 0]),
        depth=3,
        time_limit_s=1.0,
        time_source=lambda: 0.0,
    )

    action = agent.choose_action(board, mark=1)

    assert action == 4


def test_minimax_dqn_iterative_deepening_returns_legal_move_on_immediate_timeout():
    board = board_from_moves([6, 3, 6, 3, 0, 2])
    agent = MinimaxDQNAgent(
        q_network=FixedQNet([0, 0, 0, 0, 0, 0, 0]),
        depth=3,
        time_limit_s=0.0,
        time_source=lambda: 1.0,
    )

    action = agent.choose_action(board, mark=1)

    assert action in range(7)
