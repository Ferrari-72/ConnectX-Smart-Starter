import torch

from connectx_rl.board_utils import drop_piece, empty_board
from connectx_rl.minimax_dqn_agent import MinimaxDQNAgent, dqn_leaf_score


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
