from connectx_rl.board_utils import drop_piece, empty_board
from connectx_rl.search_tactics import immediate_winning_moves, tactical_move_order


def board_from_moves(moves, rows=6, columns=7):
    board = empty_board(rows=rows, columns=columns)
    mark = 1
    for column in moves:
        board = drop_piece(board, column, mark)
        mark = 2 if mark == 1 else 1
    return board


def test_immediate_winning_moves_detects_single_winning_column():
    board = board_from_moves([0, 6, 1, 6, 2, 5])

    assert immediate_winning_moves(board, mark=1) == [3]


def test_tactical_move_order_prioritizes_forced_block():
    board = board_from_moves([6, 0, 6, 1, 5, 2])

    order = tactical_move_order(
        board,
        current_mark=1,
        root_mark=1,
        leaf_evaluator=lambda child: float(child[-1][0]),
    )

    assert order[0] == 3
