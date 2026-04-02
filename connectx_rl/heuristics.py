from __future__ import annotations

from connectx_rl.board_utils import Board, board_dimensions, has_connect_n, iter_windows, opponent_mark


WIN_SCORE = 1_000_000


def _evaluate_window(window: tuple[int, ...], my_mark: int, opp_mark: int) -> int:
    score = 0
    my_count = window.count(my_mark)
    opp_count = window.count(opp_mark)
    empty_count = window.count(0)

    if my_count == 4:
        score += 10_000
    elif my_count == 3 and empty_count == 1:
        score += 100
    elif my_count == 2 and empty_count == 2:
        score += 10

    if opp_count == 3 and empty_count == 1:
        score -= 120
    elif opp_count == 2 and empty_count == 2:
        score -= 12

    return score


def evaluate_board(board: Board, my_mark: int) -> int:
    opp_mark = opponent_mark(my_mark)

    if has_connect_n(board, my_mark):
        return WIN_SCORE
    if has_connect_n(board, opp_mark):
        return -WIN_SCORE

    rows, columns = board_dimensions(board)
    center_column = columns // 2
    center_count = sum(1 for row in range(rows) if board[row][center_column] == my_mark)
    score = center_count * 6

    for window in iter_windows(board, window_size=4):
        score += _evaluate_window(window, my_mark, opp_mark)

    return score
