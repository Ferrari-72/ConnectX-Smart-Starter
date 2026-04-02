from __future__ import annotations

from typing import Iterable

Board = tuple[tuple[int, ...], ...]


def empty_board(rows: int = 6, columns: int = 7) -> Board:
    return tuple(tuple(0 for _ in range(columns)) for _ in range(rows))


def board_from_flat(flat_board: list[int] | tuple[int, ...], rows: int, columns: int) -> Board:
    if len(flat_board) != rows * columns:
        raise ValueError("Flat board size does not match rows * columns")
    return tuple(
        tuple(flat_board[row * columns + column] for column in range(columns))
        for row in range(rows)
    )


def board_dimensions(board: Board) -> tuple[int, int]:
    return len(board), len(board[0])


def valid_moves(board: Board) -> list[int]:
    rows, columns = board_dimensions(board)
    return [column for column in range(columns) if board[0][column] == 0 and rows > 0]


def is_full(board: Board) -> bool:
    return not valid_moves(board)


def drop_piece(board: Board, column: int, mark: int) -> Board:
    rows, columns = board_dimensions(board)
    if column < 0 or column >= columns:
        raise ValueError(f"Column {column} is out of range")
    if board[0][column] != 0:
        raise ValueError(f"Column {column} is full")

    mutable = [list(row) for row in board]
    for row in range(rows - 1, -1, -1):
        if mutable[row][column] == 0:
            mutable[row][column] = mark
            return tuple(tuple(cell for cell in board_row) for board_row in mutable)

    raise ValueError(f"Column {column} is full")


def opponent_mark(mark: int) -> int:
    if mark not in (1, 2):
        raise ValueError("ConnectX marks must be 1 or 2")
    return 2 if mark == 1 else 1


def iter_windows(board: Board, window_size: int = 4) -> Iterable[tuple[int, ...]]:
    rows, columns = board_dimensions(board)

    for row in range(rows):
        for start_column in range(columns - window_size + 1):
            yield tuple(board[row][start_column + offset] for offset in range(window_size))

    for column in range(columns):
        for start_row in range(rows - window_size + 1):
            yield tuple(board[start_row + offset][column] for offset in range(window_size))

    for start_row in range(rows - window_size + 1):
        for start_column in range(columns - window_size + 1):
            yield tuple(board[start_row + offset][start_column + offset] for offset in range(window_size))

    for start_row in range(window_size - 1, rows):
        for start_column in range(columns - window_size + 1):
            yield tuple(board[start_row - offset][start_column + offset] for offset in range(window_size))


def has_connect_n(board: Board, mark: int, n: int = 4) -> bool:
    return any(window.count(mark) == n for window in iter_windows(board, window_size=n))


def winner(board: Board, n: int = 4) -> int:
    for mark in (1, 2):
        if has_connect_n(board, mark, n=n):
            return mark
    return 0


def board_to_string(board: Board) -> str:
    token_map = {0: ".", 1: "X", 2: "O"}
    lines = [" ".join(token_map[cell] for cell in row) for row in board]
    footer = " ".join(str(index) for index in range(len(board[0])))
    return "\n".join([*lines, footer])
