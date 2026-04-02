from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from connectx_rl.board_utils import board_from_flat
from connectx_rl.minimax_agent import choose_minimax_action


@dataclass(frozen=True)
class KaggleObservation:
    board: list[int]
    mark: int


@dataclass(frozen=True)
class KaggleConfiguration:
    rows: int
    columns: int
    inarow: int


def _get_attr_or_key(obj: Any, name: str) -> Any:
    if hasattr(obj, name):
        return getattr(obj, name)
    return obj[name]


def agent(observation: Any, configuration: Any) -> int:
    rows = _get_attr_or_key(configuration, "rows")
    columns = _get_attr_or_key(configuration, "columns")
    mark = _get_attr_or_key(observation, "mark")
    flat_board = _get_attr_or_key(observation, "board")

    board = board_from_flat(flat_board, rows=rows, columns=columns)
    return choose_minimax_action(board, my_mark=mark, depth=3)
