from __future__ import annotations

from pathlib import Path

from connectx_rl.checkpoints import load_checkpoint


def _to_python_list(value) -> list:
    return value.detach().cpu().tolist()


def _render_model_constants(checkpoint: dict) -> tuple[list, list]:
    state_dict = checkpoint["q_network"]
    weights = [
        _to_python_list(state_dict["layers.1.weight"]),
        _to_python_list(state_dict["layers.3.weight"]),
        _to_python_list(state_dict["layers.5.weight"]),
    ]
    biases = [
        _to_python_list(state_dict["layers.1.bias"]),
        _to_python_list(state_dict["layers.3.bias"]),
        _to_python_list(state_dict["layers.5.bias"]),
    ]
    return weights, biases


def build_hybrid_submission_source(
    checkpoint_path: str | Path,
    depth: int = 2,
    time_limit_s: float | None = None,
) -> str:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    weights, biases = _render_model_constants(checkpoint)

    return f'''from __future__ import annotations

import time

MODEL_WEIGHTS = {weights!r}
MODEL_BIASES = {biases!r}
SEARCH_DEPTH = {depth}
TIME_LIMIT_SECONDS = {time_limit_s!r}
WIN_SCORE = 1_000_000.0


def _get_attr_or_key(obj, name):
    if hasattr(obj, name):
        return getattr(obj, name)
    return obj[name]


def _board_from_flat(flat_board, rows, columns):
    return tuple(
        tuple(flat_board[row * columns + column] for column in range(columns))
        for row in range(rows)
    )


def _valid_moves(board):
    return [column for column in range(len(board[0])) if board[0][column] == 0]


def _drop_piece(board, column, mark):
    mutable = [list(row) for row in board]
    for row in range(len(board) - 1, -1, -1):
        if mutable[row][column] == 0:
            mutable[row][column] = mark
            return tuple(tuple(cell for cell in board_row) for board_row in mutable)
    raise ValueError(f"Column {{column}} is full")


def _opponent_mark(mark):
    return 2 if mark == 1 else 1


def _choose_preferred_action(board, actions):
    center = len(board[0]) // 2
    return min(actions, key=lambda action: (abs(action - center), action))


def _iter_windows(board, window_size=4):
    rows = len(board)
    columns = len(board[0])
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


def _has_connect_n(board, mark, n=4):
    return any(window.count(mark) == n for window in _iter_windows(board, window_size=n))


def _is_full(board):
    return not _valid_moves(board)


def _evaluate_window(window, my_mark, opp_mark):
    score = 0.0
    my_count = window.count(my_mark)
    opp_count = window.count(opp_mark)
    empty_count = window.count(0)

    if my_count == 4:
        score += 10_000.0
    elif my_count == 3 and empty_count == 1:
        score += 100.0
    elif my_count == 2 and empty_count == 2:
        score += 10.0

    if opp_count == 3 and empty_count == 1:
        score -= 120.0
    elif opp_count == 2 and empty_count == 2:
        score -= 12.0

    return score


def _evaluate_board(board, my_mark):
    opp_mark = _opponent_mark(my_mark)
    if _has_connect_n(board, my_mark):
        return WIN_SCORE
    if _has_connect_n(board, opp_mark):
        return -WIN_SCORE

    rows = len(board)
    columns = len(board[0])
    center_column = columns // 2
    center_count = sum(1 for row in range(rows) if board[row][center_column] == my_mark)
    score = center_count * 6.0

    for window in _iter_windows(board, window_size=4):
        score += _evaluate_window(window, my_mark, opp_mark)

    return score


def _immediate_winning_moves(board, mark):
    wins = []
    for action in _valid_moves(board):
        if _has_connect_n(_drop_piece(board, action, mark), mark):
            wins.append(action)
    return wins


def _forced_tactical_action(board, mark):
    winning_moves = _immediate_winning_moves(board, mark)
    if winning_moves:
        return _choose_preferred_action(board, winning_moves)
    opponent_wins = _immediate_winning_moves(board, _opponent_mark(mark))
    if len(opponent_wins) == 1:
        return opponent_wins[0]
    return None


def _encode_board(board, my_mark):
    opp_mark = _opponent_mark(my_mark)
    encoded = []
    for target_mark in (my_mark, opp_mark):
        for row in board:
            for cell in row:
                encoded.append(1.0 if cell == target_mark else 0.0)
    return encoded


def _linear(inputs, weight, bias):
    outputs = []
    for row, row_bias in zip(weight, bias):
        total = row_bias
        for value, coefficient in zip(inputs, row):
            total += value * coefficient
        outputs.append(total)
    return outputs


def _relu(values):
    return [value if value > 0.0 else 0.0 for value in values]


def _q_values(board, root_mark):
    x = _encode_board(board, root_mark)
    x = _relu(_linear(x, MODEL_WEIGHTS[0], MODEL_BIASES[0]))
    x = _relu(_linear(x, MODEL_WEIGHTS[1], MODEL_BIASES[1]))
    return _linear(x, MODEL_WEIGHTS[2], MODEL_BIASES[2])


def _dqn_leaf_score(board, root_mark):
    legal_moves = _valid_moves(board)
    if not legal_moves:
        return 0.0
    q_values = _q_values(board, root_mark)
    return max(q_values[action] for action in legal_moves)


def _hybrid_leaf_evaluate(board, root_mark):
    opp_mark = _opponent_mark(root_mark)
    if _has_connect_n(board, root_mark):
        return WIN_SCORE
    if _has_connect_n(board, opp_mark):
        return -WIN_SCORE
    if _is_full(board):
        return 0.0
    return _evaluate_board(board, root_mark) + _dqn_leaf_score(board, root_mark)


def _ordered_moves(board, current_mark, root_mark):
    scored_moves = []
    current_wins = set(_immediate_winning_moves(board, current_mark))
    opponent_wins = set(_immediate_winning_moves(board, _opponent_mark(current_mark)))
    center = len(board[0]) // 2
    for action in _valid_moves(board):
        child = _drop_piece(board, action, current_mark)
        opponent_reply_wins = _immediate_winning_moves(child, _opponent_mark(current_mark))
        next_turn_wins = _immediate_winning_moves(child, current_mark)
        leaf_score = _hybrid_leaf_evaluate(child, root_mark)
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


def _is_terminal(board):
    return _has_connect_n(board, 1) or _has_connect_n(board, 2) or _is_full(board)


class _SearchTimeout(RuntimeError):
    pass


def _probe_transposition(cache, key, depth, alpha, beta):
    entry = cache.get(key)
    if entry is None or entry["depth"] < depth:
        return alpha, beta, None
    flag = entry["flag"]
    score = entry["score"]
    action = entry["action"]
    if flag == "exact":
        return alpha, beta, (score, action)
    if flag == "lower":
        alpha = max(alpha, score)
    else:
        beta = min(beta, score)
    if alpha >= beta:
        return alpha, beta, (score, action)
    return alpha, beta, None


def _store_transposition(cache, key, depth, score, action, original_alpha, original_beta):
    if score <= original_alpha:
        flag = "upper"
    elif score >= original_beta:
        flag = "lower"
    else:
        flag = "exact"
    cache[key] = {{"depth": depth, "score": score, "action": action, "flag": flag}}


def _minimax_dqn(board, depth, alpha, beta, current_mark, root_mark, cache, deadline=None):
    if deadline is not None and time.perf_counter() >= deadline:
        raise _SearchTimeout
    if depth == 0 or _is_terminal(board):
        return _hybrid_leaf_evaluate(board, root_mark), None

    original_alpha = alpha
    original_beta = beta
    cache_key = (board, depth, current_mark, root_mark)
    alpha, beta, cached = _probe_transposition(cache, cache_key, depth, alpha, beta)
    if cached is not None:
        return cached

    maximizing = current_mark == root_mark
    best_action = None
    ordered_moves = _ordered_moves(board, current_mark, root_mark)

    if maximizing:
        value = float("-inf")
        for action in ordered_moves:
            child = _drop_piece(board, action, current_mark)
            child_value, _ = _minimax_dqn(
                child,
                depth - 1,
                alpha,
                beta,
                _opponent_mark(current_mark),
                root_mark,
                cache,
                deadline,
            )
            if child_value > value:
                value = child_value
                best_action = action
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        _store_transposition(cache, cache_key, depth, value, best_action, original_alpha, original_beta)
        return value, best_action

    value = float("inf")
    for action in ordered_moves:
        child = _drop_piece(board, action, current_mark)
        child_value, _ = _minimax_dqn(
            child,
            depth - 1,
            alpha,
            beta,
            _opponent_mark(current_mark),
            root_mark,
            cache,
            deadline,
        )
        if child_value < value:
            value = child_value
            best_action = action
        beta = min(beta, value)
        if alpha >= beta:
            break
    _store_transposition(cache, cache_key, depth, value, best_action, original_alpha, original_beta)
    return value, best_action


def agent(observation, configuration):
    rows = _get_attr_or_key(configuration, "rows")
    columns = _get_attr_or_key(configuration, "columns")
    mark = _get_attr_or_key(observation, "mark")
    flat_board = _get_attr_or_key(observation, "board")
    board = _board_from_flat(flat_board, rows=rows, columns=columns)
    tactical_action = _forced_tactical_action(board, mark)
    if tactical_action is not None:
        return tactical_action
    if TIME_LIMIT_SECONDS is None:
        _score, action = _minimax_dqn(
            board,
            SEARCH_DEPTH,
            float("-inf"),
            float("inf"),
            mark,
            mark,
            {{}},
        )
    else:
        action = _choose_preferred_action(board, _valid_moves(board))
        deadline = time.perf_counter() + max(0.0, TIME_LIMIT_SECONDS)
        for current_depth in range(1, SEARCH_DEPTH + 1):
            try:
                _score, candidate_action = _minimax_dqn(
                    board,
                    current_depth,
                    float("-inf"),
                    float("inf"),
                    mark,
                    mark,
                    {{}},
                    deadline,
                )
            except _SearchTimeout:
                break
            if candidate_action is not None:
                action = candidate_action
    if action is None:
        raise ValueError("Cannot choose an action from a terminal position")
    return action
'''


def export_hybrid_submission(
    checkpoint_path: str | Path,
    output_path: str | Path,
    depth: int = 2,
    time_limit_s: float | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_hybrid_submission_source(checkpoint_path, depth=depth, time_limit_s=time_limit_s),
        encoding="utf-8",
    )
    return output_path
