from connectx_rl.board_utils import drop_piece, empty_board
from connectx_rl.heuristics import evaluate_board
from connectx_rl.kaggle_agent import agent as kaggle_agent
from connectx_rl.local_game import RandomAgent, play_game
from connectx_rl.minimax_agent import choose_minimax_action


def board_from_moves(moves, rows=6, columns=7):
    board = empty_board(rows=rows, columns=columns)
    mark = 1
    for column in moves:
        board = drop_piece(board, column, mark)
        mark = 2 if mark == 1 else 1
    return board


def test_evaluate_board_prefers_center_control():
    center_board = empty_board()
    center_board = drop_piece(center_board, 3, 1)

    edge_board = empty_board()
    edge_board = drop_piece(edge_board, 0, 1)

    assert evaluate_board(center_board, my_mark=1) > evaluate_board(edge_board, my_mark=1)


def test_minimax_chooses_immediate_winning_move():
    board = board_from_moves([0, 6, 1, 6, 2, 6])

    action = choose_minimax_action(board, my_mark=1, depth=3)

    assert action == 3


def test_minimax_blocks_immediate_opponent_win():
    board = board_from_moves([6, 0, 6, 1, 5, 2])

    action = choose_minimax_action(board, my_mark=1, depth=3)

    assert action == 3


def test_kaggle_agent_chooses_immediate_win():
    board = board_from_moves([0, 6, 1, 6, 2, 6])
    observation = {"board": [cell for row in board for cell in row], "mark": 1}
    configuration = {"rows": 6, "columns": 7, "inarow": 4}

    action = kaggle_agent(observation, configuration)

    assert action == 3


def test_local_game_returns_valid_winner():
    winner, history = play_game(RandomAgent(seed=1), RandomAgent(seed=2))

    assert winner in {0, 1, 2}
    assert len(history) > 0
