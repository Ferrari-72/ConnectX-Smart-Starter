from pathlib import Path

import torch

from connectx_rl.board_utils import board_from_flat, drop_piece, empty_board
from connectx_rl.kaggle_export import build_hybrid_submission_source, export_hybrid_submission
from connectx_rl.minimax_dqn_agent import MinimaxDQNAgent
from connectx_rl.q_network import QNetwork


def _write_checkpoint(path: Path, final_bias: list[float]) -> None:
    q_network = QNetwork()
    state_dict = q_network.state_dict()
    for name, tensor in state_dict.items():
        state_dict[name] = torch.zeros_like(tensor)
    state_dict["layers.5.bias"] = torch.tensor(final_bias, dtype=torch.float32)
    torch.save({"q_network": state_dict}, path)


def _load_exported_agent(source: str):
    namespace: dict[str, object] = {}
    exec(source, namespace)
    return namespace["agent"]


def test_generated_submission_matches_hybrid_agent_choice(tmp_path: Path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    _write_checkpoint(checkpoint_path, final_bias=[0.0, 0.0, 0.0, 5.0, 1.0, 2.0, 3.0])

    source = build_hybrid_submission_source(checkpoint_path, depth=1)
    exported_agent = _load_exported_agent(source)

    q_network = QNetwork()
    q_network.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["q_network"])
    project_agent = MinimaxDQNAgent(q_network=q_network, depth=1, device="cpu")
    board = empty_board()

    expected = project_agent.choose_action(board, mark=1)
    actual = exported_agent(
        {"board": [cell for row in board for cell in row], "mark": 1},
        {"rows": 6, "columns": 7, "inarow": 4},
    )

    assert actual == expected


def test_generated_submission_masks_illegal_moves(tmp_path: Path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    _write_checkpoint(checkpoint_path, final_bias=[1.0, 2.0, 3.0, 100.0, 4.0, 5.0, 6.0])

    source = build_hybrid_submission_source(checkpoint_path, depth=1)
    exported_agent = _load_exported_agent(source)
    q_network = QNetwork()
    q_network.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["q_network"])
    project_agent = MinimaxDQNAgent(q_network=q_network, depth=1, device="cpu")

    board = empty_board()
    for mark in [1, 2, 1, 2, 1, 2]:
        board = drop_piece(board, 3, mark)

    expected = project_agent.choose_action(board, mark=1)
    actual = exported_agent(
        {"board": [cell for row in board for cell in row], "mark": 1},
        {"rows": 6, "columns": 7, "inarow": 4},
    )

    assert actual == expected
    assert actual != 3


def test_export_hybrid_submission_writes_agent_file(tmp_path: Path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    output_path = tmp_path / "submission.py"
    _write_checkpoint(checkpoint_path, final_bias=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    export_hybrid_submission(checkpoint_path, output_path, depth=2)

    contents = output_path.read_text(encoding="utf-8")
    assert "def agent(" in contents
    assert "MODEL_WEIGHTS" in contents
