import torch

from connectx_rl.board_utils import drop_piece, empty_board
from connectx_rl.dqn_agent import DQNAgent, encode_board
from connectx_rl.replay_buffer import ReplayBuffer, Transition


def test_encode_board_splits_current_and_opponent_planes():
    board = empty_board()
    board = drop_piece(board, 0, 1)
    board = drop_piece(board, 1, 2)

    encoded = encode_board(board, my_mark=1)

    assert encoded.shape == (2, 6, 7)
    assert encoded[0, 5, 0].item() == 1.0
    assert encoded[1, 5, 1].item() == 1.0
    assert encoded[:, 5, 2].sum().item() == 0.0


class FixedQNet(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        values = torch.tensor([[1.0, 2.0, 3.0, 100.0, 4.0, 5.0, 6.0]])
        return values.repeat(batch_size, 1)


def test_dqn_agent_masks_illegal_actions():
    board = empty_board()
    for mark in [1, 2, 1, 2, 1, 2]:
        board = drop_piece(board, 3, mark)

    agent = DQNAgent(q_network=FixedQNet(), epsilon=0.0)
    action = agent.choose_action(board, mark=1)

    assert action != 3
    assert action == 6


def test_replay_buffer_samples_stacked_tensors():
    buffer = ReplayBuffer(capacity=10)
    state = torch.zeros((2, 6, 7), dtype=torch.float32)

    for action in range(4):
        buffer.push(
            Transition(
                state=state + action,
                action=action,
                reward=float(action),
                next_state=state + action + 1,
                done=action % 2 == 0,
            )
        )

    batch = buffer.sample(batch_size=3)

    assert batch.states.shape == (3, 2, 6, 7)
    assert batch.next_states.shape == (3, 2, 6, 7)
    assert batch.actions.shape == (3,)
    assert batch.rewards.shape == (3,)
    assert batch.dones.shape == (3,)
