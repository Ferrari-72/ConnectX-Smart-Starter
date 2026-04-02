from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import torch


@dataclass(frozen=True)
class Transition:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool


@dataclass(frozen=True)
class TransitionBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buffer)

    def push(self, transition: Transition) -> None:
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> TransitionBatch:
        batch = random.sample(list(self._buffer), batch_size)
        return TransitionBatch(
            states=torch.stack([item.state for item in batch]),
            actions=torch.tensor([item.action for item in batch], dtype=torch.long),
            rewards=torch.tensor([item.reward for item in batch], dtype=torch.float32),
            next_states=torch.stack([item.next_state for item in batch]),
            dones=torch.tensor([item.done for item in batch], dtype=torch.float32),
        )
