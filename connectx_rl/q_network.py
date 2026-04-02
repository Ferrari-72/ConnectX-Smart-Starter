from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, rows: int = 6, columns: int = 7, hidden_size: int = 128) -> None:
        super().__init__()
        input_size = 2 * rows * columns
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, columns),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
