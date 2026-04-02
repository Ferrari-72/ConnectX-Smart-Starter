from __future__ import annotations

from copy import deepcopy
from typing import Callable

from connectx_rl.local_game import AgentProtocol, play_game

OpponentFactory = Callable[[int], AgentProtocol]


def summarize_matchups(
    agent: AgentProtocol,
    opponents: dict[str, AgentProtocol],
    games_per_matchup: int = 10,
) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}

    for name, opponent in opponents.items():
        wins = losses = draws = 0
        for _ in range(games_per_matchup):
            winner, _history = play_game(agent, deepcopy(opponent))
            if winner == 1:
                wins += 1
            elif winner == 2:
                losses += 1
            else:
                draws += 1

        summary[name] = {
            "wins": wins,
            "losses": losses,
            "draws": draws,
        }

    return summary


def evaluate_agent(
    agent: AgentProtocol,
    opponents: list[tuple[str, OpponentFactory]],
    games_per_opponent: int = 10,
) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}

    for name, factory in opponents:
        wins = losses = draws = 0
        for seed in range(games_per_opponent):
            opponent = factory(seed)
            winner, _history = play_game(agent, opponent)
            if winner == 1:
                wins += 1
            elif winner == 2:
                losses += 1
            else:
                draws += 1

        summary[name] = {
            "games": games_per_opponent,
            "wins": wins,
            "losses": losses,
            "draws": draws,
        }

    return summary
