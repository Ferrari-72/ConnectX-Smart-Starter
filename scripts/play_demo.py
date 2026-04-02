from __future__ import annotations

from connectx_rl.local_game import MinimaxAgent, RandomAgent, format_history, play_game


def main() -> None:
    minimax = MinimaxAgent(depth=3)
    random_agent = RandomAgent(seed=7)

    winner, history = play_game(minimax, random_agent)
    print("Single demo game")
    print("================")
    print(format_history(history))
    print(f"\nWinner: {winner}")

    wins = {0: 0, 1: 0, 2: 0}
    for seed in range(10):
        winner, _ = play_game(MinimaxAgent(depth=3), RandomAgent(seed=seed))
        wins[winner] += 1

    print("\nTen-game quick check (Minimax goes first)")
    print("=========================================")
    print(f"Minimax wins: {wins[1]}")
    print(f"Random wins: {wins[2]}")
    print(f"Draws: {wins[0]}")


if __name__ == "__main__":
    main()
