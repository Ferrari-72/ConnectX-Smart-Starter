from __future__ import annotations

from connectx_rl.dqn_trainer import DQNTrainer
from connectx_rl.local_game import MinimaxAgent, RandomAgent, play_game
from connectx_rl.minimax_dqn_agent import MinimaxDQNAgent


def main() -> None:
    trainer = DQNTrainer(batch_size=32)
    training_stats = trainer.train(episodes=100)
    hybrid_agent = MinimaxDQNAgent(q_network=trainer.q_network, depth=2)

    print("Hybrid agent demo")
    print("=================")
    print(f"Trained DQN episodes: {training_stats.episodes}")
    print(f"DQN wins/losses/draws: {training_stats.wins}/{training_stats.losses}/{training_stats.draws}")
    print()

    wins_vs_random = 0
    wins_vs_minimax = 0
    games = 5

    for seed in range(games):
        winner, _ = play_game(hybrid_agent, RandomAgent(seed=seed))
        if winner == 1:
            wins_vs_random += 1

        winner, _ = play_game(hybrid_agent, MinimaxAgent(depth=2))
        if winner == 1:
            wins_vs_minimax += 1

    print(f"Hybrid wins vs random: {wins_vs_random}/{games}")
    print(f"Hybrid wins vs minimax(depth=2): {wins_vs_minimax}/{games}")


if __name__ == "__main__":
    main()
