from __future__ import annotations

import argparse
from pathlib import Path

from connectx_rl.dqn_agent import DQNAgent
from connectx_rl.dqn_trainer import DQNTrainer
from connectx_rl.evaluation import evaluate_agent
from connectx_rl.local_game import MinimaxAgent, RandomAgent
from connectx_rl.minimax_dqn_agent import MinimaxDQNAgent


def build_opponents():
    return [
        ("random", lambda seed: RandomAgent(seed=seed)),
        ("minimax_depth_2", lambda seed: MinimaxAgent(depth=2)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved or fresh ConnectX agents.")
    parser.add_argument(
        "--agent",
        choices=["dqn", "hybrid", "minimax"],
        default="dqn",
        help="Which agent family to evaluate.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint for DQN-based agents.")
    parser.add_argument("--games", type=int, default=10, help="Games to run per opponent.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device selection for DQN-based agents.",
    )
    args = parser.parse_args()

    if args.agent == "minimax":
        agent = MinimaxAgent(depth=3)
    else:
        trainer = DQNTrainer(device=args.device)
        if args.checkpoint is not None:
            trainer.load_checkpoint(args.checkpoint)
        elif args.agent in {"dqn", "hybrid"}:
            trainer.train(episodes=50)

        if args.agent == "dqn":
            agent = DQNAgent(q_network=trainer.q_network, epsilon=0.0)
        else:
            agent = MinimaxDQNAgent(q_network=trainer.q_network, depth=2)

    summary = evaluate_agent(agent, opponents=build_opponents(), games_per_opponent=args.games)
    print("Evaluation summary")
    print("==================")
    for name, result in summary.items():
        print(f"{name}: {result['wins']}W/{result['losses']}L/{result['draws']}D over {result['games']} games")


if __name__ == "__main__":
    main()
