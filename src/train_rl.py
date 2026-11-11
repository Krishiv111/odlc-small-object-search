"""Train the tile-selection DQN and compare it against random and oracle
tile-picking baselines, under the same tile budget.

Usage:
    python src/train_rl.py --oracle results/tile_oracle.csv --budget 4 \
        --episodes 300 --out results/rl_training.csv
"""
import argparse
import random

import numpy as np
import pandas as pd

from dqn_agent import DQNAgent
from rl_env import TileSearchEnv


def run_episode(env: TileSearchEnv, agent: DQNAgent, epsilon: float, train: bool):
    state = env.reset()
    total_reward = 0.0
    for _ in range(env.budget):
        action = agent.select_action(state, env.visited, epsilon)
        next_state, reward, done = env.step(action)
        if train:
            agent.buffer.push(state, action, reward, next_state, float(done))
            agent.train_step()
        state = next_state
        total_reward += reward
    return total_reward


def random_policy_reward(env: TileSearchEnv) -> float:
    env.reset()
    total = 0.0
    valid = list(range(env.num_tiles))
    random.shuffle(valid)
    for action in valid[: env.budget]:
        _, reward, _ = env.step(action)
        total += reward
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", default="results/tile_oracle.csv")
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--out", default="results/rl_training.csv")
    args = parser.parse_args()

    env = TileSearchEnv(args.oracle, budget=args.budget)
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim)

    epsilon_start, epsilon_end, epsilon_decay = 1.0, 0.05, 0.98
    epsilon = epsilon_start
    rows = []

    for ep in range(args.episodes):
        reward = run_episode(env, agent, epsilon, train=True)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        if ep % 10 == 0:
            agent.update_target()
        rows.append({"episode": ep, "reward": reward, "epsilon": epsilon})

    # Final evaluation: greedy agent (no exploration) vs. random vs. oracle best.
    agent_reward = run_episode(env, agent, epsilon=0.0, train=False)
    random_rewards = [random_policy_reward(env) for _ in range(50)]
    random_avg = float(np.mean(random_rewards))
    oracle_best = env.best_possible_for_budget()
    full_total = env.total_possible()

    print(f"\nBudget: {args.budget} tiles out of {env.num_tiles}")
    print(f"Trained DQN agent:     {agent_reward:.0f} detections")
    print(f"Random policy (avg):   {random_avg:.1f} detections")
    print(f"Oracle best (budget):  {oracle_best:.0f} detections")
    print(f"Full exhaustive tiling:{full_total:.0f} detections (checks all {env.num_tiles} tiles)")

    pd.DataFrame(rows).to_csv(args.out, index=False)
    summary = pd.DataFrame(
        [
            {"policy": "trained_dqn", "tiles_checked": args.budget, "detections": agent_reward},
            {"policy": "random", "tiles_checked": args.budget, "detections": random_avg},
            {"policy": "oracle_best", "tiles_checked": args.budget, "detections": oracle_best},
            {"policy": "full_exhaustive", "tiles_checked": env.num_tiles, "detections": full_total},
        ]
    )
    summary_path = args.out.replace(".csv", "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved training curve to {args.out}")
    print(f"Saved policy comparison to {summary_path}")


if __name__ == "__main__":
    main()
