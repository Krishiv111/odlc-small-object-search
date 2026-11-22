"""Train the tile-selection DQN on some images, then test it on a held-out
image it never saw during training. This is the actual generalization test
promised in RESULTS.md -- single-image training can only ever memorize, so
this is the experiment that tells us whether giving the agent real tile
features (multi_image_grid.tile_features) was enough to make it learn a
transferable rule instead.

Usage:
    python src/train_rl_generalization.py --oracle results/tile_oracle_multi.csv \
        --held-out marina_boats --budget 5 --episodes 500
"""
import argparse
import random

import numpy as np
import pandas as pd

from dqn_agent import DQNAgent
from rl_env_multi import ANOMALY_FEATURE_COLS, RAW_FEATURE_COLS, MultiTileSearchEnv


def run_episode(env, agent, epsilon, train, image_id=None, train_image_ids=None):
    state = env.reset(image_id=image_id, train_image_ids=train_image_ids)
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


def random_policy_reward(env, image_id):
    env.reset(image_id=image_id)
    total = 0.0
    valid = list(range(env.num_tiles))
    random.shuffle(valid)
    for action in valid[: env.budget]:
        _, reward, _ = env.step(action)
        total += reward
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", default="results/tile_oracle_multi.csv")
    parser.add_argument("--held-out", default="marina_boats")
    parser.add_argument(
        "--held-out-base",
        default=None,
        help="Hold out a whole scene group: this id plus every "
        "<id>_bright/_dark/_rot/_flip augmented variant of it. Takes "
        "priority over --held-out when set.",
    )
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--out", default="results/rl_generalization_summary.csv")
    parser.add_argument(
        "--feature-set",
        choices=["raw", "anomaly"],
        default="raw",
        help="raw = mean_r/g/b + edge_density (absolute, color-dependent). "
        "anomaly = color_anomaly + edge_anomaly (relative to this image's own "
        "background, no baked-in color assumption).",
    )
    args = parser.parse_args()

    feature_cols = ANOMALY_FEATURE_COLS if args.feature_set == "anomaly" else RAW_FEATURE_COLS
    env = MultiTileSearchEnv(args.oracle, budget=args.budget, feature_cols=feature_cols)

    if args.held_out_base:
        held_out_group = {args.held_out_base} | {
            f"{args.held_out_base}_{suffix}" for suffix in ["bright", "dark", "rot", "flip"]
        }
        train_image_ids = [i for i in env.image_ids if i not in held_out_group]
        held_out_display = args.held_out_base
    else:
        held_out_group = {args.held_out}
        train_image_ids = [i for i in env.image_ids if i != args.held_out]
        held_out_display = args.held_out

    print(f"Training images ({len(train_image_ids)}): {train_image_ids}")
    print(f"Held-out group (never trained on, any form): {sorted(held_out_group)}")

    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim)
    epsilon, epsilon_end, epsilon_decay = 1.0, 0.05, 0.99

    for ep in range(args.episodes):
        run_episode(env, agent, epsilon, train=True, train_image_ids=train_image_ids)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        if ep % 10 == 0:
            agent.update_target()

    rows = []
    for image_id in env.image_ids:
        seen = "seen (trained on)" if image_id not in held_out_group else "HELD OUT (never trained on)"
        agent_reward = run_episode(env, agent, epsilon=0.0, train=False, image_id=image_id)
        random_rewards = [random_policy_reward(env, image_id) for _ in range(50)]
        random_avg = float(np.mean(random_rewards))
        oracle_best = env.best_possible_for_budget(image_id)
        full_total = env.total_possible(image_id)
        print(
            f"\n[{image_id}] {seen}\n"
            f"  trained agent: {agent_reward:.0f} / oracle best: {oracle_best:.0f} "
            f"/ random avg: {random_avg:.1f} / full exhaustive: {full_total:.0f}"
        )
        rows.append(
            {
                "image_id": image_id, "status": seen,
                "trained_agent": agent_reward, "random_avg": random_avg,
                "oracle_best": oracle_best, "full_exhaustive": full_total,
            }
        )

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nSaved generalization comparison to {args.out}")


if __name__ == "__main__":
    main()
