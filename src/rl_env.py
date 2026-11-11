"""Budgeted tile-selection environment.

The agent picks `budget` tiles (out of all tiles in the grid) one at a time,
trying to maximize total detections found, using the precomputed oracle from
build_oracle.py as the reward signal. State is simply "which tiles have been
picked so far" -- a one-hot visited mask -- which is enough for the agent to
learn a tile ranking on a single fixed image.

This is the first, simplest version of the Hierarchical Method agent
sketched in the one-pager (region-selection instead of exhaustive tiling).
Scaling this up to real images needs real per-tile image features in the
state (not just a visited mask) so the policy can generalize past one photo
-- see RESULTS.md for why that's the next step, not this one.
"""
import pandas as pd


class TileSearchEnv:
    def __init__(self, oracle_csv: str, budget: int = 4):
        self.oracle = pd.read_csv(oracle_csv)
        self.num_tiles = len(self.oracle)
        self.budget = budget
        self.state_dim = self.num_tiles  # visited mask
        self.action_dim = self.num_tiles
        self.reset()

    def reset(self):
        self.visited = [0] * self.num_tiles
        self.steps_taken = 0
        return list(self.visited)

    def step(self, action: int):
        already_visited = self.visited[action] == 1
        if already_visited:
            reward = 0.0
        else:
            reward = float(self.oracle.iloc[action]["num_detections"])
            self.visited[action] = 1

        self.steps_taken += 1
        done = self.steps_taken >= self.budget
        return list(self.visited), reward, done

    def total_possible(self) -> float:
        return float(self.oracle["num_detections"].sum())

    def best_possible_for_budget(self) -> float:
        top = self.oracle["num_detections"].nlargest(self.budget)
        return float(top.sum())
