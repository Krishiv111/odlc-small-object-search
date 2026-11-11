"""Multi-image tile-selection environment.

Same budgeted "pick N of the tiles" game as rl_env.py, but now:
  - there are several images, each with its own oracle (answer key)
  - the state includes each tile's actual visual features (color + edge
    density), not just a visited mask
  - reset(image_id=...) lets training sample a random image, and evaluation
    pin a specific (possibly held-out) image

This is what makes it possible to test generalization: train only on some
image_ids, then reset(image_id=<one never trained on>) and see if the agent
still picks good tiles using what the tile LOOKS like, instead of having
memorized tile positions for one specific photo.
"""
from __future__ import annotations

import random

import pandas as pd

FEATURE_COLS = ["mean_r", "mean_g", "mean_b", "edge_density"]


class MultiTileSearchEnv:
    def __init__(self, oracle_csv: str, budget: int = 5):
        self.oracle = pd.read_csv(oracle_csv)
        self.image_ids = sorted(self.oracle["image_id"].unique())
        self.num_tiles = self.oracle["tile_id"].nunique()
        self.budget = budget
        self.state_dim = self.num_tiles * (1 + len(FEATURE_COLS))
        self.action_dim = self.num_tiles
        self.current_image = None
        self.current_df = None
        self.reset()

    def _state(self):
        state = []
        for tile_id in range(self.num_tiles):
            row = self.current_df[self.current_df.tile_id == tile_id].iloc[0]
            state.append(float(self.visited[tile_id]))
            state.extend(float(row[c]) for c in FEATURE_COLS)
        return state

    def reset(self, image_id: str | None = None, train_image_ids: list | None = None):
        if image_id is not None:
            self.current_image = image_id
        elif train_image_ids is not None:
            self.current_image = random.choice(train_image_ids)
        else:
            self.current_image = random.choice(self.image_ids)

        self.current_df = self.oracle[self.oracle.image_id == self.current_image]
        self.visited = [0] * self.num_tiles
        self.steps_taken = 0
        return self._state()

    def step(self, action: int):
        already_visited = self.visited[action] == 1
        if already_visited:
            reward = 0.0
        else:
            row = self.current_df[self.current_df.tile_id == action].iloc[0]
            reward = float(row["num_detections"])
            self.visited[action] = 1

        self.steps_taken += 1
        done = self.steps_taken >= self.budget
        return self._state(), reward, done

    def total_possible(self, image_id: str | None = None) -> float:
        df = self.current_df if image_id is None else self.oracle[self.oracle.image_id == image_id]
        return float(df["num_detections"].sum())

    def best_possible_for_budget(self, image_id: str | None = None) -> float:
        df = self.current_df if image_id is None else self.oracle[self.oracle.image_id == image_id]
        return float(df["num_detections"].nlargest(self.budget).sum())
