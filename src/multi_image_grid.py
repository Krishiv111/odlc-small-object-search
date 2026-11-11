"""Shared grid + tile-feature utilities for the multi-image generalization
experiment.

Every image gets resized to the same canvas size so the tile grid (and
therefore the action space: "pick tile 0..11") lines up across different
photos. Each tile also gets a small handcrafted feature vector (average
color + an edge-density number) -- this is the "give the agent eyes" part:
instead of only a visited/not-visited flag, the agent sees *what the tile
actually looks like*, so it has a chance to learn a rule like "greenish-dark
water tiles are empty, grey/white dock tiles have boats" instead of
memorizing tile positions.
"""
from dataclasses import dataclass

import cv2
import numpy as np

CANVAS_W, CANVAS_H = 1280, 960
TILE_SIZE = 320
GRID_COLS = CANVAS_W // TILE_SIZE  # 4
GRID_ROWS = CANVAS_H // TILE_SIZE  # 3
NUM_TILES = GRID_COLS * GRID_ROWS  # 12


@dataclass
class Tile:
    tile_id: int
    x: int
    y: int
    w: int
    h: int


def load_resized(image_path: str):
    img = cv2.imread(image_path)
    return cv2.resize(img, (CANVAS_W, CANVAS_H))


def build_grid() -> list[Tile]:
    tiles = []
    tid = 0
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            tiles.append(
                Tile(tile_id=tid, x=col * TILE_SIZE, y=row * TILE_SIZE,
                     w=TILE_SIZE, h=TILE_SIZE)
            )
            tid += 1
    return tiles


def crop(img, tile: Tile):
    return img[tile.y : tile.y + tile.h, tile.x : tile.x + tile.w]


def tile_features(tile_img) -> list[float]:
    """Mean R/G/B (0-1) + a rough edge-density score (0-1-ish)."""
    mean_bgr = tile_img.reshape(-1, 3).mean(axis=0) / 255.0
    gray = cv2.cvtColor(tile_img, cv2.COLOR_BGR2GRAY)
    edge_density = float(np.abs(cv2.Laplacian(gray, cv2.CV_64F)).mean()) / 50.0
    edge_density = min(edge_density, 1.0)
    b, g, r = mean_bgr
    return [float(r), float(g), float(b), edge_density]
