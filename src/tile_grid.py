"""Non-overlapping tile grid over an image, used by the RL tile-selection env.

Separate from tiled_inference.py's SAHI slicing (which overlaps tiles for
better edge coverage) -- the RL agent picks from a fixed, non-overlapping
grid so "which tiles did it check" is unambiguous.
"""
from dataclasses import dataclass

import cv2


@dataclass
class Tile:
    tile_id: int
    x: int
    y: int
    w: int
    h: int


def build_grid(image_path: str, tile_size: int = 320) -> list[Tile]:
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    tiles = []
    tid = 0
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tw = min(tile_size, w - x)
            th = min(tile_size, h - y)
            tiles.append(Tile(tile_id=tid, x=x, y=y, w=tw, h=th))
            tid += 1
    return tiles


def crop(image_path: str, tile: Tile):
    img = cv2.imread(image_path)
    return img[tile.y : tile.y + tile.h, tile.x : tile.x + tile.w]
