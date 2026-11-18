# ODLC small-object search (baseline)

Week-1 baseline for the ODLC small-object-at-altitude option: how much does
exhaustive sliced inference (SAHI-style tiling) help over plain full-image
inference for small/far targets, and what does it cost in tiles checked.

RL adaptive search (learning *where* to look instead of checking every tile)
is the stretch goal once this baseline and the real team dataset are in place
-- see the project one-pager for the full plan.

**We now have real data.** `models/forge_yolo11n_v19_best.pt` is Forge's own
trained detector (classes: `tent`, `mannequin`), and `data/real/*.jpg` are
clean frames pulled from Forge's own flight footage. See the "Real mission
data" section at the top of `RESULTS.md` for the current results -- the
sections below using `data/sample.jpg` and other stand-ins are from before
we had real data, kept for history, not the current story.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the baseline

```bash
python src/get_sample_image.py          # pulls a test image into data/sample.jpg
python src/tiled_inference.py --image data/sample.jpg --out results/baseline.csv --save-vis
```

`--save-vis` writes annotated images to `results/vis/` so you can eyeball what
each tiling setting actually catches.

```bash
python src/make_plots.py       # turns results/baseline.csv into results/plots/
```

See `RESULTS.md` for the writeup: what the numbers show, caveats, and the
list of experiments still needed before this is report-ready.

## RL tile-selection (first version)

```bash
python src/build_oracle.py --image data/sample.jpg --tile-size 320 --out results/tile_oracle.csv
python src/train_rl.py --oracle results/tile_oracle.csv --budget 4 --episodes 300
python src/plot_rl.py
```

First working version of the RL stretch goal: a small DQN picks a budget of
tiles to check instead of checking all of them. Results and the "what this
does/doesn't prove yet" caveats are in `RESULTS.md`.

## RL generalization check (train on 2 images, test on a held-out 3rd)

```bash
python src/get_multi_images.py   # pulls 2 more stand-in images into data/multi/
python src/build_oracle_multi.py --out results/tile_oracle_multi.csv
python src/train_rl_generalization.py --oracle results/tile_oracle_multi.csv \
    --held-out marina_boats --budget 5 --episodes 500
python src/plot_generalization.py
```

Tests whether the agent learned a real "what to look for" rule or just
memorized one photo, by training on 2 images and evaluating on a 3rd it
never saw. Results (it beats random but doesn't fully match oracle-best on
the unseen image) are in `RESULTS.md`.

## Week 1 results (stand-in image)

Ran on a single test aerial photo (SAHI's tutorial image, standing in for real
flight footage until team dataset access is confirmed). Pretrained YOLOv8n,
confidence threshold 0.25, CPU inference (no local GPU):

| method | slice size | overlap | detections | tiles checked | time (s) |
|---|---|---|---|---|---|
| full image | - | - | 12 | 1 | 4.94 |
| sliced | 320 | 0.1 | 32 | 8 | 2.86 |
| sliced | 320 | 0.2 | 34 | 12 | 3.92 |
| sliced | 512 | 0.1 | 22 | 6 | 2.14 |
| sliced | 512 | 0.2 | 22 | 6 | 2.06 |
| sliced | 640 | 0.1 | 19 | 2 | 1.05 |
| sliced | 640 | 0.2 | 19 | 2 | 0.87 |

Full-image inference found 12 of the objects in frame; the finest tiling
(320px, 0.2 overlap) found 34, nearly 3x more, by checking 12 tiles instead
of 1. This is the gap the project is actually about: exhaustive tiling buys
recall but costs tile checks, which is exactly what an RL search policy would
try to do with fewer checks.

**Caveat:** this is one stand-in image with a pretrained COCO model, not real
flight footage or a held-out test set, so these numbers are a sanity check
that the harness works, not a result to put in the final report. Also spotted
at least one false positive from a tile seam (a low-confidence "airplane" on
the smallest tiling) -- worth tracking as tiling introduces its own error
modes, see per-option guidance on duplicate/edge-of-tile problems.

## Next steps

- Swap `data/sample.jpg` for a real sample from the team dataset (or a
  VisDrone/DOTA crop if that access is still pending)
- Sweep over multiple images and report recall as a function of target pixel
  size, not just raw detection counts on one photo
- Add NMS-based duplicate merging across tile boundaries and measure its effect
