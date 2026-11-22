# ODLC small-object search

Hi, my name is Krishiv. I'm on UAV Forge's ODLC/perception team, and this is
a side project testing whether reinforcement learning can make our
competition drone's object search smarter — teaching an agent *where to
look* in a photo instead of scanning every inch of it.

My inspiration came from getting interested in reinforcement learning and
wanting to actually build something with it, not just read about it. Two
papers shaped how this is built:

- **Deep Q-Networks (DQN)** — Mnih et al., ["Playing Atari with Deep Reinforcement Learning"](https://arxiv.org/abs/1312.5602).
  This is where the core learning algorithm here comes from: a small neural
  net that learns to estimate "how good is this action" from trial, error,
  and reward, instead of being told the rules up front.
- **"Object Detection with Deep Reinforcement Learning"** — Samiei & Li, McGill (["arXiv:2208.04511"](https://arxiv.org/pdf/2208.04511)).
  This paper got me thinking about RL + object detection together. Worth
  being precise about what we borrowed and what we didn't: their agent
  moves and zooms a *single bounding box* onto one object. Ours is a
  different problem — picking *which tile* of a grid to check next, to
  search a whole image efficiently. Same RL machinery (DQN, replay buffer,
  Bellman updates), different task.

## What this project actually does, phase by phase

**Phase 1 — baseline: does tiling even help?**
Before touching RL, I needed a number to beat. Split an aerial photo into a
grid of tiles, run the detector on every tile, and compare that against
just running the detector once on the full image. On a generic stand-in
photo with a generic model, tiling nearly tripled detections. On Forge's
*real*, fine-tuned model, the gap mostly disappeared — the real model is
already good enough that full-image detection catches almost everything.
Honest finding, not the one I expected, and it reshaped the rest of the
project (see "Real mission data" in `RESULTS.md`).

**Phase 2 — can an agent learn to check *fewer* tiles and still find everything?**
Here's where RL comes in. I precompute an "oracle" — run the real detector
on every tile once, offline, and cache how many real detections are in
each one. Then I train a small DQN agent: give it a budget (say, 2 of 24
tiles), let it pick, reward it with the real detection count in whatever
tile it picks. On real data, it learned to check only 2 of 24 tiles and
still found everything — the "needle in a haystack" case this is actually
for.

**Phase 3 — does it generalize, or did it just memorize one photo?**
Training on one image can only prove memorization. So the real test: train
on some images, hide one away completely, and see if the agent still picks
good tiles on the one it's never seen. First pass: it didn't generalize at
all (0 for 2). I synthetically expanded the training data (brightness,
rotation, flip variants of the real frames) and reran it as a proper
leave-one-scene-out test across all 4 scenes. Result: 47% of the
best-possible score on unseen scenes, versus 21% for random guessing — a
real signal, but inconsistent scene-to-scene, not a solved problem.

**Phase 4 — figuring out *why* it's inconsistent, and fixing the actual flaw**
The agent doesn't see the image directly — each tile is represented by 4
cheap numbers: average red/green/blue and an edge-density (texture) score.
Checking those numbers against real detections showed the color features
barely separate "has a target" from "doesn't" (0.4 vs 0.5 average red —
basically noise). Worse: even if that weak color signal were stronger, it'd
be tied to the *one* tent color in our training footage, which doesn't
hold up for competition day if the real tent is a different color.

**Phase 5 — moving to anomaly-style features (where this is headed)**
Instead of hard-coding "a tent looks like color X," the direction we're
building toward is relative: for each image, compute what the background
(field, dirt, grass) typically looks like, and flag tiles whose color
*stands out* from that background, whatever that background and target
color actually turn out to be. That removes the baked-in color assumption
and should generalize across tent colors, since it's measuring "what's
different here" instead of "does this match a color I memorized."

## Run it locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Baseline — tiling vs. full image, on Forge's real model and real frames:**
```bash
python src/tiled_inference.py --image data/real/tent_mannequin_multi.jpg \
    --model models/forge_yolo11n_v19_best.pt --out results/baseline_real.csv --save-vis
python src/make_plots.py --csv results/baseline_real.csv --out results/plots_real
```

**RL tile-selection on a single real image:**
```bash
python src/build_oracle.py --image data/real/tent_mannequin_multi.jpg --tile-size 320 \
    --model models/forge_yolo11n_v19_best.pt --out results/tile_oracle_real.csv
python src/train_rl.py --oracle results/tile_oracle_real.csv --budget 2 --episodes 800 \
    --out results/rl_training_real.csv
python src/plot_rl.py --training results/rl_training_real.csv \
    --summary results/rl_training_real_summary.csv --out results/plots_real
```

**Generalization: synthetic variety + leave-one-scene-out:**
```bash
python src/augment_frames.py
python src/build_oracle_multi.py --dataset real --model models/forge_yolo11n_v19_best.pt \
    --out results/tile_oracle_multi_real.csv
for base in frame_multi frame_mannequin_tent frame_single_tent frame_tent_closeup; do
    python src/train_rl_generalization.py --oracle results/tile_oracle_multi_real.csv \
        --held-out-base $base --budget 5 --episodes 500 \
        --out results/rl_generalization_loso_${base}.csv
done
python src/aggregate_loso.py
python src/plot_loso.py
```

Full writeup of every result, caveat, and honest "this didn't work and
here's why" is in [`RESULTS.md`](RESULTS.md) — that's the real source of
truth, this README is just the tour.
