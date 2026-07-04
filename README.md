# Road Extraction from Satellite Imagery — A Comparative Study

Comparative evaluation of four semantic segmentation architectures — **U-Net, U-Net++,
DeepLabV3+, and LinkNet** — for road extraction from aerial imagery, with a
geo-referencing pipeline that converts detected road pixels to real-world
latitude/longitude. Target output: an IEEE-format conference paper.

**Dataset:** [Massachusetts Roads Dataset](https://www.cs.toronto.edu/~vmnih/data/)
(Mnih, 2013) — 1,171 aerial images (1500×1500 px, 1 m/px) with binary road masks.

## How it works

```
raw images ──► tile into 512×512 patches ──► train 4 models (identical recipe)
                                                      │
   IEEE paper ◄── geo-reference + OSM check ◄── evaluate & compare (IoU, F1, …)
```

All four models share the same encoder (ResNet-34), loss (Dice + BCE), augmentation,
schedule, and data — so measured differences come from the architecture alone.

## Setup

```bash
uv venv && uv pip install -r requirements.txt -e .
source .venv/bin/activate
```

## Usage

```bash
# 1. Download data (subset for local dev; --all for the full ~8 GB dataset)
python -m roadx.data.download --out data/raw --train-n 50

# 2. Tile 1500×1500 images into 512×512 patches
python -m roadx.data.tile --raw data/raw --out data/tiles

# 3. Train (one config per architecture in configs/)
python -m roadx.train --model unet          # unetpp | deeplabv3plus | linknet
python -m roadx.train --model unet --epochs 2 --limit 128   # quick sanity run

# 4. Evaluate all trained checkpoints on the test split -> CSV + LaTeX table
python -m roadx.evaluate --data data/tiles --runs runs

# 5. Full-image prediction with red road overlay
python -m roadx.predict --checkpoint runs/unet/best.pt --image <img.tiff> --out results/
```

## Cloud training

Full training runs use Kaggle's free GPUs (notebooks in `notebooks/`, generated from
this repo's sources by `scripts/make_notebooks.py` — never edit them by hand):

1. `kaggle_1_prepare_data.ipynb` — attach the public Massachusetts Roads dataset,
   tile it, save `tiles/` as notebook output (run once).
2. `kaggle_2_train.ipynb` — attach the tiles output, enable GPU, set `MODELS`,
   run twice: `['unet', 'unetpp']` then `['deeplabv3plus', 'linknet']`.

Colab variants (`colab_*.ipynb`) exist as a fallback; they stage data through
Google Drive and train one model per session.

## Roadmap

- [x] Data pipeline (download, tile, augment)
- [x] Training harness — 4 architectures, identical recipe
- [x] Local end-to-end sanity run (Apple Silicon MPS)
- [x] Kaggle/Colab notebooks for full training
- [ ] Full training runs (4 models, 30 epochs, Kaggle P100)
- [ ] Evaluation: comparison table + qualitative figures
- [ ] Geo-referencing: mask pixels → lat/lon → GeoJSON, OSM validation
- [ ] IEEE conference paper (LaTeX)
