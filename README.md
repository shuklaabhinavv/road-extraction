# Road Extraction from Satellite Imagery — Comparative Study

Comparative evaluation of four semantic segmentation architectures (U-Net, U-Net++,
DeepLabV3+, LinkNet) for road extraction from aerial imagery, with a geo-referencing
pipeline that converts detected road pixels to real-world coordinates.

Dataset: [Massachusetts Roads Dataset](https://www.cs.toronto.edu/~vmnih/data/) (Mnih, 2013).

## Setup

```bash
uv venv && uv pip install -r requirements.txt
source .venv/bin/activate
```

## Pipeline

```bash
# 1. Download data (subset for local dev; use --all for the full dataset)
python -m roadx.data.download --out data/raw --train-n 50

# 2. Tile 1500x1500 images into 512x512 patches
python -m roadx.data.tile --raw data/raw --out data/tiles

# 3. Train a model (one config per architecture in configs/)
python -m roadx.train --model unet
python -m roadx.train --model unetpp
python -m roadx.train --model deeplabv3plus
python -m roadx.train --model linknet

# 4. Evaluate all trained models on the test split
python -m roadx.evaluate --data data/tiles --runs runs

# 5. Predict + overlay on a full-size image
python -m roadx.predict --checkpoint runs/unet/best.pt --image data/raw/test/sat/xxx.tiff --out results/
```

Full training runs happen on Kaggle/Colab (see `notebooks/`); local machine is used
for development, evaluation, figures, and the paper.
