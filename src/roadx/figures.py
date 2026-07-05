"""Generate the paper's figures from training logs and trained checkpoints.

    python -m roadx.figures --data data --runs runs --out results/figures

Produces:
  val_iou_curves.(png|pdf)  - validation IoU vs epoch for all models
  qualitative_<stem>.png    - input | ground truth | each model's prediction
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from roadx.models import build_model, pick_device
from roadx.predict import overlay, predict_image

MODEL_LABELS = {
    "unet": "U-Net",
    "unetpp": "U-Net++",
    "deeplabv3plus": "DeepLabV3+",
    "linknet": "LinkNet",
}
ORDER = ["unet", "unetpp", "deeplabv3plus", "linknet"]


def curves_figure(runs: Path, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in ORDER:
        log = runs / name / "log.csv"
        if not log.exists():
            continue
        df = pd.read_csv(log)
        ax.plot(df["epoch"], df["val_iou"], label=MODEL_LABELS[name], linewidth=1.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation IoU")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "val_iou_curves.png", dpi=200)
    fig.savefig(out / "val_iou_curves.pdf")
    plt.close(fig)
    print(f"wrote {out}/val_iou_curves.png|.pdf")


def qualitative_figure(models: dict, sat_path: Path, map_path: Path, out: Path, device) -> None:
    img = np.asarray(Image.open(sat_path).convert("RGB"))
    gt = (np.asarray(Image.open(map_path).convert("L")) > 127)

    n = 2 + len(models)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.5))
    axes[0].imshow(img)
    axes[0].set_title("Input")
    axes[1].imshow(gt, cmap="gray")
    axes[1].set_title("Ground truth")
    for ax, (name, model) in zip(axes[2:], models.items()):
        prob = predict_image(model, img, device)
        ax.imshow(overlay(img, prob > 0.5))
        ax.set_title(MODEL_LABELS[name])
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    dest = out / f"qualitative_{sat_path.stem}.png"
    fig.savefig(dest, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {dest}")


def image_iou(model, sat_path: Path, map_path: Path, device) -> float:
    img = np.asarray(Image.open(sat_path).convert("RGB"))
    gt = np.asarray(Image.open(map_path).convert("L")) > 127
    pred = predict_image(model, img, device) > 0.5
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter / (union + 1e-7))


def failure_figure(model, name: str, sat_dir: Path, map_dir: Path, out: Path,
                   device, n_worst: int = 2) -> None:
    """Rank test images by IoU and render the worst cases with error maps."""
    pairs = []
    for sat_path in sorted(sat_dir.iterdir()):
        map_path = next(
            (map_dir / f"{sat_path.stem}{ext}" for ext in (".tif", ".tiff", ".png")
             if (map_dir / f"{sat_path.stem}{ext}").exists()), None)
        if map_path is None:
            continue
        iou = image_iou(model, sat_path, map_path, device)
        pairs.append((iou, sat_path, map_path))
        print(f"  {sat_path.stem}: IoU {iou:.3f}")
    pairs.sort(key=lambda t: t[0])

    worst = pairs[:n_worst]
    fig, axes = plt.subplots(n_worst, 3, figsize=(10.5, 3.6 * n_worst))
    axes = np.atleast_2d(axes)
    for row, (iou, sat_path, map_path) in enumerate(worst):
        img = np.asarray(Image.open(sat_path).convert("RGB"))
        gt = np.asarray(Image.open(map_path).convert("L")) > 127
        pred = predict_image(model, img, device) > 0.5
        # error map: green = hit, red = missed road (FN), blue = false road (FP)
        err = np.zeros((*gt.shape, 3), dtype=np.uint8)
        err[gt & pred] = (46, 125, 50)
        err[gt & ~pred] = (198, 40, 40)
        err[~gt & pred] = (21, 101, 192)
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"Input (IoU {iou:.3f})", fontsize=10)
        axes[row, 1].imshow(gt, cmap="gray")
        axes[row, 1].set_title("Ground truth", fontsize=10)
        axes[row, 2].imshow(err)
        axes[row, 2].set_title("Hit / missed / false", fontsize=10)
        for ax in axes[row]:
            ax.axis("off")
    fig.tight_layout()
    dest = out / f"failure_cases_{name}.png"
    fig.savefig(dest, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {dest}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--out", type=Path, default=Path("results/figures"))
    p.add_argument("--n-images", type=int, default=3)
    p.add_argument("--failure", action="store_true",
                   help="only render the failure-cases figure for the best model")
    args = p.parse_args()

    if args.failure:
        args.out.mkdir(parents=True, exist_ok=True)
        device = pick_device()
        ckpt = torch.load(args.runs / "unetpp" / "best.pt", map_location="cpu")
        m = build_model(ckpt["model"], ckpt["encoder"], encoder_weights=None)
        m.load_state_dict(ckpt["state_dict"])
        m.to(device).eval()
        failure_figure(m, "unetpp", args.data / "raw" / "test" / "sat",
                       args.data / "raw" / "test" / "map", args.out, device)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    curves_figure(args.runs, args.out)

    device = pick_device()
    models = {}
    for name in ORDER:
        ckpt_path = args.runs / name / "best.pt"
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu")
        m = build_model(ckpt["model"], ckpt["encoder"], encoder_weights=None)
        m.load_state_dict(ckpt["state_dict"])
        m.to(device).eval()
        models[name] = m

    sat_dir = args.data / "raw" / "test" / "sat"
    map_dir = args.data / "raw" / "test" / "map"
    sats = sorted(sat_dir.iterdir())
    # spread picks across the test set rather than taking neighbours
    picks = [sats[i] for i in np.linspace(0, len(sats) - 1, args.n_images, dtype=int)]
    for sat_path in picks:
        map_path = next(
            (map_dir / f"{sat_path.stem}{ext}" for ext in (".tif", ".tiff", ".png")
             if (map_dir / f"{sat_path.stem}{ext}").exists()),
            None,
        )
        if map_path is None:
            continue
        qualitative_figure(models, sat_path, map_path, args.out, device)


if __name__ == "__main__":
    main()
