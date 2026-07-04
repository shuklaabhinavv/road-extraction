"""Run inference on a full-size satellite image via sliding window, save the
binary road mask and a red overlay visualization.

    python -m roadx.predict --checkpoint runs/unet/best.pt --image path/to.tiff --out results/
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from roadx.data.dataset import IMAGENET_MEAN, IMAGENET_STD
from roadx.models import build_model, pick_device

TILE = 512


@torch.no_grad()
def predict_image(model: torch.nn.Module, img: np.ndarray, device: torch.device) -> np.ndarray:
    """Sliding-window prediction over an HxWx3 uint8 image -> HxW probability map."""
    h, w = img.shape[:2]
    ph = (h + TILE - 1) // TILE * TILE
    pw = (w + TILE - 1) // TILE * TILE
    padded = np.zeros((ph, pw, 3), dtype=np.uint8)
    padded[:h, :w] = img

    x = padded.astype(np.float32) / 255.0
    x = (x - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    x = torch.from_numpy(x.transpose(2, 0, 1)).float()

    prob = np.zeros((ph, pw), dtype=np.float32)
    for y0 in range(0, ph, TILE):
        for x0 in range(0, pw, TILE):
            patch = x[:, y0 : y0 + TILE, x0 : x0 + TILE].unsqueeze(0).to(device)
            logits = model(patch)
            prob[y0 : y0 + TILE, x0 : x0 + TILE] = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return prob[:h, :w]


def overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    out = img.copy()
    red = np.zeros_like(img)
    red[..., 0] = 255
    sel = mask.astype(bool)
    out[sel] = (alpha * red[sel] + (1 - alpha) * img[sel]).astype(np.uint8)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    device = pick_device()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(ckpt["model"], ckpt["encoder"], encoder_weights=None)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    img = np.asarray(Image.open(args.image).convert("RGB"))
    prob = predict_image(model, img, device)
    mask = prob > args.threshold

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    Image.fromarray((mask * 255).astype(np.uint8)).save(args.out / f"{stem}_{ckpt['model']}_mask.png")
    Image.fromarray(overlay(img, mask)).save(args.out / f"{stem}_{ckpt['model']}_overlay.png")
    print(f"road pixels: {mask.mean():.2%} of image")
    print(f"wrote {args.out}/{stem}_{ckpt['model']}_mask.png and _overlay.png")


if __name__ == "__main__":
    main()
