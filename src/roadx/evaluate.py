"""Evaluate every trained checkpoint in runs/ on the test split and write the
comparison table (CSV + LaTeX) used in the paper.

    python -m roadx.evaluate --data data/tiles --runs runs --out results
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from roadx.data.dataset import RoadDataset
from roadx.metrics import SegMetrics
from roadx.models import build_model, pick_device


@torch.no_grad()
def evaluate_checkpoint(ckpt_path: Path, data_dir: Path, device: torch.device, batch_size: int = 8) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model(ckpt["model"], ckpt["encoder"], encoder_weights=None)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    ds = RoadDataset(data_dir, "test", train=False)
    dl = DataLoader(ds, batch_size, shuffle=False, num_workers=4)

    metrics = SegMetrics()
    n_imgs = 0
    t0 = time.time()
    for img, msk in dl:
        img, msk = img.to(device), msk.to(device)
        logits = model(img)
        metrics.update(logits, msk)
        n_imgs += img.size(0)
    elapsed = time.time() - t0

    out = metrics.compute()
    out["model"] = ckpt["model"]
    out["params_m"] = sum(p.numel() for p in model.parameters()) / 1e6
    out["ms_per_tile"] = elapsed / n_imgs * 1000
    out["best_epoch"] = ckpt.get("epoch")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/tiles"))
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args()

    device = pick_device()
    rows = []
    for ckpt in sorted(args.runs.glob("*/best.pt")):
        print(f"evaluating {ckpt} ...")
        rows.append(evaluate_checkpoint(ckpt, args.data, device))

    if not rows:
        raise SystemExit(f"no checkpoints found under {args.runs}/*/best.pt")

    df = pd.DataFrame(rows)[
        ["model", "iou", "f1", "precision", "recall", "params_m", "ms_per_tile", "best_epoch"]
    ].sort_values("iou", ascending=False)

    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "comparison.csv", index=False)
    latex = df.to_latex(index=False, float_format="%.4f")
    (args.out / "comparison.tex").write_text(latex)
    print(df.to_string(index=False))
    print(f"\nwrote {args.out / 'comparison.csv'} and comparison.tex")


if __name__ == "__main__":
    main()
