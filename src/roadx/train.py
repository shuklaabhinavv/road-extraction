"""Train one segmentation model. The architecture is a config/flag choice so all
four models train under identical conditions.

    python -m roadx.train --model unet
    python -m roadx.train --model deeplabv3plus --epochs 5 --limit 200   # sanity run
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from roadx.data.dataset import RoadDataset
from roadx.losses import DiceBCELoss
from roadx.metrics import SegMetrics
from roadx.models import build_model, pick_device


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, loss_fn, device, optimizer=None) -> dict[str, float]:
    training = optimizer is not None
    model.train() if training else model.eval()
    metrics = SegMetrics()
    total_loss = 0.0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for img, msk in tqdm(loader, leave=False):
            img, msk = img.to(device), msk.to(device)
            logits = model(img)
            loss = loss_fn(logits, msk)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * img.size(0)
            metrics.update(logits, msk)
    out = metrics.compute()
    out["loss"] = total_loss / len(loader.dataset)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="unet | unetpp | deeplabv3plus | linknet")
    p.add_argument("--config", type=Path, default=None, help="defaults to configs/<model>.yaml")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="cap train samples (sanity runs)")
    p.add_argument("--out", type=Path, default=Path("runs"))
    args = p.parse_args()

    cfg_path = args.config or Path("configs") / f"{args.model}.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    if args.data_dir:
        cfg["data_dir"] = str(args.data_dir)
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.batch_size:
        cfg["batch_size"] = args.batch_size

    seed_everything(cfg.get("seed", 42))
    device = pick_device()
    print(f"model={cfg['model']} encoder={cfg['encoder']} device={device.type}")

    train_ds = RoadDataset(cfg["data_dir"], "train", train=True, limit=args.limit)
    valid_ds = RoadDataset(cfg["data_dir"], "valid", train=False)
    print(f"train tiles={len(train_ds)} valid tiles={len(valid_ds)}")

    pin = device.type == "cuda"
    train_dl = DataLoader(
        train_ds, cfg["batch_size"], shuffle=True,
        num_workers=cfg.get("num_workers", 4), pin_memory=pin, drop_last=True,
    )
    valid_dl = DataLoader(
        valid_ds, cfg["batch_size"], shuffle=False,
        num_workers=cfg.get("num_workers", 4), pin_memory=pin,
    )

    model = build_model(cfg["model"], cfg["encoder"], cfg.get("encoder_weights", "imagenet"))
    model.to(device)
    loss_fn = DiceBCELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg.get("weight_decay", 1e-4))
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    run_dir = args.out / cfg["model"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    log_path = run_dir / "log.csv"
    best_iou = -1.0

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "train_iou", "val_loss", "val_iou", "val_f1", "seconds"])
        for epoch in range(1, cfg["epochs"] + 1):
            t0 = time.time()
            tr = run_epoch(model, train_dl, loss_fn, device, optimizer)
            va = run_epoch(model, valid_dl, loss_fn, device)
            scheduler.step()
            dt = time.time() - t0
            writer.writerow([
                epoch, f"{optimizer.param_groups[0]['lr']:.2e}",
                f"{tr['loss']:.4f}", f"{tr['iou']:.4f}",
                f"{va['loss']:.4f}", f"{va['iou']:.4f}", f"{va['f1']:.4f}", f"{dt:.1f}",
            ])
            f.flush()
            marker = ""
            if va["iou"] > best_iou:
                best_iou = va["iou"]
                torch.save(
                    {"model": cfg["model"], "encoder": cfg["encoder"],
                     "state_dict": model.state_dict(), "val_iou": best_iou, "epoch": epoch},
                    run_dir / "best.pt",
                )
                marker = " *best*"
            print(
                f"epoch {epoch:3d}/{cfg['epochs']} "
                f"train loss {tr['loss']:.4f} iou {tr['iou']:.4f} | "
                f"val loss {va['loss']:.4f} iou {va['iou']:.4f} f1 {va['f1']:.4f} "
                f"({dt:.0f}s){marker}"
            )

    print(f"done. best val IoU {best_iou:.4f} -> {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
