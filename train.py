"""
train.py
--------
Training loop for all 8 models (3 baselines + 5 ablation variants).
Logs per-epoch train/val loss & accuracy to outputs/logs/<model>_log.json,
saves best checkpoint to outputs/checkpoints/<model>_best.pt.

Usage:
    python train.py --model alexnet         --epochs 15 --num_classes 20
    python train.py --model mresnet50
    python train.py --model vit
    python train.py --model v1_resnet50
    python train.py --model v2_swish
    python train.py --model v3_transformer
    python train.py --model v4_fixed_noise
    python train.py --model v5_proposed
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset import get_food101_loaders
from models import build_model, count_parameters, MODEL_BUILDERS


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, targets: torch.Tensor, topk=(1, 5)):
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1)
    correct = pred.eq(targets.unsqueeze(1).expand_as(pred))
    return {f"top{k}": correct[:, :k].any(dim=1).float().mean().item()
            for k in topk}


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_top1, total_top5, n = 0.0, 0.0, 0.0, 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        bs = x.size(0)
        accs = accuracy(out.detach(), y)
        total_loss += loss.item() * bs
        total_top1 += accs["top1"] * bs
        total_top5 += accs["top5"] * bs
        n += bs
    return {"loss": total_loss/n, "top1": total_top1/n, "top5": total_top5/n}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_top1, total_top5, n = 0.0, 0.0, 0.0, 0
    for x, y in tqdm(loader, desc="eval ", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x); loss = criterion(out, y)
        bs = x.size(0)
        accs = accuracy(out, y)
        total_loss += loss.item() * bs
        total_top1 += accs["top1"] * bs
        total_top5 += accs["top5"] * bs
        n += bs
    return {"loss": total_loss/n, "top1": total_top1/n, "top5": total_top5/n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_BUILDERS.keys()))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_classes", type=int, default=20)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--out_root", type=str, default="./outputs")
    p.add_argument("--label_smoothing", type=float, default=0.1)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader, info = get_food101_loaders(
        root=args.data_root, img_size=args.img_size,
        batch_size=args.batch_size, num_classes=args.num_classes,
        num_workers=args.num_workers, seed=args.seed,
    )
    num_classes = info["num_classes"]
    print(f"Dataset: Food-101 subset | classes={num_classes} | "
          f"train={info['num_train_samples']} | test={info['num_test_samples']}")

    model = build_model(args.model, num_classes=num_classes, pretrained=True).to(device)
    n_params_M = count_parameters(model) / 1e6
    print(f"Model: {args.model}  |  params = {n_params_M:.2f} M")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    out_root = Path(args.out_root)
    log_dir  = out_root / "logs"
    ckpt_dir = out_root / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.model}_log.json"
    cfg_path = log_dir / f"{args.model}_config.json"
    best_ckpt = ckpt_dir / f"{args.model}_best.pt"

    with open(cfg_path, "w") as f:
        json.dump({**vars(args), "num_classes_used": num_classes,
                   "params_M": n_params_M,
                   "class_names": info["kept_class_names"]}, f, indent=2)

    history = []
    best_top1 = 0.0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        ev = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0
        row = {
            "epoch": epoch,
            "train_loss": tr["loss"], "train_top1": tr["top1"], "train_top5": tr["top5"],
            "val_loss":   ev["loss"], "val_top1":   ev["top1"], "val_top5":   ev["top5"],
            "lr": optimizer.param_groups[0]["lr"], "elapsed_s": elapsed,
        }
        history.append(row)
        print(f"[{args.model}] ep {epoch:2d}/{args.epochs}  "
              f"tr_loss={tr['loss']:.3f}  tr_top1={tr['top1']*100:.1f}%  "
              f"val_loss={ev['loss']:.3f}  val_top1={ev['top1']*100:.1f}%  "
              f"val_top5={ev['top5']*100:.1f}%  ({elapsed:.0f}s)")
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)

        if ev["top1"] > best_top1:
            best_top1 = ev["top1"]
            torch.save({"model_state": model.state_dict(),
                        "epoch": epoch, "top1": best_top1}, best_ckpt)
            print(f"   ↳ new best top-1 = {best_top1*100:.2f}%, saved to {best_ckpt}")

    print(f"\nDone. Best top-1 = {best_top1*100:.2f}%  |  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
