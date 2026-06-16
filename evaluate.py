"""
evaluate.py
-----------
Loads the best checkpoint for a model and computes SEVEN metrics:

  1. Top-1 accuracy
  2. Top-5 accuracy
  3. Macro F1-score
  4. Macro Precision
  5. Macro Recall
  6. Inference time per image (ms, batch=1, GPU if available)
  7. Trainable parameter count (M)

Also saves the confusion matrix and per-class top-1 accuracy as .npy.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (confusion_matrix, f1_score,
                             precision_score, recall_score)
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset import get_food101_loaders
from models import build_model, count_parameters, MODEL_BUILDERS


@torch.no_grad()
def evaluate_full(model, loader, device, num_classes):
    model.eval()
    total_top1, total_top5, n = 0.0, 0.0, 0
    all_preds, all_targets = [], []
    for x, y in tqdm(loader, desc="eval"):
        x, y = x.to(device), y.to(device)
        out = model(x)
        bs = x.size(0)
        _, top5 = out.topk(5, dim=1)
        correct_top5 = top5.eq(y.unsqueeze(1).expand_as(top5))
        total_top5 += correct_top5.any(dim=1).float().sum().item()
        top1 = top5[:, 0]
        total_top1 += top1.eq(y).float().sum().item()
        n += bs
        all_preds.append(top1.cpu().numpy())
        all_targets.append(y.cpu().numpy())
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    per_class = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    macro_f1   = f1_score(targets, preds, average="macro", zero_division=0)
    macro_prec = precision_score(targets, preds, average="macro", zero_division=0)
    macro_rec  = recall_score(targets, preds, average="macro", zero_division=0)

    return {
        "top1": total_top1 / n,
        "top5": total_top5 / n,
        "macro_f1":        macro_f1,
        "macro_precision": macro_prec,
        "macro_recall":    macro_rec,
        "confusion_matrix": cm,
        "per_class_top1":   per_class,
    }


@torch.no_grad()
def time_inference(model, device, img_size=224, batch_size=1,
                   n_warmup=20, n_runs=200):
    model.eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    for _ in range(n_warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_runs):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return ((time.time() - t0) / n_runs) * 1000.0  # ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_BUILDERS.keys()))
    p.add_argument("--num_classes", type=int, default=20)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--out_root", type=str, default="./outputs")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_loader, info = get_food101_loaders(
        root=args.data_root, img_size=args.img_size,
        batch_size=args.batch_size, num_classes=args.num_classes,
        num_workers=args.num_workers, seed=args.seed,
    )
    num_classes = info["num_classes"]

    model = build_model(args.model, num_classes=num_classes,
                        pretrained=False).to(device)
    ckpt_path = Path(args.out_root) / "checkpoints" / f"{args.model}_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    res = evaluate_full(model, test_loader, device, num_classes)
    ms_per_image = time_inference(model, device,
                                  img_size=args.img_size, batch_size=1)
    n_params_M = count_parameters(model) / 1e6

    out_dir = Path(args.out_root) / "logs"
    np.save(out_dir / f"{args.model}_confusion.npy", res["confusion_matrix"])
    np.save(out_dir / f"{args.model}_per_class.npy", res["per_class_top1"])

    summary = {
        "model": args.model,
        "params_M": n_params_M,
        "top1": res["top1"],
        "top5": res["top5"],
        "macro_f1":        res["macro_f1"],
        "macro_precision": res["macro_precision"],
        "macro_recall":    res["macro_recall"],
        "inference_ms_per_image": ms_per_image,
        "num_classes": num_classes,
        "class_names": info["kept_class_names"],
    }
    with open(out_dir / f"{args.model}_eval.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== {args.model} ===")
    print(f"  params           : {n_params_M:.2f} M")
    print(f"  top-1 accuracy   : {res['top1']*100:.2f} %")
    print(f"  top-5 accuracy   : {res['top5']*100:.2f} %")
    print(f"  macro F1         : {res['macro_f1']*100:.2f} %")
    print(f"  macro Precision  : {res['macro_precision']*100:.2f} %")
    print(f"  macro Recall     : {res['macro_recall']*100:.2f} %")
    print(f"  inference time   : {ms_per_image:.2f} ms / image (batch=1)")


if __name__ == "__main__":
    main()
