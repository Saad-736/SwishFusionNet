"""
plot_results.py
---------------
Generates every figure used in the final paper, including the ablation study.

Outputs (PNG + PDF in outputs/figures/):
  fig1_loss_curves.png        — train/val loss curves
  fig2_accuracy_curves.png    — train/val top-1 curves
  fig3_baselines_vs_proposed.png  — proposed V5 vs three external baselines
  fig4_ablation_top1.png      — ablation: top-1 across V1→V5 with delta annotations
  fig5_ablation_metrics.png   — ablation: F1, precision, recall across V1→V5
  fig6_params_vs_acc.png      — parameter count vs accuracy scatter
  fig7_inference_time.png     — inference latency comparison
  fig8_confusion_<model>.png  — confusion matrix per model
  fig9_per_class_acc.png      — per-class top-1 accuracy across selected models
  fig10_metric_radar.png      — radar plot of 5 metrics for top models

CSV outputs:
  outputs/figures/summary_table_all.csv     — all 8 models, all 7 metrics
  outputs/figures/ablation_table.csv        — only V1-V5 with delta columns
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
})

BASELINES = ["alexnet", "mresnet50", "vit"]
ABLATION  = ["v1_resnet50", "v2_swish", "v3_transformer",
             "v4_fixed_noise", "v5_proposed"]
ALL_MODELS = BASELINES + ABLATION

DISPLAY = {
    "alexnet":         "AlexNet (Paper 1)",
    "mresnet50":       "MResNet-50 (Paper 2)",
    "vit":             "NoisyViT (Paper 3)",
    "v1_resnet50":     "V1: ResNet-50",
    "v2_swish":        "V2: + Swish",
    "v3_transformer":  "V3: + Transformer",
    "v4_fixed_noise":  "V4: + Fixed noise",
    "v5_proposed":     "V5: Proposed (Adaptive)",
}
COLORS = {
    "alexnet":         "#4C72B0",
    "mresnet50":       "#DD8452",
    "vit":             "#55A868",
    "v1_resnet50":     "#C44E52",
    "v2_swish":        "#8172B3",
    "v3_transformer":  "#937860",
    "v4_fixed_noise":  "#DA8BC3",
    "v5_proposed":     "#2C3E50",
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main(out_root: str = "./outputs"):
    out_root = Path(out_root)
    log_dir = out_root / "logs"
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    histories, evals = {}, {}
    for m in ALL_MODELS:
        log_p  = log_dir / f"{m}_log.json"
        eval_p = log_dir / f"{m}_eval.json"
        if log_p.exists():
            histories[m] = load_json(log_p)
        if eval_p.exists():
            evals[m] = load_json(eval_p)

    # ============================================================
    # Figure 1: Loss curves (all models that have histories)
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for m, hist in histories.items():
        ep = [r["epoch"] for r in hist]
        axes[0].plot(ep, [r["train_loss"] for r in hist],
                     color=COLORS[m], label=DISPLAY[m], linewidth=1.8)
        axes[1].plot(ep, [r["val_loss"] for r in hist],
                     color=COLORS[m], label=DISPLAY[m], linewidth=1.8)
    axes[0].set_title("Training loss"); axes[1].set_title("Validation loss")
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Figure 1.  Loss curves on Food-101 subset")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_loss_curves.png")
    fig.savefig(fig_dir / "fig1_loss_curves.pdf")
    plt.close(fig)

    # ============================================================
    # Figure 2: Accuracy curves
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for m, hist in histories.items():
        ep = [r["epoch"] for r in hist]
        axes[0].plot(ep, [r["train_top1"]*100 for r in hist],
                     color=COLORS[m], label=DISPLAY[m], linewidth=1.8)
        axes[1].plot(ep, [r["val_top1"]*100 for r in hist],
                     color=COLORS[m], label=DISPLAY[m], linewidth=1.8)
    axes[0].set_title("Training top-1 accuracy")
    axes[1].set_title("Validation top-1 accuracy")
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.set_ylabel("Top-1 accuracy (%)")
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Figure 2.  Top-1 accuracy curves on Food-101 subset")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_accuracy_curves.png")
    fig.savefig(fig_dir / "fig2_accuracy_curves.pdf")
    plt.close(fig)

    # ============================================================
    # Figure 3: Proposed (V5) vs three external baselines
    # ============================================================
    keep = [m for m in BASELINES + ["v5_proposed"] if m in evals]
    if keep:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        names = [DISPLAY[m] for m in keep]
        top1  = [evals[m]["top1"]*100 for m in keep]
        top5  = [evals[m]["top5"]*100 for m in keep]
        x = np.arange(len(names)); w = 0.36
        b1 = ax.bar(x - w/2, top1, w, label="Top-1", color="#4C72B0")
        b5 = ax.bar(x + w/2, top5, w, label="Top-5", color="#55A868")
        for b, v in zip(b1, top1):
            ax.text(b.get_x()+b.get_width()/2, v+0.5, f"{v:.1f}",
                    ha="center", fontsize=9)
        for b, v in zip(b5, top5):
            ax.text(b.get_x()+b.get_width()/2, v+0.5, f"{v:.1f}",
                    ha="center", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=12)
        ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 105); ax.legend()
        ax.set_title("Figure 3.  Proposed model vs three external baselines")
        fig.tight_layout()
        fig.savefig(fig_dir / "fig3_baselines_vs_proposed.png")
        fig.savefig(fig_dir / "fig3_baselines_vs_proposed.pdf")
        plt.close(fig)

    # ============================================================
    # Figure 4: Ablation — top-1 across V1→V5 with delta annotations
    # ============================================================
    abl = [m for m in ABLATION if m in evals]
    if len(abl) >= 2:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        names = [DISPLAY[m] for m in abl]
        top1  = [evals[m]["top1"]*100 for m in abl]
        bars = ax.bar(names, top1, color=[COLORS[m] for m in abl])
        for i, (b, v) in enumerate(zip(bars, top1)):
            ax.text(b.get_x()+b.get_width()/2, v+0.4, f"{v:.2f}",
                    ha="center", fontsize=10, fontweight="bold")
            if i > 0:
                delta = top1[i] - top1[i-1]
                arrow = "▲" if delta >= 0 else "▼"
                color = "green" if delta >= 0 else "red"
                ax.text(b.get_x()+b.get_width()/2, v-3,
                        f"{arrow} {abs(delta):.2f}",
                        ha="center", fontsize=9, color=color)
        ax.set_ylabel("Top-1 accuracy (%)")
        ax.set_ylim(0, max(top1) * 1.15)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=18, ha="right")
        ax.set_title("Figure 4.  Ablation study — top-1 accuracy "
                     "with marginal gain of each component")
        fig.tight_layout()
        fig.savefig(fig_dir / "fig4_ablation_top1.png")
        fig.savefig(fig_dir / "fig4_ablation_top1.pdf")
        plt.close(fig)

    # ============================================================
    # Figure 5: Ablation — F1, Precision, Recall across V1→V5
    # ============================================================
    if len(abl) >= 2:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        names = [DISPLAY[m] for m in abl]
        f1   = [evals[m]["macro_f1"]*100 for m in abl]
        prec = [evals[m]["macro_precision"]*100 for m in abl]
        rec  = [evals[m]["macro_recall"]*100 for m in abl]
        x = np.arange(len(names)); w = 0.27
        ax.bar(x - w, f1,   w, label="Macro F1",        color="#4C72B0")
        ax.bar(x,     prec, w, label="Macro Precision", color="#DD8452")
        ax.bar(x + w, rec,  w, label="Macro Recall",    color="#55A868")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=18, ha="right")
        ax.set_ylabel("Score (%)"); ax.set_ylim(0, 105); ax.legend()
        ax.set_title("Figure 5.  Ablation study — F1 / Precision / Recall")
        fig.tight_layout()
        fig.savefig(fig_dir / "fig5_ablation_metrics.png")
        fig.savefig(fig_dir / "fig5_ablation_metrics.pdf")
        plt.close(fig)

    # ============================================================
    # Figure 6: Parameters vs accuracy (all models)
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in ALL_MODELS:
        if m not in evals: continue
        marker = "o" if m in ABLATION else "s"
        size = 250 if m == "v5_proposed" else 160
        ax.scatter(evals[m]["params_M"], evals[m]["top1"]*100,
                   s=size, color=COLORS[m], label=DISPLAY[m],
                   edgecolor="black", marker=marker, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Number of parameters (M, log scale)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Figure 6.  Parameter efficiency: params vs accuracy")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig6_params_vs_acc.png")
    fig.savefig(fig_dir / "fig6_params_vs_acc.pdf")
    plt.close(fig)

    # ============================================================
    # Figure 7: Inference time comparison
    # ============================================================
    keep = [m for m in ALL_MODELS if m in evals]
    if keep:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        names = [DISPLAY[m] for m in keep]
        times = [evals[m]["inference_ms_per_image"] for m in keep]
        bars = ax.bar(names, times, color=[COLORS[m] for m in keep])
        for b, v in zip(bars, times):
            ax.text(b.get_x()+b.get_width()/2, v + max(times)*0.01,
                    f"{v:.2f} ms", ha="center", fontsize=9)
        ax.set_ylabel("Inference time per image (ms, batch=1)")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_title("Figure 7.  Inference latency comparison")
        fig.tight_layout()
        fig.savefig(fig_dir / "fig7_inference_time.png")
        fig.savefig(fig_dir / "fig7_inference_time.pdf")
        plt.close(fig)

    # ============================================================
    # Figure 8: Confusion matrices (one per model)
    # ============================================================
    for m in ALL_MODELS:
        cm_p = log_dir / f"{m}_confusion.npy"
        if not cm_p.exists() or m not in evals:
            continue
        cm = np.load(cm_p)
        class_names = evals[m]["class_names"]
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(cm_norm, cmap="Blues", vmin=0, vmax=1,
                    xticklabels=class_names, yticklabels=class_names, ax=ax,
                    cbar_kws={"label": "Fraction"})
        ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
        ax.set_title(f"Figure 8.  Confusion matrix — {DISPLAY[m]}")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)
        fig.tight_layout()
        fig.savefig(fig_dir / f"fig8_confusion_{m}.png")
        fig.savefig(fig_dir / f"fig8_confusion_{m}.pdf")
        plt.close(fig)

    # ============================================================
    # Figure 9: Per-class top-1 accuracy (3 baselines + V5 proposed)
    # ============================================================
    sel = [m for m in (BASELINES + ["v5_proposed"]) if m in evals]
    sel = [m for m in sel if (log_dir / f"{m}_per_class.npy").exists()]
    if sel:
        all_names = evals[sel[0]]["class_names"]
        fig, ax = plt.subplots(figsize=(12, 5))
        width = 0.85 / max(len(sel), 1)
        for i, m in enumerate(sel):
            per = np.load(log_dir / f"{m}_per_class.npy") * 100
            x = np.arange(len(per)) + (i - (len(sel)-1)/2) * width
            ax.bar(x, per, width, color=COLORS[m], label=DISPLAY[m])
        ax.set_xticks(np.arange(len(all_names)))
        ax.set_xticklabels(all_names, rotation=60, ha="right")
        ax.set_ylabel("Per-class top-1 accuracy (%)")
        ax.set_ylim(0, 105); ax.legend(fontsize=9)
        ax.set_title("Figure 9.  Per-class top-1 accuracy: baselines vs proposed")
        fig.tight_layout()
        fig.savefig(fig_dir / "fig9_per_class_acc.png")
        fig.savefig(fig_dir / "fig9_per_class_acc.pdf")
        plt.close(fig)

    # ============================================================
    # Figure 10: Radar of 5 metrics for top models
    # ============================================================
    radar_models = [m for m in (BASELINES + ["v5_proposed"]) if m in evals]
    if radar_models:
        labels = ["Top-1", "Top-5", "F1", "Precision", "Recall"]
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
        for m in radar_models:
            e = evals[m]
            vals = [e["top1"]*100, e["top5"]*100, e["macro_f1"]*100,
                    e["macro_precision"]*100, e["macro_recall"]*100]
            vals += vals[:1]
            ax.plot(angles, vals, label=DISPLAY[m],
                    color=COLORS[m], linewidth=2)
            ax.fill(angles, vals, color=COLORS[m], alpha=0.10)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
        ax.set_ylim(0, 100)
        ax.set_title("Figure 10.  Five-metric radar: baselines vs proposed", pad=22)
        ax.legend(loc="lower right", bbox_to_anchor=(1.25, 0.0), fontsize=9)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig10_metric_radar.png")
        fig.savefig(fig_dir / "fig10_metric_radar.pdf")
        plt.close(fig)

    # ============================================================
    # CSV tables
    # ============================================================
    rows_all, rows_abl = [], []
    for m in ALL_MODELS:
        if m not in evals: continue
        e = evals[m]
        row = {
            "Model": DISPLAY[m],
            "Params (M)":            round(e["params_M"], 2),
            "Top-1 (%)":             round(e["top1"]*100, 2),
            "Top-5 (%)":             round(e["top5"]*100, 2),
            "Macro F1 (%)":          round(e["macro_f1"]*100, 2),
            "Macro Precision (%)":   round(e["macro_precision"]*100, 2),
            "Macro Recall (%)":      round(e["macro_recall"]*100, 2),
            "Inference (ms/image)":  round(e["inference_ms_per_image"], 2),
        }
        rows_all.append(row)
        if m in ABLATION:
            rows_abl.append(row)

    pd.DataFrame(rows_all).to_csv(fig_dir / "summary_table_all.csv", index=False)
    df_abl = pd.DataFrame(rows_abl)
    if len(df_abl):
        df_abl["Δ Top-1 vs prev (%)"] = df_abl["Top-1 (%)"].diff().round(2)
        df_abl["Δ Top-1 vs V1 (%)"]   = (df_abl["Top-1 (%)"] - df_abl["Top-1 (%)"].iloc[0]).round(2)
        df_abl.to_csv(fig_dir / "ablation_table.csv", index=False)

    print("\n=== Summary table (all 8 models) ===")
    if rows_all:
        print(pd.DataFrame(rows_all).to_string(index=False))
    print("\n=== Ablation table (V1-V5) ===")
    if len(df_abl):
        print(df_abl.to_string(index=False))
    print(f"\nAll figures saved to {fig_dir.resolve()}")


if __name__ == "__main__":
    main()
