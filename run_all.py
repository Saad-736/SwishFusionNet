"""
run_all.py
----------
Runs the entire experiment with ONE command:

    python run_all.py

What it does (in order):
    1. Quick pipeline smoke test (~2 min, no Food-101 needed)
    2. Trains all 8 models      (~3-5 hours on GPU)
    3. Evaluates all 8 models   (~15 min)
    4. Generates all 10 figures (~30 sec)

Resumable: if a model is already trained (checkpoint exists), it skips training.
If a model is already evaluated (eval JSON exists), it skips evaluation.
So you can stop with Ctrl+C and re-run later without losing progress.

To skip the smoke test:        python run_all.py --skip-smoke
To run only some models:       python run_all.py --models v5_proposed alexnet
To change num_classes/epochs:  python run_all.py --num-classes 20 --epochs 15
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


ALL_MODELS = [
    "alexnet", "mresnet50", "vit",
    "v1_resnet50", "v2_swish", "v3_transformer",
    "v4_fixed_noise", "v5_proposed",
]


def run(cmd, label):
    """Run a subprocess command and stream its output live."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'='*70}\n")
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    print(f"\n[done in {elapsed/60:.1f} min, exit code {result.returncode}]")
    if result.returncode != 0:
        print(f"!!! Command failed: {label}")
        sys.exit(result.returncode)
    return elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=ALL_MODELS,
                   help="Which models to run. Default = all 8.")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--num-classes", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--skip-smoke", action="store_true",
                   help="Skip the smoke test step.")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip training (assumes checkpoints already exist).")
    p.add_argument("--skip-eval", action="store_true",
                   help="Skip evaluation (assumes eval JSONs already exist).")
    p.add_argument("--skip-plot", action="store_true",
                   help="Skip figure generation.")
    args = p.parse_args()

    py = sys.executable   # use the same Python that launched this script
    out_dir = Path("outputs")
    ckpt_dir = out_dir / "checkpoints"
    log_dir  = out_dir / "logs"

    total_start = time.time()

    # ----------------------------------------------------------------------
    # 1. Smoke test
    # ----------------------------------------------------------------------
    if not args.skip_smoke:
        run([py, "smoke_test.py"], "STEP 1 / 4 — Smoke test (synthetic data)")
        # Smoke test pollutes outputs/; clean it before real training
        import shutil
        if out_dir.exists():
            shutil.rmtree(out_dir)
        print("\n[cleaned smoke-test outputs before real training]")

    # ----------------------------------------------------------------------
    # 2. Train all models
    # ----------------------------------------------------------------------
    if not args.skip_train:
        for i, model in enumerate(args.models, 1):
            ckpt = ckpt_dir / f"{model}_best.pt"
            if ckpt.exists():
                print(f"\n[skip training {model} — checkpoint already exists at {ckpt}]")
                continue
            run(
                [py, "src/train.py",
                 "--model", model,
                 "--epochs", str(args.epochs),
                 "--num_classes", str(args.num_classes),
                 "--batch_size", str(args.batch_size)],
                f"STEP 2 / 4 — Train ({i}/{len(args.models)}): {model}"
            )

    # ----------------------------------------------------------------------
    # 3. Evaluate all models
    # ----------------------------------------------------------------------
    if not args.skip_eval:
        for i, model in enumerate(args.models, 1):
            eval_file = log_dir / f"{model}_eval.json"
            if eval_file.exists():
                print(f"\n[skip eval {model} — already evaluated at {eval_file}]")
                continue
            run(
                [py, "src/evaluate.py",
                 "--model", model,
                 "--num_classes", str(args.num_classes)],
                f"STEP 3 / 4 — Evaluate ({i}/{len(args.models)}): {model}"
            )

    # ----------------------------------------------------------------------
    # 4. Generate figures
    # ----------------------------------------------------------------------
    if not args.skip_plot:
        run([py, "src/plot_results.py"],
            "STEP 4 / 4 — Generate all figures and tables")

    # ----------------------------------------------------------------------
    # Done
    # ----------------------------------------------------------------------
    total_min = (time.time() - total_start) / 60
    print(f"\n\n{'='*70}")
    print(f"  ALL DONE in {total_min:.1f} minutes ({total_min/60:.2f} hours)")
    print(f"{'='*70}")
    print(f"\n  Results:")
    print(f"    outputs/figures/   -> 10 figures (PNG + PDF) + 2 CSV tables")
    print(f"    outputs/logs/      -> per-model JSON logs + eval metrics")
    print(f"    outputs/checkpoints/ -> best model weights")
    print(f"\n  Send the entire 'outputs/figures/' and 'outputs/logs/' folders")
    print(f"  back to write the final paper.")


if __name__ == "__main__":
    main()
