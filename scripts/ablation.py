"""Ablation and comparison experiment runner.

Runs all experiments in the correct order:
    1. Baseline FasterNet-T0
    2. + MSCA only
    3. + Fusion only
    4. + MSCA + Fusion (full model)
    5. Comparison models (MobileNetV3, ShuffleNetV2, GhostNetV2, EfficientNet-Lite0)

Usage:
    python scripts/ablation.py --dataset ip102 --gpu 0
    python scripts/ablation.py --dataset plantvillage --gpu 0
    python scripts/ablation.py --compare-only --dataset ip102
"""

import os
import sys
import argparse
import json
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    parser = argparse.ArgumentParser(description="Run Ablation & Comparison Experiments")

    parser.add_argument("--dataset", type=str, default="ip102",
                        choices=["ip102", "plantvillage"])
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override default epochs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)

    # Experiment selection
    parser.add_argument("--ablation-only", action="store_true",
                        help="Run only ablation experiments")
    parser.add_argument("--compare-only", action="store_true",
                        help="Run only comparison experiments")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline (if already trained)")

    return parser.parse_args()


# Ablation experiment configurations
ABLATION_CONFIGS = [
    {
        "name": "baseline",
        "model": "baseline",
        "description": "A: FasterNet-T0 原版（基线）",
    },
    {
        "name": "msca_only",
        "model": "msca",
        "description": "B: A + MSCA（验证注意力模块贡献）",
    },
    {
        "name": "fusion_only",
        "model": "fusion",
        "description": "C: A + 跨层融合（验证融合策略贡献）",
    },
    {
        "name": "full_model",
        "model": "full",
        "description": "D: A + MSCA + 融合（完整模型）",
    },
]


def run_training(args, model_name, model_type):
    """Run a single training experiment."""
    epochs = args.epochs
    if epochs is None:
        epochs = 150 if args.dataset == "ip102" else 100

    cmd = [
        sys.executable, "scripts/train.py",
        "--dataset", args.dataset,
        "--model", model_type,
        "--epochs", str(epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--workers", str(args.workers),
        "--gpu", args.gpu,
        "--seed", str(args.seed),
        "--output-dir", args.output_dir,
        "--save-freq", "10",
    ]

    if args.data_dir:
        cmd.extend(["--data-dir", args.data_dir])

    print(f"\n{'='*60}")
    print(f"Running: {model_name} ({model_type})")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=os.getcwd())

    if result.returncode != 0:
        print(f"ERROR: Training {model_name} failed with return code {result.returncode}")
        return False

    return True


def run_comparison_model(args, model_name):
    """Run training for a comparison model (from timm)."""
    epochs = args.epochs
    if epochs is None:
        epochs = 150 if args.dataset == "ip102" else 100

    num_classes = 102 if args.dataset == "ip102" else 38

    cmd = [
        sys.executable, "scripts/train_comparison.py",
        "--dataset", args.dataset,
        "--model", model_name,
        "--num-classes", str(num_classes),
        "--epochs", str(epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--workers", str(args.workers),
        "--gpu", args.gpu,
        "--seed", str(args.seed),
        "--output-dir", os.path.join(args.output_dir, "comparison"),
    ]

    if args.data_dir:
        cmd.extend(["--data-dir", args.data_dir])

    print(f"\n{'='*60}")
    print(f"Running comparison: {model_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=os.getcwd())

    if result.returncode != 0:
        print(f"ERROR: Training {model_name} failed")
        return False

    return True


def collect_results(args):
    """Collect and summarize all experiment results."""
    results = {}

    for config in ABLATION_CONFIGS:
        results_path = os.path.join(
            args.output_dir,
            f"{args.dataset}_{config['model']}",
            "results.json",
        )
        if os.path.exists(results_path):
            with open(results_path) as f:
                results[config["name"]] = json.load(f)

    # Comparison models
    comparison_dir = os.path.join(args.output_dir, "comparison")
    if os.path.exists(comparison_dir):
        for model_dir in os.listdir(comparison_dir):
            results_path = os.path.join(comparison_dir, model_dir, "results.json")
            if os.path.exists(results_path):
                with open(results_path) as f:
                    results[model_dir] = json.load(f)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"EXPERIMENT RESULTS SUMMARY - {args.dataset.upper()}")
    print("=" * 80)

    header = f"{'Model':<30} {'Top-1(%)':<10} {'Top-5(%)':<10} {'Params(M)':<12} {'FLOPs(G)':<10}"
    print(header)
    print("-" * 80)

    for config in ABLATION_CONFIGS:
        if config["name"] in results:
            r = results[config["name"]]
            print(f"{config['description']:<30} "
                  f"{r.get('test_acc1', 0):<10.2f} "
                  f"{r.get('test_acc5', 0):<10.2f} "
                  f"{r.get('total_params_M', 0):<12.2f} "
                  f"{r.get('flops_G', 'N/A'):<10}")

    print("-" * 80)

    # Save summary
    summary_path = os.path.join(args.output_dir, f"summary_{args.dataset}.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Summary saved to {summary_path}")


def main():
    args = parse_args()

    # === Ablation Experiments ===
    if not args.compare_only:
        for config in ABLATION_CONFIGS:
            if args.skip_baseline and config["model"] == "baseline":
                print(f"Skipping baseline (already trained)")
                continue

            success = run_training(args, config["name"], config["model"])
            if not success:
                print(f"Warning: {config['name']} failed. Continuing...")

    # === Comparison Experiments ===
    if not args.ablation_only:
        comparison_models = [
            "mobilenetv3_small_100",
            "shufflenetv2_x0.5",
            "ghostnetv2_100",
            "efficientnet_lite0",
        ]

        for model_name in comparison_models:
            success = run_comparison_model(args, model_name)
            if not success:
                print(f"Warning: {model_name} failed. Continuing...")

    # === Collect Results ===
    collect_results(args)


if __name__ == "__main__":
    main()
