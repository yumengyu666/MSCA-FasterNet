"""Multi-seed experiment runner with statistical significance tests.

Runs each experiment configuration with multiple random seeds (default: 3),
then performs pairwise statistical tests (paired t-test, Wilcoxon signed-rank)
to determine whether performance differences are significant.

Establishes statistical confidence in results beyond a single random seed.

Usage:
    # Run all experiments with 3 seeds:
    python scripts/run_multiseed.py --dataset ip102 --seeds 42 123 456

    # Analyze existing results:
    python scripts/run_multiseed.py --analyze-only --results-dir checkpoints/multiseed

Output:
    results/multiseed/
        summary.csv            - All results across seeds
        statistical_tests.csv  - Pairwise significance tests
        significance_table.png - Publication-ready comparison table with significance markers
"""

import os
import sys
import argparse
import json
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# Experiment configurations
# ============================================================

# Ablation experiments
ABLATION_CONFIGS = [
    {"model": "baseline", "label": "Baseline (FasterNet-T0)"},
    {"model": "msca", "label": "+ MSCA"},
    {"model": "fusion", "label": "+ Fusion"},
    {"model": "full", "label": "+ MSCA + Fusion (Full)"},
]

# Attention method comparison (for #7)
ATTENTION_CONFIGS = [
    {"model": "attention_se", "label": "FasterNet + SE"},
    {"model": "attention_cbam", "label": "FasterNet + CBAM"},
    {"model": "attention_eca", "label": "FasterNet + ECA"},
    {"model": "attention_sk", "label": "FasterNet + SK-Net"},
    {"model": "full", "label": "MSCA-FasterNet (Ours)"},
]

# Lightweight model comparison
COMPARISON_CONFIGS = [
    {"model": "mobilenetv3_small_100", "label": "MobileNetV3-Small", "is_comparison": True},
    {"model": "shufflenetv2_x0.5", "label": "ShuffleNetV2-x0.5", "is_comparison": True},
    {"model": "ghostnetv2_100", "label": "GhostNetV2", "is_comparison": True},
    {"model": "efficientnet_lite0", "label": "EfficientNet-Lite0", "is_comparison": True},
]

ALL_CONFIGS = ABLATION_CONFIGS + ATTENTION_CONFIGS + COMPARISON_CONFIGS


def run_single_experiment(
    model_name: str,
    dataset: str,
    seed: int,
    epochs: int = 150,
    gpu: str = "0",
    output_dir: str = "checkpoints/multiseed",
    data_dir: str = None,
    is_comparison: bool = False,
) -> str:
    """Run a single training experiment and return the results directory path."""
    run_name = f"{dataset}_{model_name}_seed{seed}"
    run_output = os.path.join(output_dir, run_name)

    # Check if already completed
    results_file = os.path.join(run_output, "results.json")
    if os.path.exists(results_file):
        print(f"  [SKIP] {run_name} already completed")
        return run_output

    print(f"  [RUN] {run_name} @ {datetime.now().strftime('%H:%M:%S')}")

    if is_comparison:
        cmd = [
            sys.executable, "scripts/train_comparison.py",
            "--model", model_name,
            "--dataset", dataset,
            "--seed", str(seed),
            "--epochs", str(epochs),
            "--gpu", gpu,
            "--output-dir", run_output,
        ]
    else:
        cmd = [
            sys.executable, "scripts/train.py",
            "--model", model_name,
            "--dataset", dataset,
            "--seed", str(seed),
            "--epochs", str(epochs),
            "--gpu", gpu,
            "--output-dir", run_output,
        ]

    if data_dir:
        cmd.extend(["--data-dir", data_dir])

    result = subprocess.run(cmd, cwd=os.getcwd())
    if result.returncode != 0:
        print(f"  [ERROR] {run_name} failed with return code {result.returncode}")

    return run_output


def collect_results(output_dir: str, configs: list, seeds: list, dataset: str) -> List[Dict]:
    """Collect results from all completed experiments."""
    all_results = []

    for config in configs:
        model_name = config["model"]
        label = config["label"]
        is_comp = config.get("is_comparison", False)

        for seed in seeds:
            run_name = f"{dataset}_{model_name}_seed{seed}"
            results_file = os.path.join(output_dir, run_name, "results.json")

            if not os.path.exists(results_file):
                # Try alternative path (comparison models save in different structure)
                alt_dir = os.path.join(output_dir, run_name, model_name)
                alt_file = os.path.join(alt_dir, "results.json")
                if os.path.exists(alt_file):
                    results_file = alt_file
                else:
                    print(f"  [MISSING] {results_file}")
                    continue

            with open(results_file) as f:
                result = json.load(f)

            result["model_label"] = label
            result["model_name"] = model_name
            result["seed"] = seed
            result["is_comparison"] = is_comp
            all_results.append(result)

    return all_results


def statistical_tests(
    results: List[Dict],
    metric: str = "test_acc1",
    baseline_label: str = "Baseline (FasterNet-T0)",
) -> Tuple[List[Dict], Dict]:
    """Perform pairwise statistical significance tests.

    For each model variant vs baseline:
        1. Paired t-test (if >= 3 seeds)
        2. Wilcoxon signed-rank test (non-parametric, if >= 3 seeds)

    Returns:
        test_results: List of dicts with test statistics
        summary: Dict with per-model mean, std, and significance markers
    """
    try:
        from scipy import stats
    except ImportError:
        print("WARNING: scipy not installed. Install with: pip install scipy")
        print("Running without statistical tests - will only report mean ± std")
        stats = None

    # Group results by model
    model_scores = {}
    for r in results:
        label = r["model_label"]
        if metric not in r:
            continue
        if label not in model_scores:
            model_scores[label] = []
        model_scores[label].append(r[metric])

    # Compute summary statistics
    summary = {}
    for label, scores in model_scores.items():
        scores = np.array(scores)
        summary[label] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "n": len(scores),
            "scores": scores.tolist(),
        }

    # Pairwise tests against baseline
    test_results = []
    baseline_scores = model_scores.get(baseline_label, [])

    if len(baseline_scores) < 2:
        print(f"  [WARN] Baseline has {len(baseline_scores)} seeds, cannot perform tests")
        return test_results, summary

    baseline_arr = np.array(baseline_scores)

    for label, scores in model_scores.items():
        if label == baseline_label:
            continue

        other_arr = np.array(scores)
        if len(other_arr) < 2:
            continue

        # Ensure same length (paired test)
        min_len = min(len(baseline_arr), len(other_arr))
        b = baseline_arr[:min_len]
        o = other_arr[:min_len]

        result = {
            "model": label,
            "baseline_mean": float(np.mean(b)),
            "model_mean": float(np.mean(o)),
            "diff": float(np.mean(o) - np.mean(b)),
        }

        if stats is not None and min_len >= 3:
            # Paired t-test
            t_stat, t_pval = stats.ttest_rel(o, b)
            result["t_statistic"] = float(t_stat)
            result["t_pvalue"] = float(t_pval)

            # Wilcoxon signed-rank test (non-parametric)
            try:
                w_stat, w_pval = stats.wilcoxon(o, b)
                result["wilcoxon_statistic"] = float(w_stat)
                result["wilcoxon_pvalue"] = float(w_pval)
            except ValueError:
                # Wilcoxon requires non-zero differences
                result["wilcoxon_statistic"] = None
                result["wilcoxon_pvalue"] = None

            # Significance markers
            p = min(result["t_pvalue"], result.get("wilcoxon_pvalue", 1.0) or 1.0)
            if p < 0.001:
                result["significance"] = "***"
            elif p < 0.01:
                result["significance"] = "**"
            elif p < 0.05:
                result["significance"] = "*"
            else:
                result["significance"] = "n.s."
        else:
            result["significance"] = "N/A"

        test_results.append(result)

    return test_results, summary


def generate_significance_table(
    summary: Dict,
    test_results: List[Dict],
    save_path: str = "results/multiseed/significance_table.png",
    metric_name: str = "Top-1 Acc (%)",
):
    """Generate publication-ready comparison table with significance markers.

    Format: Model | Mean ± Std | Δ vs Baseline | Significance
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Prepare table data
    rows = []
    baseline_mean = None

    for label, stats in summary.items():
        mean = stats["mean"]
        std = stats["std"]
        if baseline_mean is None:
            baseline_mean = mean

        delta = mean - baseline_mean
        sig = ""
        for tr in test_results:
            if tr["model"] == label:
                sig = tr.get("significance", "")
                break

        rows.append((label, mean, std, delta, sig))

    # Sort by mean (descending)
    rows.sort(key=lambda x: -x[1])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(3, len(rows) * 0.5 + 1.5)))
    ax.axis("off")

    col_labels = ["Model", f"Mean {metric_name}", "± Std", "Δ vs Baseline", "Sig."]
    cell_text = []
    cell_colors = []

    for label, mean, std, delta, sig in rows:
        cell_text.append([
            label,
            f"{mean:.2f}",
            f"{std:.2f}",
            f"{delta:+.2f}" if delta != 0 else "—",
            sig,
        ])
        # Highlight best result
        if mean == max(r[1] for r in rows):
            cell_colors.append(["#C8E6C9"] * 5)
        else:
            cell_colors.append(["white"] * 5)

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=["#E3F2FD"] * 5,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Bold header
    for j in range(5):
        table[0, j].set_text_props(fontweight="bold")

    ax.set_title(f"Multi-Seed Experiment Results (significance: * p<0.05, ** p<0.01, *** p<0.001)",
                 fontsize=12, fontweight="bold", pad=20)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Significance table saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Experiment Runner")

    parser.add_argument("--dataset", type=str, default="ip102",
                        choices=["ip102", "plantvillage"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                        help="Random seeds for repeated experiments")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="checkpoints/multiseed")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Override results directory for --analyze-only")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only analyze existing results, don't train")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "ablation", "attention", "comparison"],
                        help="Which experiments to run")
    parser.add_argument("--metric", type=str, default="test_acc1",
                        help="Metric for statistical tests")

    args = parser.parse_args()

    # Select experiment configs
    if args.experiment == "ablation":
        configs = ABLATION_CONFIGS
    elif args.experiment == "attention":
        configs = ATTENTION_CONFIGS
    elif args.experiment == "comparison":
        configs = COMPARISON_CONFIGS
    else:
        configs = ALL_CONFIGS

    results_dir = args.results_dir or args.output_dir

    if not args.analyze_only:
        # === Run all experiments ===
        print(f"\n{'='*60}")
        print(f"Multi-Seed Experiment: {args.dataset}")
        print(f"Seeds: {args.seeds}")
        print(f"Total runs: {len(configs) * len(args.seeds)}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for config in configs:
            model_name = config["model"]
            is_comp = config.get("is_comparison", False)
            for seed in args.seeds:
                run_single_experiment(
                    model_name=model_name,
                    dataset=args.dataset,
                    seed=seed,
                    epochs=args.epochs,
                    gpu=args.gpu,
                    output_dir=args.output_dir,
                    data_dir=args.data_dir,
                    is_comparison=is_comp,
                )

        elapsed = time.time() - start_time
        print(f"\nTotal training time: {elapsed/3600:.1f} hours")

    # === Analyze results ===
    print(f"\n{'='*60}")
    print("Analyzing results...")
    print(f"{'='*60}\n")

    results = collect_results(results_dir, configs, args.seeds, args.dataset)

    if not results:
        print("No results found! Run training first.")
        return

    # Statistical tests
    baseline_label = "Baseline (FasterNet-T0)"
    test_results, summary = statistical_tests(results, args.metric, baseline_label)

    # Print summary
    print(f"\n{'Model':<35} {'Mean':>8} {'± Std':>8} {'Sig.':>6}")
    print("-" * 60)
    for label, s in sorted(summary.items(), key=lambda x: -x[1]["mean"]):
        sig = ""
        for tr in test_results:
            if tr["model"] == label:
                sig = tr.get("significance", "")
        print(f"{label:<35} {s['mean']:>8.2f} {s['std']:>8.2f} {sig:>6}")

    # Save results
    output_results_dir = os.path.join(results_dir, "analysis")
    os.makedirs(output_results_dir, exist_ok=True)

    # Save summary CSV
    import csv
    with open(os.path.join(output_results_dir, "summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "mean", "std", "n", "seeds"])
        for label, s in summary.items():
            writer.writerow([label, f"{s['mean']:.4f}", f"{s['std']:.4f}", s["n"], s["scores"]])

    # Save test results
    with open(os.path.join(output_results_dir, "statistical_tests.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "baseline_mean", "model_mean", "diff",
                         "t_statistic", "t_pvalue", "wilcoxon_statistic", "wilcoxon_pvalue", "significance"])
        for tr in test_results:
            writer.writerow([
                tr["model"], f"{tr['baseline_mean']:.4f}", f"{tr['model_mean']:.4f}",
                f"{tr['diff']:.4f}",
                tr.get("t_statistic", ""), tr.get("t_pvalue", ""),
                tr.get("wilcoxon_statistic", ""), tr.get("wilcoxon_pvalue", ""),
                tr.get("significance", ""),
            ])

    # Generate table
    generate_significance_table(
        summary, test_results,
        save_path=os.path.join(output_results_dir, "significance_table.png"),
        metric_name="Top-1 Acc (%)" if args.metric == "test_acc1" else args.metric,
    )

    # Save full results as JSON
    with open(os.path.join(output_results_dir, "full_results.json"), "w") as f:
        json.dump({
            "summary": {k: {kk: vv for kk, vv in v.items() if kk != "scores"}
                       for k, v in summary.items()},
            "statistical_tests": test_results,
            "config": {
                "dataset": args.dataset,
                "seeds": args.seeds,
                "metric": args.metric,
            },
        }, f, indent=2)

    print(f"\nResults saved to {output_results_dir}/")


if __name__ == "__main__":
    main()
