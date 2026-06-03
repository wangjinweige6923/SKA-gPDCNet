import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


EXPERIMENTS = (
    {
        "stage": "pdc",
        "backbone": "resnet34_pdc_c_v15",
        "label": "ResNet34-C-[V]x15",
        "output_name": "resnet34_pdc_c_v15_fixed5seed",
        "uses_vessel_prior": False,
    },
    {
        "stage": "gpdc",
        "backbone": "resnet34_gpdc_c_v15",
        "label": "ResNet34-gPDC-C-[V]x15",
        "output_name": "resnet34_gpdc_c_v15_fixed5seed",
        "uses_vessel_prior": False,
    },
    {
        "stage": "struct_prior",
        "backbone": "resnet34_gpdc_c_v15_struct_prior",
        "label": "ResNet34-gPDC-C-[V]x15-StructPrior",
        "output_name": "resnet34_gpdc_c_v15_struct_prior_fixed5seed",
        "uses_vessel_prior": True,
    },
)

PAPER_COLUMNS = [
    "Model",
    "Acc",
    "Macro Recall",
    "Macro F1",
    "Macro AUC",
    "Glaucoma Recall",
    "Macro Specificity",
    "Weighted F1",
    "Params (M)",
    "FLOPs (G)",
    "Time (ms/image)",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ResNet34 PDC -> gPDC -> gPDC+structure-prior improvement ablation."
    )
    parser.add_argument("--data-dir", default="dataset", help="Dataset root containing one folder per class.")
    parser.add_argument("--output-root", default="result", help="Root directory for generated experiment outputs.")
    parser.add_argument(
        "--experiment-name",
        default="resnet34_pdc_improvement_ablation",
        help="Subdirectory created under output-root for this ablation.",
    )
    parser.add_argument(
        "--vessel-prior-root",
        default=None,
        help="Root containing vessel/structure-prior PNGs matching dataset relative paths.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("pdc", "gpdc", "struct_prior"),
        default=["pdc", "gpdc", "struct_prior"],
        help="Ablation stages to run. Use '--stages pdc gpdc' for improved-PDC-only comparison.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[880, 220, 20, 900, 660])
    parser.add_argument("--train-ratio", default=0.7, type=float)
    parser.add_argument("--val-ratio", default=0.15, type=float)
    parser.add_argument("--test-ratio", default=0.15, type=float)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--img-size", nargs=2, type=int, default=[224, 224], metavar=("H", "W"))
    parser.add_argument("--device", default="cuda", help="Device to use: cuda/cpu/auto.")
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--grouping-mode", choices=("class_base_id", "filepath"), default="class_base_id")
    parser.add_argument("--representative-seed", default=220, type=int)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip an experiment when summary/paper_table.csv already exists.",
    )
    return parser.parse_args()


def run_command(command, cwd: Path):
    print(f"Running: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def has_finished_summary(run_root: Path):
    return (run_root / "summary" / "paper_table.csv").exists()


def resolve_path(path_arg: str, project_dir: Path):
    path = Path(path_arg)
    if path.is_absolute():
        return path
    return (project_dir / path).resolve()


def run_experiment(args, project_dir: Path, output_root: Path, experiment: dict):
    run_root = output_root / experiment["output_name"]
    if args.skip_existing and has_finished_summary(run_root):
        print(f"Skipping existing run: {run_root}")
        return run_root

    command = [
        sys.executable,
        "run_paper_experiment_backbone.py",
        "--data-dir",
        str(resolve_path(args.data_dir, project_dir)),
        "--output-root",
        str(run_root.resolve()),
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--train-ratio",
        str(args.train_ratio),
        "--val-ratio",
        str(args.val_ratio),
        "--test-ratio",
        str(args.test_ratio),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--img-size",
        str(args.img_size[0]),
        str(args.img_size[1]),
        "--device",
        str(args.device),
        "--num-workers",
        str(args.num_workers),
        "--grouping-mode",
        str(args.grouping_mode),
        "--backbone",
        experiment["backbone"],
        "--model-label",
        experiment["label"],
        "--representative-seed",
        str(args.representative_seed),
    ]
    if experiment["uses_vessel_prior"]:
        command.extend(["--vessel-prior-root", str(resolve_path(args.vessel_prior_root, project_dir))])

    run_command(command, cwd=project_dir)
    return run_root


def dataframe_to_markdown(dataframe: pd.DataFrame):
    columns = list(dataframe.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in dataframe.iterrows():
        values = ["" if pd.isna(row[column]) else str(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def read_paper_row(run_root: Path, expected_label: str):
    table_path = run_root / "summary" / "paper_table.csv"
    if not table_path.exists():
        raise FileNotFoundError(f"Missing aggregate table: {table_path}")
    dataframe = pd.read_csv(table_path)
    if len(dataframe) != 1:
        raise ValueError(f"Expected exactly one row in {table_path}, got {len(dataframe)}")
    row = dataframe.iloc[[0]].copy()
    row.loc[:, "Model"] = expected_label
    return row[PAPER_COLUMNS]


def parse_mean_std(value: str):
    mean_text, std_text = str(value).replace(" ", "").split("+/-", maxsplit=1)
    return float(mean_text), float(std_text)


def plot_macro_f1(comparison_df: pd.DataFrame, summary_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    means = []
    stds = []
    for value in comparison_df["Macro F1"]:
        mean, std = parse_mean_std(value)
        means.append(mean)
        stds.append(std)

    labels = comparison_df["Model"].tolist()
    x_positions = range(len(labels))
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    bars = ax.bar(
        x_positions,
        means,
        yerr=stds,
        capsize=4,
        color="#4C78A8",
        edgecolor="#1F2937",
        linewidth=0.8,
        error_kw={"elinewidth": 1.0, "ecolor": "#111827"},
    )
    ax.set_title("ResNet34-PDC Improvement Ablation Macro F1")
    ax.set_ylabel("Macro F1 (mean +/- std)")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)

    lower_bound = max(0.0, min(mean - std for mean, std in zip(means, stds)) - 0.01)
    upper_bound = min(1.0, max(mean + std for mean, std in zip(means, stds)) + 0.01)
    ax.set_ylim(lower_bound, upper_bound)

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{mean:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    png_path = summary_dir / "resnet34_pdc_improvement_ablation_macro_f1.png"
    pdf_path = summary_dir / "resnet34_pdc_improvement_ablation_macro_f1.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def save_comparison(output_root: Path, experiments, run_roots):
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for experiment, run_root in zip(experiments, run_roots):
        rows.append(read_paper_row(run_root, experiment["label"]))

    comparison_df = pd.concat(rows, ignore_index=True)
    csv_path = summary_dir / "resnet34_pdc_improvement_ablation_comparison.csv"
    markdown_path = summary_dir / "resnet34_pdc_improvement_ablation_comparison.md"
    json_path = summary_dir / "resnet34_pdc_improvement_ablation_comparison.json"

    comparison_df.to_csv(csv_path, index=False, encoding="utf-8")
    markdown_path.write_text(dataframe_to_markdown(comparison_df) + "\n", encoding="utf-8")
    png_path, pdf_path = plot_macro_f1(comparison_df, summary_dir)

    payload = {
        "experiments": [
            {
                "model": experiment["label"],
                "backbone": experiment["backbone"],
                "run_root": str(run_root),
                "uses_vessel_prior": experiment["uses_vessel_prior"],
            }
            for experiment, run_root in zip(experiments, run_roots)
        ],
        "comparison_csv": str(csv_path),
        "comparison_markdown": str(markdown_path),
        "macro_f1_png": str(png_path),
        "macro_f1_pdf": str(pdf_path),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved comparison table: {csv_path}")
    print(f"Saved markdown table: {markdown_path}")
    print(f"Saved Macro F1 plot: {png_path}")
    print(f"Saved Macro F1 plot: {pdf_path}")


def main():
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    output_base = resolve_path(args.output_root, project_dir) / args.experiment_name
    output_base.mkdir(parents=True, exist_ok=True)

    selected_stages = set(args.stages)
    experiments = [experiment for experiment in EXPERIMENTS if experiment["stage"] in selected_stages]
    needs_prior = any(experiment["uses_vessel_prior"] for experiment in experiments)
    if needs_prior:
        if not args.vessel_prior_root:
            raise ValueError("--vessel-prior-root is required when --stages includes struct_prior")
        vessel_prior_root = resolve_path(args.vessel_prior_root, project_dir)
        if not vessel_prior_root.exists():
            raise FileNotFoundError(f"Missing vessel-prior root: {vessel_prior_root}")

    run_roots = []
    for experiment in experiments:
        run_roots.append(run_experiment(args, project_dir, output_base, experiment))

    save_comparison(output_base, experiments, run_roots)
    print(f"Finished ResNet34-PDC improvement ablation under {output_base}")


if __name__ == "__main__":
    main()
