import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


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

COMPARISON_SOURCES = (
    {
        "label": "ResNet34",
        "candidates": (
            Path("runs/resnet34_5seed"),
            Path("result/resnet34_5seed"),
        ),
    },
    {
        "label": "ResNet34-C-[V]x15",
        "candidates": (
            Path("result/resnet34_pdc_c_v15_vs_gpdc_c_v15/resnet34_pdc_c_v15_fixed5seed"),
            Path("result/resnet34_pdc_improvement_ablation/resnet34_pdc_c_v15_fixed5seed"),
            Path("result/resnet34_pdc_c_v15_4seed"),
        ),
    },
    {
        "label": "ResNet34-gPDC-C-[V]x15",
        "candidates": (
            Path("result/resnet34_pdc_c_v15_vs_gpdc_c_v15/resnet34_gpdc_c_v15_fixed5seed"),
            Path("result/resnet34_pdc_improvement_ablation/resnet34_gpdc_c_v15_fixed5seed"),
        ),
    },
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run RGB-only SKA-gPDCNet and summarize it with existing baselines.")
    parser.add_argument("--data-dir", default="dataset", help="Dataset root containing one folder per class.")
    parser.add_argument("--output-root", default="result", help="Root directory for generated experiment outputs.")
    parser.add_argument("--experiment-name", default="ska_gpdcnet_no_vessel")
    parser.add_argument("--seeds", nargs="+", type=int, default=[880, 220, 20, 900, 660])
    parser.add_argument("--train-ratio", default=0.7, type=float)
    parser.add_argument("--val-ratio", default=0.15, type=float)
    parser.add_argument("--test-ratio", default=0.15, type=float)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--img-size", nargs=2, type=int, default=[224, 224], metavar=("H", "W"))
    parser.add_argument("--device", default="cuda", help="Device to use: cuda/cpu/auto.")
    parser.add_argument(
        "--num-workers",
        default=0,
        type=int,
        help="DataLoader worker processes. Use 0 on Windows to avoid shared file mapping errors.",
    )
    parser.add_argument("--grouping-mode", choices=("class_base_id", "filepath"), default="class_base_id")
    parser.add_argument("--representative-seed", default=220, type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--only-compare", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only seed 220. This is intended for pipeline validation before the full 5-seed run.",
    )
    return parser.parse_args()


def resolve_path(path_arg: str, project_dir: Path):
    path = Path(path_arg)
    if path.is_absolute():
        return path
    return (project_dir / path).resolve()


def run_command(command, cwd: Path):
    print(f"Running: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


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
        table_path = run_root / "summary" / "table2_overall_5seed.csv"
    if not table_path.exists():
        raise FileNotFoundError(f"Missing paper table under {run_root / 'summary'}")
    dataframe = pd.read_csv(table_path)
    if dataframe.empty:
        raise ValueError(f"Empty paper table: {table_path}")
    row = dataframe.iloc[[0]].copy()
    row.loc[:, "Model"] = expected_label
    return row[PAPER_COLUMNS]


def find_source_row(project_dir: Path, source: dict):
    missing = []
    for candidate in source["candidates"]:
        run_root = resolve_path(str(candidate), project_dir)
        try:
            return read_paper_row(run_root, source["label"]), run_root
        except FileNotFoundError as exc:
            missing.append(str(exc))
    raise FileNotFoundError("\n".join(missing))


def run_ska_experiment(args, project_dir: Path, run_root: Path, seeds):
    if args.skip_existing and (run_root / "summary" / "paper_table.csv").exists():
        print(f"Skipping existing SKA-gPDCNet run: {run_root}")
        return

    representative_seed = args.representative_seed
    if representative_seed not in seeds:
        representative_seed = seeds[0]

    run_command(
        [
            sys.executable,
            "run_paper_experiment_backbone.py",
            "--data-dir",
            str(resolve_path(args.data_dir, project_dir)),
            "--output-root",
            str(run_root),
            "--seeds",
            *[str(seed) for seed in seeds],
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
            "resnet34_ska_gpdc_c_v15",
            "--model-label",
            "SKA-gPDCNet",
            "--representative-seed",
            str(representative_seed),
        ],
        cwd=project_dir,
    )


def save_comparison(project_dir: Path, output_base: Path, ska_run_root: Path):
    rows = []
    source_payload = []
    for source in COMPARISON_SOURCES:
        row, run_root = find_source_row(project_dir, source)
        rows.append(row)
        source_payload.append({"model": source["label"], "run_root": str(run_root)})

    rows.append(read_paper_row(ska_run_root, "SKA-gPDCNet"))
    source_payload.append({"model": "SKA-gPDCNet", "run_root": str(ska_run_root)})

    summary_dir = output_base / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    comparison_df = pd.concat(rows, ignore_index=True)
    for column in ("Params (M)", "FLOPs (G)", "Time (ms/image)"):
        comparison_df[column] = comparison_df[column].apply(lambda value: f"{float(value):.2f}")

    csv_path = summary_dir / "ska_gpdcnet_comparison.csv"
    markdown_path = summary_dir / "ska_gpdcnet_comparison.md"
    json_path = summary_dir / "ska_gpdcnet_comparison.json"

    comparison_df.to_csv(csv_path, index=False, encoding="utf-8")
    markdown_path.write_text(dataframe_to_markdown(comparison_df) + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "sources": source_payload,
                "comparison_csv": str(csv_path),
                "comparison_markdown": str(markdown_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Saved comparison table: {csv_path}")
    print(f"Saved markdown table: {markdown_path}")


def main():
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    output_root = resolve_path(args.output_root, project_dir)
    output_base = output_root / args.experiment_name
    output_base.mkdir(parents=True, exist_ok=True)

    seeds = [220] if args.smoke else args.seeds
    ska_run_root = output_base / ("ska_gpdcnet_smoke_seed220" if args.smoke else "ska_gpdcnet_fixed5seed")

    if not args.only_compare:
        run_ska_experiment(args, project_dir, ska_run_root, seeds)
    save_comparison(project_dir, output_base, ska_run_root)
    print(f"Finished SKA-gPDCNet workflow under {output_base}")


if __name__ == "__main__":
    main()
