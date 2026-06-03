import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from torch_utils_backbone import BACKBONE_CHOICES, BACKBONE_MODEL_LABELS, normalize_backbone_name


METRIC_KEYS = (
    "accuracy",
    "macro_recall",
    "macro_f1",
    "macro_auc",
    "weighted_f1",
)
TEST_SPLITS = {
    "offsite": "odir_4class_offsite",
    "onsite": "odir_4class_onsite",
}


def parse_args():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Retrain a backbone on ODIR-derived four-class train/val splits and evaluate on "
            "ODIR Off-site and On-site test splits."
        )
    )
    parser.add_argument(
        "--split-root",
        default=str(project_dir / "result" / "external_splits"),
        help="Directory containing odir_4class_trainval/offsite/onsite split folders.",
    )
    parser.add_argument(
        "--output-root",
        default=str(project_dir / "result" / "odir_4class_retraining" / "ours_ska_gpdcnet"),
        help="Directory for ODIR retraining checkpoints, evaluations, and summary tables.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[110, 220, 900, 1100, 1110])
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
    parser.add_argument(
        "--backbone",
        default="resnet34_ska_gpdc_c_v15",
        choices=BACKBONE_CHOICES,
        help="Backbone/method to retrain on ODIR.",
    )
    parser.add_argument("--model-label", default=None, help="Display name for summary tables.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip a seed if best_model.pth already exists.")
    parser.add_argument("--only-summary", action="store_true", help="Only aggregate existing evaluation outputs.")
    parser.add_argument(
        "--skip-split-build",
        action="store_true",
        help="Do not regenerate ODIR split files before training.",
    )
    return parser.parse_args()


def run_command(command, cwd: Path):
    print(f"Running: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_report_row(report_path: Path, label: str):
    with report_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        label_column = reader.fieldnames[0]
        for row in reader:
            if row[label_column] == label:
                return row
    return None


def load_predictions(predictions_path: Path):
    with predictions_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def infer_class_names(prediction_rows):
    if not prediction_rows:
        return []
    return [column[len("prob_") :] for column in prediction_rows[0] if column.startswith("prob_")]


def compute_specificity(prediction_rows, class_name: str):
    true_positive_class = [row["class_name"] == class_name for row in prediction_rows]
    pred_positive_class = [row["pred_class_name"] == class_name for row in prediction_rows]
    false_positive = sum((not true_value) and pred_value for true_value, pred_value in zip(true_positive_class, pred_positive_class))
    true_negative = sum((not true_value) and (not pred_value) for true_value, pred_value in zip(true_positive_class, pred_positive_class))
    denominator = true_negative + false_positive
    return true_negative / denominator if denominator else 0.0


def compute_macro_specificity(prediction_rows):
    class_names = infer_class_names(prediction_rows)
    values = [compute_specificity(prediction_rows, class_name) for class_name in class_names]
    return sum(values) / len(values) if values else 0.0


def format_mean_std(values):
    mean = sum(values) / len(values)
    std = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
    return mean, std, f"{mean:.4f} +/- {std:.4f}"


def write_markdown_table(path: Path, rows):
    columns = list(rows[0]) if rows else []
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_split_rows(output_root: Path, split_key: str, seeds):
    rows = []
    for seed in seeds:
        eval_dir = output_root / f"seed_{seed}" / split_key
        metrics_path = eval_dir / "metrics.json"
        report_path = eval_dir / "classification_report.csv"
        predictions_path = eval_dir / "predictions.csv"
        if not (metrics_path.exists() and report_path.exists() and predictions_path.exists()):
            continue

        metrics = read_json(metrics_path)
        glaucoma_row = load_report_row(report_path, "glaucoma")
        prediction_rows = load_predictions(predictions_path)
        rows.append(
            {
                "seed": seed,
                "accuracy": float(metrics["accuracy"]),
                "macro_recall": float(metrics["macro_recall"]),
                "macro_f1": float(metrics["macro_f1"]),
                "macro_auc": float(metrics["macro_auc"]),
                "glaucoma_recall": float(glaucoma_row["recall"]) if glaucoma_row else 0.0,
                "macro_specificity": compute_macro_specificity(prediction_rows),
                "weighted_f1": float(metrics["weighted_f1"]),
            }
        )
    return rows


def save_summary(output_root: Path, model_label: str, seeds):
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    all_payload = {"model": model_label, "splits": {}}
    table_rows = []

    metric_order = (
        "accuracy",
        "macro_recall",
        "macro_f1",
        "macro_auc",
        "glaucoma_recall",
        "macro_specificity",
        "weighted_f1",
    )
    display_names = {
        "accuracy": "Acc",
        "macro_recall": "Macro Recall",
        "macro_f1": "Macro F1",
        "macro_auc": "Macro AUC",
        "glaucoma_recall": "Glaucoma Recall",
        "macro_specificity": "Macro Specificity",
        "weighted_f1": "Weighted F1",
    }

    for split_key in TEST_SPLITS:
        split_rows = collect_split_rows(output_root, split_key, seeds)
        if not split_rows:
            continue

        seed_csv_path = summary_dir / f"{split_key}_seed_metrics.csv"
        with seed_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(split_rows[0]))
            writer.writeheader()
            writer.writerows(split_rows)

        split_summary = {}
        table_row = {"Model": model_label, "Test Split": split_key}
        for metric in metric_order:
            values = [row[metric] for row in split_rows]
            mean, std, formatted = format_mean_std(values)
            split_summary[metric] = {"mean": mean, "std": std, "formatted": formatted}
            table_row[display_names[metric]] = formatted
        table_rows.append(table_row)
        all_payload["splits"][split_key] = {
            "seed_metrics_csv": str(seed_csv_path),
            "metrics": split_summary,
        }

    if not table_rows:
        raise FileNotFoundError(f"No evaluation outputs found under {output_root}")

    table_csv_path = summary_dir / "odir_4class_retraining_summary.csv"
    with table_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    write_markdown_table(summary_dir / "odir_4class_retraining_summary.md", table_rows)
    (summary_dir / "odir_4class_retraining_summary.json").write_text(
        json.dumps(all_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved ODIR retraining summary: {table_csv_path}")


def main():
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    split_root = Path(args.split_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    canonical_backbone = normalize_backbone_name(args.backbone)
    model_label = args.model_label or BACKBONE_MODEL_LABELS[canonical_backbone]

    if not args.skip_split_build and not args.only_summary:
        run_command([sys.executable, "build_odir_4class_splits.py"], cwd=project_dir)

    trainval_split = split_root / "odir_4class_trainval"
    if not trainval_split.exists():
        raise FileNotFoundError(f"Missing ODIR train/val split: {trainval_split}")
    for split_name in TEST_SPLITS.values():
        if not (split_root / split_name).exists():
            raise FileNotFoundError(f"Missing ODIR test split: {split_root / split_name}")

    if not args.only_summary:
        for seed in args.seeds:
            seed_dir = output_root / f"seed_{seed}"
            checkpoint_dir = seed_dir / "checkpoint"
            checkpoint_path = checkpoint_dir / "best_model.pth"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            if not (args.skip_existing and checkpoint_path.exists()):
                run_command(
                    [
                        sys.executable,
                        "train_backbone.py",
                        "--backbone",
                        canonical_backbone,
                        "--split-dir",
                        str(trainval_split),
                        "--epochs",
                        str(args.epochs),
                        "--batch-size",
                        str(args.batch_size),
                        "--img-size",
                        str(args.img_size[0]),
                        str(args.img_size[1]),
                        "--output-dir",
                        str(checkpoint_dir),
                        "--seed",
                        str(seed),
                        "--device",
                        str(args.device),
                        "--num-workers",
                        str(args.num_workers),
                    ],
                    cwd=project_dir,
                )
            else:
                print(f"Skipping existing checkpoint: {checkpoint_path}")

            for split_key, split_dir_name in TEST_SPLITS.items():
                run_command(
                    [
                        sys.executable,
                        "evaluate_backbone.py",
                        "--backbone",
                        canonical_backbone,
                        "--split-dir",
                        str(split_root / split_dir_name),
                        "--model",
                        str(checkpoint_path),
                        "--output-dir",
                        str(seed_dir / split_key),
                        "--batch-size",
                        str(args.batch_size),
                        "--device",
                        str(args.device),
                        "--num-workers",
                        str(args.num_workers),
                    ],
                    cwd=project_dir,
                )

    save_summary(output_root, model_label=model_label, seeds=args.seeds)


if __name__ == "__main__":
    main()
