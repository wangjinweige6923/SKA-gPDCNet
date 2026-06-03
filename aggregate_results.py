import argparse
import json
import math
import os
import time
from pathlib import Path

import pandas as pd


METRICS = [
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "macro_auc",
]

SCI_MAIN_COLUMNS = [
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

CLASS_DISPLAY_NAMES = {
    "cataract": "Cataract",
    "diabetic_retinopathy": "DR",
    "glaucoma": "Glaucoma",
    "normal": "Normal",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed experiment results and export SCI-ready tables.")
    parser.add_argument("--run-root", required=True, help="Directory containing seed_* experiment folders.")
    parser.add_argument(
        "--representative-seed",
        default=40,
        type=int,
        help="Representative seed used for class-wise table and figures.",
    )
    parser.add_argument("--model-label", default="ResNet50", help="Display name used in the SCI main table.")
    parser.add_argument(
        "--profile-device",
        default="auto",
        help="Device for latency profiling. FLOPs are computed on CPU; latency is only measured on CUDA.",
    )
    parser.add_argument("--profile-warmup", default=20, type=int, help="Warmup iterations for latency profiling.")
    parser.add_argument("--profile-runs", default=100, type=int, help="Measured iterations for latency profiling.")
    parser.add_argument(
        "--profile-repeats",
        default=5,
        type=int,
        help="Repeated latency measurements; the median is reported to reduce one-off timing noise.",
    )
    return parser.parse_args()


def parse_seed_value(seed_dir: Path):
    try:
        return int(seed_dir.name.split("_", 1)[1])
    except (IndexError, ValueError):
        return seed_dir.name


def iter_seed_dirs(run_root: Path):
    seed_dirs = [path for path in run_root.glob("seed_*") if path.is_dir()]
    return sorted(seed_dirs, key=parse_seed_value)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_seed_metrics(run_root: Path):
    rows = []
    for seed_dir in iter_seed_dirs(run_root):
        metrics_path = seed_dir / "results" / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = load_json(metrics_path)
        row = {
            "seed": parse_seed_value(seed_dir),
            "seed_dir": Path(os.path.relpath(seed_dir, run_root)).as_posix(),
        }
        row.update({metric: metrics.get(metric) for metric in METRICS})
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No results/metrics.json files found under {run_root}")
    return pd.DataFrame(rows)


def make_summary(seed_df: pd.DataFrame):
    summary_rows = []
    for metric in METRICS:
        values = seed_df[metric].astype(float)
        mean = values.mean()
        std = values.std(ddof=0)
        summary_rows.append(
            {
                "metric": metric,
                "mean": mean,
                "std": std,
                "formatted": f"{mean:.4f}+/-{std:.4f}",
            }
        )
    return pd.DataFrame(summary_rows)


def make_legacy_paper_table(summary_df: pd.DataFrame):
    row = {}
    for _, record in summary_df.iterrows():
        row[record["metric"]] = record["formatted"]
    return pd.DataFrame([row])


def load_classification_report(seed_dir: Path):
    report_path = seed_dir / "results" / "classification_report.csv"
    report_df = pd.read_csv(report_path)
    first_column = report_df.columns[0]
    report_df = report_df.rename(
        columns={
            first_column: "label",
            "f1-score": "f1_score",
        }
    )
    return report_df


def load_predictions(seed_dir: Path):
    predictions_path = seed_dir / "results" / "predictions.csv"
    return pd.read_csv(predictions_path)


def infer_class_names(prediction_df: pd.DataFrame):
    probability_columns = [column for column in prediction_df.columns if column.startswith("prob_")]
    return [column[len("prob_") :] for column in probability_columns]


def compute_specificity(prediction_df: pd.DataFrame, class_name: str):
    true_mask = prediction_df["class_name"] == class_name
    pred_mask = prediction_df["pred_class_name"] == class_name
    fp = int((~true_mask & pred_mask).sum())
    tn = int((~true_mask & ~pred_mask).sum())
    denominator = tn + fp
    return (tn / denominator) if denominator else 0.0


def compute_macro_specificity(prediction_df: pd.DataFrame, class_names):
    values = [compute_specificity(prediction_df, class_name) for class_name in class_names]
    return sum(values) / len(values) if values else 0.0


def collect_seed_sci_metrics(run_root: Path):
    rows = []
    for seed_dir in iter_seed_dirs(run_root):
        metrics_path = seed_dir / "results" / "metrics.json"
        report_path = seed_dir / "results" / "classification_report.csv"
        predictions_path = seed_dir / "results" / "predictions.csv"
        if not (metrics_path.exists() and report_path.exists() and predictions_path.exists()):
            continue

        metrics = load_json(metrics_path)
        report_df = load_classification_report(seed_dir)
        prediction_df = load_predictions(seed_dir)
        class_names = infer_class_names(prediction_df)

        glaucoma_row = report_df[report_df["label"] == "glaucoma"]
        glaucoma_recall = float(glaucoma_row.iloc[0]["recall"]) if not glaucoma_row.empty else None
        macro_specificity = compute_macro_specificity(prediction_df, class_names)

        rows.append(
            {
                "seed": parse_seed_value(seed_dir),
                "seed_dir": Path(os.path.relpath(seed_dir, run_root)).as_posix(),
                "accuracy": float(metrics["accuracy"]),
                "macro_recall": float(metrics["macro_recall"]),
                "macro_f1": float(metrics["macro_f1"]),
                "macro_auc": float(metrics["macro_auc"]),
                "glaucoma_recall": glaucoma_recall,
                "macro_specificity": macro_specificity,
                "weighted_f1": float(metrics["weighted_f1"]),
            }
        )

    if not rows:
        raise FileNotFoundError(f"No SCI-ready seed artifacts found under {run_root}")
    return pd.DataFrame(rows)


def format_mean_std(values: pd.Series):
    values = values.astype(float)
    mean = values.mean()
    std = values.std(ddof=0)
    return f"{mean:.4f} +/- {std:.4f}"


def load_params_m(seed_dir: Path):
    summary_path = seed_dir / "checkpoint" / "model_summary.txt"
    if not summary_path.exists():
        return None
    payload = load_json(summary_path)
    return float(payload["total_params"]) / 1_000_000.0


def format_scalar(value, decimals=2):
    if value is None:
        return "TBD"
    return f"{value:.{decimals}f}"


def resolve_representative_seed_dir(run_root: Path, representative_seed: int):
    seed_dir = run_root / f"seed_{representative_seed}"
    if seed_dir.exists():
        return seed_dir
    raise FileNotFoundError(f"Representative seed directory not found: {seed_dir}")


def load_model_for_profiling(seed_dir: Path):
    try:
        import torch

        from torch_utils import load_checkpoint
    except Exception as exc:
        return None, None, {"profiling_error": f"Failed to import torch stack: {exc}"}

    checkpoint_path = seed_dir / "checkpoint" / "best_model.pth"
    if not checkpoint_path.exists():
        return None, None, {"profiling_error": f"Missing checkpoint: {checkpoint_path}"}

    model, _, image_size, _ = load_checkpoint(checkpoint_path, map_location=torch.device("cpu"))
    model.eval()
    return model, image_size, {}


def get_torchvision_meta_flops_g(model, image_size):
    if tuple(image_size) != (224, 224):
        return None

    backbone_name = getattr(model, "backbone_name", None)
    if backbone_name is None:
        backbone_name = getattr(getattr(model, "backbone", None), "backbone_name", None)
    if backbone_name is None:
        return None

    try:
        from torchvision import models
    except Exception:
        return None

    weights_by_backbone = {
        "convnext_tiny": models.ConvNeXt_Tiny_Weights.DEFAULT,
        "densenet121": models.DenseNet121_Weights.DEFAULT,
        "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT,
        "mobilenet_v3_large": models.MobileNet_V3_Large_Weights.DEFAULT,
        "resnet34": models.ResNet34_Weights.DEFAULT,
        "resnet50": models.ResNet50_Weights.DEFAULT,
        "resnet101": models.ResNet101_Weights.DEFAULT,
        "swin_tiny": models.Swin_T_Weights.DEFAULT,
    }
    weights = weights_by_backbone.get(backbone_name)
    if weights is None:
        return None

    ops_gmac = weights.meta.get("_ops")
    if ops_gmac is None:
        return None
    return float(ops_gmac) * 2.0


def compute_model_flops_g(model, image_size):
    meta_flops_g = get_torchvision_meta_flops_g(model, image_size)
    if meta_flops_g is not None:
        return meta_flops_g

    try:
        import torch
        from torch import nn
    except Exception:
        return None

    total_flops = 0.0
    handles = []

    def conv_hook(module, inputs, output):
        nonlocal total_flops
        if not isinstance(output, torch.Tensor):
            return
        batch_size = output.shape[0]
        out_channels = output.shape[1]
        out_h = output.shape[2]
        out_w = output.shape[3]
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels / module.groups)
        bias_ops = 1 if module.bias is not None else 0
        total_flops += batch_size * out_channels * out_h * out_w * (kernel_ops * 2 + bias_ops)

    def linear_hook(module, inputs, output):
        nonlocal total_flops
        if not isinstance(output, torch.Tensor):
            return
        output_positions = output.numel() // module.out_features
        bias_ops = 1 if module.bias is not None else 0
        total_flops += output_positions * module.out_features * (module.in_features * 2 + bias_ops)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    input_channels = getattr(getattr(model, "backbone", model), "expected_input_channels", None)
    if input_channels is None:
        input_channels = 3
    dummy = torch.randn(1, int(input_channels), image_size[0], image_size[1])
    with torch.no_grad():
        model(dummy)

    for handle in handles:
        handle.remove()

    return total_flops / 1_000_000_000.0


def median(values):
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    if count == 0:
        return None
    midpoint = count // 2
    if count % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def benchmark_latency_ms(model, image_size, device_name, warmup, runs, repeats):
    try:
        import torch

        from torch_utils import resolve_device
    except Exception as exc:
        return None, {"latency_error": f"Failed to import profiling dependencies: {exc}"}

    device = resolve_device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        return None, {"latency_note": "CUDA unavailable; Time (ms/image) left as TBD."}

    model = model.to(device)
    input_channels = getattr(getattr(model, "backbone", model), "expected_input_channels", None)
    if input_channels is None:
        input_channels = 3
    dummy = torch.randn(1, int(input_channels), image_size[0], image_size[1], device=device)

    repeat_times = []
    with torch.no_grad():
        for _ in range(max(1, warmup)):
            model(dummy)
        torch.cuda.synchronize()

        for _ in range(max(1, repeats)):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(max(1, runs)):
                model(dummy)
            end_event.record()
            torch.cuda.synchronize()
            repeat_times.append(start_event.elapsed_time(end_event) / max(1, runs))

    return median(repeat_times), {
        "latency_device": str(device),
        "latency_batch_size": 1,
        "latency_warmup": int(warmup),
        "latency_runs": int(runs),
        "latency_repeats": int(max(1, repeats)),
        "latency_repeat_ms_image": repeat_times,
        "latency_statistic": "median",
    }


def collect_profile_summary(seed_dir: Path, profile_device: str, profile_warmup: int, profile_runs: int, profile_repeats: int):
    params_m = load_params_m(seed_dir)
    model, image_size, notes = load_model_for_profiling(seed_dir)
    flops_g = None
    time_ms = None

    if model is not None and image_size is not None:
        try:
            input_channels = getattr(getattr(model, "backbone", model), "expected_input_channels", 3)
            meta_flops_g = get_torchvision_meta_flops_g(model, image_size)
            if meta_flops_g is not None:
                flops_g = meta_flops_g
                notes["flops_method"] = (
                    "Torchvision weights _ops metadata converted from GMACs to FLOPs "
                    f"on input size 1x{int(input_channels)}x{image_size[0]}x{image_size[1]}."
                )
            else:
                flops_g = compute_model_flops_g(model, image_size)
                notes["flops_method"] = (
                    "Conv2d/Linear forward-hook FLOPs with shape-aware Linear positions "
                    f"on input size 1x{int(input_channels)}x{image_size[0]}x{image_size[1]}."
                )
        except Exception as exc:
            notes["flops_error"] = str(exc)

        try:
            time_ms, latency_notes = benchmark_latency_ms(
                model=model,
                image_size=image_size,
                device_name=profile_device,
                warmup=profile_warmup,
                runs=profile_runs,
                repeats=profile_repeats,
            )
            notes.update(latency_notes)
        except Exception as exc:
            notes["latency_error"] = str(exc)

    return {
        "params_m": params_m,
        "flops_g": flops_g,
        "time_ms_image": time_ms,
        "notes": notes,
    }


def build_sci_main_table(seed_sci_df: pd.DataFrame, model_label: str, profile_summary: dict):
    row = {
        "Model": model_label,
        "Acc": format_mean_std(seed_sci_df["accuracy"]),
        "Macro Recall": format_mean_std(seed_sci_df["macro_recall"]),
        "Macro F1": format_mean_std(seed_sci_df["macro_f1"]),
        "Macro AUC": format_mean_std(seed_sci_df["macro_auc"]),
        "Glaucoma Recall": format_mean_std(seed_sci_df["glaucoma_recall"]),
        "Macro Specificity": format_mean_std(seed_sci_df["macro_specificity"]),
        "Weighted F1": format_mean_std(seed_sci_df["weighted_f1"]),
        "Params (M)": format_scalar(profile_summary.get("params_m"), decimals=2),
        "FLOPs (G)": format_scalar(profile_summary.get("flops_g"), decimals=2),
        "Time (ms/image)": format_scalar(profile_summary.get("time_ms_image"), decimals=2),
    }
    return pd.DataFrame([row], columns=SCI_MAIN_COLUMNS)


def build_classwise_table(seed_dir: Path):
    report_df = load_classification_report(seed_dir)
    prediction_df = load_predictions(seed_dir)
    metrics = load_json(seed_dir / "results" / "metrics.json")
    class_names = infer_class_names(prediction_df)
    report_df = report_df[report_df["label"].isin(class_names)].copy()

    rows = []
    for class_name in class_names:
        report_row = report_df[report_df["label"] == class_name]
        if report_row.empty:
            continue
        report_row = report_row.iloc[0]
        rows.append(
            {
                "Class": CLASS_DISPLAY_NAMES.get(class_name, class_name),
                "Precision": f"{float(report_row['precision']):.4f}",
                "Recall / Sensitivity": f"{float(report_row['recall']):.4f}",
                "F1-score": f"{float(report_row['f1_score']):.4f}",
                "Specificity": f"{compute_specificity(prediction_df, class_name):.4f}",
                "AUC": f"{float(metrics['per_class_auc'][class_name]):.4f}",
                "Support": int(float(report_row["support"])),
            }
        )

    return pd.DataFrame(
        rows,
        columns=["Class", "Precision", "Recall / Sensitivity", "F1-score", "Specificity", "AUC", "Support"],
    )


def save_csv(dataframe: pd.DataFrame, path: Path):
    dataframe.to_csv(path, index=False, encoding="utf-8")


def main():
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    summary_dir = run_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    seed_df = load_seed_metrics(run_root)
    save_csv(seed_df, summary_dir / "per_seed_metrics.csv")

    summary_df = make_summary(seed_df)
    save_csv(summary_df, summary_dir / "summary_metrics.csv")

    legacy_paper_df = make_legacy_paper_table(summary_df)
    save_csv(legacy_paper_df, summary_dir / "legacy_paper_table.csv")

    payload = {
        "per_seed": seed_df.to_dict(orient="records"),
        "aggregate": summary_df.to_dict(orient="records"),
    }
    (summary_dir / "summary_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    representative_seed_dir = resolve_representative_seed_dir(run_root, args.representative_seed)
    seed_sci_df = collect_seed_sci_metrics(run_root)
    profile_summary = collect_profile_summary(
        representative_seed_dir,
        profile_device=args.profile_device,
        profile_warmup=args.profile_warmup,
        profile_runs=args.profile_runs,
        profile_repeats=args.profile_repeats,
    )

    sci_main_df = build_sci_main_table(seed_sci_df, args.model_label, profile_summary)
    save_csv(sci_main_df, summary_dir / "paper_table.csv")
    save_csv(sci_main_df, summary_dir / "table2_overall_5seed.csv")

    classwise_df = build_classwise_table(representative_seed_dir)
    save_csv(classwise_df, summary_dir / "table3_seed40_classwise.csv")

    sci_payload = {
        "schema": {
            "main_table_columns": SCI_MAIN_COLUMNS,
            "classwise_table_columns": classwise_df.columns.tolist(),
        },
        "representative_seed": args.representative_seed,
        "per_seed_sci_metrics": seed_sci_df.to_dict(orient="records"),
        "profile_summary": profile_summary,
    }
    (summary_dir / "sci_summary.json").write_text(
        json.dumps(sci_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved aggregate artifacts to {summary_dir}")


if __name__ == "__main__":
    main()
