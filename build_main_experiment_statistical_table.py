import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelConfig:
    label: str
    root: Path


@dataclass(frozen=True)
class TableRowConfig:
    before: str
    after: str
    metric: str
    display_metric: str
    interpretation: str


@dataclass
class SeedPredictions:
    model: str
    seed: int
    path: Path
    class_names: List[str]
    y_true: np.ndarray
    y_pred: np.ndarray
    probabilities: np.ndarray
    unit_to_rows: Dict[str, np.ndarray]

    def row_indices_for_units(self, units: Sequence[str]) -> np.ndarray:
        parts = [self.unit_to_rows[str(unit)] for unit in units]
        return np.concatenate(parts).astype(np.int64, copy=False)


MODEL_CONFIGS = [
    ModelConfig("ResNet34-[V]x16", Path("train_pred/archive_unused/result_dirs/resnet34_pdc_v16_4seed")),
    ModelConfig(
        "ResNet34-C-[V]x15",
        Path("train_pred/result/resnet34_pdc_c_v15_vs_gpdc_c_v15/resnet34_pdc_c_v15_fixed5seed"),
    ),
    ModelConfig(
        "SKA-gPDCNet",
        Path("train_pred/result/ska_gpdcnet_no_vessel_11seed/ska_gpdcnet_fixed5seed"),
    ),
]

TABLE_ROWS = [
    TableRowConfig(
        "ResNet34-[V]x16",
        "ResNet34-C-[V]x15",
        "macro_f1",
        "F1",
        "small mean gain; stability assessed by std/CV",
    ),
    TableRowConfig("ResNet34-[V]x16", "SKA-gPDCNet", "macro_f1", "F1", "mean improvement"),
    TableRowConfig("ResNet34-[V]x16", "SKA-gPDCNet", "macro_recall", "Sensitivity", "improved recall trend"),
    TableRowConfig("ResNet34-[V]x16", "SKA-gPDCNet", "macro_auc", "AUC", "comparable ranking; no Holm-significant gain"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the main-experiment paired bootstrap statistical table."
    )
    parser.add_argument(
        "--output-dir",
        default="train_pred/result/main_experiment_statistical_table",
        help="Output directory for CSV and Markdown tables.",
    )
    parser.add_argument("--n-boot", type=int, default=5000, help="Number of bootstrap replicates.")
    parser.add_argument("--ci", type=float, default=0.95, help="Confidence level.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for bootstrap.")
    parser.add_argument(
        "--unit-col",
        default="group_id",
        help="Bootstrap unit column. Falls back to filepath, sample_id, then row index.",
    )
    return parser.parse_args()


def parse_seed(seed_dir: Path) -> int:
    try:
        return int(seed_dir.name.split("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Cannot parse seed from directory name: {seed_dir.name}") from exc


def iter_prediction_files(run_root: Path) -> Iterable[Tuple[int, Path]]:
    for seed_dir in sorted(run_root.glob("seed_*"), key=parse_seed):
        path = seed_dir / "results" / "predictions.csv"
        if path.exists():
            yield parse_seed(seed_dir), path


def infer_class_names(df: pd.DataFrame) -> List[str]:
    probability_columns = [column for column in df.columns if column.startswith("prob_")]
    if not probability_columns:
        raise ValueError("predictions.csv must contain columns named prob_<class>.")
    return [column[len("prob_") :] for column in probability_columns]


def resolve_unit_labels(df: pd.DataFrame, preferred: str) -> np.ndarray:
    for candidate in (preferred, "filepath", "image_id", "sample_id"):
        if candidate and candidate in df.columns:
            return df[candidate].astype(str).to_numpy()
    return np.arange(len(df)).astype(str)


def build_unit_index(unit_labels: np.ndarray) -> Dict[str, np.ndarray]:
    return {str(unit): np.flatnonzero(unit_labels == unit).astype(np.int64) for unit in pd.unique(unit_labels)}


def load_seed_predictions(path: Path, model: str, seed: int, unit_col: str) -> SeedPredictions:
    df = pd.read_csv(path)
    required = {"class_name", "pred_class_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    class_names = infer_class_names(df)
    class_to_index = {class_name: i for i, class_name in enumerate(class_names)}
    probability_columns = [f"prob_{class_name}" for class_name in class_names]

    unknown_true = sorted(set(df["class_name"].astype(str)) - set(class_names))
    unknown_pred = sorted(set(df["pred_class_name"].astype(str)) - set(class_names))
    if unknown_true or unknown_pred:
        raise ValueError(f"{path} labels do not match probability columns.")

    unit_labels = resolve_unit_labels(df, unit_col)
    return SeedPredictions(
        model=model,
        seed=seed,
        path=path,
        class_names=class_names,
        y_true=df["class_name"].astype(str).map(class_to_index).to_numpy(dtype=np.int64),
        y_pred=df["pred_class_name"].astype(str).map(class_to_index).to_numpy(dtype=np.int64),
        probabilities=df[probability_columns].to_numpy(dtype=np.float64),
        unit_to_rows=build_unit_index(unit_labels),
    )


def load_model(config: ModelConfig, unit_col: str) -> Dict[int, SeedPredictions]:
    predictions: Dict[int, SeedPredictions] = {}
    for seed, path in iter_prediction_files(config.root):
        predictions[seed] = load_seed_predictions(path, config.label, seed, unit_col)
    if not predictions:
        raise FileNotFoundError(f"No seed_*/results/predictions.csv files found under {config.root}")
    return predictions


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.zeros_like(numerator, dtype=np.float64)
    mask = denominator != 0
    out[mask] = numerator[mask] / denominator[mask]
    return out


def binary_auc(y_binary: np.ndarray, scores: np.ndarray) -> float:
    y_binary = y_binary.astype(bool, copy=False)
    n_pos = int(y_binary.sum())
    n_total = int(len(y_binary))
    n_neg = n_total - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(n_total, dtype=np.float64)
    start = 0
    while start < n_total:
        end = start + 1
        while end < n_total and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end

    sum_pos_ranks = float(ranks[y_binary].sum())
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def compute_metric(predictions: SeedPredictions, row_indices: np.ndarray, metric: str) -> float:
    y_true = predictions.y_true[row_indices]
    y_pred = predictions.y_pred[row_indices]
    probabilities = predictions.probabilities[row_indices]
    n_classes = len(predictions.class_names)
    n_samples = len(y_true)
    if n_samples == 0:
        return float("nan")

    confusion = np.bincount(
        y_true * n_classes + y_pred,
        minlength=n_classes * n_classes,
    ).reshape(n_classes, n_classes)
    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0).astype(np.float64) - tp
    fn = confusion.sum(axis=1).astype(np.float64) - tp
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)

    if metric == "accuracy":
        return float(tp.sum() / n_samples)
    if metric == "macro_recall":
        return float(np.mean(recall))
    if metric == "macro_f1":
        return float(np.mean(f1))
    if metric == "macro_auc":
        aucs = [binary_auc(y_true == class_index, probabilities[:, class_index]) for class_index in range(n_classes)]
        finite = [value for value in aucs if math.isfinite(value)]
        return float(np.mean(finite)) if finite else float("nan")
    raise ValueError(f"Unsupported metric: {metric}")


def mean_finite(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def std_finite(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) <= 1:
        return 0.0 if len(finite) == 1 else float("nan")
    return float(np.std(finite, ddof=1))


def quantile_ci(values: np.ndarray, ci: float) -> Tuple[float, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan"), float("nan")
    alpha = 1.0 - ci
    return float(np.quantile(finite, alpha / 2.0)), float(np.quantile(finite, 1.0 - alpha / 2.0))


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    indexed = [(i, float(p)) for i, p in enumerate(p_values)]
    finite = [(i, p) for i, p in indexed if math.isfinite(p)]
    adjusted = [float("nan")] * len(indexed)
    previous = 0.0
    m = len(finite)
    for rank, (i, p_value) in enumerate(sorted(finite, key=lambda item: item[1]), start=1):
        current = min(1.0, (m - rank + 1) * p_value)
        current = max(previous, current)
        adjusted[i] = current
        previous = current
    return adjusted


def common_units(seed_predictions: Sequence[SeedPredictions]) -> np.ndarray:
    if not seed_predictions:
        return np.asarray([], dtype=object)
    units = set(seed_predictions[0].unit_to_rows)
    for predictions in seed_predictions[1:]:
        units &= set(predictions.unit_to_rows)
    return np.asarray(sorted(units), dtype=object)


def all_units(predictions: SeedPredictions) -> np.ndarray:
    return np.asarray(sorted(predictions.unit_to_rows), dtype=object)


def seed_metric_values(seed_predictions: Sequence[SeedPredictions], metric: str) -> List[float]:
    values = []
    for predictions in seed_predictions:
        row_indices = predictions.row_indices_for_units(all_units(predictions))
        values.append(compute_metric(predictions, row_indices, metric))
    return values


def bootstrap_row(
    row_config: TableRowConfig,
    all_predictions: Dict[str, Dict[int, SeedPredictions]],
    n_boot: int,
    ci: float,
    rng: np.random.Generator,
) -> Dict[str, object]:
    common_seeds = sorted(set(all_predictions[row_config.before]) & set(all_predictions[row_config.after]))
    if not common_seeds:
        raise ValueError(f"No common seeds for {row_config.before} vs {row_config.after}")

    paired_units_by_seed: Dict[int, np.ndarray] = {}
    before_seed_values = []
    after_seed_values = []
    for seed in common_seeds:
        before_predictions = all_predictions[row_config.before][seed]
        after_predictions = all_predictions[row_config.after][seed]
        units = common_units([before_predictions, after_predictions])
        if len(units) == 0:
            continue
        paired_units_by_seed[seed] = units
        before_rows = before_predictions.row_indices_for_units(units)
        after_rows = after_predictions.row_indices_for_units(units)
        before_seed_values.append(compute_metric(before_predictions, before_rows, row_config.metric))
        after_seed_values.append(compute_metric(after_predictions, after_rows, row_config.metric))

    if not paired_units_by_seed:
        raise ValueError(f"No common bootstrap units for {row_config.before} vs {row_config.after}")

    before_mean = mean_finite(before_seed_values)
    after_mean = mean_finite(after_seed_values)

    deltas = np.empty(n_boot, dtype=np.float64)
    for boot_index in range(n_boot):
        seed_deltas = []
        for seed, units in paired_units_by_seed.items():
            sampled_units = rng.choice(units, size=len(units), replace=True)
            before_predictions = all_predictions[row_config.before][seed]
            after_predictions = all_predictions[row_config.after][seed]
            before_rows = before_predictions.row_indices_for_units(sampled_units)
            after_rows = after_predictions.row_indices_for_units(sampled_units)
            before_boot = compute_metric(before_predictions, before_rows, row_config.metric)
            after_boot = compute_metric(after_predictions, after_rows, row_config.metric)
            if math.isfinite(before_boot) and math.isfinite(after_boot):
                seed_deltas.append(after_boot - before_boot)
        deltas[boot_index] = mean_finite(seed_deltas)

    ci_low, ci_high = quantile_ci(deltas, ci)
    finite_deltas = deltas[np.isfinite(deltas)]
    if len(finite_deltas):
        p_value = 2.0 * min(float(np.mean(finite_deltas <= 0.0)), float(np.mean(finite_deltas >= 0.0)))
        p_value = min(1.0, p_value)
    else:
        p_value = float("nan")

    delta = after_mean - before_mean
    return {
        "Comparison": f"{row_config.before} vs {row_config.after}",
        "Metric": row_config.display_metric,
        "Metric source": row_config.metric,
        "Before mean": before_mean,
        "After mean": after_mean,
        "Delta mean": delta,
        "Delta mean (pp)": delta * 100.0,
        "95% bootstrap CI low": ci_low,
        "95% bootstrap CI high": ci_high,
        "95% bootstrap CI (pp)": f"[{ci_low * 100.0:+.2f}, {ci_high * 100.0:+.2f}]",
        "p-value": p_value,
        "n_boot": n_boot,
        "n_common_seeds": len(paired_units_by_seed),
        "mean_common_units_per_seed": float(np.mean([len(units) for units in paired_units_by_seed.values()])),
        "common_seeds": ";".join(str(seed) for seed in paired_units_by_seed),
        "before_n_seeds": len(all_predictions[row_config.before]),
        "after_n_seeds": len(all_predictions[row_config.after]),
        "before_seed_std": std_finite(before_seed_values),
        "after_seed_std": std_finite(after_seed_values),
        "Interpretation": row_config.interpretation,
    }


def build_stability_rows(all_predictions: Dict[str, Dict[int, SeedPredictions]]) -> List[Dict[str, object]]:
    rows = []
    for model_label, seed_map in all_predictions.items():
        seed_predictions = [seed_map[seed] for seed in sorted(seed_map)]
        for metric, display in (("macro_f1", "F1"), ("macro_recall", "Sensitivity"), ("macro_auc", "AUC")):
            values = seed_metric_values(seed_predictions, metric)
            mean = mean_finite(values)
            std = std_finite(values)
            rows.append(
                {
                    "Model": model_label,
                    "Metric": display,
                    "Mean": mean,
                    "Std": std,
                    "CV (%)": (std / mean * 100.0) if mean else float("nan"),
                    "N seeds": len(seed_predictions),
                    "Mean units/seed": float(np.mean([len(all_units(predictions)) for predictions in seed_predictions])),
                }
            )
    return rows


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(value) for value in row.tolist()) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.n_boot <= 0:
        raise ValueError("--n-boot must be positive")
    if not 0.0 < args.ci < 1.0:
        raise ValueError("--ci must be between 0 and 1")

    all_predictions = {config.label: load_model(config, args.unit_col) for config in MODEL_CONFIGS}
    for label, seed_map in all_predictions.items():
        print(f"Loaded {label}: seeds={sorted(seed_map)}")

    rng = np.random.default_rng(args.random_state)
    rows = [bootstrap_row(row_config, all_predictions, args.n_boot, args.ci, rng) for row_config in TABLE_ROWS]
    adjusted = holm_adjust([row["p-value"] for row in rows])
    for row, p_holm in zip(rows, adjusted):
        row["Holm-adjusted p"] = p_holm

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(output_dir / "main_experiment_paired_bootstrap_stats.csv", index=False, encoding="utf-8-sig")

    table_df = raw_df[
        [
            "Comparison",
            "Metric",
            "Delta mean (pp)",
            "95% bootstrap CI (pp)",
            "p-value",
            "Holm-adjusted p",
            "Interpretation",
        ]
    ].copy()
    table_df["Delta mean (pp)"] = table_df["Delta mean (pp)"].map(lambda value: f"{value:+.2f}")
    table_df["p-value"] = table_df["p-value"].map(lambda value: f"{value:.4f}" if math.isfinite(value) else "")
    table_df["Holm-adjusted p"] = table_df["Holm-adjusted p"].map(
        lambda value: f"{value:.4f}" if math.isfinite(value) else ""
    )
    table_df.to_csv(output_dir / "main_experiment_statistical_table.csv", index=False, encoding="utf-8-sig")
    (output_dir / "main_experiment_statistical_table.md").write_text(
        dataframe_to_markdown(table_df) + "\n",
        encoding="utf-8",
    )

    stability_df = pd.DataFrame(build_stability_rows(all_predictions))
    stability_df.to_csv(output_dir / "main_experiment_seed_stability.csv", index=False, encoding="utf-8-sig")

    print(f"Saved raw stats: {output_dir / 'main_experiment_paired_bootstrap_stats.csv'}")
    print(f"Saved paper table: {output_dir / 'main_experiment_statistical_table.csv'}")
    print(f"Saved stability table: {output_dir / 'main_experiment_seed_stability.csv'}")


if __name__ == "__main__":
    main()
