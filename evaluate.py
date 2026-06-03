import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from tqdm import tqdm

from torch_utils import build_dataloader, load_checkpoint, load_splits, resolve_device


def extract_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["logits"]
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the baseline model on the held-out test set.")
    parser.add_argument("--split-dir", required=True, help="Directory containing splits.csv.")
    parser.add_argument("--model", required=True, help="Path to the trained PyTorch checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Directory to store evaluation artifacts.")
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size.")
    parser.add_argument("--device", default="auto", help="Device to use: auto/cuda/cpu.")
    parser.add_argument("--num-workers", default=4, type=int, help="DataLoader worker processes.")
    parser.add_argument(
        "--vessel-prior-root",
        default=None,
        help="Optional root containing vessel-prior PNGs matching dataset relative paths.",
    )
    return parser.parse_args()


def save_predictions(output_dir: Path, test_df: pd.DataFrame, probabilities: np.ndarray, class_names):
    pred_indices = probabilities.argmax(axis=1)
    pred_labels = [class_names[index] for index in pred_indices]
    max_confidence = probabilities.max(axis=1)

    prediction_df = test_df.copy()
    if "resolved_filepath" in prediction_df.columns:
        prediction_df = prediction_df.drop(columns=["resolved_filepath"])
    prediction_df["pred_class_index"] = pred_indices
    prediction_df["pred_class_name"] = pred_labels
    prediction_df["confidence"] = max_confidence
    for class_index, class_name in enumerate(class_names):
        prediction_df[f"prob_{class_name}"] = probabilities[:, class_index]

    prediction_df.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8")
    return pred_indices


def save_classification_report(output_dir: Path, y_true, y_pred, class_names):
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(output_dir / "classification_report.csv", encoding="utf-8")
    return report_dict


def save_confusion_matrix(output_dir: Path, y_true, y_pred, class_names):
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_roc_curves(output_dir: Path, y_true, probabilities, class_names):
    one_hot = np.eye(len(class_names))[y_true]
    per_class_auc = {}
    fig, ax = plt.subplots(figsize=(8, 6))
    for class_index, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(one_hot[:, class_index], probabilities[:, class_index])
        auc_score = roc_auc_score(one_hot[:, class_index], probabilities[:, class_index])
        per_class_auc[class_name] = float(auc_score)
        ax.plot(fpr, tpr, label=f"{class_name} (AUC={auc_score:.4f})")

    macro_auc = roc_auc_score(one_hot, probabilities, multi_class="ovr", average="macro")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves (macro AUC={macro_auc:.4f})")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return float(macro_auc), per_class_auc


def save_metrics(output_dir: Path, report_dict, accuracy, macro_auc, per_class_auc, model_path, device, checkpoint):
    metrics = {
        "framework": "pytorch",
        "model_path": Path(os.path.relpath(model_path, output_dir)).as_posix(),
        "model_path_base": "eval_dir",
        "device": str(device),
        "best_epoch": checkpoint.get("epoch"),
        "accuracy": float(accuracy),
        "macro_precision": float(report_dict["macro avg"]["precision"]),
        "macro_recall": float(report_dict["macro avg"]["recall"]),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_precision": float(report_dict["weighted avg"]["precision"]),
        "weighted_recall": float(report_dict["weighted avg"]["recall"]),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
        "macro_auc": float(macro_auc),
        "per_class_auc": per_class_auc,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    split_dir = Path(args.split_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    model_path = Path(args.model).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    test_df, _ = load_splits(split_dir)
    test_df = test_df[test_df["subset"] == "test"].reset_index(drop=True)
    if test_df.empty:
        raise RuntimeError("test subset is empty")

    device = resolve_device(args.device)
    model, class_names, image_size, checkpoint = load_checkpoint(model_path, map_location=device)
    vessel_prior_root = args.vessel_prior_root or checkpoint.get("vessel_prior_root")
    model = model.to(device)
    model.eval()

    test_loader = build_dataloader(
        test_df,
        image_size=image_size,
        batch_size=args.batch_size,
        training=False,
        seed=0,
        num_workers=args.num_workers,
        vessel_prior_root=vessel_prior_root,
    )

    all_probabilities = []
    progress = tqdm(test_loader, leave=False, desc="eval")
    with torch.no_grad():
        for images, _ in progress:
            images = images.to(device, non_blocking=True)
            logits = extract_logits(model(images))
            probabilities = torch.softmax(logits, dim=1)
            all_probabilities.append(probabilities.cpu().numpy())

    probabilities = np.concatenate(all_probabilities, axis=0)
    y_true = test_df["class_index"].astype("int32").to_numpy()
    y_pred = save_predictions(output_dir, test_df, probabilities, class_names)
    accuracy = float((y_true == y_pred).mean())

    report_dict = save_classification_report(output_dir, y_true, y_pred, class_names)
    save_confusion_matrix(output_dir, y_true, y_pred, class_names)
    macro_auc, per_class_auc = save_roc_curves(output_dir, y_true, probabilities, class_names)
    save_metrics(output_dir, report_dict, accuracy, macro_auc, per_class_auc, model_path, device, checkpoint)

    print(f"Saved evaluation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
