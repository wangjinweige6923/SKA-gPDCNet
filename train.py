import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from torch_utils import (
    build_dataloader,
    load_splits,
    resolve_device,
    save_checkpoint,
    save_model_summary,
    set_seed,
    create_model,
)


def extract_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["logits"]
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def compute_model_loss(model, outputs, labels, criterion):
    loss_fn = getattr(model, "compute_loss", None)
    if callable(loss_fn):
        return loss_fn(outputs, labels, criterion=criterion)
    return criterion(extract_logits(outputs), labels)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the baseline ResNet50 classifier with PyTorch.")
    parser.add_argument("--split-dir", required=True, help="Directory containing splits.csv.")
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size.")
    parser.add_argument("--img-size", nargs=2, type=int, default=[224, 224], metavar=("H", "W"))
    parser.add_argument("--output-dir", required=True, help="Directory to store training artifacts.")
    parser.add_argument("--seed", required=True, type=int, help="Random seed.")
    parser.add_argument("--device", default="auto", help="Device to use: auto/cuda/cpu.")
    parser.add_argument("--num-workers", default=4, type=int, help="DataLoader worker processes.")
    parser.add_argument(
        "--vessel-prior-root",
        default=None,
        help="Optional root containing vessel-prior PNGs matching dataset relative paths.",
    )
    parser.add_argument(
        "--ska-lambda-proto",
        default=0.1,
        type=float,
        help="Weight for the SKA semantic prototype auxiliary loss.",
    )
    parser.add_argument(
        "--ska-lambda-attr",
        default=0.1,
        type=float,
        help="Weight for the SKA attribute prediction auxiliary loss.",
    )
    return parser.parse_args()


def run_epoch(model, dataloader, criterion, optimizer, device, scaler, train_mode):
    if train_mode:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    amp_enabled = device.type == "cuda"
    progress = tqdm(dataloader, leave=False, desc="train" if train_mode else "val")

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            with torch.amp.autocast(device.type, enabled=amp_enabled):
                outputs = model(images)
                logits = extract_logits(outputs)
                loss = compute_model_loss(model, outputs, labels, criterion)

            if train_mode:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += (predictions == labels).sum().item()
        total += batch_size

        progress.set_postfix(
            loss=f"{running_loss / max(total, 1):.4f}",
            acc=f"{correct / max(total, 1):.4f}",
        )

    epoch_loss = running_loss / max(total, 1)
    epoch_accuracy = correct / max(total, 1)
    return epoch_loss, epoch_accuracy


def plot_history(history_df, output_dir: Path):
    history_df.to_csv(output_dir / "history.csv", index=False, encoding="utf-8")
    payload = {
        "epoch": [int(value) for value in history_df["epoch"].tolist()],
        "history": {
            column: [float(value) for value in history_df[column].tolist()]
            for column in history_df.columns
            if column != "epoch"
        },
    }
    (output_dir / "history.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history_df["epoch"], history_df["accuracy"], label="train_accuracy")
    axes[0].plot(history_df["epoch"], history_df["val_accuracy"], label="val_accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["loss"], label="train_loss")
    axes[1].plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_training_metadata(output_dir: Path, args, class_names, device, best_metrics):
    metadata = {
        "framework": "pytorch",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "class_names": class_names,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "best_metrics": best_metrics,
        "vessel_prior_root": args.vessel_prior_root,
        "ska_loss_weights": {
            "lambda_proto": float(args.ska_lambda_proto),
            "lambda_attr": float(args.ska_lambda_attr),
        },
    }
    (output_dir / "train_config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2 because the classifier head uses BatchNorm.")

    split_dir = Path(args.split_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    os.environ["SKA_LAMBDA_PROTO"] = str(args.ska_lambda_proto)
    os.environ["SKA_LAMBDA_ATTR"] = str(args.ska_lambda_attr)

    split_df, class_names = load_splits(split_dir)
    train_df = split_df[split_df["subset"] == "train"].reset_index(drop=True)
    val_df = split_df[split_df["subset"] == "val"].reset_index(drop=True)
    if train_df.empty or val_df.empty:
        raise RuntimeError("train/val subsets must not be empty")

    image_size = tuple(args.img_size)
    train_loader = build_dataloader(
        train_df,
        image_size=image_size,
        batch_size=args.batch_size,
        training=True,
        seed=args.seed,
        num_workers=args.num_workers,
        vessel_prior_root=args.vessel_prior_root,
    )
    val_loader = build_dataloader(
        val_df,
        image_size=image_size,
        batch_size=args.batch_size,
        training=False,
        seed=args.seed,
        num_workers=args.num_workers,
        vessel_prior_root=args.vessel_prior_root,
    )

    model = create_model(num_classes=len(class_names)).to(device)
    save_model_summary(output_dir, model, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-7)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    history_rows = []
    checkpoint_path = output_dir / "best_model.pth"
    best_val_loss = float("inf")
    best_metrics = {}
    patience = 5
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler, train_mode=True
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, optimizer, device, scaler, train_mode=False
        )
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        history_rows.append(
            {
                "epoch": epoch,
                "loss": train_loss,
                "accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "lr": current_lr,
            }
        )

        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"loss: {train_loss:.4f} - accuracy: {train_accuracy:.4f} - "
            f"val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f} - lr: {current_lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_metrics = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
            save_checkpoint(
                checkpoint_path,
                model,
                class_names=class_names,
                image_size=image_size,
                epoch=epoch,
                metrics=best_metrics,
                vessel_prior_root=args.vessel_prior_root,
            )
            print(f"Saved best checkpoint to {checkpoint_path}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    history_df = pd.DataFrame(history_rows)
    plot_history(history_df, output_dir)
    save_training_metadata(output_dir, args, class_names, device, best_metrics)
    print(f"Best model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
