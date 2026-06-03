import argparse
import json
import math
import os
import random
import re
from pathlib import Path

import pandas as pd


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif"}
EYE_SUFFIX_RE = re.compile(r"([_-](left|right|os|od))$", re.IGNORECASE)
SUBSET_NAMES = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create leakage-safe train/val/test splits for the fundus classification dataset."
    )
    parser.add_argument("--data-dir", required=True, help="Dataset root containing one folder per class.")
    parser.add_argument("--output-dir", required=True, help="Directory to write split artifacts.")
    parser.add_argument("--seed", required=True, type=int, help="Random seed used for split generation.")
    parser.add_argument("--train-ratio", required=True, type=float, help="Train ratio.")
    parser.add_argument("--val-ratio", required=True, type=float, help="Validation ratio.")
    parser.add_argument("--test-ratio", required=True, type=float, help="Test ratio.")
    parser.add_argument(
        "--grouping-mode",
        choices=("class_base_id", "filepath"),
        default="class_base_id",
        help=(
            "How to keep related samples together. "
            "'class_base_id' strips common eye laterality suffixes such as _left/_right "
            "from the filename stem and keeps each class-scoped base ID in one subset. "
            "'filepath' treats each image independently."
        ),
    )
    return parser.parse_args()


def validate_ratios(train_ratio, val_ratio, test_ratio):
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {total:.6f}")
    for name, value in {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")


def scan_dataset(data_dir: Path) -> pd.DataFrame:
    class_dirs = sorted([path for path in data_dir.iterdir() if path.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(f"No class directories found in {data_dir}")

    rows = []
    for class_index, class_dir in enumerate(class_dirs):
        files = sorted(
            [
                path
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ]
        )
        if not files:
            raise FileNotFoundError(f"No image files found in class directory {class_dir}")
        for file_path in files:
            rows.append(
                {
                    "filepath": file_path.relative_to(data_dir).as_posix(),
                    "class_name": class_dir.name,
                    "class_index": class_index,
                }
            )

    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        raise RuntimeError(f"No image samples found under {data_dir}")
    return dataframe


def extract_sample_metadata(stored_path: str):
    stem = Path(stored_path).stem
    match = EYE_SUFFIX_RE.search(stem)
    if not match:
        return stem, ""
    sample_id = stem[: match.start()] or stem
    return sample_id, match.group(2).lower()


def add_group_columns(dataframe: pd.DataFrame, grouping_mode: str) -> pd.DataFrame:
    dataframe = dataframe.copy()
    sample_metadata = dataframe["filepath"].map(extract_sample_metadata)
    dataframe["sample_id"] = sample_metadata.map(lambda value: value[0])
    dataframe["laterality"] = sample_metadata.map(lambda value: value[1])

    if grouping_mode == "filepath":
        dataframe["group_id"] = dataframe["filepath"].astype(str)
    elif grouping_mode == "class_base_id":
        dataframe["group_id"] = dataframe["class_name"].astype(str) + "/" + dataframe["sample_id"].astype(str)
    else:
        raise ValueError(f"Unsupported grouping mode: {grouping_mode}")

    return dataframe


def allocate_subset_targets(total_samples: int, ratios: dict) -> dict:
    raw_targets = {subset: total_samples * float(ratios[subset]) for subset in SUBSET_NAMES}
    targets = {subset: int(math.floor(raw_targets[subset])) for subset in SUBSET_NAMES}
    remainder = total_samples - sum(targets.values())
    ranked_subsets = sorted(
        SUBSET_NAMES,
        key=lambda subset: (raw_targets[subset] - targets[subset], ratios[subset], -SUBSET_NAMES.index(subset)),
        reverse=True,
    )
    for subset in ranked_subsets[:remainder]:
        targets[subset] += 1
    return targets


def choose_group_ids_for_target(group_table: pd.DataFrame, target_count: int, seed: int):
    if target_count <= 0 or group_table.empty:
        return set()

    groups = list(group_table[["group_id", "sample_count"]].itertuples(index=False, name=None))
    rng = random.Random(seed)
    rng.shuffle(groups)

    reachable = {0: None}
    for group_index, (_, sample_count) in enumerate(groups):
        for running_total in sorted(reachable.keys(), reverse=True):
            next_total = running_total + int(sample_count)
            if next_total not in reachable:
                reachable[next_total] = (running_total, group_index)

    def subset_sum_score(total):
        tie_break = -total if total <= target_count else total
        return abs(total - target_count), total > target_count, tie_break

    best_total = min(reachable.keys(), key=subset_sum_score)
    selected_group_ids = set()
    while best_total != 0:
        previous_total, group_index = reachable[best_total]
        selected_group_ids.add(groups[group_index][0])
        best_total = previous_total
    return selected_group_ids


def stratified_group_split(
    dataframe: pd.DataFrame,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
):
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    split_frames = []
    class_table = (
        dataframe[["class_name", "class_index"]]
        .drop_duplicates()
        .sort_values("class_index")
        .reset_index(drop=True)
    )

    for class_offset, row in enumerate(class_table.itertuples(index=False)):
        class_df = dataframe[dataframe["class_name"] == row.class_name].copy()
        group_table = (
            class_df.groupby("group_id", sort=False)
            .size()
            .reset_index(name="sample_count")
        )
        subset_targets = allocate_subset_targets(len(class_df), ratios)

        test_groups = choose_group_ids_for_target(
            group_table=group_table,
            target_count=subset_targets["test"],
            seed=seed * 1009 + class_offset * 2 + 1,
        )
        remaining_group_table = group_table[~group_table["group_id"].isin(test_groups)].reset_index(drop=True)
        val_groups = choose_group_ids_for_target(
            group_table=remaining_group_table,
            target_count=subset_targets["val"],
            seed=seed * 1009 + class_offset * 2 + 2,
        )

        class_df["subset"] = "train"
        class_df.loc[class_df["group_id"].isin(val_groups), "subset"] = "val"
        class_df.loc[class_df["group_id"].isin(test_groups), "subset"] = "test"
        split_frames.append(class_df)

    return pd.concat(split_frames, ignore_index=True)


def audit_group_leakage(split_df: pd.DataFrame):
    overlap_df = (
        split_df.groupby("group_id")["subset"]
        .nunique()
        .reset_index(name="subset_count")
    )
    overlap_df = overlap_df[overlap_df["subset_count"] > 1].sort_values(["subset_count", "group_id"], ascending=[False, True])
    example_rows = []
    if not overlap_df.empty:
        overlap_group_ids = set(overlap_df["group_id"].tolist())
        leaking_rows = split_df[split_df["group_id"].isin(overlap_group_ids)].copy()
        leaking_rows = leaking_rows.sort_values(["group_id", "subset", "filepath"])
        for group_id, group_frame in leaking_rows.groupby("group_id"):
            example_rows.append(
                {
                    "group_id": group_id,
                    "subsets": sorted(group_frame["subset"].unique().tolist()),
                    "files": group_frame["filepath"].tolist(),
                }
            )
            if len(example_rows) >= 10:
                break
    return {
        "overlap_group_count": int(len(overlap_df)),
        "examples": example_rows,
    }


def make_summary(
    split_df: pd.DataFrame,
    output_dir: Path,
    data_dir: Path,
    seed: int,
    ratios: dict,
    grouping_mode: str,
    leakage_audit: dict,
):
    subset_counts = split_df.groupby("subset").size().to_dict()
    class_counts = (
        split_df.groupby(["subset", "class_name"])
        .size()
        .unstack(fill_value=0)
        .sort_index(axis=1)
        .to_dict(orient="index")
    )
    subset_group_counts = split_df.groupby("subset")["group_id"].nunique().to_dict()
    group_size_distribution = split_df.groupby("group_id").size().value_counts().sort_index().to_dict()
    summary = {
        "seed": seed,
        "ratios": ratios,
        "grouping_mode": grouping_mode,
        "dataset_root_from_split_dir": Path(os.path.relpath(data_dir, output_dir)).as_posix(),
        "total_samples": int(len(split_df)),
        "total_groups": int(split_df["group_id"].nunique()),
        "classes": sorted(split_df["class_name"].unique().tolist()),
        "subset_counts": {key: int(value) for key, value in subset_counts.items()},
        "subset_group_counts": {key: int(value) for key, value in subset_group_counts.items()},
        "group_size_distribution": {str(key): int(value) for key, value in group_size_distribution.items()},
        "group_leakage": leakage_audit,
        "class_distribution": {
            subset: {class_name: int(count) for class_name, count in counts.items()}
            for subset, counts in class_counts.items()
        },
    }
    summary_path = output_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_df = scan_dataset(data_dir)
    dataset_df = add_group_columns(dataset_df, grouping_mode=args.grouping_mode)
    split_df = stratified_group_split(
        dataset_df,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    split_df = split_df[
        ["filepath", "class_name", "class_index", "sample_id", "laterality", "group_id", "subset"]
    ]
    split_df = split_df.sort_values(["subset", "class_index", "group_id", "filepath"]).reset_index(drop=True)

    leakage_audit = audit_group_leakage(split_df)
    if leakage_audit["overlap_group_count"] > 0:
        raise RuntimeError(
            "Detected grouped leakage after split generation: "
            f"{leakage_audit['overlap_group_count']} group(s) span multiple subsets."
        )

    split_df.to_csv(output_dir / "splits.csv", index=False, encoding="utf-8")
    make_summary(
        split_df,
        output_dir=output_dir,
        data_dir=data_dir,
        seed=args.seed,
        ratios={
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
        },
        grouping_mode=args.grouping_mode,
        leakage_audit=leakage_audit,
    )

    summary_counts = split_df.groupby(["subset", "class_name"]).size().unstack(fill_value=0)
    summary_group_counts = split_df.groupby(["subset", "class_name"])["group_id"].nunique().unstack(fill_value=0)
    print(f"Saved splits to {output_dir / 'splits.csv'}")
    print("Sample counts per subset/class:")
    print(summary_counts)
    print("Grouped sample-ID counts per subset/class:")
    print(summary_group_counts)


if __name__ == "__main__":
    main()
