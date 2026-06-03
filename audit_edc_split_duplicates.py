import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha256_file(path: Path, cache: dict[str, str]) -> str:
    key = str(path.resolve())
    if key in cache:
        return cache[key]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    cache[key] = digest.hexdigest()
    return cache[key]


def audit_split_dir(split_dir: Path, hash_cache: dict[str, str]) -> dict:
    split_path = split_dir / "splits.csv"
    summary_path = split_dir / "split_summary.json"
    if not split_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"Missing splits.csv or split_summary.json under {split_dir}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    data_root = (split_dir / summary["dataset_root_from_split_dir"]).resolve()
    by_hash = defaultdict(list)
    subset_counts = defaultdict(int)

    with split_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            subset = row["subset"]
            image_path = data_root / row["filepath"]
            image_hash = sha256_file(image_path, hash_cache)
            subset_counts[subset] += 1
            by_hash[image_hash].append(
                {
                    "subset": subset,
                    "filepath": row["filepath"],
                    "class_name": row["class_name"],
                    "group_id": row.get("group_id", ""),
                }
            )

    cross_subset_duplicates = []
    within_subset_duplicates = []
    for image_hash, rows in by_hash.items():
        subsets = sorted({row["subset"] for row in rows})
        if len(rows) > 1 and len(subsets) > 1:
            cross_subset_duplicates.append({"sha256": image_hash, "subsets": subsets, "rows": rows})
        elif len(rows) > 1:
            within_subset_duplicates.append({"sha256": image_hash, "subsets": subsets, "rows": rows})

    return {
        "split_dir": str(split_dir),
        "dataset_root": str(data_root),
        "sample_count": sum(subset_counts.values()),
        "subset_counts": dict(sorted(subset_counts.items())),
        "unique_sha256_count": len(by_hash),
        "cross_subset_exact_duplicate_count": len(cross_subset_duplicates),
        "within_subset_exact_duplicate_count": len(within_subset_duplicates),
        "cross_subset_examples": cross_subset_duplicates[:10],
        "within_subset_examples": within_subset_duplicates[:10],
    }


def main():
    parser = argparse.ArgumentParser(description="Audit exact image duplicate leakage across EDC split subsets.")
    parser.add_argument(
        "--split-root",
        default="result/literature_adapted_comparison/splits",
        help="Directory containing seed_* split folders.",
    )
    parser.add_argument(
        "--output",
        default="result/dataset_audits/edc_exact_duplicate_audit.json",
        help="JSON file to write the audit summary.",
    )
    args = parser.parse_args()

    split_root = Path(args.split_root)
    split_dirs = sorted(path for path in split_root.glob("seed_*") if path.is_dir())
    if not split_dirs:
        raise FileNotFoundError(f"No seed_* split directories found under {split_root}")

    hash_cache = {}
    audits = [audit_split_dir(split_dir, hash_cache) for split_dir in split_dirs]
    payload = {
        "split_root": str(split_root),
        "split_count": len(audits),
        "total_cross_subset_exact_duplicate_count": sum(
            item["cross_subset_exact_duplicate_count"] for item in audits
        ),
        "audits": audits,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
