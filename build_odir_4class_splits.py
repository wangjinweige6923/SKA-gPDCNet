import argparse
import csv
import json
import os
import random
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


CLASS_ORDER = ("cataract", "diabetic_retinopathy", "glaucoma", "normal")
CLASS_TO_INDEX = {class_name: index for index, class_name in enumerate(CLASS_ORDER)}

ODIR_LABEL_COLUMNS = ("N", "D", "G", "C", "A", "H", "M", "O")
ODIR_LABEL_TO_CLASS = {
    "C": "cataract",
    "D": "diabetic_retinopathy",
    "G": "glaucoma",
    "N": "normal",
}

SOURCE_CONFIGS = (
    {
        "key": "trainval",
        "source_set": "Training Set",
        "annotation_file": "training annotation (English).xlsx",
        "output_name": "odir_4class_trainval",
        "subset_mode": "trainval",
    },
    {
        "key": "offsite",
        "source_set": "Off-site Test Set",
        "annotation_file": "off-site test annotation (English).xlsx",
        "output_name": "odir_4class_offsite",
        "subset_mode": "test",
    },
    {
        "key": "onsite",
        "source_set": "On-site Test Set",
        "annotation_file": "on-site test annotation (English).xlsx",
        "output_name": "odir_4class_onsite",
        "subset_mode": "test",
    },
)

XLSX_NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

NON_FUNDUS_TERMS = {
    "anterior segment image",
    "no fundus image",
}
NON_DECISIVE_TERMS = {
    "lens dust",
    "low image quality",
    "optic disk photographically invisible",
    "image offset",
}
NORMAL_TERMS = {"normal fundus"}


def parse_args():
    default_root = Path(__file__).resolve().parent / "external_datasets" / "OIA-ODIR"
    default_output_root = Path(__file__).resolve().parent / "result" / "external_splits"
    parser = argparse.ArgumentParser(
        description=(
            "Build four-class ODIR external-validation split files compatible with evaluate.py. "
            "Only ODIR N/D/G/C single-label cases are converted to image-level labels."
        )
    )
    parser.add_argument("--odir-root", default=str(default_root), help="Root directory containing OIA-ODIR subsets.")
    parser.add_argument(
        "--output-root",
        default=str(default_output_root),
        help="Directory where odir_4class_offsite/onsite split folders will be written.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=[config["key"] for config in SOURCE_CONFIGS],
        default=[config["key"] for config in SOURCE_CONFIGS],
        help="ODIR subsets to convert.",
    )
    parser.add_argument("--seed", default=2026, type=int, help="Seed for patient-level train/val splitting.")
    parser.add_argument("--val-ratio", default=0.15, type=float, help="Validation ratio for ODIR Training Set.")
    return parser.parse_args()


def column_index(cell_ref: str) -> int:
    letters = re.sub(r"[^A-Z]", "", cell_ref.upper())
    index = 0
    for char in letters:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def read_shared_strings(zip_file: zipfile.ZipFile):
    try:
        with zip_file.open("xl/sharedStrings.xml") as handle:
            root = ET.parse(handle).getroot()
    except KeyError:
        return []
    strings = []
    for item in root.findall(".//x:si", XLSX_NS):
        strings.append("".join(item.itertext()))
    return strings


def read_cell_value(cell, shared_strings):
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(cell.itertext()).strip()

    value_node = cell.find("x:v", XLSX_NS)
    if value_node is None or value_node.text is None:
        return ""

    value = value_node.text.strip()
    if cell_type == "s":
        return shared_strings[int(value)].strip()
    return value


def read_xlsx_records(path: Path):
    with zipfile.ZipFile(path) as zip_file:
        shared_strings = read_shared_strings(zip_file)
        with zip_file.open("xl/worksheets/sheet1.xml") as handle:
            root = ET.parse(handle).getroot()

    rows = []
    for row_node in root.findall(".//x:sheetData/x:row", XLSX_NS):
        row_values = []
        for cell in row_node.findall("x:c", XLSX_NS):
            cell_ref = cell.attrib.get("r", "")
            index = column_index(cell_ref)
            while len(row_values) <= index:
                row_values.append("")
            row_values[index] = read_cell_value(cell, shared_strings)
        rows.append(row_values)

    if not rows:
        raise ValueError(f"No rows found in annotation file: {path}")

    headers = [value.strip() for value in rows[0]]
    records = []
    for values in rows[1:]:
        values = values + [""] * (len(headers) - len(values))
        records.append({header: values[index].strip() for index, header in enumerate(headers)})
    return records


def parse_label_value(value: str) -> int:
    if value == "":
        return 0
    return int(float(value))


def normalize_patient_id(value: str) -> str:
    value = str(value).strip()
    if not value:
        return value
    try:
        return str(int(float(value)))
    except ValueError:
        return value


def normalize_keywords(value: str) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("\uff0c", ",").replace("ï¼Œ", ",")
    text = text.replace(";", ",").replace("\uff1b", ",")
    text = re.sub(r"\s+", " ", text)
    return text


def split_keyword_parts(text: str):
    return [part.strip() for part in text.split(",") if part.strip()]


def infer_eye_class(keywords: str):
    text = normalize_keywords(keywords)
    if not text:
        return None
    if any(term in text for term in NON_FUNDUS_TERMS):
        return None

    disease_hits = set()
    if "cataract" in text:
        disease_hits.add("cataract")
    if "glaucoma" in text:
        disease_hits.add("glaucoma")
    if (
        "diabetic retinopathy" in text
        or "nonproliferative retinopathy" in text
        or "non proliferative retinopathy" in text
        or "proliferative diabetic retinopathy" in text
    ):
        disease_hits.add("diabetic_retinopathy")

    if len(disease_hits) == 1:
        return next(iter(disease_hits))
    if len(disease_hits) > 1:
        return None

    allowed_normal_parts = NORMAL_TERMS | NON_DECISIVE_TERMS
    parts = split_keyword_parts(text)
    if parts and all(part in allowed_normal_parts for part in parts):
        return "normal"
    return None


def get_single_patient_label(record):
    labels = {label: parse_label_value(record.get(label, "")) for label in ODIR_LABEL_COLUMNS}
    positive_labels = [label for label, value in labels.items() if value == 1]
    if len(positive_labels) != 1:
        return None, positive_labels
    label = positive_labels[0]
    if label not in ODIR_LABEL_TO_CLASS:
        return None, positive_labels
    return label, positive_labels


def init_stats():
    return {
        "patients_seen": 0,
        "patients_selected": 0,
        "patients_skipped_multilabel_or_empty": 0,
        "patients_skipped_unsupported_label": 0,
        "eyes_seen_from_selected_patients": 0,
        "eyes_selected": 0,
        "eyes_skipped_missing_image": 0,
        "eyes_skipped_unusable_keywords": 0,
        "eyes_skipped_label_mismatch": 0,
        "eyes_skipped_duplicate": 0,
    }


def make_image_row(record, laterality: str, source_set: str):
    prefix = "Left" if laterality == "left" else "Right"
    return {
        "image_name": record.get(f"{prefix}-Fundus", "").strip(),
        "diagnostic_keywords": record.get(f"{prefix}-Diagnostic Keywords", "").strip(),
        "laterality": laterality,
        "source_set": source_set,
        "patient_id": normalize_patient_id(record.get("ID", "")),
    }


def build_source_rows(odir_root: Path, source_config: dict):
    source_dir = odir_root / source_config["source_set"]
    image_root = source_dir / "Images"
    annotation_path = source_dir / "Annotation" / source_config["annotation_file"]
    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing ODIR annotation file: {annotation_path}")
    if not image_root.exists():
        raise FileNotFoundError(f"Missing ODIR image directory: {image_root}")

    records = read_xlsx_records(annotation_path)
    stats = init_stats()
    rows = []
    seen_filepaths = set()

    for record in records:
        stats["patients_seen"] += 1
        odir_label, positive_labels = get_single_patient_label(record)
        if odir_label is None:
            if len(positive_labels) == 1:
                stats["patients_skipped_unsupported_label"] += 1
            else:
                stats["patients_skipped_multilabel_or_empty"] += 1
            continue

        patient_class = ODIR_LABEL_TO_CLASS[odir_label]
        patient_selected = False
        for laterality in ("left", "right"):
            stats["eyes_seen_from_selected_patients"] += 1
            eye_row = make_image_row(record, laterality, source_config["source_set"])
            image_name = eye_row["image_name"]
            if not image_name or not (image_root / image_name).exists():
                stats["eyes_skipped_missing_image"] += 1
                continue

            eye_class = infer_eye_class(eye_row["diagnostic_keywords"])
            if eye_class is None:
                stats["eyes_skipped_unusable_keywords"] += 1
                continue
            if eye_class != patient_class:
                stats["eyes_skipped_label_mismatch"] += 1
                continue
            if image_name in seen_filepaths:
                stats["eyes_skipped_duplicate"] += 1
                continue

            seen_filepaths.add(image_name)
            patient_selected = True
            rows.append(
                {
                    "filepath": image_name,
                    "class_name": eye_class,
                    "class_index": CLASS_TO_INDEX[eye_class],
                    "subset": "test",
                    "patient_id": eye_row["patient_id"],
                    "laterality": laterality,
                    "diagnostic_keywords": eye_row["diagnostic_keywords"],
                    "source_set": eye_row["source_set"],
                    "odir_patient_label": odir_label,
                }
            )
            stats["eyes_selected"] += 1

        if patient_selected:
            stats["patients_selected"] += 1

    rows.sort(key=lambda row: (row["class_index"], int(row["patient_id"]) if row["patient_id"].isdigit() else row["patient_id"], row["laterality"]))
    return rows, stats, annotation_path, image_root


def class_counts(rows):
    counts = {class_name: 0 for class_name in CLASS_ORDER}
    for row in rows:
        counts[row["class_name"]] += 1
    return counts


def subset_counts(rows):
    counts = {}
    for row in rows:
        counts[row["subset"]] = counts.get(row["subset"], 0) + 1
    return {subset: counts[subset] for subset in sorted(counts)}


def class_distribution(rows):
    distribution = {}
    for subset in sorted({row["subset"] for row in rows}):
        distribution[subset] = class_counts([row for row in rows if row["subset"] == subset])
    return distribution


def split_trainval_by_patient(rows, val_ratio: float, seed: int):
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

    grouped = {}
    for row in rows:
        grouped.setdefault(row["class_name"], {}).setdefault(row["patient_id"], []).append(row)

    for class_index, class_name in enumerate(CLASS_ORDER):
        patient_ids = list(grouped.get(class_name, {}))
        if not patient_ids:
            continue
        patient_ids.sort(key=lambda value: int(value) if str(value).isdigit() else str(value))
        rng = random.Random(seed * 1009 + class_index)
        rng.shuffle(patient_ids)
        val_count = int(round(len(patient_ids) * val_ratio))
        if len(patient_ids) > 1:
            val_count = min(max(1, val_count), len(patient_ids) - 1)
        else:
            val_count = 0
        val_patients = set(patient_ids[:val_count])
        for patient_id, patient_rows in grouped[class_name].items():
            subset = "val" if patient_id in val_patients else "train"
            for row in patient_rows:
                row["subset"] = subset
    rows.sort(
        key=lambda row: (
            {"train": 0, "val": 1, "test": 2}.get(row["subset"], 99),
            row["class_index"],
            int(row["patient_id"]) if row["patient_id"].isdigit() else row["patient_id"],
            row["laterality"],
        )
    )
    return rows


def write_split(rows, output_dir: Path, image_root: Path, annotation_path: Path, stats: dict, source_config: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    distribution = class_distribution(rows)
    for subset, counts in distribution.items():
        empty_classes = [class_name for class_name, count in counts.items() if count == 0]
        if empty_classes:
            raise RuntimeError(
                f"{source_config['source_set']} subset {subset} has empty target classes after filtering: {empty_classes}"
            )

    fieldnames = [
        "filepath",
        "class_name",
        "class_index",
        "subset",
        "patient_id",
        "laterality",
        "diagnostic_keywords",
        "source_set",
        "odir_patient_label",
    ]
    with (output_dir / "splits.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    dataset_root_from_split_dir = Path(os.path.relpath(image_root, output_dir)).as_posix()
    summary = {
        "source_dataset": "ODIR-2019/OIA-ODIR",
        "source_set": source_config["source_set"],
        "annotation_file": str(annotation_path),
        "dataset_root_from_split_dir": dataset_root_from_split_dir,
        "classes": list(CLASS_ORDER),
        "subset_mode": source_config["subset_mode"],
        "subset_counts": subset_counts(rows),
        "class_distribution": distribution,
        "selection_scope": "single-label ODIR N/D/G/C patients converted to image-level labels by eye keywords",
        "selection_rules": {
            "patient_level": "Keep only patients with exactly one positive ODIR label in N/D/G/C and no A/H/M/O positives.",
            "image_level": "Keep only eyes whose diagnostic keywords map to the same class as the selected patient label.",
            "normal": "normal fundus or only non-decisive quality terms",
            "diabetic_retinopathy": "diabetic retinopathy, nonproliferative retinopathy, or proliferative diabetic retinopathy",
            "glaucoma": "glaucoma or suspected glaucoma",
            "cataract": "cataract",
        },
        "stats": stats,
    }
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main():
    args = parse_args()
    odir_root = Path(args.odir_root).resolve()
    output_root = Path(args.output_root).resolve()
    if not odir_root.exists():
        raise FileNotFoundError(f"Missing ODIR root: {odir_root}")

    selected_sources = {key.lower() for key in args.sources}
    summaries = {}
    for source_config in SOURCE_CONFIGS:
        if source_config["key"] not in selected_sources:
            continue
        rows, stats, annotation_path, image_root = build_source_rows(odir_root, source_config)
        if source_config["subset_mode"] == "trainval":
            rows = split_trainval_by_patient(rows, val_ratio=args.val_ratio, seed=args.seed)
        output_dir = output_root / source_config["output_name"]
        summary = write_split(rows, output_dir, image_root, annotation_path, stats, source_config)
        summaries[source_config["key"]] = summary
        print(
            f"{source_config['source_set']}: wrote {len(rows)} image-level samples to "
            f"{output_dir / 'splits.csv'}"
        )
        print(f"  class distribution: {summary['class_distribution']}")

    if not summaries:
        raise RuntimeError("No ODIR sources were selected.")


if __name__ == "__main__":
    main()
