"""Local CSV validation. No transcript contents appear in error messages."""

import csv
import hashlib
import itertools
from pathlib import Path


def read_rows(path, text_col="input", label_col="output", group_col=None):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {text_col, label_col} | ({group_col} if group_col else set())
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV is missing required columns.")
        rows = []
        for row in reader:
            if any(not isinstance(row.get(k), str) or not row[k].strip() for k in required):
                raise ValueError("CSV contains empty text, labels, or group identifiers.")
            rows.append({"text": row[text_col].strip(), "label": row[label_col].strip(),
                         "group": row[group_col].strip() if group_col else None})
    if not rows:
        raise ValueError("CSV contains no records.")
    return rows


def fingerprint(text):
    normalized = " ".join(text.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def check_splits(splits, grouped=False):
    for left, right in itertools.combinations(splits.values(), 2):
        if {fingerprint(r["text"]) for r in left} & {fingerprint(r["text"]) for r in right}:
            raise ValueError("Duplicate transcripts cross dataset splits; revise the splits.")
        if grouped and {r["group"] for r in left} & {r["group"] for r in right}:
            raise ValueError("Participant identifiers cross dataset splits; revise the splits.")


def label_mapping(rows):
    labels = sorted({row["label"] for row in rows})
    if len(labels) != 2:
        raise ValueError("This screening sample requires exactly two training classes.")
    return {label: i for i, label in enumerate(labels)}


def encode_labels(rows, mapping):
    if any(row["label"] not in mapping for row in rows):
        raise ValueError("A split contains a label absent from the training mapping.")
    return [mapping[row["label"]] for row in rows]
