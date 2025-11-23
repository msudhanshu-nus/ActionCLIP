#!/usr/bin/env python3
"""
Utility to convert MultiBypass140 clip metadata CSV files into the txt format
expected by ActionCLIP.

For every hospital in HOSPITALS, each fold in FOLDS, and each split in SPLITS,
the script reads the metadata CSV from
`/mnt/nas/mishra/datasets/MultiBypass140/labels/{hospital}/labels_by70_splits`
and writes the aggregated rows into the matching txt file under
`/mnt/iMVR/sudhanshu/Projects/ActionCLIP/lists/mby140`.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Tuple

HOSPITALS = ("bern", "strasbourg")
FOLDS = (0, 1, 2, 3, 4)
SPLITS = ("test", "train", "val")

LABEL_ROOT = Path("/mnt/nas/mishra/datasets/MultiBypass140/labels")
FRAME_ROOT = Path("/mnt/nas/mishra/datasets/MultiBypass140/mby140_clip_level_frames")
OUTPUT_ROOT = Path("/mnt/iMVR/sudhanshu/Projects/ActionCLIP/lists/mby140")

# Allow a few alias spellings so the script remains robust to column naming.
FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "clip_name": ("clip_name", "clip"),
    "start_frame_id": ("start_frame_id", "start_frame"),
    "end_frame_id": ("end_frame_id", "end_frame"),
    "num_frames": ("num_frames", "num_frame"),
    "step_GT": ("step_GT", "step_gt"),
    "Phase_GT": ("Phase_GT", "phase_GT", "phase_gt"),
    "Bleeding": ("Bleeding",),
    "Mechanical": ("Mechanical",),
    "Thermal": ("Thermal",),
    "Bleeding_severity": ("Bleeding_severity", "Bleeding severity"),
    "Mechanical_severity": ("Mechanical_severity", "Mechanical severity"),
    "Thermal_severity": (
        "Thermal_severity",
        "Thermal severity",
        "thermal severity",
    ),
}


def read_csv_rows(csv_path: Path) -> Iterable[Dict[str, str]]:
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def get_value(row: Dict[str, str], field: str, csv_path: Path) -> str:
    aliases = FIELD_ALIASES.get(field, (field,))
    for alias in aliases:
        if alias in row and row[alias] not in (None, ""):
            return row[alias].strip()
    raise KeyError(f"Missing required column {aliases} in {csv_path}")


def process_split(hospital: str, fold: int, split: str) -> None:
    csv_path = (
        LABEL_ROOT
        / hospital
        / "labels_by70_splits"
        / f"clips_metadata_{split}_fold{fold}.csv"
    )
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing metadata csv: {csv_path}")

    out_dir = OUTPUT_ROOT / f"fold{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mb_{split}.txt"

    rows_written = 0
    with out_path.open("a", newline="") as dst:
        for row in read_csv_rows(csv_path):
            clip_name = get_value(row, "clip_name", csv_path)
            num_frames = get_value(row, "num_frames", csv_path)
            bleeding = get_value(row, "Bleeding", csv_path)
            mechanical = get_value(row, "Mechanical", csv_path)
            thermal = get_value(row, "Thermal", csv_path)
            bleed_sev = get_value(row, "Bleeding_severity", csv_path)
            mech_sev = get_value(row, "Mechanical_severity", csv_path)
            therm_sev = get_value(row, "Thermal_severity", csv_path)
            step_gt = get_value(row, "step_GT", csv_path)
            phase_gt = get_value(row, "Phase_GT", csv_path)

            # Read (but do not currently emit) the frame range columns to ensure
            # the CSV contains them.
            _ = (
                get_value(row, "start_frame_id", csv_path),
                get_value(row, "end_frame_id", csv_path),
            )

            clip_path = FRAME_ROOT / hospital / clip_name
            line = " ".join(
                [
                    str(clip_path),
                    num_frames,
                    bleeding,
                    mechanical,
                    thermal,
                    bleed_sev,
                    mech_sev,
                    therm_sev,
                    step_gt,
                    phase_gt,
                ]
            )
            dst.write(f"{line}\n")
            rows_written += 1

    print(f"[{hospital}] fold {fold} {split}: wrote {rows_written} rows -> {out_path}")


def main() -> None:
    for hospital in HOSPITALS:
        for fold in FOLDS:
            for split in SPLITS:
                process_split(hospital, fold, split)


if __name__ == "__main__":
    main()
