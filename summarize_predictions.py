#!/usr/bin/env python3
import argparse
import sys

import pandas as pd


CLASS_CONFIG = {
    "bleeding": {
        "gt": "bleeding_gt",
        "pred": "pred_bleeding",
        "prob": "prob_bleeding",
    },
    "mechanical": {
        "gt": "mechanical_injury_gt",
        "pred": "pred_mechanical_injury",
        "prob": "prob_mechanical_injury",
    },
    "thermal": {
        "gt": "thermal_injury_gt",
        "pred": "pred_thermal_injury",
        "prob": "prob_thermal_injury",
    },
}


def _validate_columns(df: pd.DataFrame) -> None:
    missing = set()
    for config in CLASS_CONFIG.values():
        for col in config.values():
            if col not in df.columns:
                missing.add(col)
    if missing:
        cols = ", ".join(sorted(missing))
        raise ValueError(f"Missing expected column(s): {cols}")


def _compute_confusion(gt: pd.Series, pred: pd.Series) -> dict:
    gt = gt.fillna(0)
    pred = pred.fillna(0)

    tp = int(((pred == 1) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def _probability_range(prob: pd.Series) -> tuple:
    prob = prob.dropna()
    if prob.empty:
        return None, None
    return float(prob.min()), float(prob.max())


def summarize(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    _validate_columns(df)

    for name, config in CLASS_CONFIG.items():
        gt_col = config["gt"]
        pred_col = config["pred"]
        prob_col = config["prob"]

        confusion = _compute_confusion(df[gt_col], df[pred_col])
        prob_min, prob_max = _probability_range(df[prob_col])

        print(f"{name.capitalize()}")
        print(f"  TP (pred=1, gt=1): {confusion['TP']}")
        print(f"  TN (pred=0, gt=0): {confusion['TN']}")
        print(f"  FP (pred=1, gt=0): {confusion['FP']}")
        print(f"  FN (pred=0, gt=1): {confusion['FN']}")
        if prob_min is None:
            print("  Probability range: no probability values")
        else:
            print(f"  Probability range: min={prob_min:.4f}, max={prob_max:.4f}")
        print()


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize confusion metrics and probability ranges per adverse event class."
    )
    parser.add_argument("csv", help="Path to the predictions CSV file.")
    args = parser.parse_args(argv)

    summarize(args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
