#!/usr/bin/env python3
"""
Draw precision-recall curves for each adverse event from a prediction CSV.

CSV columns (per your description):
    clip_id,
    bleeding_gt, bleeding_severity,
    mechanical_injury_gt, mechanical_injury_severity,
    thermal_injury_gt, thermal_injury_severity,
    prob_bleeding, pred_bleeding,
    prob_mechanical_injury, pred_mechanical_injury,
    prob_thermal_injury, pred_thermal_injury

Usage:
    python draw_pr_curves.py /mnt/iMVR/sudhanshu/Projects/ActionCLIP/predictions/fold1_bce_wt_v1.csv
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


def plot_pr_for_event(
    df,
    event_name,
    base_name,
    out_dir,
):
    """
    Plot and save PR curve for one adverse event.

    event_name: one of ["bleeding", "mechanical_injury", "thermal_injury"]
    """

    gt_col = f"{event_name}_gt"
    prob_col = f"prob_{event_name}"
    pred_col = f"pred_{event_name}"

    if gt_col not in df.columns or prob_col not in df.columns or pred_col not in df.columns:
        print(f"[WARN] Missing columns for event '{event_name}'. Skipping.")
        return

    y_true = df[gt_col].values.astype(int)
    y_score = df[prob_col].values.astype(float)
    y_pred = df[pred_col].values.astype(int)

    n_pos = int(y_true.sum())
    n_total = len(y_true)

    if n_pos == 0:
        print(f"[WARN] No positive samples for event '{event_name}' "
              f"({n_total} samples, {n_pos} positives). Skipping PR curve.")
        return

    # Precision–recall curve using score/probabilities
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    # F1, precision, recall at your existing prediction (pred_* column)
    f1_default = f1_score(y_true, y_pred, zero_division=0)
    prec_default = precision_score(y_true, y_pred, zero_division=0)
    rec_default = recall_score(y_true, y_pred, zero_division=0)

    # Best F1 obtainable by scanning the PR curve
    f1_curve = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.nanargmax(f1_curve)
    best_f1 = f1_curve[best_idx]
    best_prec = precision[best_idx]
    best_rec = recall[best_idx]

    print(f"== {event_name} ==")
    print(f"  #samples: {n_total}, #positives: {n_pos}")
    print(f"  F1 @pred (thresholded): {f1_default:.4f}")
    print(f"  Precision @pred: {prec_default:.4f}, Recall @pred: {rec_default:.4f}")
    print(f"  Best F1 from PR curve: {best_f1:.4f} (P={best_prec:.4f}, R={best_rec:.4f})")
    print(f"  Average precision (AP): {ap:.4f}")
    print()

    # Plot PR curve
    plt.figure(figsize=(6, 5))
    plt.step(recall, precision, where="post", label=f"PR curve (AP={ap:.3f})")
    plt.scatter(
        [best_rec],
        [best_prec],
        marker="o",
        label=f"Best F1={best_f1:.3f}",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        f"Precision-Recall for {event_name.replace('_', ' ')}\n"
        f"{base_name}"
    )
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left")
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base_name}_{event_name}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[INFO] Saved PR curve to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Draw precision-recall curves for bleeding, mechanical_injury, thermal_injury."
    )
    parser.add_argument("csv_path", help="Path to prediction CSV file.")
    parser.add_argument(
        "--out_dir",
        default="./PR_curves",
        help="Directory to save PR curve PNGs (default: ./PR_curves).",
    )
    args = parser.parse_args()

    csv_path = args.csv_path
    out_dir = args.out_dir

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    base_name = os.path.splitext(os.path.basename(csv_path))[0]

    print(f"[INFO] Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Events to process
    events = ["bleeding", "mechanical_injury", "thermal_injury"]

    for event in events:
        plot_pr_for_event(df, event, base_name, out_dir)


if __name__ == "__main__":
    main()
