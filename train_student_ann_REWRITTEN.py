#!/usr/bin/env python3
"""
train_student_ann_REWRITTEN.py

Purpose
-------
Train a *single-output* forward ANN classifier for weldability:
    y = Weldable (0/1)

This removes the confusing auxiliary head (Crack_OK) and makes the model and
OOD (out-of-distribution) robustness checks reviewer-proof.

What it does
------------
1) Loads the in-distribution training dataset (ann_dataset_full_v23.csv)
2) Trains an MLPClassifier on standardized inputs
3) Saves:
      - student_ann_model_weldable.pkl
      - scaler_weldable.pkl
      - feature_cols_weldable.json (feature order for inference)
4) (Optional) If an OOD dataset is present (extreme_cases.csv or extreme_results_final.csv),
   evaluates OOD performance overall and per regime and writes:
      - ood_metrics_weldable.csv
      - ood_confusion_by_regime.txt

Run
---
python train_student_ann_REWRITTEN.py \
  --train_csv ann_dataset_full_v23.csv \
  --model_out student_ann_model_weldable.pkl \
  --scaler_out scaler_weldable.pkl

Optional OOD eval:
python train_student_ann_REWRITTEN.py --ood_csv extreme_cases.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    classification_report,
)

import joblib


# ----------------------------
# Utilities
# ----------------------------
def resolve_columns(df: pd.DataFrame) -> dict:
    """
    Resolve column names robustly across slightly different naming conventions.
    Returns a dict with keys: feature_cols, y_col, regime_col (optional), mode_col (optional).
    """
    # Candidate name maps (training csv uses parentheses/spaces; OOD often uses underscores)
    candidates = {
        "material_flag": ["Material_Flag", "material_flag"],
        "V1": ["V1 (m/s)", "V1_mps", "V1"],
        "w_top": ["w_top (mm)", "w_top_mm", "w_top"],
        "w_bot": ["w_bot (mm)", "w_bot_mm", "w_bot"],
        "d12": ["d12 (mm)", "d12_mm", "d12"],
        "d23": ["d23 (mm)", "d23_mm", "d23"],
        "h1": ["h1 (mm)", "h1_mm", "h1"],
        "h2": ["h2 (mm)", "h2_mm", "h2"],
        "h3": ["h3 (mm)", "h3_mm", "h3"],
        "weldable": ["Weldable", "weldable", "weldable_label", "weldable_label_partial"],
        "regime": ["regime", "Regime"],
        "failure_mode": ["failure_mode", "Failure_Mode"],
    }

    def pick(name_list):
        for n in name_list:
            if n in df.columns:
                return n
        return None

    cols = {k: pick(v) for k, v in candidates.items()}

    feature_cols = [
        cols["material_flag"],
        cols["V1"],
        cols["w_top"],
        cols["w_bot"],
        cols["d12"],
        cols["d23"],
        cols["h1"],
        cols["h2"],
        cols["h3"],
    ]
    if any(c is None for c in feature_cols):
        missing = [str(feature_cols[i]) for i, c in enumerate(feature_cols) if c is None]
        raise ValueError(
            "Could not resolve all feature columns. "
            f"Resolved list = {feature_cols}. Missing entries at positions: {missing}. "
            "Please check your CSV headers."
        )

    y_col = cols["weldable"]  # for training set
    regime_col = cols["regime"]
    mode_col = cols["failure_mode"]

    return {
        "feature_cols": feature_cols,
        "y_col": y_col,
        "regime_col": regime_col,
        "mode_col": mode_col,
    }


def bin_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "cm": cm}


def ensure_binary(series: pd.Series) -> pd.Series:
    """Force labels to 0/1 integers."""
    # Common cases: already 0/1; or boolean; or strings "Weldable"/"Not Weldable"
    if series.dtype == bool:
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return (series.astype(float) > 0.5).astype(int)
    # Strings
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["1", "true", "yes", "weldable"]).astype(int)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="ann_dataset_full_v23.csv")
    ap.add_argument("--ood_csv", default=None, help="Optional OOD CSV (e.g., extreme_cases.csv)")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)

    ap.add_argument("--hidden", default="128,64",
                    help="Comma-separated hidden layer sizes, e.g., '128,64' or '64,64,32'")
    ap.add_argument("--max_iter", type=int, default=1500)
    ap.add_argument("--alpha", type=float, default=1e-4, help="L2 regularization strength")
    ap.add_argument("--early_stopping", action="store_true", help="Enable early stopping (recommended)")

    ap.add_argument("--model_out", default="student_ann_model_weldable.pkl")
    ap.add_argument("--scaler_out", default="scaler_weldable.pkl")
    ap.add_argument("--feature_out", default="feature_cols_weldable.json")
    ap.add_argument("--ood_metrics_out", default="ood_metrics_weldable.csv")
    ap.add_argument("--ood_confusion_out", default="ood_confusion_by_regime.txt")

    args = ap.parse_args()

    train_path = Path(args.train_csv)
    if not train_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_path.resolve()}")

    print(f"--- Loading training data: {train_path} ---")
    df = pd.read_csv(train_path)

    colinfo = resolve_columns(df)
    feature_cols = colinfo["feature_cols"]
    y_col = colinfo["y_col"]

    if y_col is None:
        raise ValueError(
            "Could not find a weldability target column in the training CSV. "
            "Expected one of: Weldable / weldable / weldable_label / weldable_label_partial."
        )

    X = df[feature_cols].copy()
    y = ensure_binary(df[y_col])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    hidden = tuple(int(x.strip()) for x in args.hidden.split(",") if x.strip())
    print(f"--- Training forward ANN (single-output Weldable) ---")
    print(f"Hidden layers: {hidden} | max_iter={args.max_iter} | alpha={args.alpha} | early_stopping={args.early_stopping}")

    model = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        max_iter=args.max_iter,
        random_state=args.random_state,
        alpha=args.alpha,
        early_stopping=args.early_stopping,
        n_iter_no_change=25,
        verbose=False,
    )
    model.fit(X_train_scaled, y_train)

    # In-distribution evaluation
    y_pred = model.predict(X_test_scaled)
    m = bin_metrics(y_test.to_numpy(), y_pred)
    print("\n--- In-distribution test performance (held-out) ---")
    print(f"Accuracy={m['accuracy']:.3f} | Precision={m['precision']:.3f} | Recall={m['recall']:.3f} | F1={m['f1']:.3f}")
    print("Confusion matrix [[TN FP],[FN TP]]:\n", m["cm"])
    print("\nClassification report:\n", classification_report(y_test, y_pred, zero_division=0))

    # Save artifacts
    joblib.dump(model, args.model_out)
    joblib.dump(scaler, args.scaler_out)
    with open(args.feature_out, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    print(f"\nSaved model  : {args.model_out}")
    print(f"Saved scaler : {args.scaler_out}")
    print(f"Saved feature order: {args.feature_out}")

    # Optional OOD evaluation
    if args.ood_csv:
        ood_path = Path(args.ood_csv)
        if not ood_path.exists():
            raise FileNotFoundError(f"OOD CSV not found: {ood_path.resolve()}")

        print(f"\n--- OOD evaluation on: {ood_path} ---")
        ood = pd.read_csv(ood_path)

        ood_colinfo = resolve_columns(ood)
        ood_feature_cols = ood_colinfo["feature_cols"]

        # Ensure same feature order as training
        # Map OOD columns into training feature names by position
        X_ood = ood[ood_feature_cols].copy()
        X_ood.columns = feature_cols  # rename to training order for clarity
        X_ood_scaled = scaler.transform(X_ood)

        # Determine OOD ground truth weldable:
        # Prefer explicit weldability column; else derive from failure_mode.
        ood_y_col = ood_colinfo["y_col"]
        if ood_y_col is not None:
            y_true = ensure_binary(ood[ood_y_col]).to_numpy()
        else:
            mode_col = ood_colinfo["mode_col"]
            if mode_col is None:
                raise ValueError(
                    "OOD CSV has no weldability label column and no failure_mode column. "
                    "Need at least one to compute ground truth."
                )
            y_true = (ood[mode_col].astype(str).str.strip() == "Weldable").astype(int).to_numpy()

        y_hat = model.predict(X_ood_scaled).astype(int)

        # Overall
        overall = bin_metrics(y_true, y_hat)
        print(f"OOD Overall: Accuracy={overall['accuracy']:.3f} | Precision={overall['precision']:.3f} | Recall={overall['recall']:.3f} | F1={overall['f1']:.3f}")

        # By regime (if present)
        rows = []
        confusion_lines = []
        regime_col = ood_colinfo["regime_col"]
        if regime_col and regime_col in ood.columns:
            for reg, sub in ood.groupby(regime_col):
                sub_idx = sub.index.to_numpy()
                mm = bin_metrics(y_true[sub_idx], y_hat[sub_idx])
                rows.append({
                    "regime": reg,
                    "n": int(len(sub_idx)),
                    "accuracy": mm["accuracy"],
                    "precision": mm["precision"],
                    "recall": mm["recall"],
                    "f1": mm["f1"],
                    "TN": int(mm["cm"][0,0]),
                    "FP": int(mm["cm"][0,1]),
                    "FN": int(mm["cm"][1,0]),
                    "TP": int(mm["cm"][1,1]),
                })
                confusion_lines.append(f"REGIME: {reg} (n={len(sub_idx)})\n{mm['cm']}\n")
        else:
            rows.append({
                "regime": "ALL",
                "n": int(len(y_true)),
                "accuracy": overall["accuracy"],
                "precision": overall["precision"],
                "recall": overall["recall"],
                "f1": overall["f1"],
                "TN": int(overall["cm"][0,0]),
                "FP": int(overall["cm"][0,1]),
                "FN": int(overall["cm"][1,0]),
                "TP": int(overall["cm"][1,1]),
            })
            confusion_lines.append(f"REGIME: ALL (n={len(y_true)})\n{overall['cm']}\n")

        out_df = pd.DataFrame(rows).sort_values("regime")
        out_df.to_csv(args.ood_metrics_out, index=False)

        with open(args.ood_confusion_out, "w", encoding="utf-8") as f:
            f.write("\n".join(confusion_lines))

        print(f"Saved OOD metrics table: {args.ood_metrics_out}")
        print(f"Saved OOD confusion matrices: {args.ood_confusion_out}")


if __name__ == "__main__":
    main()
