"""
B) Monte Carlo Dropout inference: mean + std + 95% prediction intervals

Typical usage:
  python mc_dropout_infer.py --artifacts artifacts_uq --cases data/cases.csv --out data/uq_predictions.csv --T 50

cases.csv must contain X_COLS (same as training).

The output CSV includes:
  - mu_<target>   : predictive mean
  - sigma_<target>: predictive std (epistemic proxy)
  - lo95_<target>, hi95_<target>: 95% PI assuming ~Gaussian

This is what you cite in the rebuttal as "prediction uncertainty quantified by MC Dropout."
"""
from __future__ import annotations

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import tensorflow as tf

from uq_utils import mc_dropout_predict

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# EDIT THESE to match your dataset columns
X_COLS = [
    # e.g., "Vc12", "Vc23", "beta1", "TI1", "TI2", "mat_flag"
]
Y_COLS = [
    # e.g., "V1", "w_top", "w_bot", "d23", "h1", "h2", "h3"
]
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True, help="Directory with model.keras + scalers")
    ap.add_argument("--cases", required=True, help="CSV with X_COLS")
    ap.add_argument("--out", required=True, help="Output CSV")
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--batch", type=int, default=2048)
    args = ap.parse_args()

    art = Path(args.artifacts)
    model = tf.keras.models.load_model(art/"model.keras")
    sx = joblib.load(art/"scaler_X.joblib")
    sy = joblib.load(art/"scaler_Y.joblib")

    df = pd.read_csv(args.cases)
    assert len(X_COLS) > 0 and len(Y_COLS) > 0, "Set X_COLS and Y_COLS in this script."
    X = df[X_COLS].to_numpy(dtype=float)
    Xs = sx.transform(X)

    mu_s, sig_s = mc_dropout_predict(model, Xs, T=args.T, batch_size=args.batch)

    mu  = sy.inverse_transform(mu_s)
    # std needs scaling by target scaler stds (StandardScaler): y = mu_s*std + mean
    # so sigma_y = sigma_s * std_y
    y_std = sy.scale_
    sig = sig_s * y_std[None, :]

    out = df.copy()
    for j, name in enumerate(Y_COLS):
        out[f"mu_{name}"] = mu[:, j]
        out[f"sigma_{name}"] = sig[:, j]
        out[f"lo95_{name}"] = mu[:, j] - 1.96 * sig[:, j]
        out[f"hi95_{name}"] = mu[:, j] + 1.96 * sig[:, j]

    out.to_csv(args.out, index=False)
    print(f"Saved UQ predictions: {args.out}")

if __name__ == "__main__":
    main()
