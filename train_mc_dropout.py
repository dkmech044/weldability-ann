"""
B) Train a regression HPINN (inverse design) with Dropout layers for MC-Dropout UQ

This reuses your stated architecture (64-64-32) but inserts dropout after each hidden layer.
At inference, you keep dropout ON and sample T passes to estimate uncertainty.

Typical usage:
  python train_mc_dropout.py --data data/aug.csv --outdir artifacts_uq

Outputs:
  - artifacts_uq/model.keras
  - artifacts_uq/scaler_X.joblib
  - artifacts_uq/scaler_Y.joblib
  - artifacts_uq/train_history.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# EDIT THESE to match your dataset columns
X_COLS = [
    # e.g., "Vc12", "Vc23", "beta1", "TI1", "TI2", "mat_flag"
]
Y_COLS = [
    # e.g., "V1", "w_top", "w_bot", "d23", "h1", "h2", "h3"
]
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def build_model(input_dim: int, output_dim: int, dropout_p: float = 0.15) -> keras.Model:
    inp = keras.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(inp)
    x = layers.Dropout(dropout_p)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout_p)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(dropout_p)(x)
    out = layers.Dense(output_dim, activation="linear")(x)
    return keras.Model(inp, out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV dataset path (augmented recommended)")
    ap.add_argument("--outdir", required=True, help="Directory to save model + scalers")
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    assert len(X_COLS) > 0 and len(Y_COLS) > 0, "Set X_COLS and Y_COLS in this script."

    X = df[X_COLS].to_numpy(dtype=float)
    Y = df[Y_COLS].to_numpy(dtype=float)

    # split 80/10/10 as stated in the manuscript
    X_train, X_tmp, Y_train, Y_tmp = train_test_split(
        X, Y, test_size=0.20, random_state=args.seed
    )
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_tmp, Y_tmp, test_size=0.50, random_state=args.seed
    )

    sx = StandardScaler().fit(X_train)
    sy = StandardScaler().fit(Y_train)

    X_train_s = sx.transform(X_train)
    X_val_s   = sx.transform(X_val)
    X_test_s  = sx.transform(X_test)

    Y_train_s = sy.transform(Y_train)
    Y_val_s   = sy.transform(Y_val)
    Y_test_s  = sy.transform(Y_test)

    model = build_model(X_train_s.shape[1], Y_train_s.shape[1], dropout_p=args.dropout)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")]
    )

    cb = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5),
    ]

    hist = model.fit(
        X_train_s, Y_train_s,
        validation_data=(X_val_s, Y_val_s),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=cb,
        verbose=2
    )

    # Evaluate
    test = model.evaluate(X_test_s, Y_test_s, verbose=0)
    print({"test_loss_mse": float(test[0]), "test_mae_scaled": float(test[1])})

    # Save artifacts
    model.save(outdir/"model.keras")
    joblib.dump(sx, outdir/"scaler_X.joblib")
    joblib.dump(sy, outdir/"scaler_Y.joblib")

    pd.DataFrame(hist.history).to_csv(outdir/"train_history.csv", index=False)

    # Also export a small split for reproducibility
    np.savez_compressed(
        outdir/"splits.npz",
        X_train=X_train, Y_train=Y_train,
        X_val=X_val,     Y_val=Y_val,
        X_test=X_test,   Y_test=Y_test,
    )

    print(f"Saved model + scalers in: {outdir}")

if __name__ == "__main__":
    main()
