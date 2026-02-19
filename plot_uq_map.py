"""
Quick plotting helper for UQ maps / boundary effects.

You can use this in two modes:
1) scatter: show uncertainty (sigma) over two chosen input dimensions (e.g., V1-beta1 or V1-d23)
2) grid: interpolate a heatmap over a 2D grid (simple nearest interpolation)

Usage:
  python plot_uq_map.py --uq data/uq_predictions.csv --x V1 --y d23 --z sigma_V1 --out fig_uq.png

NOTE: x and y here must exist as columns in the UQ CSV. If your inverse-model inputs are not (V1, d23),
choose your GUI input features (e.g., Vc12, Vc23, beta1, ...). The point is to show "uncertainty spikes near boundaries".
"""
from __future__ import annotations

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uq", required=True, help="CSV produced by mc_dropout_infer.py")
    ap.add_argument("--x", required=True, help="x-axis column name")
    ap.add_argument("--y", required=True, help="y-axis column name")
    ap.add_argument("--z", required=True, help="uncertainty column name (e.g., sigma_V1 or sigma_d23)")
    ap.add_argument("--out", required=True, help="output image path (png)")
    ap.add_argument("--s", type=float, default=14.0, help="marker size")
    args = ap.parse_args()

    df = pd.read_csv(args.uq)
    x = df[args.x].to_numpy()
    y = df[args.y].to_numpy()
    z = df[args.z].to_numpy()

    plt.figure()
    sc = plt.scatter(x, y, c=z, s=args.s)
    plt.xlabel(args.x)
    plt.ylabel(args.y)
    plt.colorbar(sc, label=args.z)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
