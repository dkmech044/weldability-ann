#!/usr/bin/env python3

"""
Run a literature tri-material case study (Material_Flag=2) through the analytical forward model.

Example usage:
  python run_trimetal_case_study.py --out meng_trimetal_results.csv

If the literature does not report die widths (w_top/w_bot), we sweep a small grid to find if a weldable window exists.
"""
import argparse
import pandas as pd

from forward_model_full_v23_trimetal import forward_model_v2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="meng_trimetal_results.csv")
    ap.add_argument("--wmin", type=float, default=10.0)
    ap.add_argument("--wmax", type=float, default=30.0)
    ap.add_argument("--wstep", type=float, default=5.0)
    args = ap.parse_args()

    # Literature inputs (edit if needed)
    V1_list = [675, 720, 855, 960]          # m/s
    h1, h2, h3 = 2.0, 1.0, 2.0              # mm
    d12, d23 = 3.0, 1.5                     # mm
    mat_flag = 2                             # 5A06 / 3003 / 321SS

    ws = []
    w = args.wmin
    while w <= args.wmax + 1e-9:
        ws.append(round(w, 6))
        w += args.wstep

    rows = []
    for V1 in V1_list:
        for w_top in ws:
            for w_bot in ws:
                res = forward_model_v2(
                    material_flag=mat_flag,
                    V1=V1, w_top=w_top, w_bot=w_bot,
                    d12=d12, d23=d23, h1=h1, h2=h2, h3=h3
                )
                rows.append({
                    "Material_Flag": mat_flag,
                    "V1 (m/s)": V1,
                    "w_top (mm)": w_top,
                    "w_bot (mm)": w_bot,
                    "h1 (mm)": h1, "h2 (mm)": h2, "h3 (mm)": h3,
                    "d12 (mm)": d12, "d23 (mm)": d23,
                    **res
                })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"[OK] wrote {args.out}  (rows={len(df)})")

    # quick console summary: best (max) weldable probability is not available; we print weldable counts
    print("\nWeldable counts by V1:")
    print(df.groupby("V1 (m/s)")["Weldable"].sum())

if __name__ == "__main__":
    main()
