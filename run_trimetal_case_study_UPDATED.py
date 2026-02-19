#!/usr/bin/env python3
"""run_trimetal_case_study.py (UPDATED)

Why this exists
---------------
Most literature tri-material VFAW papers report (V1, h1/h2/h3, d12/d23, materials),
but they typically do NOT report the die geometry used in *this* paper's wedge/die-based
analytical model (w_top, w_bot).

So, for the reviewer “generalizability” check, we treat (w_top, w_bot) as DESIGN variables
and report the *feasible die-width window* where the literature stack is predicted weldable.

Recommended run (your paper's die ranges)
-----------------------------------------
python run_trimetal_case_study_UPDATED.py --out meng_trimetal_results_wPaperRange.csv \
  --wtop_min 6 --wtop_max 14 --wtop_step 1 \
  --wbot_min 0.75 --wbot_max 5 --wbot_step 0.25 \
  --summary_out meng_trimetal_results_wPaperRange_summary.csv

Notes
-----
- Material properties are NOT passed via CSV. They live in the material card for Material_Flag=2
  inside forward_model_full_v23_trimetal.py.
- We enforce w_bot < w_top because w_bot >= w_top degenerates the wedge and can create singular
  angles in the dynamic-geometry formulation.
"""

import argparse
import pandas as pd

from forward_model_full_v23_trimetal import forward_model_v2


def frange(a: float, b: float, step: float):
    """Inclusive float range."""
    xs = []
    x = float(a)
    b = float(b)
    step = float(step)
    if step <= 0:
        raise ValueError("step must be > 0")
    while x <= b + 1e-9:
        xs.append(round(x, 6))
        x += step
    return xs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="meng_trimetal_results.csv", help="Full grid results CSV")
    ap.add_argument("--summary_out", default=None, help="Summary CSV: feasible die-width window per V1")
    ap.add_argument("--wtop_min", type=float, default=6.0)
    ap.add_argument("--wtop_max", type=float, default=14.0)
    ap.add_argument("--wtop_step", type=float, default=1.0)
    ap.add_argument("--wbot_min", type=float, default=0.75)
    ap.add_argument("--wbot_max", type=float, default=5.0)
    ap.add_argument("--wbot_step", type=float, default=0.25)
    ap.add_argument("--allow_wbot_ge_wtop", action="store_true",
                    help="If set, do NOT enforce w_bot < w_top (NOT recommended).")
    args = ap.parse_args()

    # -----------------------------
    # Literature inputs (EDIT HERE if you switch to another paper)
    # -----------------------------
    V1_list = [675, 720, 855, 960]          # m/s
    h1, h2, h3 = 2.0, 1.0, 2.0              # mm
    d12, d23 = 3.0, 1.5                     # mm
    mat_flag = 2                             # 5A06 / 3003 / 321SS (see forward_model_full_v23_trimetal.py)

    w_top_list = frange(args.wtop_min, args.wtop_max, args.wtop_step)
    w_bot_list = frange(args.wbot_min, args.wbot_max, args.wbot_step)

    rows = []
    for V1 in V1_list:
        for w_top in w_top_list:
            for w_bot in w_bot_list:
                if (not args.allow_wbot_ge_wtop) and (w_bot >= w_top):
                    continue

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

    # -----------------------------
    # The “range of (w_top, w_bot) that works” (not a full map in the paper text)
    # -----------------------------
    print("\nWeldable counts by V1:")
    print(df.groupby("V1 (m/s)")["Weldable"].sum())

    weld_df = df[df["Weldable"] == 1].copy()

    if args.summary_out is None:
        args.summary_out = args.out.replace(".csv", "_summary.csv")

    if len(weld_df) == 0:
        summary = pd.DataFrame([{
            "V1 (m/s)": v,
            "n_weldable": 0,
            "w_top_min": None, "w_top_max": None,
            "w_bot_min": None, "w_bot_max": None
        } for v in V1_list])
    else:
        summary = (weld_df.groupby("V1 (m/s)")
                   .agg(n_weldable=("Weldable", "size"),
                        w_top_min=("w_top (mm)", "min"),
                        w_top_max=("w_top (mm)", "max"),
                        w_bot_min=("w_bot (mm)", "min"),
                        w_bot_max=("w_bot (mm)", "max"))
                   .reset_index())

    summary.to_csv(args.summary_out, index=False)
    print(f"[OK] wrote {args.summary_out}")
    print("\nFeasible die-width window (from the sweep):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
