
import argparse
import math
import numpy as np
import joblib
from tensorflow.keras.models import load_model


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def compute_energy_kJ(material_flag, h1_mm, V1):
    """
    Compute approximate capacitor input energy (kJ) from V1 and h1,
    using the same form as §2.3.4:

        E_input = (1/(2 η)) * ρ1 * A_eff * h1 * V1^2

    We take:
      - ρ1 = 2700 kg/m^3 for both A6451P-T4 and AA1060 (Table 2)
      - A_eff = 30 mm^2 = 30e-6 m^2  (active zone area from experiments)
      - η ≈ 0.02 chosen so that V1 ≈ 1403 m/s at h1 = 1 mm corresponds to ~4 kJ
    """
    rho1 = 2700.0  # kg/m^3, both Al alloys in this work
    A_eff = 30e-6  # m^2
    eta = 0.02

    h1_m = h1_mm / 1000.0
    E_joule = 0.5 * rho1 * A_eff * h1_m * (V1 ** 2) / eta
    return E_joule / 1000.0  # kJ


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Inverse ANN (geometry -> process): given material stack and plate "
            "thicknesses (h1, h2, h3 in mm), predict a weldable combination of "
            "V1, w_top, w_bot, d23, and corresponding input energy."
        )
    )

    parser.add_argument(
        "--material_flag",
        type=int,
        required=True,
        help="0 = Al–Fe–Fe, 1 = Al–Cu–Cu",
    )
    parser.add_argument(
        "--h1_mm", type=float, required=True, help="Flyer thickness h1 in mm"
    )
    parser.add_argument(
        "--h2_mm", type=float, required=True, help="Intermediate thickness h2 in mm"
    )
    parser.add_argument(
        "--h3_mm", type=float, required=True, help="Target thickness h3 in mm"
    )

    args = parser.parse_args()

    # Load scalers + model
    scaler_X = joblib.load("scaler_X_geom2proc_v23.pkl")
    scaler_Y = joblib.load("scaler_Y_geom2proc_v23.pkl")
    model = load_model("weldability_inverse_geom2proc_v23.h5")

    # Build raw input vector (same order as training)
    X_raw = np.array([[
        float(args.material_flag),
        float(args.h1_mm),
        float(args.h2_mm),
        float(args.h3_mm),
    ]], dtype=np.float32)

    # Scale, predict, inverse-transform
    X_s = scaler_X.transform(X_raw)
    Y_pred_s = model.predict(X_s, verbose=0)
    Y_pred = scaler_Y.inverse_transform(Y_pred_s)[0]

    V1, w_top, w_bot, d23 = Y_pred

    # Clamp to ranges used in dataset
    V1    = clamp(V1,    600.0, 1800.0)
    w_top = clamp(w_top,   6.0,   14.0)
    w_bot = clamp(w_bot,   0.5,    5.0)
    d23   = clamp(d23,     0.5,    4.0)

    # d12 is fixed by material system in the forward model: 3 mm or 2 mm
    d12 = 3.0 if int(args.material_flag) == 0 else 2.0

    # Compute corresponding approximate input energy (kJ)
    E_kJ = compute_energy_kJ(args.material_flag, args.h1_mm, V1)

    print("\nPredicted weldable process/geometry (ANN inverse design):")
    print(f"  Material_Flag = {int(args.material_flag)}  (0 = Al–Fe–Fe, 1 = Al–Cu–Cu)")
    print(f"  h1            ≈ {args.h1_mm:.2f} mm")
    print(f"  h2            ≈ {args.h2_mm:.2f} mm")
    print(f"  h3            ≈ {args.h3_mm:.2f} mm")
    print()
    print(f"  V1            ≈ {V1:.1f} m/s")
    print(f"  w_top         ≈ {w_top:.2f} mm")
    print(f"  w_bot         ≈ {w_bot:.2f} mm")
    print(f"  d12           ≈ {d12:.2f} mm  (fixed by stack)")
    print(f"  d23           ≈ {d23:.2f} mm")
    print()
    print(f"  Estimated input energy ≈ {E_kJ:.2f} kJ")

if __name__ == "__main__":
    main()
