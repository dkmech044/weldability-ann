import argparse
import numpy as np
import joblib
from tensorflow.keras.models import load_model


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def main():
    parser = argparse.ArgumentParser(
        description="Inverse ANN: predict V1, w_top, w_bot, d12, d23, h1, h2, h3 "
                    "from physics-level inputs (β1, T_int, Vc12, Vimpact)."
    )

    parser.add_argument("--material_flag", type=int, required=True,
                        help="0 = Al–Fe–Fe, 1 = Al–Cu–Cu")
    parser.add_argument("--beta1_deg", type=float, required=True,
                        help="Representative β1 in degrees")
    parser.add_argument("--T_int1", type=float, required=True,
                        help="Interface 1 temperature (K)")
    parser.add_argument("--T_int2", type=float, required=True,
                        help="Interface 2 temperature (K)")
    parser.add_argument("--clearance_ok", type=int, default=1,
                        help="1 if clearance constraints should be satisfied, else 0")
    parser.add_argument("--Vc12", type=float, required=True,
                        help="Collision velocity at Interface 1 (m/s)")
    parser.add_argument("--Vimpact23", type=float, required=True,
                        help="Impact velocity of Plate 2 on Plate 3 (m/s)")

    args = parser.parse_args()

    # Load scalers + model
    scaler_X = joblib.load("scaler_X_v23.pkl")
    scaler_Y = joblib.load("scaler_Y_v23.pkl")
    model    = load_model("weldability_inverse_ann_model_v23.h5")

    # Build input vector
    X_raw = np.array([[
        float(args.material_flag),
        float(args.beta1_deg),
        float(args.T_int1),
        float(args.T_int2),
        float(args.clearance_ok),
        float(args.Vc12),
        float(args.Vimpact23),
    ]], dtype=np.float32)

    # Scale, predict, inverse-scale
    X_s = scaler_X.transform(X_raw)
    Y_pred_s = model.predict(X_s, verbose=0)
    Y_pred = scaler_Y.inverse_transform(Y_pred_s)[0]

    V1, w_top, w_bot, d12, d23, h1, h2, h3 = Y_pred

    # Clamp to physics / design ranges used in the dataset
    V1    = clamp(V1,    600.0, 1800.0)
    w_top = clamp(w_top,   6.0,   14.0)
    w_bot = clamp(w_bot,   0.5,    5.0)
    d12   = clamp(d12,     2.0,    3.0)
    d23   = clamp(d23,     0.5,    4.0)
    h1    = clamp(h1,      0.8,    1.5)
    h2    = clamp(h2,      0.8,    1.5)
    h3    = clamp(h3,      0.8,    1.5)

    print("\nPredicted process / geometry (ANN inverse design, clamped to bounds):")
    print(f"  V1         ≈ {V1:.1f} m/s")
    print(f"  w_top      ≈ {w_top:.2f} mm")
    print(f"  w_bot      ≈ {w_bot:.2f} mm")
    print(f"  d12        ≈ {d12:.2f} mm")
    print(f"  d23        ≈ {d23:.2f} mm")
    print(f"  h1         ≈ {h1:.2f} mm")
    print(f"  h2         ≈ {h2:.2f} mm")
    print(f"  h3         ≈ {h3:.2f} mm")


if __name__ == "__main__":
    main()
