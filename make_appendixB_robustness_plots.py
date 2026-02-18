import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ----------------------------
# USER SETTINGS (match your manuscript conventions)
# ----------------------------
V1_MIN, V1_MAX = 600, 1800
H_MIN, H_MAX   = 0.25, 3.0
WT_MIN, WT_MAX = 4.0, 22.0

# Fixed geometry for thickness scaling plot
FIX_WTOP_THICK = 10.0
FIX_WBOT_THICK = 1.5
FIX_D23_THICK  = 1.5

# Fixed geometry for w_top sweep plot
FIX_H_WTOP  = 1.0
FIX_WBOT_WT = 1.5
FIX_D23_WT  = 1.5

# d12 depends on material flag (per your Table 4)
def d12_for_flag(material_flag: int) -> float:
    return 3.0 if material_flag == 0 else 2.0  # 0=Al-Fe-Fe, 1=Al-Cu-Cu

# ----------------------------
# LOAD MODEL + SCALER + FEATURE ORDER
# ----------------------------
model  = joblib.load("student_ann_model_weldable.pkl")
scaler = joblib.load("scaler_weldable.pkl")
with open("feature_cols_weldable.json", "r") as f:
    feature_cols = json.load(f)

# ----------------------------
# LOAD OOD DATA (optional overlay points)
# ----------------------------
ood = pd.read_csv("extreme_results_final.csv")
ood["weldable_true"] = (ood["failure_mode"].str.lower() == "weldable").astype(int)

# Helper: build model input with correct feature order
def build_X(material_flag, V1, w_top, w_bot, d12, d23, h1, h2, h3):
    row = {
        "material_flag": material_flag,
        "V1_mps": V1,
        "w_top_mm": w_top,
        "w_bot_mm": w_bot,
        "d12_mm": d12,
        "d23_mm": d23,
        "h1_mm": h1,
        "h2_mm": h2,
        "h3_mm": h3,
    }
    # Some datasets use slightly different column names; map if needed:
    # If your feature_cols are exactly as above, you’re fine.
    return np.array([[row[c] for c in feature_cols]], dtype=float)

def predict_prob_grid(X):
    Xs = scaler.transform(X)
    # sklearn MLPClassifier: predict_proba exists
    return model.predict_proba(Xs)[:, 1]

# ----------------------------
# PLOTTERS
# ----------------------------
def plot_V1_vs_h(material_flag: int, out_png: str):
    d12 = d12_for_flag(material_flag)

    v = np.linspace(V1_MIN, V1_MAX, 220)
    h = np.linspace(H_MIN,  H_MAX,  180)
    V, H = np.meshgrid(v, h)

    # Build grid inputs: h1=h2=h3=H
    X_list = []
    for hi in h:
        for vi in v:
            X_list.append(build_X(material_flag, vi, FIX_WTOP_THICK, FIX_WBOT_THICK, d12, FIX_D23_THICK, hi, hi, hi)[0])
    X = np.array(X_list, dtype=float)
    Z = predict_prob_grid(X).reshape(H.shape)

    plt.figure(figsize=(9.5, 7.0), dpi=220)
    cf = plt.contourf(V, H, Z, levels=20)
    plt.colorbar(cf, label="ANN weldability probability")

    # ANN decision boundary (0.5)
    plt.contour(V, H, Z, levels=[0.5], linestyles="--", linewidths=2.0)

    # Overlay OOD points (ground truth) near fixed settings
    tol = 1e-6
    sub = ood[
        (ood["material_flag"] == material_flag) &
        (np.isclose(ood["w_top_mm"], FIX_WTOP_THICK, atol=0.25)) &
        (np.isclose(ood["w_bot_mm"], FIX_WBOT_THICK, atol=0.25)) &
        (np.isclose(ood["d12_mm"], d12, atol=0.25)) &
        (np.isclose(ood["d23_mm"], FIX_D23_THICK, atol=0.25)) &
        (np.isclose(ood["h1_mm"], ood["h2_mm"], atol=tol)) &
        (np.isclose(ood["h1_mm"], ood["h3_mm"], atol=tol))
    ]
    if len(sub) > 0:
        ok = sub[sub["weldable_true"] == 1]
        ng = sub[sub["weldable_true"] == 0]
        plt.scatter(ok["V1_mps"], ok["h1_mm"], s=22, marker="o", edgecolors="k", linewidths=0.3, label="Physical success (OOD)")
        plt.scatter(ng["V1_mps"], ng["h1_mm"], s=18, marker="x", linewidths=0.6, label="Physical failure (OOD)")
        plt.legend(loc="lower right", frameon=True)

    title_mat = "Al–Fe–Fe" if material_flag == 0 else "Al–Cu–Cu"
    plt.title(f"ANN Robustness: V1 vs h (h1=h2=h3) | {title_mat}\n"
              f"Fixed: w_top={FIX_WTOP_THICK}, w_bot={FIX_WBOT_THICK}, d12={d12}, d23={FIX_D23_THICK}")
    plt.xlabel("Flyer velocity V1 (m/s)")
    plt.ylabel("h1=h2=h3 (mm)")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def plot_V1_vs_wtop(material_flag: int, out_png: str):
    d12 = d12_for_flag(material_flag)

    v  = np.linspace(V1_MIN, V1_MAX, 220)
    wt = np.linspace(WT_MIN,  WT_MAX, 180)
    V, WT = np.meshgrid(v, wt)

    X_list = []
    for wti in wt:
        for vi in v:
            X_list.append(build_X(material_flag, vi, wti, FIX_WBOT_WT, d12, FIX_D23_WT,
                                  FIX_H_WTOP, FIX_H_WTOP, FIX_H_WTOP)[0])
    X = np.array(X_list, dtype=float)
    Z = predict_prob_grid(X).reshape(WT.shape)

    plt.figure(figsize=(9.5, 7.0), dpi=220)
    cf = plt.contourf(V, WT, Z, levels=20)
    plt.colorbar(cf, label="ANN weldability probability")
    plt.contour(V, WT, Z, levels=[0.5], linestyles="--", linewidths=2.0)

    # Overlay OOD points near fixed settings
    sub = ood[
        (ood["material_flag"] == material_flag) &
        (np.isclose(ood["w_bot_mm"], FIX_WBOT_WT, atol=0.25)) &
        (np.isclose(ood["d12_mm"], d12, atol=0.25)) &
        (np.isclose(ood["d23_mm"], FIX_D23_WT, atol=0.25)) &
        (np.isclose(ood["h1_mm"], FIX_H_WTOP, atol=1e-6)) &
        (np.isclose(ood["h2_mm"], FIX_H_WTOP, atol=1e-6)) &
        (np.isclose(ood["h3_mm"], FIX_H_WTOP, atol=1e-6))
    ]
    if len(sub) > 0:
        ok = sub[sub["weldable_true"] == 1]
        ng = sub[sub["weldable_true"] == 0]
        plt.scatter(ok["V1_mps"], ok["w_top_mm"], s=22, marker="o", edgecolors="k", linewidths=0.3, label="Physical success (OOD)")
        plt.scatter(ng["V1_mps"], ng["w_top_mm"], s=18, marker="x", linewidths=0.6, label="Physical failure (OOD)")
        plt.legend(loc="lower right", frameon=True)

    title_mat = "Al–Fe–Fe" if material_flag == 0 else "Al–Cu–Cu"
    plt.title(f"ANN Robustness: V1 vs w_top | {title_mat}\n"
              f"Fixed: h1=h2=h3={FIX_H_WTOP}, w_bot={FIX_WBOT_WT}, d12={d12}, d23={FIX_D23_WT}")
    plt.xlabel("Flyer velocity V1 (m/s)")
    plt.ylabel("w_top (mm)")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

# ----------------------------
# RUN (4 plots)
# ----------------------------
plot_V1_vs_h(material_flag=0, out_png="FigB_V1_h_equal_AlFeFe.png")
plot_V1_vs_h(material_flag=1, out_png="FigB_V1_h_equal_AlCuCu.png")
plot_V1_vs_wtop(material_flag=0, out_png="FigB_V1_wtop_AlFeFe.png")
plot_V1_vs_wtop(material_flag=1, out_png="FigB_V1_wtop_AlCuCu.png")

print("Saved: FigB_V1_h_equal_AlFeFe.png, FigB_V1_h_equal_AlCuCu.png, FigB_V1_wtop_AlFeFe.png, FigB_V1_wtop_AlCuCu.png")