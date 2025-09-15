# --- VFAW-ANN: Sobol + SHAP + Uncertainty ---
import numpy as np, pandas as pd, joblib, matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from SALib.sample import saltelli
from SALib.analyze import sobol
import shap

# --------------------------
# 0) Load data/model/scalers
# --------------------------
DF = pd.read_csv("final_ann_dataset_material_specific_thickness.csv")
model = load_model("weldability_inverse_ann_model.h5", compile=False)
scaler_X = joblib.load("scaler_X.pkl")
scaler_Y = joblib.load("scaler_Y.pkl")

# Match your training script feature order
X_cols = ["Material_Flag","beta1 (deg)","T_Interface1 (K)","T_Interface2 (K)","Clearance_OK","Vc12 (m/s)","Vc23 (m/s)"]
Y_cols = ["V1 (m/s)","b (mm)","d12 (mm)","d23 (mm)","h1 (mm)","h2 (mm)","h3 (mm)"]

X = DF[X_cols].values

# --------------------------
# 1) Define bounds for Sobol
# --------------------------
# Use physical min/max seen in your dataset (uniform). We’ll run per-material.
bounds = []
for c in X_cols:
    lo, hi = DF[c].min(), DF[c].max()
    bounds.append([float(lo), float(hi)])

problem = {"num_vars": len(X_cols), "names": X_cols, "bounds": bounds}

# Choose output to analyze (index into Y_cols)
OUTPUT = "V1 (m/s)"
j = Y_cols.index(OUTPUT)

# Choose material (run twice: 0=Al–Fe–Fe, 1=Al–Cu–Cu)
fixed_material_flag = 0  # change to 1 for Al–Cu–Cu

# --------------------------
# 2) Saltelli sampling
# --------------------------
N = 1000
S = saltelli.sample(problem, N, calc_second_order=True)

# Enforce categorical/binary values deterministically to avoid nonsense combos
# Material flag (col 0)
S[:, 0] = fixed_material_flag
# Clearance_OK (col 4): fix to 1 for “weldable” subset sensitivity, or set to 0 for failure-side analysis
S[:, 4] = 1

# Scale and predict (model expects scaled X; outputs scaled, then invert to physical)
X_scaled = scaler_X.transform(S)
Y_scaled = model.predict(X_scaled, verbose=0)
Y_phys = scaler_Y.inverse_transform(Y_scaled)
Y_target = Y_phys[:, j]

# --------------------------
# 3) Sobol global sensitivity
# --------------------------
Si = sobol.analyze(problem, Y_target, calc_second_order=True, print_to_console=False)

sobol_df = pd.DataFrame({
    "Parameter": X_cols,
    "S1": Si["S1"],
    "ST": Si["ST"]
}).sort_values("ST", ascending=False)

print("\n== Sobol Indices ({}; material_flag={}) ==".format(OUTPUT, fixed_material_flag))
print(sobol_df.to_string(index=False))

# Plot S1 and ST
plt.figure(figsize=(9,5))
plt.bar(sobol_df["Parameter"], sobol_df["ST"], label="Total (ST)", alpha=0.8)
plt.bar(sobol_df["Parameter"], sobol_df["S1"], label="First-order (S1)", alpha=0.8)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Sobol index")
plt.title(f"Sobol Sensitivity for {OUTPUT} (material_flag={fixed_material_flag})")
plt.legend(); plt.tight_layout()
plt.savefig(f"sobol_{OUTPUT.replace(' ','_')}_mat{fixed_material_flag}.png", dpi=300)

# --------------------------
# 4) SHAP (model interpretability)
# --------------------------
# Use a representative subset as background for speed
rng = np.random.default_rng(42)
idx = rng.choice(len(DF), size=min(4000, len(DF)), replace=False)
X_bg = DF.loc[idx, X_cols].values
X_bg_scaled = scaler_X.transform(X_bg)

# Multi-output model: SHAP values shape = (n, features, outputs)
explainer = shap.Explainer(model, X_bg_scaled)
sv = explainer(X_bg_scaled)

# Summary for the chosen output
plt.figure()
shap.summary_plot(sv.values[:, :, j], X_bg, feature_names=X_cols, show=False)
plt.title(f"SHAP Summary for {OUTPUT} (material_flag={fixed_material_flag})")
plt.tight_layout()
plt.savefig(f"shap_{OUTPUT.replace(' ','_')}_mat{fixed_material_flag}.png", dpi=300)

# --------------------------
# 5) Uncertainty quantification
# --------------------------
# Input-variability (aleatoric): uniform within data bounds
M = 5000
U = np.empty((M, len(X_cols)))
for k, (lo, hi) in enumerate(problem["bounds"]):
    U[:, k] = rng.uniform(lo, hi, size=M)

# Fix categories again
U[:, 0] = fixed_material_flag
U[:, 4] = 1

U_scaled = scaler_X.transform(U)
Yp_scaled = model.predict(U_scaled, verbose=0)
Yp = scaler_Y.inverse_transform(Yp_scaled)
target = Yp[:, j]

mu, sigma = float(np.mean(target)), float(np.std(target))
print(f"\n== UQ (aleatoric) for {OUTPUT}, material_flag={fixed_material_flag} ==")
print(f"mean={mu:.2f}, std={sigma:.2f}")

plt.figure(figsize=(7,4.5))
plt.hist(target, bins=50, density=True, alpha=0.8)
plt.xlabel(OUTPUT); plt.ylabel("PDF")
plt.title(f"UQ (Input Variability) for {OUTPUT} (material_flag={fixed_material_flag})")
plt.tight_layout()
plt.savefig(f"uq_{OUTPUT.replace(' ','_')}_mat{fixed_material_flag}.png", dpi=300)

print("\nSaved: sobol_*.png, shap_*.png, uq_*.png")
