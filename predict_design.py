
import argparse
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load saved model and scalers
model = load_model("weldability_inverse_ann_model.h5")
scaler_X = joblib.load("scaler_X.pkl")
scaler_Y = joblib.load("scaler_Y.pkl")

# CLI argument parser
parser = argparse.ArgumentParser(description="Inverse Design ANN for Weldability")
parser.add_argument("--material", type=int, required=True, help="Material flag: 0=Al-Fe-Fe, 1=Al-Cu-Cu")
parser.add_argument("--beta1", type=float, required=True, help="Target impact angle β1 in degrees")
parser.add_argument("--temp1", type=float, required=True, help="Max interface 1 temperature (K)")
parser.add_argument("--temp2", type=float, required=True, help="Max interface 2 temperature (K)")
parser.add_argument("--vc12", type=float, required=True, help="Target collision velocity at interface 1 (m/s)")
parser.add_argument("--vc23", type=float, required=True, help="Target collision velocity at interface 2 (m/s)")
parser.add_argument("--clearance", type=int, choices=[0,1], required=True, help="Clearance OK (1) or Not OK (0)")

args = parser.parse_args()

# Prepare input
x_input = np.array([[args.material, args.beta1, args.temp1, args.temp2, args.clearance, args.vc12, args.vc23]])
x_scaled = scaler_X.transform(x_input)

# Predict and inverse scale
y_scaled = model.predict(x_scaled)
y_pred = scaler_Y.inverse_transform(y_scaled)

# Output
param_names = ["V1 (m/s)", "b (mm)", "d12 (mm)", "d23 (mm)", "h1 (mm)", "h2 (mm)", "h3 (mm)"]
print("\n🔧 Predicted Design Parameters:")
for name, val in zip(param_names, y_pred[0]):
    print(f"{name}: {val:.3f}")
